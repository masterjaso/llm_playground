"""Fail-closed treatment matching, using checkpoint metadata as the authority."""

from __future__ import annotations

import copy
import math

from .config import FlashMiniConfig


def shared_config(config):
    values = FlashMiniConfig.from_dict(config).to_dict()
    if values["architecture_version"] >= 3:
        values["ple"].pop("enabled", None)
        values["ple"].pop("offload", None)
    else:
        values.pop("ple")
    values.pop("use_ple")
    return values


def _require_equal(a, b, key):
    if key not in a or key not in b or a[key] is None or b[key] is None:
        raise ValueError(f"comparison missing {key}")
    if a[key] != b[key]:
        raise ValueError(f"comparison {key} mismatch")


def _valid_number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def validate_ple_pair(baseline, candidate, evaluation_manifest_hash):
    """Validate complete checkpoint envelopes before loading weights or evaluating."""
    bc, cc = baseline["config"], candidate["config"]
    if bc.get("use_ple") is not False:
        raise ValueError("baseline must explicitly have PLE OFF")
    if cc.get("use_ple") is not True:
        raise ValueError("candidate must explicitly have PLE ON")
    _require_equal(baseline, candidate, "architecture_version")
    for saved in (baseline, candidate):
        if saved["architecture_version"] != saved["config"].get("architecture_version"):
            raise ValueError("checkpoint/config architecture mismatch")
    if shared_config(bc) != shared_config(cc):
        raise ValueError("shared backbone/config mismatch (including sequence length)")
    b, c = baseline.get("extra", {}), candidate.get("extra", {})
    for key in ("tokens_seen", "real_tokens_seen", "data_manifest_sha256"):
        _require_equal(b, c, key)
    if b["data_manifest_sha256"] != evaluation_manifest_hash:
        raise ValueError("evaluation dataset manifest mismatch")
    bt, ct = b.get("training", {}), c.get("training", {})
    for key in ("seed", "batch_size", "seq_len", "grad_accum", "schedule", "dataset"):
        _require_equal(bt, ct, key)
    if bt["seq_len"] != bc["max_seq_len"]:
        raise ValueError("recorded sequence length differs from config")
    for key in ("tokenizer", "tokenizer_revision", "dataset_revision"):
        _require_equal(bt["dataset"], ct["dataset"], key)
    bm, cm = bt.get("run_metadata", {}), ct.get("run_metadata", {})
    for key in ("source_sha256", "shared_optimizer", "data_contract"):
        _require_equal(bm, cm, key)
    if baseline["architecture_version"] >= 3:
        _require_equal(bm, cm, "execution_policy")
        policy = bm["execution_policy"]
        required = {"router_aux_loss_coef", "precision", "shared_parameter_dtypes", "gradient_clip_max_norm",
                    "clipping_policy", "optimizer_recipe"}
        if not isinstance(policy, dict) or not required.issubset(policy):
            raise ValueError("comparison requires complete execution_policy")
        recipe = policy["optimizer_recipe"]
        if not isinstance(recipe, dict) or not {"dense_family", "table_family", "base_lr",
                "ple_lr_multiplier", "dense_weight_decay", "table_weight_decay", "betas", "eps"}.issubset(recipe):
            raise ValueError("comparison requires complete recorded optimizer recipe")
        if (recipe["dense_family"] != "AdamW" or recipe["table_family"] not in ("AdamW", "SparseAdam")
                or any(not _valid_number(recipe[key], positive=True) for key in ("base_lr", "ple_lr_multiplier", "eps"))
                or any(not _valid_number(recipe[key]) for key in ("dense_weight_decay", "table_weight_decay"))):
            raise ValueError("comparison optimizer recipe contains invalid values")
        betas = recipe["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
                not _valid_number(beta) or beta >= 1 for beta in betas):
            raise ValueError("comparison optimizer recipe has invalid betas")
        if (not _valid_number(policy["router_aux_loss_coef"])
                or not _valid_number(policy["gradient_clip_max_norm"], positive=True)
                or policy["precision"] not in ("no_autocast", "cuda_bfloat16_autocast")
                or policy["clipping_policy"] != "independent_shared_ple_dense_ple_sparse_v3"
                or not isinstance(policy["shared_parameter_dtypes"], list)
                or not policy["shared_parameter_dtypes"]
                or any(dtype not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")
                       for dtype in policy["shared_parameter_dtypes"])):
            raise ValueError("comparison execution_policy contains invalid values")
    return {
        "comparison_type": "B_vs_C_PLE_treatment",
        "ablation_type": "within_model_memory_reliance_diagnostic_not_B_baseline",
        "training_seed": bt["seed"],
        "matched_seed_count": 1,
        "seed_scope": "paired_single_seed_screening",
        "confirmation_policy": "at_least_3_full_matched_seeds_for_small_effects",
        "uncertainty_scope": "fixed_holdout_blocks_not_training_seed_variance",
        "long_context_validated": False,
        "final_go_eligible": False,
        "data_contract": copy.deepcopy(bm["data_contract"]),
    }


def _treatment_signature(config: dict) -> tuple[str, bool]:
    """Return the (mixer, ple) treatment signature for a v3 config.

    A: full attention (gdn_per_attention == 0), PLE off.
    B: hybrid 3 GDN : 1 full attention (gdn_per_attention == 3), PLE off.
    C: identical to B with PLE on.
    """
    if config.get("architecture_version", 0) < 3:
        raise ValueError("generic comparator requires architecture version 3")
    gdn = config.get("gdn_per_attention")
    if gdn not in (0, 3):
        raise ValueError(f"unexpected gdn_per_attention {gdn!r}; expected 0 (A) or 3 (B/C)")
    return ("full_attention" if gdn == 0 else "hybrid_gdn", bool(config.get("use_ple")))


def _expected_difference(sig_a: tuple[str, bool], sig_b: tuple[str, bool]) -> str | None:
    """Return the allowed treatment difference label, or None if unrelated."""
    if sig_a == sig_b:
        return None
    # A vs B: mixer differs, PLE off on both.
    if sig_a[1] is False and sig_b[1] is False and sig_a[0] != sig_b[0]:
        return "mixer_treatment"
    # B vs C: same mixer, PLE differs.
    if sig_a[0] == sig_b[0] and sig_a[1] != sig_b[1]:
        return "ple_treatment"
    # A vs C: both mixer and PLE differ (the intended A/C contrast).
    if sig_a[1] is False and sig_b[1] is True and sig_a[0] != sig_b[0]:
        return "mixer_and_ple"
    return None


def validate_generic_pair(baseline, candidate, evaluation_manifest_hash):
    """Fail-closed generic A/B/C treatment matching.

    Verifies that the two checkpoints share every shared control (data bytes,
    tokenizer provenance, dataset revision, seed, token count, batch size,
    sequence length, schedule, optimizer recipe, precision, clipping policy,
    source/freeze SHA, execution environment, reserved evaluation slice) and
    that the only difference is the intended treatment.

    Returns a control record describing the allowed difference. Raises
    ``ValueError`` on any unrelated config difference.
    """
    bc, cc = baseline["config"], candidate["config"]
    if bc.get("architecture_version") != cc.get("architecture_version"):
        raise ValueError("comparison architecture_version mismatch")
    sig_b, sig_c = _treatment_signature(bc), _treatment_signature(cc)
    difference = _expected_difference(sig_b, sig_c)
    if difference is None:
        raise ValueError(
            "comparison has no valid treatment difference; "
            f"baseline={sig_b}, candidate={sig_c}"
        )
    # The shared backbone must be identical once the treatment knobs are
    # removed. ``gdn_per_attention`` and ``attention_layers`` are the mixer
    # treatment; everything else must match exactly.
    shared_b = shared_config(bc)
    shared_c = shared_config(cc)
    shared_b["gdn_per_attention"] = None
    shared_c["gdn_per_attention"] = None
    shared_b["attention_layers"] = None
    shared_c["attention_layers"] = None
    if shared_b != shared_c:
        raise ValueError("shared backbone/config mismatch beyond the treatment")
    b, c = baseline.get("extra", {}), candidate.get("extra", {})
    for key in ("tokens_seen", "real_tokens_seen", "data_manifest_sha256"):
        _require_equal(b, c, key)
    if b["data_manifest_sha256"] != evaluation_manifest_hash:
        raise ValueError("evaluation dataset manifest mismatch")
    bt, ct = b.get("training", {}), c.get("training", {})
    for key in ("seed", "batch_size", "seq_len", "grad_accum", "schedule", "dataset"):
        _require_equal(bt, ct, key)
    if bt["seq_len"] != bc["max_seq_len"]:
        raise ValueError("recorded sequence length differs from config")
    for key in ("tokenizer", "tokenizer_revision", "dataset_revision"):
        _require_equal(bt["dataset"], ct["dataset"], key)
    bm, cm = bt.get("run_metadata", {}), ct.get("run_metadata", {})
    for key in ("source_sha256", "shared_optimizer", "data_contract"):
        _require_equal(bm, cm, key)
    # Execution environment must match (fingerprint).
    fp_b, fp_c = bm.get("execution_fingerprint"), cm.get("execution_fingerprint")
    if fp_b is None or fp_c is None:
        raise ValueError("comparison requires recorded execution fingerprints")
    if fp_b.get("fingerprint_sha256") != fp_c.get("fingerprint_sha256"):
        raise ValueError("comparison execution environment fingerprint mismatch")
    if bc.get("architecture_version") >= 3:
        _require_equal(bm, cm, "execution_policy")
        policy = bm["execution_policy"]
        required = {"router_aux_loss_coef", "precision", "shared_parameter_dtypes",
                    "gradient_clip_max_norm", "clipping_policy", "optimizer_recipe"}
        if not isinstance(policy, dict) or not required.issubset(policy):
            raise ValueError("comparison requires complete execution_policy")
    return {
        "comparison_type": difference,
        "baseline_treatment": sig_b,
        "candidate_treatment": sig_c,
        "training_seed": bt["seed"],
        "matched_seed_count": 1,
        "seed_scope": "paired_single_seed_screening",
        "confirmation_policy": "at_least_3_full_matched_seeds_for_small_effects",
        "uncertainty_scope": "fixed_holdout_blocks_not_training_seed_variance",
        "long_context_validated": False,
        "final_go_eligible": False,
        "data_contract": copy.deepcopy(bm["data_contract"]),
    }
