"""Run the bounded, resumable p16/top6 50%-reduction development workstream.

The runner deliberately keeps the scientific phases explicit:

``validate-inputs`` -> ``rebaseline`` -> ``smoke`` -> ``capacity`` ->
``learned`` -> ``robustness``.  FIT-DEV is never used to alter a recipe.  A
completed receipt is write-once and every phase stops on its declared gates.
Protected tiers and exact LM evaluation are not opened by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dense2moe.config import MoEProfile
from dense2moe.evaluation.receipts import (
    build_structural_generalization_receipt,
    validate_structural_receipt,
    write_immutable_receipt,
)
from dense2moe.evaluation.replay import iter_paired_activation_batches, iter_paired_activation_shards, sha256_file
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import partition_indices
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import train_torch_layer


EPIC_ID = "d2m-p16-top6-50-selection"
DEFAULT_RUN_ROOT = REPO_ROOT / ".nsp" / "artifacts" / "runs" / EPIC_ID
DEFAULT_PREREG = DEFAULT_RUN_ROOT / "preregistration.json"
DEFAULT_DEVICE = "cuda:1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _event(run_root: Path, event: str, **fields: Any) -> None:
    path = run_root / "logs" / "events.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "run_id": EPIC_ID, "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_prereg(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "IMMUTABLE_PREREGISTRATION":
        raise ValueError("preregistration is not immutable")
    if payload.get("run_id") != EPIC_ID:
        raise ValueError("preregistration run identity mismatch")
    candidate = payload.get("candidate", {})
    expected = {
        "source_candidate": "ht-2e2d35188fc77dea",
        "routed_experts": 16,
        "top_k": 6,
        "dense_width": 17408,
        "shared_width": 3072,
        "expert_width": 896,
        "residual_width": 256,
        "residual_scope": "static",
        "fallback_mode": "none",
        "active_width": 8704,
        "active_ffn_reduction": 0.5,
    }
    for key, value in expected.items():
        if candidate.get(key) != value:
            raise ValueError(f"preregistered candidate mismatch for {key}: {candidate.get(key)!r} != {value!r}")
    return payload


def _runtime_identity(prereg: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch

    runtime = dict(prereg["runtime"])
    lock_path = _resolve(str(runtime["lock_path"]))
    file_hash = sha256_file(lock_path)
    if file_hash.lower() != str(runtime["lock_sha256"]).lower():
        raise RuntimeError(f"runtime lock file hash mismatch: {file_hash} != {runtime['lock_sha256']}")
    if str(device) != str(runtime["device"]):
        raise RuntimeError(f"scientific device must be {runtime['device']}, got {device}")
    if os.name != "nt":
        raise RuntimeError("native Windows runtime is required for this workstream")
    executable = Path(sys.executable).resolve()
    expected_executable = _resolve(str(runtime["python"]))
    if executable != expected_executable.resolve():
        raise RuntimeError(f"locked Python mismatch: {executable} != {expected_executable}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("approved CUDA runtime with two GPUs is required")
    current = current_git_commit()
    if str(runtime.get("code_commit_at_registration", "")) != current:
        raise RuntimeError("code commit differs from preregistered runtime identity")
    lock_payload = _load_json(lock_path)
    return {"path": str(lock_path), "sha256": file_hash, "device": device, "lock": lock_payload, "code_commit": current}


def _manifest_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError(f"activation manifest schema must be 2: {path}")
    if payload.get("status") not in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED"}:
        raise ValueError(f"activation manifest is not complete: {path}")
    if not bool(payload.get("quality_gate_eligible", False)):
        raise ValueError(f"activation manifest is not quality-gate eligible: {path}")
    if str(payload.get("evidence_class", "")).lower().find("protected") >= 0:
        raise ValueError(f"protected-tier activation manifest is forbidden: {path}")
    return payload


def _validate_inputs(prereg: Mapping[str, Any], run_root: Path, *, device: str) -> dict[str, Any]:
    inputs = prereg["inputs"]
    train_path = _resolve(str(inputs["fit_train_manifest"]))
    dev_path = _resolve(str(inputs["fit_dev_manifest"]))
    train = _manifest_payload(train_path)
    dev = _manifest_payload(dev_path)
    expected = ((train, "FIT-TRAIN", train_path, inputs["fit_train_sha256"], inputs["fit_train_rows"]), (dev, "FIT-DEV", dev_path, inputs["fit_dev_sha256"], inputs["fit_dev_rows"]))
    groups: list[set[str]] = []
    summaries: list[dict[str, Any]] = []
    for payload, split, path, expected_hash, expected_count in expected:
        if payload.get("split") != split:
            raise ValueError(f"split identity mismatch for {path}: {payload.get('split')}")
        if sha256_file(path).lower() != str(expected_hash).lower():
            raise ValueError(f"manifest hash mismatch for {path}")
        for key in ("source_revision", "dataset_hash", "tokenizer_hash"):
            if str(payload.get(key, "")) != str(inputs[key if key != "source_revision" else "source_revision"]):
                raise ValueError(f"{key} mismatch for {path}")
        if int(payload.get("count", 0)) != int(expected_count):
            raise ValueError(f"retained row count mismatch for {path}")
        groups_for_split: set[str] = set()
        finite = True
        hidden_sizes: set[int] = set()
        target_shapes: set[tuple[int, ...]] = set()
        loaded_count = 0
        for batch in iter_paired_activation_shards(path, expected_split=split, repo_root=REPO_ROOT):
            import torch

            loaded_count += int(batch.inputs.shape[0])
            hidden_sizes.add(int(batch.inputs.shape[-1]))
            target_shapes.add(tuple(int(value) for value in batch.targets.shape))
            finite = finite and bool(torch.isfinite(batch.inputs).all().item()) and bool(torch.isfinite(batch.targets).all().item())
            groups_for_split.update(str(item["independent_group"]) for item in batch.metadata)
        if loaded_count != int(expected_count) or not finite or hidden_sizes != {int(inputs["hidden_size"])}:
            raise ValueError(f"finite/geometry/retained-row validation failed for {path}")
        if any(len(shape) != 2 or shape[1] != int(inputs["hidden_size"]) for shape in target_shapes):
            raise ValueError(f"target geometry mismatch for {path}")
        groups.append(groups_for_split)
        summaries.append({"path": str(path), "split": split, "sha256": sha256_file(path), "count": loaded_count, "hidden_size": sorted(hidden_sizes), "finite": finite, "group_count": len(groups_for_split), "source_revision": payload.get("source_revision"), "dataset_hash": payload.get("dataset_hash"), "tokenizer_hash": payload.get("tokenizer_hash")})
    overlap = sorted(groups[0] & groups[1])
    if overlap:
        raise ValueError(f"FIT-TRAIN/FIT-DEV group overlap: {overlap[:3]}")
    runtime = _runtime_identity(prereg, device)
    result = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-input-validation", "status": "INPUTS_VALID", "run_id": EPIC_ID, "runtime": runtime, "manifests": summaries, "group_overlap_count": 0, "protected_tier_membership": False, "prediction_result": {"classifier": "expected-match", "reason": "manifest and runtime identities match the frozen preregistration"}}
    _write_once(run_root / "results" / "input-validation.json", result)
    _event(run_root, "inputs_validated", manifests=summaries)
    return result


def _rebaseline(prereg: Mapping[str, Any], run_root: Path, *, device: str) -> dict[str, Any]:
    _runtime_identity(prereg, device)
    result_path = run_root / "results" / "per-config" / "ht-2e2d35188fc77dea.json"
    if not result_path.exists():
        raise RuntimeError("run the existing V2 frontier command before rebaseline verification")
    result = _load_json(result_path)
    expected = prereg["oracle_rebaseline"]["expected_fit_dev"]
    fit_dev = result.get("fit_dev", {})
    checks = {
        "cosine": (fit_dev.get("cosine_similarity"), expected["cosine"]),
        "normalized_mse": (fit_dev.get("normalized_mse"), expected["normalized_mse"]),
        "target_relative_norm_error": (fit_dev.get("target_relative_norm_error"), expected["target_relative_norm_error"]),
        "mean_prediction_target_norm_ratio": (fit_dev.get("mean_prediction_to_target_norm_ratio"), expected["mean_prediction_target_norm_ratio"]),
        "p95_abs_relative_norm_error": (fit_dev.get("p95_abs_relative_norm_error"), expected["p95_abs_relative_norm_error"]),
        "q4_cosine": (fit_dev.get("target_norm_buckets", {}).get("q4", {}).get("cosine_similarity"), expected["q4_cosine"]),
        "q4_normalized_mse": (fit_dev.get("target_norm_buckets", {}).get("q4", {}).get("normalized_mse"), expected["q4_normalized_mse"]),
        "oracle_load_cv": (fit_dev.get("oracle_load_cv"), expected["oracle_load_cv"]),
    }
    for name, (observed, target) in checks.items():
        if observed is None or not math.isclose(float(observed), float(target), rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(f"oracle rebaseline mismatch: {name}={observed} expected {target}")
    if int(fit_dev.get("oracle_dead_expert_count", -1)) != 0 or any(int(fit_dev.get(key, -1)) != 0 for key in ("dropped_token_count", "invalid_token_count", "non_finite_token_count")):
        raise RuntimeError("oracle rebaseline safety metrics are not zero")
    compute = result.get("compute", {})
    widths = compute.get("active_width_summary", {}).get("fit_dev", {})
    if any(float(widths.get(key, -1)) != 8704.0 for key in ("mean", "p50", "p95", "max")) or float(compute.get("average_reduction", {}).get("fit_dev", -1)) != 0.5:
        raise RuntimeError("oracle rebaseline compute accounting mismatch")
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-selection-rebaseline-verification", "status": "REBASELINE_GREEN", "run_id": EPIC_ID, "oracle_only": True, "promotion_eligible": False, "source_candidate": prereg["candidate"]["source_candidate"], "v2_result": {"path": str(result_path.relative_to(run_root)), "sha256": sha256_file(result_path)}, "fit_dev": fit_dev, "compute": compute, "prediction_result": {"classifier": "expected-match", "reason": "current V2 receipt reproduces the frozen oracle result"}}
    _write_once(run_root / "results" / "rebaseline-verification.json", payload)
    _event(run_root, "oracle_rebaseline_verified", result=str(result_path))
    return payload


def _profile(prereg: Mapping[str, Any]) -> MoEProfile:
    candidate = prereg["candidate"]
    return MoEProfile(name="d2m_p16_top6_50", hidden_size=5120, dense_intermediate_size=int(candidate["dense_width"]), num_hidden_layers=64, routed_experts=int(candidate["routed_experts"]), expert_intermediate_size=int(candidate["expert_width"]), shared_intermediate_size=int(candidate["shared_width"]), top_k=int(candidate["top_k"]), revision=str(prereg["inputs"]["source_revision"]), routing_mode="independent_positive")


def _partition_artifact(prereg: Mapping[str, Any], run_root: Path) -> Path:
    candidate = prereg["candidate"]
    plan = partition_indices(int(candidate["dense_width"]), int(candidate["routed_experts"]), int(candidate["expert_width"]), int(candidate["shared_width"]))
    payload = plan.as_dict()
    payload["initial_expert_scales"] = [1.0] * int(candidate["routed_experts"])
    path = run_root / "planning" / "partition-p16-top6-50.json"
    _write_once(path, payload)
    return path


def _load_dense_layer(source_dir: Path) -> tuple[Any, Any, Any]:
    import torch
    from safetensors import safe_open

    index = _load_json(source_dir / "model.safetensors.index.json")
    prefix = "model.language_model.layers.0.mlp."
    names = {key[len(prefix):]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    if set(names) != {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
        raise ValueError("source layer-0 MLP inventory mismatch")
    values: dict[str, Any] = {}
    for short, shard in names.items():
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **kwargs) as handle:
            values[short] = handle.get_tensor(prefix + short).float()
    return values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"]


def _reload_equivalence(prereg: Mapping[str, Any], checkpoint_dir: Path, *, device: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    candidate = prereg["candidate"]
    source = _resolve(str(prereg["inputs"]["source_snapshot"]))
    gate, up, down = _load_dense_layer(source)
    metadata = _load_json(checkpoint_dir / "layer-0000.json")
    config = metadata.get("training_config", {})
    model_kwargs = {
        "routed_experts": int(candidate["routed_experts"]),
        "shared_intermediate_size": int(candidate["shared_width"]),
        "top_k": int(candidate["top_k"]),
        "routing_mode": "independent_positive",
        "router_hidden_size": config.get("router_hidden_size"),
        "router_feature_mode": str(config.get("router_feature_mode", "none")),
        "residual_intermediate_size": int(candidate["residual_width"]),
        "residual_scope": str(candidate["residual_scope"]),
        "fallback_mode": "none",
        "fallback_rate_budget": 0.0,
    }
    first = TorchQwen35SwiGLUMoE.from_dense(gate, up, down, **model_kwargs)
    second = TorchQwen35SwiGLUMoE.from_dense(gate, up, down, **model_kwargs)
    raw = load_file(str(checkpoint_dir / "layer-0000.safetensors"), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise RuntimeError("checkpoint namespace is not layer-0 compatible")
    first.load_state_dict(state, strict=True)
    second.load_state_dict(state, strict=True)
    first.to(device).eval()
    second.to(device).eval()
    dev_path = _resolve(str(prereg["inputs"]["fit_dev_manifest"]))
    batch = next(iter_paired_activation_batches(dev_path, expected_split="FIT-DEV", repo_root=REPO_ROOT, batch_tokens=4))
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32, device=device)
    with torch.inference_mode():
        first_output, first_info = first(inputs, return_router=True)
        second_output, second_info = second(inputs, return_router=True)
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_info["indices"], second_info["indices"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_info["weights"], second_info["weights"], rtol=0.0, atol=0.0)
    return {"outputs_equivalent": True, "routing_equivalent": True, "token_count": int(inputs.shape[0]), "checkpoint_sha256": sha256_file(checkpoint_dir / "layer-0000.safetensors"), "residual_executed": bool(first_info["residual_executed"]), "active_width": int(first_info["active_intermediate_width"]), "teacher_dependent_inference": False}


def _gate_failures(metrics: Mapping[str, Any], prereg: Mapping[str, Any], *, require_learned: bool = True) -> list[str]:
    gates = prereg["metric_gates"]
    q4 = metrics.get("target_norm_buckets", {}).get("q4", {}) if isinstance(metrics.get("target_norm_buckets"), Mapping) else {}
    learned_cv = metrics.get("learned_load_cv", metrics.get("load_cv"))
    dead = metrics.get("dead_expert_count", metrics.get("dead_experts"))
    values = {
        "global_cosine": (metrics.get("cosine_similarity", metrics.get("cosine")), float(gates["global_cosine_min"]), lambda a, b: a >= b),
        "global_normalized_mse": (metrics.get("normalized_mse"), float(gates["global_normalized_mse_max"]), lambda a, b: a <= b),
        "target_relative_norm_error": (metrics.get("target_relative_norm_error"), float(gates["target_relative_norm_error_max"]), lambda a, b: a <= b),
        "mean_prediction_target_norm_ratio": (metrics.get("mean_prediction_to_target_norm_ratio"), float(gates["mean_prediction_target_norm_ratio_min"]), lambda a, b: a >= b),
        "p95_abs_relative_norm_error": (metrics.get("p95_abs_relative_norm_error"), float(gates["p95_abs_relative_norm_error_max"]), lambda a, b: a <= b),
        "load_cv": (learned_cv, float(gates["learned_load_cv_max"]), lambda a, b: a <= b),
        "dead_experts": (dead, int(gates["dead_experts_max"]), lambda a, b: int(a) <= int(b)),
        "dropped_tokens": (metrics.get("dropped_token_count"), int(gates["dropped_tokens_max"]), lambda a, b: int(a) <= int(b)),
        "invalid_tokens": (metrics.get("invalid_token_count"), int(gates["invalid_tokens_max"]), lambda a, b: int(a) <= int(b)),
        "nonfinite_tokens": (metrics.get("non_finite_token_count"), int(gates["nonfinite_tokens_max"]), lambda a, b: int(a) <= int(b)),
        "q4_cosine": (q4.get("cosine_similarity"), float(gates["q4_cosine_min"]), lambda a, b: a >= b),
        "q4_normalized_mse": (q4.get("normalized_mse"), float(gates["q4_normalized_mse_max"]), lambda a, b: a <= b),
        "mean_active_width": (metrics.get("active_intermediate_width_mean"), float(gates["mean_active_width_max"]), lambda a, b: a <= b),
        "p50_active_width": (metrics.get("active_intermediate_width_p50"), float(gates["p50_active_width_max"]), lambda a, b: a <= b),
        "p95_active_width": (metrics.get("active_intermediate_width_p95"), float(gates["p95_active_width_max"]), lambda a, b: a <= b),
        "maximum_active_width": (metrics.get("active_intermediate_width_max"), float(gates["maximum_active_width_max"]), lambda a, b: a <= b),
        "average_ffn_reduction": (metrics.get("average_ffn_reduction"), float(gates["average_ffn_reduction_min"]), lambda a, b: a >= b),
    }
    failures: list[str] = []
    for name, (observed, threshold, predicate) in values.items():
        if observed is None:
            failures.append(f"{name}=missing")
            continue
        try:
            finite = math.isfinite(float(observed))
        except (TypeError, ValueError):
            finite = False
        if not finite or not predicate(float(observed), threshold):
            failures.append(f"{name}={observed} threshold={threshold}")
    if bool(metrics.get("dense_fallback_used", False)):
        failures.append("dense_fallback_used=true")
    if require_learned and metrics.get("learned_router_status") not in {None, "COMPUTED"}:
        failures.append(f"learned_router_status={metrics.get('learned_router_status')}")
    return failures


def _recipe_receipt(prereg: Mapping[str, Any], run_root: Path, *, recipe_id: str, seed: int, result: Mapping[str, Any], evidence_class: str, oracle_only: bool, runtime: Mapping[str, Any]) -> dict[str, Any]:
    fit_train = dict(result.get("final_selection") or result.get("final_fit") or {})
    fit_dev = dict(result.get("validation_b_metrics") or {})
    if fit_dev.get("normalized_mse") is None:
        fit_dev = dict(result.get("final_selection") or {})
    candidate = dict(prereg["candidate"])
    candidate.update({"recipe_id": recipe_id, "seed": seed, "oracle_only": oracle_only, "promotion_eligible": False, "checkpoint": result.get("tensor_file"), "teacher_dependent_inference": False})
    fit_train_identity = {"manifest": str(prereg["inputs"]["fit_train_manifest"]), "manifest_sha256": str(prereg["inputs"]["fit_train_sha256"]), "split": "FIT-TRAIN", "dataset_hash": str(prereg["inputs"]["dataset_hash"])}
    fit_dev_identity = {"manifest": str(prereg["inputs"]["fit_dev_manifest"]), "manifest_sha256": str(prereg["inputs"]["fit_dev_sha256"]), "split": "FIT-DEV", "dataset_hash": str(prereg["inputs"]["dataset_hash"])}
    train_cos = fit_train.get("cosine_similarity", fit_train.get("cosine"))
    dev_cos = fit_dev.get("cosine_similarity", fit_dev.get("cosine"))
    generalization = {"classification": "TRAINING_OVERFIT_SIGNAL" if train_cos is not None and dev_cos is not None and float(train_cos) - float(dev_cos) > 0.02 else "NO_MATERIAL_GAP", "cosine_gap": (float(train_cos) - float(dev_cos)) if train_cos is not None and dev_cos is not None else None, "normalized_mse_gap": (float(fit_dev.get("normalized_mse")) - float(fit_train.get("normalized_mse"))) if fit_dev.get("normalized_mse") is not None and fit_train.get("normalized_mse") is not None else None}
    payload = build_structural_generalization_receipt(source_model={"family": "Qwen3.5", "revision": prereg["inputs"]["source_revision"], "path": prereg["inputs"]["source_snapshot"]}, layer=0, candidate=candidate, fit_train=fit_train, fit_dev=fit_dev, generalization=generalization, evidence_class=evidence_class, source_receipt_lineage={"run_id": EPIC_ID, "preregistration_sha256": sha256_file(run_root / "preregistration.json"), "oracle_only": oracle_only}, code_science_identity={"code_commit": current_git_commit(), "runner": "scripts/run_p16_top6_50_selection.py"}, runtime_lock_identity=runtime, lm_evaluation_eligibility={"status": "NOT_COMPUTED", "reason": "protected exact LM evaluation is sealed"}, fit_train_identity=fit_train_identity, fit_dev_identity=fit_dev_identity)
    receipt_path = run_root / "receipts" / recipe_id / f"seed-{seed}.json"
    receipt = write_immutable_receipt(payload, receipt_path)
    validation = validate_structural_receipt(receipt)
    if not validation["valid"]:
        raise RuntimeError(f"invalid V2 receipt for {recipe_id}/seed-{seed}: {validation}")
    return {"path": str(receipt_path), "sha256": receipt["receipt_sha256"], "compatibility": validation["compatibility"]}


def _stage_schedule(raw: Sequence[Mapping[str, Any]], *, learned: bool = False, quantile_balanced: bool = False) -> list[dict[str, Any]]:
    if not learned:
        return [dict(stage) for stage in raw]
    return [
        {"name": "learned_oracle_label_warm_start", "epochs": 2, "train_shared": False, "train_experts": False, "train_residual": False, "train_scales": False, "train_selection_router": True, "train_amplitude_router": True, "use_oracle_targets": True, "oracle_target_mode": "residual_correlation", "oracle_loss_mode": "repeated_cross_entropy", "oracle_amplitude_mode": "teacher_forced", "teacher_forcing_ratio": 1.0, "quantile_balanced": False, "loss_coefficients": {"mse": 1.0, "cosine": 0.05, "load_balance": 0.05, "hard_load_balance": 0.025, "router_z_loss": 0.001, "oracle": 0.1, "oracle_amplitude": 0.05}},
        {"name": "learned_joint_structural_refinement", "epochs": 2, "train_shared": True, "train_experts": True, "train_residual": True, "train_scales": True, "train_selection_router": True, "train_amplitude_router": True, "use_oracle_targets": False, "quantile_balanced": bool(quantile_balanced), "loss_coefficients": {"mse": 1.0, "cosine": 0.05, "load_balance": 0.05, "hard_load_balance": 0.025, "router_z_loss": 0.001}},
    ]


def _inject_policy_learning_rates(stages: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply the immutable training-policy router rate without changing stages."""

    policy = prereg.get("training_policy", {})
    router_rate = float(policy.get("router_learning_rate", policy.get("base_learning_rate", 0.0)))
    if router_rate <= 0.0:
        raise ValueError("preregistered router_learning_rate must be positive")
    output: list[dict[str, Any]] = []
    for stage in stages:
        item = dict(stage)
        rates = {str(name): float(value) for name, value in (item.get("learning_rates") or {}).items()}
        rates.setdefault("selection_router", router_rate)
        rates.setdefault("amplitude_router", router_rate)
        item["learning_rates"] = rates
        output.append(item)
    return output


def _run_recipe(prereg: Mapping[str, Any], run_root: Path, *, recipe: Mapping[str, Any], seed: int, device: str, initial_checkpoint_dir: Path | None = None, learned: bool = False) -> dict[str, Any]:
    recipe_id = str(recipe["id"])
    result_path = run_root / "results" / "recipes" / recipe_id / f"seed-{seed}.json"
    if result_path.exists():
        return _load_json(result_path)
    runtime = _runtime_identity(prereg, device)
    candidate = prereg["candidate"]
    train_manifest = _resolve(str(prereg["inputs"]["fit_train_manifest"]))
    dev_manifest = _resolve(str(prereg["inputs"]["fit_dev_manifest"]))
    partition = _partition_artifact(prereg, run_root)
    profile = _profile(prereg)
    if learned:
        raw_stages = _stage_schedule([], learned=True, quantile_balanced=bool(recipe.get("quantile_balanced", False)))
        router_hidden_size = 128
        router_feature_mode = "shared_output"
    else:
        raw_stages = _stage_schedule(recipe["stages"], learned=False)
        router_hidden_size = None
        router_feature_mode = "none"
    raw_stages = _inject_policy_learning_rates(raw_stages, prereg)
    output_dir = run_root / "checkpoints" / recipe_id / f"seed-{seed}"
    _event(run_root, "recipe_started", recipe=recipe_id, seed=seed, learned=learned)
    result = train_torch_layer(source_dir=_resolve(str(prereg["inputs"]["source_snapshot"])), activation_manifest=train_manifest, selection_manifest=train_manifest, selection_split="FIT-TRAIN", validation_b_manifest=dev_manifest, validation_b_split="FIT-DEV", output_dir=output_dir, layer=0, profile=profile, partition_path=partition, epochs=2, microbatch=int(prereg["training_policy"]["microbatch"]), learning_rate=float(prereg["training_policy"]["base_learning_rate"]), device=device, seed=int(seed), source_revision=str(prereg["inputs"]["source_revision"]), code_commit=current_git_commit(), stage_schedule=raw_stages, evaluate_holdout=False, initial_checkpoint_dir=initial_checkpoint_dir, router_hidden_size=router_hidden_size, router_feature_mode=router_feature_mode, residual_intermediate_size=int(candidate["residual_width"]), residual_scope=str(candidate["residual_scope"]), fallback_mode="none", fallback_rate_budget=0.0)
    fit_train = result.get("final_selection") or result.get("final_fit") or {}
    fit_dev = result.get("validation_b_metrics") or {}
    reload_result = _reload_equivalence(prereg, output_dir, device=device)
    failures = _gate_failures(fit_dev, prereg, require_learned=True)
    result_payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-selection-recipe-result", "status": "GATE_GREEN" if not failures else "GATE_FAILED", "run_id": EPIC_ID, "recipe_id": recipe_id, "seed": seed, "learned": learned, "oracle_only": not learned, "promotion_eligible": False, "fit_train": fit_train, "fit_dev": fit_dev, "gate_failures": failures, "reload": reload_result, "training": result.get("training_config", {}), "checkpoint": {"directory": str(output_dir), "metadata": str(output_dir / "layer-0000.json"), "tensor": str(output_dir / "layer-0000.safetensors"), "tensor_sha256": sha256_file(output_dir / "layer-0000.safetensors")}, "runtime": runtime, "prediction_result": {"classifier": "expected-match" if not failures else "counterexample", "reason": "all declared gates cleared" if not failures else "one or more declared absolute gates failed", "failed_gates": failures}}
    receipt = _recipe_receipt(prereg, run_root, recipe_id=recipe_id, seed=seed, result={**result, "final_selection": fit_train, "validation_b_metrics": fit_dev, "tensor_file": str(output_dir / "layer-0000.safetensors")}, evidence_class="trained-capacity-oracle-supervision" if not learned else "learned-router", oracle_only=not learned, runtime=runtime)
    result_payload["receipt"] = receipt
    _write_once(result_path, result_payload)
    _event(run_root, "recipe_completed", recipe=recipe_id, seed=seed, status=result_payload["status"], failures=failures)
    return result_payload


def _capacity_phase(prereg: Mapping[str, Any], run_root: Path, *, device: str) -> dict[str, Any]:
    recipes = prereg["capacity_recipes"]
    rows = [_run_recipe(prereg, run_root, recipe=recipe, seed=17, device=device, learned=False) for recipe in recipes]
    passing = [row for row in rows if row.get("status") == "GATE_GREEN"]
    if not passing:
        decision = "ORACLE_TRAINED_CAPACITY_FAILED"
        selected = None
    else:
        def rank(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
            metric = row["fit_train"]
            return (float(metric.get("cosine_similarity", metric.get("cosine", -1.0))), -float(metric.get("normalized_mse", 1e9)), -float(metric.get("learned_load_cv", metric.get("load_cv", 1e9))), str(row["recipe_id"]))
        selected = sorted(passing, key=rank, reverse=True)[0]
        decision = "ORACLE_TRAINED_CAPACITY_GREEN"
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-oracle-trained-capacity", "status": decision, "run_id": EPIC_ID, "oracle_only": True, "promotion_eligible": False, "recipes": rows, "selected_recipe": selected["recipe_id"] if selected else None, "stop_rule": "stop before learned-router training when no capacity recipe clears every absolute gate", "prediction_result": {"classifier": "expected-match" if selected else "counterexample", "reason": "at least one actual-capacity recipe cleared" if selected else "neither actual-capacity recipe cleared FIT-DEV absolute gates"}}
    _write_once(run_root / "results" / "oracle-trained-capacity.json", payload)
    _event(run_root, "capacity_phase_completed", status=decision, selected_recipe=payload["selected_recipe"])
    return payload


def _learned_phase(prereg: Mapping[str, Any], run_root: Path, capacity: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    selected_id = capacity.get("selected_recipe")
    if not selected_id:
        raise RuntimeError("learned-router phase is not authorized when trained capacity failed")
    capacity_row = next(row for row in capacity["recipes"] if row["recipe_id"] == selected_id)
    initial_checkpoint_dir = Path(capacity_row["checkpoint"]["directory"])
    rows = [_run_recipe(prereg, run_root, recipe=recipe, seed=17, device=device, initial_checkpoint_dir=initial_checkpoint_dir, learned=True) for recipe in prereg["learned_router_recipes"]]
    passing = [row for row in rows if row.get("status") == "GATE_GREEN"]
    if not passing:
        decision = "LEARNED_ROUTER_FAILED"
        selected = None
    else:
        def rank(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
            metric = row["fit_dev"]
            return (float(metric.get("cosine_similarity", metric.get("cosine", -1.0))), -float(metric.get("normalized_mse", 1e9)), -float(metric.get("learned_load_cv", metric.get("load_cv", 1e9))), str(row["recipe_id"]))
        selected = sorted(passing, key=rank, reverse=True)[0]
        decision = "LEARNED_ROUTER_GREEN"
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-learned-router-fit-dev", "status": decision, "run_id": EPIC_ID, "oracle_only": False, "promotion_eligible": False, "capacity_recipe": selected_id, "recipes": rows, "selected_recipe": selected["recipe_id"] if selected else None, "prediction_result": {"classifier": "expected-match" if selected else "counterexample", "reason": "learned router cleared all absolute gates" if selected else "no learned-router recipe cleared every absolute gate"}}
    _write_once(run_root / "results" / "learned-router-fit-dev.json", payload)
    _event(run_root, "learned_phase_completed", status=decision, selected_recipe=payload["selected_recipe"])
    return payload


def _robustness_phase(prereg: Mapping[str, Any], run_root: Path, learned: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    selected_id = learned.get("selected_recipe")
    if not selected_id:
        raise RuntimeError("seed robustness is not authorized when learned routing failed")
    recipe = next(recipe for recipe in prereg["learned_router_recipes"] if recipe["id"] == selected_id)
    rows = [_run_recipe(prereg, run_root, recipe=recipe, seed=seed, device=device, learned=True) for seed in prereg["seeds"]]
    passing = [row for row in rows if row.get("status") == "GATE_GREEN"]
    decision = "D2M_50_DEVELOPMENT_SELECTED" if len(passing) == len(rows) else "SEED_ROBUSTNESS_FAILED"
    canonical = None
    if passing:
        canonical = sorted(passing, key=lambda row: (float(row["fit_dev"].get("cosine_similarity", row["fit_dev"].get("cosine", -1.0))), -float(row["fit_dev"].get("normalized_mse", 1e9)), int(row["seed"])), reverse=True)[0]
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-seed-robustness", "status": decision, "run_id": EPIC_ID, "recipe_id": selected_id, "seeds": rows, "canonical_checkpoint": canonical["checkpoint"] if canonical else None, "protected_tiers_opened": False, "exact_lm_evaluation": False, "prediction_result": {"classifier": "expected-match" if decision == "D2M_50_DEVELOPMENT_SELECTED" else "counterexample", "reason": "all three frozen seeds independently cleared" if decision == "D2M_50_DEVELOPMENT_SELECTED" else "at least one frozen seed failed", "failed_seeds": [row["seed"] for row in rows if row.get("status") != "GATE_GREEN"]}}
    _write_once(run_root / "results" / "seed-robustness.json", payload)
    _event(run_root, "robustness_phase_completed", status=decision, canonical_seed=canonical.get("seed") if canonical else None)
    return payload


def _write_capacity_stop_artifacts(run_root: Path, *, capacity: Mapping[str, Any]) -> None:
    """Close the downstream artifact surface without fabricating unevaluated results."""

    learned_path = run_root / "results" / "learned-router-fit-dev.json"
    learned_payload = {
        "schema_version": 1,
        "artifact_type": "dense2moe-p16-top6-50-learned-router-fit-dev",
        "status": "NOT_RUN",
        "run_id": EPIC_ID,
        "oracle_only": False,
        "promotion_eligible": False,
        "capacity_decision": str(capacity.get("status")),
        "recipes": [],
        "selected_recipe": None,
        "stop_reason": "The two preregistered actual-capacity recipes both failed FIT-DEV absolute gates; learned-router training was not authorized.",
        "protected_tiers_opened": False,
        "exact_lm_evaluation": False,
        "prediction_result": {"classifier": "counterexample", "reason": "capacity stop rule prevented learned-router evaluation"},
    }
    _write_once(learned_path, learned_payload)
    seed_path = run_root / "results" / "seed-robustness.json"
    seed_payload = {
        "schema_version": 1,
        "artifact_type": "dense2moe-p16-top6-50-seed-robustness",
        "status": "NOT_RUN",
        "run_id": EPIC_ID,
        "recipe_id": None,
        "seeds": [],
        "canonical_checkpoint": None,
        "upstream_decision": str(capacity.get("status")),
        "stop_reason": "Seed robustness was not authorized after the oracle-trained capacity stop rule.",
        "protected_tiers_opened": False,
        "exact_lm_evaluation": False,
        "prediction_result": {"classifier": "counterexample", "reason": "capacity stop rule prevented seed robustness"},
    }
    _write_once(seed_path, seed_payload)
    for seed in (17, 29, 41):
        _write_once(
            run_root / "results" / "per-seed" / f"seed-{seed}.json",
            {
                "schema_version": 1,
                "artifact_type": "dense2moe-p16-top6-50-seed-result",
                "status": "NOT_RUN",
                "run_id": EPIC_ID,
                "seed": seed,
                "upstream_decision": str(capacity.get("status")),
                "reason": "Learned recipe was not authorized.",
            },
        )


def _finalize(run_root: Path, *, decision: str, capacity: Mapping[str, Any] | None, learned: Mapping[str, Any] | None, robustness: Mapping[str, Any] | None) -> dict[str, Any]:
    allowed = {"D2M_50_DEVELOPMENT_SELECTED", "ORACLE_TRAINED_CAPACITY_FAILED", "LEARNED_ROUTER_FAILED", "SEED_ROBUSTNESS_FAILED", "INVALID_OR_BLOCKED"}
    if decision not in allowed:
        raise ValueError(f"invalid final decision: {decision}")
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-final-decision", "decision": decision, "run_id": EPIC_ID, "candidate": {"source_candidate": "ht-2e2d35188fc77dea", "layer": 0, "topology": "p16/top6", "active_width": 8704, "average_ffn_reduction": 0.5}, "capacity_status": capacity.get("status") if capacity else None, "learned_status": learned.get("status") if learned else None, "robustness_status": robustness.get("status") if robustness else None, "protected_tiers_opened": False, "exact_lm_evaluation": False, "production_promotion": False}
    _write_once(run_root / "results" / "final-decision.json", payload)
    if decision == "ORACLE_TRAINED_CAPACITY_FAILED" and capacity is not None:
        _write_capacity_stop_artifacts(run_root, capacity=capacity)
    summary = "# p16/top6 50% development selection\n\n" + f"Final decision: **{decision}**.\n\n" + "Scope is layer 0, the declared FIT-TRAIN/FIT-DEV activations, fixed p16/top6 geometry, and preregistered training recipes. Oracle routing is diagnostic/training supervision only. Protected tiers, exact LM evaluation, release, and production promotion remain unopened.\n"
    summary_path = run_root / "results" / "executive-summary.md"
    closeout_amendment = run_root / "planning" / "phase-02-closeout-amendment.json"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != summary and not closeout_amendment.exists():
        raise RuntimeError(f"refusing to overwrite immutable summary: {summary_path}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text(summary, encoding="utf-8")
    return payload


def _smoke(prereg: Mapping[str, Any], run_root: Path, *, device: str) -> dict[str, Any]:
    import torch

    torch.manual_seed(17)
    hidden = 8
    dense = 32
    gate = torch.randn(dense, hidden)
    up = torch.randn(dense, hidden)
    down = torch.randn(hidden, dense)
    model = TorchQwen35SwiGLUMoE.from_dense(gate, up, down, routed_experts=4, shared_intermediate_size=16, top_k=2, routing_mode="independent_positive", learnable_scales=True, residual_intermediate_size=4, residual_scope="static", fallback_mode="none", fallback_rate_budget=0.0).to(device)
    inputs = torch.randn(5, hidden, device=device, requires_grad=True)
    output, info = model(inputs, return_router=True)
    output.square().mean().backward()
    if info["active_intermediate_width"] != 28 or model.residual_corrector.output_proj.weight.grad is None:
        raise RuntimeError("tiny residual smoke failed")
    destination = run_root / "checkpoints" / "fixture-smoke"
    model.save_pretrained(destination)
    restored = TorchQwen35SwiGLUMoE.from_pretrained(destination, strict=True, device=device).eval()
    with torch.inference_mode():
        observed, observed_info = restored(inputs.detach(), return_router=True)
    torch.testing.assert_close(observed, model(inputs.detach()), rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed_info["indices"], model(inputs.detach(), return_router=True)[1]["indices"], rtol=0.0, atol=0.0)
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-tiny-smoke", "status": "SMOKE_GREEN", "run_id": EPIC_ID, "device": device, "active_width": 28, "residual_executed": True, "reload_equivalent": True}
    _write_once(run_root / "results" / "tiny-fixture-smoke.json", payload)
    _event(run_root, "tiny_smoke_completed", status="SMOKE_GREEN")
    return payload


def _real_smoke(prereg: Mapping[str, Any], run_root: Path, *, device: str) -> dict[str, Any]:
    """Run one bounded forward/backward/reload pass on retained real rows."""

    import torch

    candidate = prereg["candidate"]
    source = _resolve(str(prereg["inputs"]["source_snapshot"]))
    gate, up, down = _load_dense_layer(source)
    model = TorchQwen35SwiGLUMoE.from_dense(gate, up, down, routed_experts=int(candidate["routed_experts"]), shared_intermediate_size=int(candidate["shared_width"]), top_k=int(candidate["top_k"]), routing_mode="independent_positive", learnable_scales=True, residual_intermediate_size=int(candidate["residual_width"]), residual_scope="static", fallback_mode="none", fallback_rate_budget=0.0).to(device)
    manifest = _resolve(str(prereg["inputs"]["fit_train_manifest"]))
    batch = next(iter_paired_activation_batches(manifest, expected_split="FIT-TRAIN", repo_root=REPO_ROOT, batch_tokens=4))
    inputs = torch.as_tensor(batch.inputs, dtype=torch.float32, device=device)
    targets = torch.as_tensor(batch.targets, dtype=torch.float32, device=device)
    output, info = model(inputs, return_router=True)
    loss = torch.mean((output - targets).square())
    loss.backward()
    if int(info["active_intermediate_width"]) != int(candidate["active_width"]):
        raise RuntimeError("real-data smoke active-width mismatch")
    if model.residual_corrector is None or model.residual_corrector.output_proj.weight.grad is None:
        raise RuntimeError("real-data smoke residual gradient is missing")
    destination = run_root / "checkpoints" / "real-data-smoke"
    model.save_pretrained(destination)
    restored = TorchQwen35SwiGLUMoE.from_pretrained(destination, strict=True, device=device).eval()
    with torch.inference_mode():
        reloaded, reloaded_info = restored(inputs, return_router=True)
    torch.testing.assert_close(reloaded, model.eval()(inputs), rtol=0.0, atol=0.0)
    torch.testing.assert_close(reloaded_info["indices"], info["indices"], rtol=0.0, atol=0.0)
    payload = {"schema_version": 1, "artifact_type": "dense2moe-p16-top6-50-real-data-smoke", "status": "REAL_DATA_SMOKE_GREEN", "run_id": EPIC_ID, "device": device, "split": "FIT-TRAIN", "rows": int(inputs.shape[0]), "active_width": int(info["active_intermediate_width"]), "residual_executed": bool(info["residual_executed"]), "residual_gradient_finite": bool(torch.isfinite(model.residual_corrector.output_proj.weight.grad).all().item()), "reload_equivalent": True, "teacher_dependent_inference": False}
    _write_once(run_root / "results" / "real-data-smoke.json", payload)
    _event(run_root, "real_data_smoke_completed", status="REAL_DATA_SMOKE_GREEN", rows=payload["rows"])
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate-inputs", "rebaseline", "smoke", "real-smoke", "capacity", "learned", "robustness", "all"), default="validate-inputs")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = _resolve(args.run_root)
    prereg_path = _resolve(args.preregistration)
    prereg = _load_prereg(prereg_path)
    run_root.mkdir(parents=True, exist_ok=True)
    phase = str(args.phase)
    if phase == "validate-inputs":
        _validate_inputs(prereg, run_root, device=str(args.device))
    elif phase == "rebaseline":
        _rebaseline(prereg, run_root, device=str(args.device))
    elif phase == "smoke":
        _smoke(prereg, run_root, device=str(args.device))
    elif phase == "real-smoke":
        _validate_inputs(prereg, run_root, device=str(args.device))
        _real_smoke(prereg, run_root, device=str(args.device))
    elif phase == "capacity":
        _validate_inputs(prereg, run_root, device=str(args.device))
        _rebaseline(prereg, run_root, device=str(args.device))
        capacity = _capacity_phase(prereg, run_root, device=str(args.device))
        if capacity["status"] != "ORACLE_TRAINED_CAPACITY_GREEN":
            _finalize(run_root, decision="ORACLE_TRAINED_CAPACITY_FAILED", capacity=capacity, learned=None, robustness=None)
    elif phase == "learned":
        capacity = _load_json(run_root / "results" / "oracle-trained-capacity.json")
        learned = _learned_phase(prereg, run_root, capacity, device=str(args.device))
        if learned["status"] != "LEARNED_ROUTER_GREEN":
            _finalize(run_root, decision="LEARNED_ROUTER_FAILED", capacity=capacity, learned=learned, robustness=None)
    elif phase == "robustness":
        capacity = _load_json(run_root / "results" / "oracle-trained-capacity.json")
        learned = _load_json(run_root / "results" / "learned-router-fit-dev.json")
        robustness = _robustness_phase(prereg, run_root, learned, device=str(args.device))
        _finalize(run_root, decision=str(robustness["status"]), capacity=capacity, learned=learned, robustness=robustness)
    elif phase == "all":
        _validate_inputs(prereg, run_root, device=str(args.device))
        _rebaseline(prereg, run_root, device=str(args.device))
        _smoke(prereg, run_root, device=str(args.device))
        _real_smoke(prereg, run_root, device=str(args.device))
        capacity = _capacity_phase(prereg, run_root, device=str(args.device))
        if capacity["status"] != "ORACLE_TRAINED_CAPACITY_GREEN":
            _finalize(run_root, decision="ORACLE_TRAINED_CAPACITY_FAILED", capacity=capacity, learned=None, robustness=None)
        else:
            learned = _learned_phase(prereg, run_root, capacity, device=str(args.device))
            if learned["status"] != "LEARNED_ROUTER_GREEN":
                _finalize(run_root, decision="LEARNED_ROUTER_FAILED", capacity=capacity, learned=learned, robustness=None)
            else:
                robustness = _robustness_phase(prereg, run_root, learned, device=str(args.device))
                _finalize(run_root, decision=str(robustness["status"]), capacity=capacity, learned=learned, robustness=robustness)
    print(json.dumps({"ok": True, "phase": phase, "run_root": str(run_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
