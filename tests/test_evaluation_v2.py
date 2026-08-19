from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from dense2moe.evaluation.bootstrap import evaluate_threshold_with_confidence, grouped_bootstrap
from dense2moe.evaluation.contracts import validate_evaluation_contracts
from dense2moe.evaluation.decision import decide_candidate
from dense2moe.evaluation.generalization import classify_generalization, compute_generalization_gaps
from dense2moe.evaluation.inventory import discover_candidate_inventory
from dense2moe.evaluation.lm import StreamingLMMetrics, compute_lm_output_metrics
from dense2moe.evaluation.receipts import (
    build_lm_output_receipt,
    build_structural_generalization_receipt,
    classify_receipt_compatibility,
    validate_lm_receipt,
    validate_structural_receipt,
    write_immutable_receipt,
)
from dense2moe.evaluation.runner import evaluate_lm_pair_v2, evaluate_structural_pair_v2
from dense2moe.evaluation.suffix_replay import BLOCKED_EXACT_KL_RESOURCE_LIMIT, run_exact_layer_patch_replay
from dense2moe.evaluation.registry import METRIC_REGISTRY, POLICY_HASH, metric_policy_hash, validate_metric_registry
from dense2moe.evaluation.structural import StructuralMetricsAccumulator, compute_structural_metrics


def _structural(*, cosine: float = 0.99, nmse: float = 0.01, load: float = 0.1, **overrides):
    value = {
        "cosine_similarity": cosine,
        "normalized_mse": nmse,
        "target_relative_norm_error": 0.01,
        "mean_prediction_to_target_norm_ratio": 1.0,
        "p95_abs_relative_norm_error": 0.05,
        "learned_load_cv": load,
        "dead_expert_count": 0,
        "dropped_token_count": 0,
        "invalid_token_count": 0,
        "non_finite_token_count": 0,
        "scored_token_count": 32,
        "independent_group_count": 4,
    }
    value.update(overrides)
    return value


def _lm(*, mean_kl: float = 0.01, **overrides):
    value = {
        "mean_forward_kl": mean_kl,
        "p95_forward_kl": mean_kl,
        "top1_agreement": 0.99,
        "top5_set_recall": 0.99,
        "teacher_top5_mass_retention": 0.99,
        "high_margin_top1_flip_rate": 0.01,
        "absolute_nll_delta": 0.01,
        "relative_nll_increase": 0.01,
        "non_finite_token_count": 0,
    }
    value.update(overrides)
    return value


def test_registry_is_canonical_and_gate_covered():
    assert validate_metric_registry()["valid"]
    assert POLICY_HASH == metric_policy_hash()
    changed = dict(METRIC_REGISTRY)
    changed["structural.cosine_similarity"] = replace(changed["structural.cosine_similarity"], green=0.981)
    assert metric_policy_hash(changed) != POLICY_HASH


def test_structural_fit_dev_are_independent_and_streaming_matches():
    teacher = np.eye(4, dtype=np.float32)
    candidate = teacher.copy()
    candidate[1, 1] = 0.5
    direct = compute_structural_metrics(teacher, candidate, learned_assignments=[0, 1, 0, 1])
    accumulator = StructuralMetricsAccumulator()
    accumulator.update(teacher[:2], candidate[:2], learned_assignments=[0, 1])
    accumulator.update(teacher[2:], candidate[2:], learned_assignments=[0, 1])
    streamed = accumulator.finalize()
    assert direct["cosine_similarity"] == pytest.approx(streamed["cosine_similarity"])
    assert direct["normalized_mse"] == pytest.approx(streamed["normalized_mse"])
    assert direct["scored_token_count"] == 4


def test_generalization_gap_arithmetic_and_classification():
    gaps = compute_generalization_gaps(_structural(), _structural(cosine=0.9, nmse=0.1, load=0.2))
    assert gaps["cosine_gap"] == pytest.approx(0.09)
    assert gaps["absolute_nmse_increase"] == pytest.approx(0.09)
    assert gaps["nmse_ratio"] == pytest.approx(10.0)
    result = classify_generalization(_structural(), _structural(cosine=0.9, nmse=0.1, load=0.2))
    assert result["classification"] == "GENERALIZATION_REJECT"


def test_lm_identical_logits_zero_forward_kl_and_partition_invariant():
    logits = np.asarray([[5.0, 1.0, 0.0, -1.0, -2.0], [0.0, 4.0, 1.0, -1.0, -2.0]], dtype=np.float32)
    whole = compute_lm_output_metrics(logits, logits, targets=[0, 1], baseline_logits=logits)
    split = StreamingLMMetrics()
    split.update(logits[:1], logits[:1], targets=[0], baseline_logits=logits[:1])
    split.update(logits[1:], logits[1:], targets=[1], baseline_logits=logits[1:])
    partitioned = split.finalize()
    assert whole["mean_forward_kl"] == pytest.approx(0.0, abs=1e-7)
    assert whole["dense_repeat_mean_forward_kl"] == pytest.approx(0.0, abs=1e-7)
    assert whole["top1_agreement"] == 1.0
    assert whole["teacher_top5_mass_retention"] == pytest.approx(1.0)
    assert partitioned["mean_forward_kl"] == pytest.approx(whole["mean_forward_kl"])
    assert not any("logit" in key for key in split.__dict__)


def test_lm_forward_kl_direction_and_masking():
    teacher = np.asarray([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    candidate = np.asarray([[0.0, 4.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    forward = compute_lm_output_metrics(teacher, candidate, mask=[True])
    reverse = compute_lm_output_metrics(candidate, teacher, mask=[True])
    assert forward["mean_forward_kl"] != pytest.approx(reverse["mean_forward_kl"])
    masked = compute_lm_output_metrics(np.vstack([teacher, teacher]), np.vstack([candidate, candidate]), mask=[True, False])
    assert masked["valid_token_count"] == 1
    assert masked["masked_token_count"] == 1


def test_lm_dense_repeat_and_source_domain_slices_are_partition_safe():
    logits = np.asarray([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [3.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=np.float32)
    candidate = logits.copy()
    candidate[2, 0] -= 0.5
    metadata = [
        {"source_family": "a", "domain": "x"},
        {"source_family": "a", "domain": "y"},
        {"source_family": "b", "domain": "x"},
        {"source_family": "b", "domain": "y"},
    ]
    whole = compute_lm_output_metrics(logits, candidate, metadata=metadata, baseline_logits=logits)
    split = StreamingLMMetrics()
    split.update(logits[:2], candidate[:2], baseline_logits=logits[:2], metadata=metadata[:2])
    split.update(logits[2:], candidate[2:], baseline_logits=logits[2:], metadata=metadata[2:])
    partitioned = split.finalize()
    assert whole["dense_repeat_mean_forward_kl"] == pytest.approx(0.0, abs=1e-7)
    assert partitioned["dense_repeat_mean_forward_kl"] == pytest.approx(whole["dense_repeat_mean_forward_kl"], abs=1e-7)
    assert set(partitioned["source_slices"]) == {"a", "b"}
    assert set(partitioned["domain_slices"]) == {"x", "y"}
    assert partitioned["source_slices"]["a"]["scored_token_count"] == 2


def test_grouped_bootstrap_samples_groups_and_conservative_bound():
    result = grouped_bootstrap({"g1": [1.0] * 8, "g2": [3.0] * 8}, seed=7, repetitions=100, minimum_sample_count=2)
    assert result["bootstrap_unit"] == "independent_group"
    assert result["independent_group_count"] == 2
    assert result["lower"] <= result["point_estimate"] <= result["upper"]
    gate = evaluate_threshold_with_confidence(result, direction="lower-is-better", threshold=4.0)
    assert gate["classification"] == "GREEN"


def test_decision_green_fit_red_dev_is_generalization_reject():
    result = decide_candidate(_structural(), _structural(cosine=0.9, nmse=0.1, load=0.2))
    assert result["status"] == "GENERALIZATION_REJECT"
    assert result["decision"] == "REJECT"
    assert result["fit_dev_averaged"] is False


def test_dev_green_can_survive_fit_warning_and_lm_red_vetoes():
    fit_warning = _structural(cosine=0.97, nmse=0.06)
    dev_green = _structural()
    survived = decide_candidate(fit_warning, dev_green)
    assert survived["decision"] == "PROMOTE"
    vetoed = decide_candidate(_structural(), dev_green, lm_metrics=_lm(mean_kl=0.9, p95_forward_kl=0.9))
    assert vetoed["status"] == "LM_VETO"


def test_source_collapse_and_overrides_are_narrow_and_protected():
    source_failed = _structural(source_family_slices={"code": {"gate_eligible": True, "overall": "RED"}})
    result = decide_candidate(_structural(), source_failed)
    assert result["status"] == "SOURCE_SLICE_COLLAPSE"
    near_cosine = _structural(cosine=0.977)
    pending = decide_candidate(_structural(), near_cosine, lm_metrics=_lm(), override_requested=True)
    assert pending["override_eligible"] is True
    assert pending["decision"] == "OVERRIDE_PENDING_PROTECTED_CONFIRMATION"
    confirmed = decide_candidate(_structural(), near_cosine, lm_metrics=_lm(), override_requested=True, protected_confirmation=True)
    assert confirmed["decision"] == "PROMOTE_OVERRIDE"
    dual = decide_candidate(_structural(), _structural(cosine=0.977, nmse=0.055), lm_metrics=_lm(), override_requested=True)
    assert dual["override_eligible"] is False


def test_structural_slice_without_routing_fields_is_not_false_collapse():
    # Slice reducers intentionally omit aggregate learned-router counts.  The
    # decision engine must gate their quality metrics without treating absent
    # slice routing fields as a source collapse.
    slice_payload = _structural()
    slice_payload.pop("learned_load_cv")
    slice_payload.pop("dead_expert_count")
    slice_payload["gate_eligible"] = True
    result = decide_candidate(
        _structural(),
        _structural(source_family_slices={"code": slice_payload}),
    )
    assert result["source_slice_collapses"] == []


def test_blocked_exact_lm_preserves_structural_result_without_lm_veto():
    result = decide_candidate(
        _structural(),
        _structural(),
        lm_metrics={
            "status": "BLOCKED_EXACT_KL_RESOURCE_LIMIT",
            "scientific_evaluation_performed": False,
        },
    )
    assert result["status"] == "LM_EVALUATION_BLOCKED"
    assert result["decision"] == "RESEARCH_ONLY"
    assert result["lm_gate"]["overall"] == "BLOCKED_RESOURCE_LIMIT"
    assert result["lm_gate"]["status"] == "BLOCKED_EXACT_KL_RESOURCE_LIMIT"


def test_receipts_round_trip_and_legacy_compatibility(tmp_path):
    structural = build_structural_generalization_receipt(
        source_model={"model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "layers": 64, "hidden_size": 5120, "dense_intermediate": 17408},
        layer=0,
        candidate={"candidate_id": "fixture"},
        fit_train=_structural(),
        fit_dev=_structural(),
        generalization={"classification": "STABLE_CANDIDATE"},
        runtime_lock_identity={"sha256": "runtime"},
    )
    assert validate_structural_receipt(structural)["valid"]
    path = tmp_path / "structural.json"
    write_immutable_receipt(structural, path)
    write_immutable_receipt(structural, path)
    with pytest.raises(ValueError):
        write_immutable_receipt({**structural, "status": "different"}, path)
    lm = build_lm_output_receipt(
        source_model={"model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
        candidate={"candidate_id": "fixture"},
        layer=0,
        dataset={"split": "FIT-DEV"},
        raw_metrics=_lm(),
        runtime_lock_identity={"sha256": "runtime"},
    )
    assert validate_lm_receipt(lm)["valid"]
    assert classify_receipt_compatibility({"receipt_type": "dense2moe-v2.3-equal-budget"})["classification"] == "LEGACY_REQUIRES_RECOMPUTATION"


def test_contract_meta_validation_rejects_unknown_metric():
    result = validate_evaluation_contracts(emitted_metric_ids=["structural.cosine_similarity", "unknown.metric"])
    assert result["valid"] is False
    assert "unknown.metric" in result["emitted_metrics"]["unknown_metric_ids"]


def test_candidate_inventory_records_terminal_replayability(tmp_path):
    run = tmp_path / "runs" / "design-A" / "seed-17"
    run.mkdir(parents=True)
    tensor = run / "layer-0000.safetensors"
    tensor.write_bytes(b"fixture")
    for split in ("FIT-TRAIN", "FIT-DEV"):
        manifest = run / f"layer-0000-{split}.json"
        manifest.write_text(json.dumps({"split": split, "dataset_hash": "data", "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}), encoding="utf-8")
    (run / "seed-result.json").write_text(json.dumps({"artifact_type": "dense2moe-v2.4", "candidate_id": "A-fixture", "seed": 17, "checkpoint": {"tensor_file": str(tensor), "tensor_sha256": "hash", "strict_reload": {"passed": True}}, "config": {"profile": {"name": "v24-a", "routing_mode": "independent_positive", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}}}), encoding="utf-8")
    records = discover_candidate_inventory([tmp_path / "runs"])
    assert len(records) == 1
    assert records[0].rerun_status == "PARTIAL_METRICS_ONLY"
    assert records[0].design_id == "A"


def test_shared_runner_emits_policy_bound_structural_and_lm_receipts():
    teacher = np.eye(4, dtype=np.float32)
    structural = evaluate_structural_pair_v2(
        teacher_fit_train=teacher,
        candidate_fit_train=teacher,
        teacher_fit_dev=teacher,
        candidate_fit_dev=teacher,
        source_model={"model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
        candidate={"candidate_id": "runner-fixture", "routing_mode": "learned"},
        layer=0,
    )
    assert structural["fit_train"]["scored_token_count"] == 4
    assert structural["fit_dev"]["scored_token_count"] == 4
    assert structural["receipt"]["fit_dev_averaged"] is False
    logits = np.asarray([[4.0, 1.0, 0.0, -1.0], [1.0, 4.0, 0.0, -1.0]], dtype=np.float32)
    lm = evaluate_lm_pair_v2(
        teacher_logits=logits,
        candidate_logits=logits,
        source_model={"model": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
        candidate={"candidate_id": "runner-fixture", "routing_mode": "learned"},
        layer=0,
        dataset={"split": "FIT-DEV", "hash": "fixture"},
        metric_options={"targets": [0, 1]},
    )
    assert lm["metrics"]["mean_forward_kl"] == pytest.approx(0.0, abs=1e-7)
    assert lm["receipt"]["full_vocabulary_kl"] is True


def test_partial_replayable_cohort_is_not_complete(tmp_path):
    run = tmp_path / "runs" / "design-A" / "seed-17"
    run.mkdir(parents=True)
    tensor = run / "layer-0000.safetensors"
    tensor.write_bytes(b"fixture")
    for split in ("FIT-TRAIN", "FIT-DEV"):
        (run / f"layer-0000-{split}.json").write_text(
            json.dumps({"split": split, "dataset_hash": "data", "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}),
            encoding="utf-8",
        )
    (run / "seed-result.json").write_text(
        json.dumps({"artifact_type": "dense2moe-v2.4", "candidate_id": "A-fixture", "seed": 17, "checkpoint": {"tensor_file": str(tensor), "strict_reload": {"passed": True}}, "config": {"profile": {"name": "v24-a", "routing_mode": "independent_positive"}}}),
        encoding="utf-8",
    )
    from dense2moe.evaluation.inventory import inventory_payload

    payload = inventory_payload(discover_candidate_inventory([tmp_path / "runs"]))
    assert payload["cohort_complete"] is False
    assert payload["replayable_unresolved_count"] == 1


def test_exact_suffix_replay_blocks_when_full_vocab_chunk_is_unsafe():
    identity = {
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "root_family": "qwen3_5",
        "text_model_type": "qwen3_5_text",
        "layers": 64,
        "hidden_size": 5120,
        "dense_intermediate": 17408,
    }
    logits = np.zeros((1, 4, 8), dtype=np.float32)
    result = run_exact_layer_patch_replay(
        source_identity=identity,
        expected_layer=0,
        sequences=[{"input_ids": np.zeros((1, 4), dtype=np.int64)}],
        dense_forward=lambda _batch: logits,
        candidate_forward=lambda _batch: logits,
        max_vocab_chunk=4,
    )
    assert result["status"] == BLOCKED_EXACT_KL_RESOURCE_LIMIT
    assert result["scientific_evaluation_performed"] is False
