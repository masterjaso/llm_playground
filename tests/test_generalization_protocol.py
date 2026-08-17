from __future__ import annotations

import hashlib

import pytest

from dense2moe.data import (
    assign_grouped_tiers,
    audit_corpus_tier_disjointness,
    new_contamination_ledger,
    open_evaluation_tier,
    retire_evaluation_tier,
    validate_contamination_ledger,
)
from dense2moe.evaluation import amplitude_metrics, evaluate_promotion_metrics


def _row(index: int, *, tier: str, text: str | None = None, task: str | None = None) -> dict[str, object]:
    value = text or f"independent task {index} has a unique implementation trajectory and result"
    return {
        "id": hashlib.sha256(value.encode()).hexdigest()[:32],
        "text": value,
        "tier": tier,
        "source_name": "source-a",
        "source_revision": "a" * 40,
        "source_license": "MIT",
        "source_record_id": f"record-{index}",
        "repo": f"repo-{index}",
        "task_id": task or f"task-{index}",
        "trajectory_id": f"trajectory-{index}",
        "document_id": f"document-{index}",
    }


def test_tier_audit_catches_group_and_exact_leakage() -> None:
    rows = [_row(1, tier="FIT-TRAIN"), _row(2, tier="G1", task="task-1")]
    rows[1]["repo"] = rows[0]["repo"]
    rows[1]["source_record_id"] = rows[0]["source_record_id"]
    rows[1]["text"] = rows[0]["text"]
    audit = audit_corpus_tier_disjointness(rows, tiers=("FIT-TRAIN", "G1"))
    assert audit["status"] == "FAIL"
    assert audit["group_conflicts"]
    assert audit["exact_conflicts"]


def test_grouped_assignment_keeps_trajectory_together() -> None:
    rows = [_row(1, tier=""), _row(2, tier="", task="task-1")]
    rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    assigned, receipt = assign_grouped_tiers(
        rows,
        tier_fractions={"FIT-TRAIN": 0.5, "FIT-DEV": 0.5},
        seed=7,
    )
    assert receipt["status"] == "PASS"
    assert assigned[0]["tier"] == assigned[1]["tier"]


def test_contamination_ledger_is_one_way_and_retires_failed_tier() -> None:
    ledger = new_contamination_ledger(
        method_version="method-v1",
        code_commit="commit-a",
        thresholds_fingerprint="thresholds-a",
        runtime_lock_sha256="runtime-a",
        corpus_hashes={"G1": "data-a"},
    )
    opened = open_evaluation_tier(
        ledger,
        tier="G1",
        dataset_hash="data-a",
        method_version="method-v1",
        code_commit="commit-a",
        thresholds_fingerprint="thresholds-a",
    )
    with pytest.raises(ValueError, match="already opened"):
        open_evaluation_tier(
            opened,
            tier="G1",
            dataset_hash="data-a",
            method_version="method-v1",
            code_commit="commit-a",
            thresholds_fingerprint="thresholds-a",
        )
    retired = retire_evaluation_tier(opened, tier="G1", reason="candidate failed")
    assert validate_contamination_ledger(retired)["status"] == "PASS"
    assert retired["tiers"]["G1"]["status"] == "RETIRED"


def test_amplitude_and_external_gate_require_all_metrics_and_domains() -> None:
    metrics = amplitude_metrics([[3.0, 4.0], [0.0, 2.0]], [[3.0, 4.0], [0.0, 2.0]])
    assert metrics["median_norm_ratio"] == pytest.approx(1.0)
    assert metrics["p95_relative_norm_error"] == pytest.approx(0.0)
    values = {
        "nmse": 0.04,
        "cosine": 0.985,
        "loadcv": 0.4,
        "dead_experts": 0,
        "oracle_regret": 0.05,
        "repeat_variation": 0.02,
        "median_norm_ratio_error": 0.02,
        "p95_relative_norm_error": 0.1,
    }
    result = evaluate_promotion_metrics(values, domain_slices={"code": values, "agentic": {**values, "cosine": 0.90}})
    assert result["overall"] == "red"
    assert "agentic" in result["failed_domains"]
    alias_result = evaluate_promotion_metrics(
        {
            "normalized_mse": 0.04,
            "cosine": 0.985,
            "load_cv": 0.4,
            "dead_expert_count": 0,
            "oracle_regret_nmse": 0.05,
            "repeat_variation": 0.02,
            "median_norm_ratio_delta": 0.02,
            "p95_abs_relative_norm_error": 0.1,
        }
    )
    assert alias_result["overall"] == "green"
