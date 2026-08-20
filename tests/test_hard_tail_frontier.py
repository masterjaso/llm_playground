from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dense2moe.evaluation.replay import ReplayInputError
from scripts.run_hard_tail_frontier import (
    _generalization,
    _metric_payload,
    _prediction,
    _read_manifest_rows,
    _write_immutable_json,
)


def test_frontier_help_is_available() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_hard_tail_frontier.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "resumable oracle frontier" in result.stdout.lower()


def test_frontier_immutable_json_is_idempotent_and_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"schema_version": 1, "values": [1, 2, 3]}
    assert _write_immutable_json(path, payload) == payload
    assert _write_immutable_json(path, payload) == payload
    with pytest.raises(RuntimeError, match="different immutable"):
        _write_immutable_json(path, {"schema_version": 1, "values": [4]})


def test_frontier_prediction_reconstructs_selected_weighted_routes() -> None:
    shared = np.zeros((2, 3), dtype=np.float32)
    routed = np.asarray(
        [
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
        ],
        dtype=np.float32,
    )
    result = {"indices": np.asarray([[0, 2], [1, 2]]), "weights": np.asarray([[0.5, 2.0], [1.5, 0.25]])}
    output = _prediction(shared, routed, result)
    np.testing.assert_allclose(output, [[0.5, 0, 6], [0, 7.5, 1.5]])


def test_frontier_generalization_is_split_specific() -> None:
    fit = {"cosine_similarity": 0.9, "normalized_mse": 0.2, "target_relative_norm_error": 0.3}
    dev = {"cosine_similarity": 0.8, "normalized_mse": 0.4, "target_relative_norm_error": 0.5}
    result = _generalization(fit, dev)
    assert result["classification"] == "ORACLE_ONLY_DIAGNOSTIC"
    assert result["cosine_gap"] == pytest.approx(0.1)
    assert result["nmse_ratio"] == pytest.approx(2.0)


def test_frontier_metric_payload_preserves_oracle_health() -> None:
    metrics = {"cosine_similarity": 0.9, "normalized_mse": 0.2}
    oracle = {
        "load_cv": 0.25,
        "dead_experts": 0,
        "expert_usage_counts": [2, 3],
        "method": "frozen_slice_load_aware_oracle",
        "assurance": "exact_candidate_sets",
        "gate_feasible": False,
    }
    result = _metric_payload(metrics, oracle_result=oracle)
    assert result["oracle_load_cv"] == pytest.approx(0.25)
    assert result["oracle_expert_counts"] == [2, 3]
    assert result["oracle_method"] == "frozen_slice_load_aware_oracle"


def test_frontier_rejects_malformed_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 2, "status": "CAPTURE_COMPLETE", "shards": []}), encoding="utf-8")
    with pytest.raises(ReplayInputError):
        _read_manifest_rows(path, expected_split="FIT-TRAIN")
