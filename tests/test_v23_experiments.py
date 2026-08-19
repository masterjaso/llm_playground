from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dense2moe.science.v23_designs import iter_designs
from dense2moe.science.v23_experiments import (
    V23ExperimentBlocked,
    build_equal_budget_receipt,
    build_oracle_ceiling_receipt,
    build_pareto_frontier_receipt,
    validate_paired_activation_manifests,
)
from scripts.run_v23_comparison import _resolve_tensor_path
from scripts.run_v23_comparison import main as comparison_main


def _write_manifest(root: Path, split: str, *, promotion: bool = False) -> Path:
    artifact = root / f"{split}-shard.bin"
    artifact.write_bytes(f"activation-{split}".encode())
    payload = {
        "status": "CAPTURE_COMPLETE",
        "split": split,
        "layer": 0,
        "count": 16,
        "dataset_hash": "dataset-v23",
        "source_revision": "a" * 40,
        "input_tensor": "ffn_input",
        "target_tensor": "dense_ffn_target",
        "capture_kind": "real_qwen_layer0_ffn_input_target",
        "shards": [
            {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "count": 16,
                "input_tensor": "ffn_input",
                "target_tensor": "dense_ffn_target",
            }
        ],
    }
    if promotion:
        payload["opened_evaluation_tiers"] = ["GATE-A"]
    path = root / f"{split}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    return _write_manifest(tmp_path, "FIT-TRAIN"), _write_manifest(tmp_path, "FIT-DEV")


def _oracle_attempts() -> dict[str, dict[str, object]]:
    return {
        design.design_id: {
            "status": "PASS",
            "oracle_gate_pass": True,
            "metrics": {
                "nmse": 0.04 + index * 0.001,
                "cosine": 0.98 - index * 0.001,
                "oracle_regret": 0.03 + index * 0.001,
            },
        }
        for index, design in enumerate(iter_designs())
    }


def _equal_results() -> dict[str, dict[str, object]]:
    return {
        design.design_id: {
            "status": "PASS",
            "tokens": 128,
            "flops": 1024,
            "metrics": {
                "nmse": 0.05 + index * 0.001,
                "cosine": 0.97 - index * 0.001,
                "oracle_regret": 0.04 + index * 0.001,
                "load_cv": 0.1 + index * 0.001,
            },
            "seeds": {
                str(seed): {
                    "finite_gradients": True,
                    "reload_verified": True,
                    "dead_experts": 0,
                }
                for seed in (17, 29, 41)
            },
        }
        for index, design in enumerate(iter_designs())
    }


def test_paired_manifests_require_complete_dense_targets_and_lineage(tmp_path: Path) -> None:
    train, dev = _pair(tmp_path)
    pair = validate_paired_activation_manifests(train, dev)
    assert pair.train["split"] == "FIT-TRAIN"
    assert pair.dev["split"] == "FIT-DEV"
    assert pair.train_sha256

    broken = json.loads(train.read_text(encoding="utf-8"))
    broken["target_tensor"] = None
    broken["shards"][0]["target_tensor"] = None
    train.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(V23ExperimentBlocked, match="dense FFN target"):
        validate_paired_activation_manifests(train, dev)


def test_oracle_receipt_records_all_five_and_is_deterministic(tmp_path: Path) -> None:
    train, dev = _pair(tmp_path)
    first = build_oracle_ceiling_receipt(train, dev, attempts=_oracle_attempts())
    second = build_oracle_ceiling_receipt(train, dev, attempts=_oracle_attempts())

    assert first == second
    assert first["status"] == "ORACLE_CEILINGS_COMPLETE"
    assert first["all_five_designs_attempted"] is True
    assert [item["design_id"] for item in first["designs"]] == ["A", "B", "C", "D", "E"]
    assert all(item["oracle_eligible"] for item in first["designs"])
    assert first["receipt_sha256"]


def test_missing_oracle_attempt_is_recorded_and_pruned_equal_budget(tmp_path: Path) -> None:
    train, dev = _pair(tmp_path)
    attempts = _oracle_attempts()
    del attempts["D"]
    oracle = build_oracle_ceiling_receipt(train, dev, attempts=attempts)
    assert oracle["status"] == "ORACLE_CEILING_BLOCKED"
    assert oracle["designs"][3]["failure_reasons"] == ["ORACLE_ATTEMPT_MISSING"]

    equal = build_equal_budget_receipt(
        oracle,
        train,
        dev,
        results=_equal_results(),
        token_budget=128,
        flops_budget=1024,
    )
    assert equal["status"] == "EQUAL_BUDGET_COMPLETE"
    assert equal["designs"][3]["status"] == "PRUNED"
    assert "ORACLE_ATTEMPT_MISSING" in equal["designs"][3]["failure_reasons"]


def test_equal_budget_and_pareto_receipts_are_fit_only(tmp_path: Path) -> None:
    train, dev = _pair(tmp_path)
    oracle = build_oracle_ceiling_receipt(train, dev, attempts=_oracle_attempts())
    equal = build_equal_budget_receipt(
        oracle,
        train,
        dev,
        results=_equal_results(),
        token_budget=128,
        flops_budget=1024,
    )
    frontier = build_pareto_frontier_receipt(equal)
    assert frontier["status"] == "PARETO_FRONTIER_COMPLETE"
    assert frontier["fit_only"] is True
    assert frontier["opened_evaluation_tiers"] == []
    assert set(frontier["frontier_design_ids"]).issubset({"A", "B", "C", "D", "E"})


def test_promotion_tier_input_is_rejected_before_experiment_receipts(tmp_path: Path) -> None:
    train = _write_manifest(tmp_path, "FIT-TRAIN", promotion=True)
    dev = _write_manifest(tmp_path, "FIT-DEV")
    with pytest.raises(V23ExperimentBlocked, match="promotion-tier"):
        build_oracle_ceiling_receipt(train, dev, attempts=_oracle_attempts())


def test_missing_activation_data_fails_closed(tmp_path: Path) -> None:
    train = _write_manifest(tmp_path, "FIT-TRAIN")
    with pytest.raises(V23ExperimentBlocked, match="does not exist"):
        build_oracle_ceiling_receipt(train, tmp_path / "missing-dev.json", attempts=_oracle_attempts())


def test_comparison_accepts_separate_immutable_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "closed-input"
    output_root = tmp_path / "new-output"
    input_root.mkdir()
    with pytest.raises(RuntimeError, match="complete paired V2.3 capture manifests"):
        comparison_main(
            [
                "--run-dir",
                str(output_root),
                "--input-run-dir",
                str(input_root),
                "--source-dir",
                str(tmp_path / "source"),
                "--source-revision",
                "a" * 40,
            ]
        )
    assert not output_root.exists()


def test_checkpoint_tensor_path_resolution_avoids_duplicate_output_root(tmp_path: Path) -> None:
    seed_dir = tmp_path / "comparison" / "pilots" / "design-A" / "seed-17"
    seed_dir.mkdir(parents=True)
    metadata = seed_dir / "metadata.json"
    tensor = seed_dir / "layer-0000.safetensors"
    tensor.write_bytes(b"checkpoint")

    assert _resolve_tensor_path(metadata, str(tensor)) == tensor
    assert _resolve_tensor_path(metadata, tensor.name) == tensor
