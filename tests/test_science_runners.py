from __future__ import annotations

import json

from dense2moe.config import load_config
from scripts.evaluate_promotion import run_promotion
from scripts.merge_development_finalists import merge_finalists
from scripts.run_candidate_search import _p32_expert_pool_size, run_candidate_search
from scripts.run_full64_training import _layer_lineage, run_full64
from scripts.run_representative_transfer import run_transfer


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_candidate_search_contract_is_development_only(tmp_path) -> None:
    train = tmp_path / "FIT-TRAIN.json"
    dev = tmp_path / "FIT-DEV.json"
    _write(train, {"split": "train", "count": 12, "dataset_hash": "fit"})
    _write(dev, {"split": "train", "count": 8, "dataset_hash": "dev"})
    result = run_candidate_search(
        run_dir=tmp_path / "run",
        activation_manifest=train,
        dev_manifest=dev,
        topology="p16/top4",
        exhaustive=True,
        expected_combinations=1820,
    )
    assert result["status"] == "CANDIDATE_SEARCH_READY"
    assert result["opened_evaluation_tiers"] == []
    assert result["search_class"] == "EXHAUSTIVE_C(16,4)"


def test_promotion_stops_and_seals_later_tiers_after_failure(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write(
        run_dir / "development" / "finalist-lock.json",
        {
            "status": "METHOD_LOCKED",
            "method_version": "moe-v22-m01",
            "external_tuning_forbidden": True,
        },
    )
    passing = {
        "nmse": 0.04,
        "cosine": 0.985,
        "loadcv": 0.4,
        "dead_experts": 0,
        "oracle_regret": 0.05,
        "repeat_variation": 0.02,
        "median_norm_ratio_error": 0.02,
        "p95_relative_norm_error": 0.1,
    }
    failing = {**passing, "cosine": 0.90}
    _write(run_dir / "promotion" / "GATE-A.json", {"dataset_hash": "gate-a", "metrics": passing})
    _write(run_dir / "promotion" / "SHADOW-B.json", {"dataset_hash": "shadow-b", "metrics": failing})
    _write(run_dir / "promotion" / "SHADOW-C.json", {"dataset_hash": "shadow-c", "metrics": passing})
    result = run_promotion(run_dir=run_dir, method_version="moe-v22-m01", tiers=["GATE-A", "SHADOW-B", "SHADOW-C"])
    assert result["status"] == "PROMOTION_REJECTED"
    assert result["tiers"]["GATE-A"]["status"] == "GREEN"
    assert result["tiers"]["SHADOW-B"]["status"] == "REJECTED"
    assert result["tiers"]["SHADOW-C"]["status"] == "SEALED"
    ledger = json.loads((run_dir / "promotion" / "contamination-ledger.json").read_text(encoding="utf-8"))
    assert ledger["tiers"]["SHADOW-B"]["status"] == "RETIRED"
    assert "SHADOW-C" not in ledger["tiers"]


def test_development_finalists_merge_is_sealed(tmp_path) -> None:
    run_dir = tmp_path / "run"
    for name, profile in (("p16-exhaustive-receipt.json", "qwen38_p16s1_top4"), ("p32-bounded-pool-receipt.json", "qwen38_p32s1_top5")):
        _write(
            run_dir / "development" / name,
            {
                "status": "DEV_FINALISTS",
                "method_version": "moe-v22-m01",
                "opened_evaluation_tiers": [],
                "profiles": {
                    profile: {
                        "status": "DEV_FINALIST",
                        "checkpoint_sha256": f"checkpoint-{profile}",
                        "dataset_hash": f"dataset-{profile}",
                    }
                },
                "finalists": [],
            },
        )
    result = merge_finalists(run_dir=run_dir, method_version="moe-v22-m01")
    assert result["status"] == "DEV_FINALISTS"
    merged = json.loads((run_dir / "development" / "finalists.json").read_text(encoding="utf-8"))
    assert sorted(merged["profiles"]) == ["qwen38_p16s1_top4", "qwen38_p32s1_top5"]


def test_full64_layer_lineage_invalidates_changed_inputs(tmp_path) -> None:
    train = tmp_path / "FIT-TRAIN.json"
    dev = tmp_path / "FIT-DEV.json"
    partition = tmp_path / "partition.json"
    train.write_text("train-v1", encoding="utf-8")
    dev.write_text("dev-v1", encoding="utf-8")
    partition.write_text("partition-v1", encoding="utf-8")
    profile = load_config("configs/qwen38_p16s1_top4.yaml")
    first = _layer_lineage(
        method_lock_sha256="lock-v1",
        train_manifest=train,
        dev_manifest=dev,
        profile=profile,
        partition=partition,
        layer=0,
        seed=17,
        device="cpu",
        epochs=1,
        microbatch=8,
        learning_rate=1e-3,
    )
    dev.write_text("dev-v2", encoding="utf-8")
    second = _layer_lineage(
        method_lock_sha256="lock-v1",
        train_manifest=train,
        dev_manifest=dev,
        profile=profile,
        partition=partition,
        layer=0,
        seed=17,
        device="cpu",
        epochs=1,
        microbatch=8,
        learning_rate=1e-3,
    )
    assert first["dev_manifest_sha256"] != second["dev_manifest_sha256"]


def test_p32_pool_budgets_expand_real_expert_search_space() -> None:
    budgets = (1024, 2048, 4096, 8192)
    pools = [_p32_expert_pool_size(routed_experts=32, top_k=5, candidate_budget=budget) for budget in budgets]
    assert pools == sorted(set(pools))
    assert pools == [13, 15, 16, 18]


def test_representative_execution_requires_distinct_untouched_external_root(tmp_path) -> None:
    run_dir = tmp_path / "run"
    for profile in ("qwen38_p16s1_top4", "qwen38_p32s1_top5"):
        path = run_dir / "method-locks" / f"{profile}.json"
        _write(path, {"external_tuning_forbidden": True})
    result = run_transfer(
        run_dir=run_dir,
        layers="0-3,28-31,60-63",
        profiles=["qwen38_p16s1_top4", "qwen38_p32s1_top5"],
        seeds=[17, 29, 41],
        execute=True,
        source_dir=tmp_path / "source",
        activation_root=tmp_path / "development",
    )
    assert result["status"] == "BLOCKED"
    assert result["blocker_code"] == "REPRESENTATIVE_EXTERNAL_INPUTS_REQUIRED"


def test_full64_execution_requires_the_frozen_representative_winner(tmp_path) -> None:
    run_dir = tmp_path / "run"
    lock = run_dir / "method-locks" / "qwen38_p16s1_top4.json"
    _write(lock, {"profile": "qwen38_p16s1_top4"})
    result = run_full64(
        run_dir=run_dir,
        profile="qwen38_p16s1_top4",
        layers="0-63",
        execute=True,
        source_dir=tmp_path / "source",
        activation_root=tmp_path / "fit",
        dev_activation_root=tmp_path / "dev",
    )
    assert result["status"] == "BLOCKED"
    assert result["blocker_code"] == "FULL64_WINNER_RECEIPT_REQUIRED"
