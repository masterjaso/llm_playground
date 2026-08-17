from __future__ import annotations

import json

from scripts.evaluate_promotion import run_promotion
from scripts.merge_development_finalists import merge_finalists
from scripts.run_candidate_search import run_candidate_search


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
