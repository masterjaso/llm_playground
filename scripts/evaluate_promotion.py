#!/usr/bin/env python3
"""Open sealed promotion tiers and evaluate frozen finalists without fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    new_contamination_ledger,
    open_evaluation_tier,
    retire_evaluation_tier,
    validate_contamination_ledger,
    write_immutable_json,
)
from dense2moe.evaluation import evaluate_promotion_metrics
from dense2moe.provenance import current_git_commit


def run_promotion(*, run_dir: Path, method_version: str, tiers: list[str]) -> dict[str, Any]:
    finalist_lock = run_dir / "promotion" / "finalist-lock.json"
    if not finalist_lock.exists():
        return {"status": "BLOCKED", "blocker_code": "FINALIST_LOCK_REQUIRED", "message": "frozen development finalists and thresholds must be locked before opening promotion tiers"}
    ledger_path = run_dir / "promotion" / "contamination-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = new_contamination_ledger(
            method_version=method_version,
            code_commit=current_git_commit(),
            thresholds_fingerprint="sealed-qwen38-promotion-v1",
            runtime_lock_sha256="",
            corpus_hashes={tier: "" for tier in tiers},
        )
    results: dict[str, Any] = {}
    for tier in tiers:
        metrics_path = run_dir / "promotion" / f"{tier}.json"
        if not metrics_path.exists():
            results[tier] = {"status": "BLOCKED", "blocker_code": "FROZEN_METRICS_REQUIRED", "message": f"no frozen metric receipt for {tier}; no tier opening occurred"}
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_hash = str(payload.get("dataset_hash", ""))
        metrics = payload.get("metrics")
        if not dataset_hash or not isinstance(metrics, dict):
            results[tier] = {"status": "BLOCKED", "blocker_code": "INVALID_METRIC_RECEIPT", "message": f"{metrics_path} must contain dataset_hash and metrics"}
            continue
        try:
            ledger = open_evaluation_tier(
                ledger,
                tier=tier,
                dataset_hash=dataset_hash,
                method_version=method_version,
                code_commit=current_git_commit(),
                thresholds_fingerprint="sealed-qwen38-promotion-v1",
            )
            gate = evaluate_promotion_metrics(metrics, domain_slices=payload.get("domain_slices"), development_metrics=payload.get("development_metrics"))
            result = {"status": "GREEN" if gate["overall"] == "green" else "REJECTED", "tier": tier, "gate": gate}
            results[tier] = result
            if result["status"] != "GREEN":
                ledger = retire_evaluation_tier(ledger, tier=tier, reason="frozen finalist failed promotion gate")
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            results[tier] = {"status": "BLOCKED", "tier": tier, "message": str(exc)}
    write_immutable_json(ledger_path, ledger)
    return {"status": "PROMOTION_GREEN" if results and all(item.get("status") == "GREEN" for item in results.values()) else "PROMOTION_REJECTED", "method_version": method_version, "tiers": results, "contamination_ledger": str(ledger_path), "ledger_validation": validate_contamination_ledger(ledger)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--tier", action="append", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_promotion(run_dir=args.run_dir, method_version=args.method_version, tiers=args.tier)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PROMOTION_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())

