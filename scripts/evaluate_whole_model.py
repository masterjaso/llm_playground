#!/usr/bin/env python3
"""Evaluate frozen whole-model receipts on internal and fresh corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json
from dense2moe.evaluation import evaluate_whole_model_metrics


def evaluate(*, run_dir: Path, profile: str, tiers: list[str]) -> dict[str, Any]:
    assembly = run_dir / "BF16_SPARSE_MASTER" / profile / "assembly-receipt.json"
    if not assembly.exists():
        return {"status": "BLOCKED", "blocker_code": "BF16_ASSEMBLY_REQUIRED", "profile": profile, "message": "whole-model evaluation requires a strict BF16 assembly"}
    try:
        assembly_payload = json.loads(assembly.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED", "blocker_code": "BF16_ASSEMBLY_INVALID", "profile": profile, "message": str(exc)}
    if assembly_payload.get("status") != "BF16_SPARSE_MASTER_ASSEMBLED" or assembly_payload.get("layer_checkpoint_application", {}).get("status") != "FULL64_LAYER_CHECKPOINTS_APPLIED":
        return {"status": "BLOCKED", "blocker_code": "BF16_FULL64_APPLICATION_REQUIRED", "profile": profile, "message": "whole-model evaluation requires a full64 checkpoint-applied BF16 assembly"}
    if "POST-ASSEMBLY-FRESH" not in tiers:
        return {"status": "BLOCKED", "blocker_code": "POST_ASSEMBLY_FRESH_TIER_REQUIRED", "profile": profile, "message": "whole-model green requires a fresh post-assembly corpus"}
    metric_paths = {tier: run_dir / "validation" / f"{profile}-{tier}.json" for tier in tiers}
    missing = [tier for tier, path in metric_paths.items() if not path.exists()]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "WHOLE_MODEL_METRICS_REQUIRED", "profile": profile, "missing_tiers": missing, "message": "no synthetic whole-model metrics are emitted"}
    results: dict[str, Any] = {}
    for tier, path in metric_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return {"status": "BLOCKED", "blocker_code": "INVALID_WHOLE_MODEL_METRICS", "profile": profile, "tier": tier}
        if not str(payload.get("dataset_hash", "")) or payload.get("external_data_untouched") is not True:
            return {"status": "BLOCKED", "blocker_code": "WHOLE_MODEL_PROVENANCE_REQUIRED", "profile": profile, "tier": tier, "message": "whole-model metrics must bind a dataset hash and prove untouched evaluation data"}
        results[tier] = evaluate_whole_model_metrics(metrics, domain_slices=payload.get("domain_slices"))
    status = "WHOLE_MODEL_GREEN" if all(item["overall"] == "green" for item in results.values()) else "WHOLE_MODEL_REJECTED"
    result = {"status": status, "profile": profile, "tiers": results, "fresh_post_assembly_required": "POST-ASSEMBLY-FRESH" in tiers, "assembly_receipt": str(assembly)}
    path = run_dir / "validation" / f"{profile}-whole-model.json"
    write_immutable_json(path, result)
    return result | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tier", action="append", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate(run_dir=args.run_dir, profile=args.profile, tiers=args.tier)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "WHOLE_MODEL_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
