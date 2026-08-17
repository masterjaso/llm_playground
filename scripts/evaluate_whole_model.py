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
from dense2moe.evaluation import evaluate_promotion_metrics


def evaluate(*, run_dir: Path, profile: str, tiers: list[str]) -> dict[str, Any]:
    assembly = run_dir / "BF16_SPARSE_MASTER" / profile / "assembly-receipt.json"
    if not assembly.exists():
        return {"status": "BLOCKED", "blocker_code": "BF16_ASSEMBLY_REQUIRED", "profile": profile, "message": "whole-model evaluation requires a strict BF16 assembly"}
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
        results[tier] = evaluate_promotion_metrics(metrics, domain_slices=payload.get("domain_slices"), development_metrics=payload.get("development_metrics"))
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

