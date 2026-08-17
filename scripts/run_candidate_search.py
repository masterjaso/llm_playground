#!/usr/bin/env python3
"""Run the development-only topology candidate search contract.

This command performs deterministic pool construction and records the exact
search class.  It never opens GATE/SHADOW/G1/G2 and never treats a bounded p32
pool as exhaustive.  Actual score receipts must be supplied by the training
worker before a finalist can be promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import sha256_file, write_immutable_json


def _record_count(path: Path) -> int:
    if path.suffix.casefold() == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("rows", "records", "selected_rows", "shards"):
            if isinstance(payload.get(key), list):
                return len(payload[key])
        return 1
    return 0


def run_candidate_search(
    *,
    run_dir: Path,
    activation_manifest: Path,
    dev_manifest: Path,
    topology: str,
    exhaustive: bool = False,
    bounded: bool = False,
    expected_combinations: int | None = None,
    candidate_pool_size: int | None = None,
) -> dict[str, Any]:
    if topology not in {"p16/top4", "p32/top5"}:
        raise ValueError("only p16/top4 and p32/top5 are active")
    if not activation_manifest.exists() or not dev_manifest.exists():
        return {
            "status": "BLOCKED",
            "blocker_code": "DEVELOPMENT_ACTIVATIONS_REQUIRED",
            "topology": topology,
            "message": "FIT-TRAIN and FIT-DEV activation manifests are required; evaluation data is not a fallback",
        }
    if topology == "p16/top4":
        total = sum(1 for _ in itertools.combinations(range(16), 4))
        if not exhaustive or expected_combinations != total:
            raise ValueError(f"p16 requires explicit exhaustive C(16,4)={total} search")
        search_class = "EXHAUSTIVE_C(16,4)"
        pool_size = total
    else:
        if not bounded or not candidate_pool_size or candidate_pool_size <= 0:
            raise ValueError("p32 requires an explicitly bounded positive candidate pool")
        search_class = "BOUNDED_CORRELATION_POOL_NOT_EXHAUSTIVE"
        pool_size = int(candidate_pool_size)
    receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-development-candidate-search",
        "status": "CANDIDATE_SEARCH_READY",
        "topology": topology,
        "partition": "FIT-TRAIN",
        "ranking_partition": "FIT-DEV",
        "activation_manifest": {"path": str(activation_manifest), "sha256": sha256_file(activation_manifest), "records": _record_count(activation_manifest)},
        "dev_manifest": {"path": str(dev_manifest), "sha256": sha256_file(dev_manifest), "records": _record_count(dev_manifest)},
        "search_class": search_class,
        "exhaustive": topology == "p16/top4",
        "candidate_pool_size": pool_size,
        "basis_freeze_required": True,
        "selector_seeds_required": [17, 29, 41],
        "amplitude_router_required": True,
        "opened_evaluation_tiers": [],
        "promotion_status": "DEV_FINALIST_NOT_YET_EVALUATED",
        "code_commit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = run_dir / "development" / ("p16-exhaustive-receipt.json" if topology == "p16/top4" else "p32-bounded-pool-receipt.json")
    write_immutable_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), required=True)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--expected-combinations", type=int)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_candidate_search(
        run_dir=args.run_dir,
        activation_manifest=args.activation_manifest,
        dev_manifest=args.dev_manifest,
        topology=args.topology,
        exhaustive=args.exhaustive,
        bounded=args.bounded,
        expected_combinations=args.expected_combinations,
        candidate_pool_size=args.candidate_pool_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

