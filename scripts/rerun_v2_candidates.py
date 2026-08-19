"""Emit terminal rerun statuses without overwriting historical receipts.

This orchestration surface deliberately does not retrain or fabricate missing
raw evidence.  A later structural/LM worker can replace ``PARTIAL_METRICS_ONLY``
with a V2 receipt when exact manifests/checkpoints are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dense2moe.evaluation.inventory import TERMINAL_RERUN_STATUSES, CandidateInventoryRecord, terminal_status_for_record


def _records(payload: dict) -> list[CandidateInventoryRecord]:
    return [CandidateInventoryRecord(**item) for item in payload.get("records", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structural-metrics", action="store_true")
    parser.add_argument(
        "--valid-v2-receipts",
        action="store_true",
        help="assert that immutable V2 structural receipts were emitted for eligible records",
    )
    parser.add_argument("--resource-blocked", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    records = _records(inventory)
    results = []
    for record in records:
        status = terminal_status_for_record(
            record,
            structural_metrics_available=args.structural_metrics,
            valid_v2_receipt=args.valid_v2_receipts,
            resource_blocked=args.resource_blocked,
        )
        if status not in TERMINAL_RERUN_STATUSES:
            raise RuntimeError(f"non-terminal rerun status: {status}")
        results.append({**record.as_dict(), "rerun_status": status})
    replayable_unresolved = [
        item["candidate_id"]
        for item in results
        if item["rerun_eligibility"] == "eligible" and item["rerun_status"] != "RERUN_COMPLETE"
    ]
    non_replayable = {"not_replayable", "not_applicable", "blocked_runtime_identity"}
    payload = {
        "schema_version": 2,
        "receipt_type": "dense2moe-v2-candidate-rerun-status-v2",
        "source_inventory": str(args.inventory),
        "candidate_count": len(results),
        "cohort_complete": bool(results) and all(
            item["rerun_status"] == "RERUN_COMPLETE" or item["rerun_eligibility"] in non_replayable
            for item in results
        ),
        "replayable_unresolved_count": len(replayable_unresolved),
        "replayable_unresolved_candidates": sorted(replayable_unresolved),
        "records": results,
        "protected_tiers_opened": False,
        "historical_receipts_overwritten": False,
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite existing rerun status: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
