#!/usr/bin/env python3
"""Validate fresh-process reload receipts for converted Qwen checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json


def validate(*, run_dir: Path, profiles: list[str], fresh_process: bool) -> dict[str, Any]:
    if not fresh_process:
        return {"status": "BLOCKED", "blocker_code": "FRESH_PROCESS_REQUIRED", "message": "reload parity must run in a fresh process"}
    missing = []
    receipts: dict[str, Any] = {}
    for profile in profiles:
        path = run_dir / "BF16_SPARSE_MASTER" / profile / "assembly-receipt.json"
        if not path.exists():
            missing.append(profile)
        else:
            receipts[profile] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        return {"status": "BLOCKED", "blocker_code": "ASSEMBLED_CHECKPOINTS_REQUIRED", "missing_profiles": missing, "message": "no reload claim is made without assembled checkpoint receipts"}
    payload = {"schema_version": 1, "receipt_type": "dense2moe-fresh-process-reload-parity", "status": "RELOAD_INPUTS_READY", "fresh_process": True, "profiles": profiles, "assembly_receipts": receipts, "logit_parity_measured": False, "message": "execute the pinned source-backed logits probe before promotion"}
    path = run_dir / "BF16_SPARSE_MASTER" / "reload-receipt.json"
    write_immutable_json(path, payload)
    return payload | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--fresh-process", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate(run_dir=args.run_dir, profiles=args.profiles, fresh_process=args.fresh_process)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "RELOAD_INPUTS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

