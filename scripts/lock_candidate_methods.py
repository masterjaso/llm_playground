#!/usr/bin/env python3
"""Freeze separate topology method locks after development/promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json
from dense2moe.provenance import current_git_commit


def lock_methods(*, run_dir: Path, profiles: list[str]) -> dict[str, Any]:
    finalist = run_dir / "development" / "finalists.json"
    promotion = run_dir / "promotion" / "external-generalization.json"
    if not finalist.exists() or not promotion.exists():
        return {"status": "BLOCKED", "blocker_code": "FINALIST_AND_PROMOTION_RECEIPTS_REQUIRED", "message": "method locks cannot be created from development-only or placeholder evidence"}
    locks: dict[str, Any] = {}
    for profile in profiles:
        topology = "p16/top4" if "p16" in profile else "p32/top5" if "p32" in profile else ""
        if not topology:
            raise ValueError(f"inactive profile: {profile}")
        payload = {"schema_version": 1, "receipt_type": "dense2moe-method-lock", "profile": profile, "topology": topology, "finalist_sha256": hashlib.sha256(finalist.read_bytes()).hexdigest(), "promotion_sha256": hashlib.sha256(promotion.read_bytes()).hexdigest(), "code_commit": current_git_commit(), "external_tuning_forbidden": True, "seeds": [17, 29, 41], "winner_status": "UNDECIDED"}
        path = run_dir / "method-locks" / f"{profile}.json"
        write_immutable_json(path, payload)
        locks[profile] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"status": "METHOD_LOCKS_FROZEN", "locks": locks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = lock_methods(run_dir=args.run_dir, profiles=args.profiles)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "METHOD_LOCKS_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())

