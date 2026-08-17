#!/usr/bin/env python3
"""Freeze development finalists before any promotion tier is opened."""

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


def lock_development_finalists(*, run_dir: Path, profiles: list[str], method_version: str, threshold_fingerprint: str = "sealed-qwen38-promotion-v1", finalists_path: Path | None = None) -> dict[str, Any]:
    source = finalists_path or run_dir / "development" / "finalists.json"
    if not source.exists():
        return {"status": "BLOCKED", "blocker_code": "DEVELOPMENT_FINALISTS_REQUIRED", "message": "development finalists must be frozen before opening any promotion tier"}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") not in {"DEV_FINALISTS", "DEV_FINALIST"}:
        return {"status": "BLOCKED", "blocker_code": "INVALID_DEVELOPMENT_FINALISTS", "message": "finalists receipt must be a development-only finalist receipt"}
    listed = payload.get("profiles")
    if not isinstance(listed, dict):
        return {"status": "BLOCKED", "blocker_code": "FINALIST_PROFILE_MAP_REQUIRED", "message": "finalists receipt must contain one entry per active profile"}
    missing = [profile for profile in profiles if profile not in listed]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "FINALIST_PROFILE_MISSING", "missing_profiles": missing}
    for profile in profiles:
        entry = listed[profile]
        if not isinstance(entry, dict) or entry.get("status") != "DEV_FINALIST":
            return {"status": "BLOCKED", "blocker_code": "PROFILE_NOT_DEV_FINALIST", "profile": profile}
        if not str(entry.get("checkpoint_sha256", "")) or not str(entry.get("dataset_hash", "")):
            return {"status": "BLOCKED", "blocker_code": "FINALIST_IDENTITY_INCOMPLETE", "profile": profile}
    lock = {
        "schema_version": 1,
        "receipt_type": "dense2moe-development-finalist-lock",
        "status": "METHOD_LOCKED",
        "method_version": method_version,
        "threshold_fingerprint": threshold_fingerprint,
        "profiles": profiles,
        "finalists_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "finalists": listed,
        "seeds": [17, 29, 41],
        "external_tuning_forbidden": True,
        "opened_evaluation_tiers": [],
        "code_commit": current_git_commit(),
    }
    path = run_dir / "development" / "finalist-lock.json"
    write_immutable_json(path, lock)
    return {"status": lock["status"], "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "method_version": method_version}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--threshold-fingerprint", default="sealed-qwen38-promotion-v1")
    parser.add_argument("--finalists", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = lock_development_finalists(run_dir=args.run_dir, profiles=args.profiles, method_version=args.method_version, threshold_fingerprint=args.threshold_fingerprint, finalists_path=args.finalists)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["status"] == "METHOD_LOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
