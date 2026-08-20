#!/usr/bin/env python3
"""Merge the two topology-specific development receipts before promotion lock."""

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


def merge_finalists(*, run_dir: Path, p16_receipt: Path | None = None, p32_receipt: Path | None = None, method_version: str | None = None) -> dict[str, Any]:
    paths = [p16_receipt or run_dir / "development" / "p16-exhaustive-receipt.json", p32_receipt or run_dir / "development" / "p32-bounded-pool-receipt.json"]
    payloads: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            return {"status": "BLOCKED", "blocker_code": "TOPOLOGY_FINALISTS_REQUIRED", "missing": str(path)}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") not in {"DEV_FINALISTS", "REUSED"}:
            return {"status": "BLOCKED", "blocker_code": "TOPOLOGY_FINALIST_NOT_READY", "path": str(path), "status_seen": payload.get("status")}
        if payload.get("opened_evaluation_tiers"):
            return {"status": "BLOCKED", "blocker_code": "DEVELOPMENT_RECEIPT_OPENED_EVALUATION", "path": str(path)}
        payloads.append(payload)
    versions = {str(payload.get("method_version", method_version or "")) for payload in payloads}
    if len(versions) != 1 or not next(iter(versions)):
        return {"status": "BLOCKED", "blocker_code": "METHOD_VERSION_MISMATCH"}
    if method_version is not None and next(iter(versions)) != method_version:
        return {"status": "BLOCKED", "blocker_code": "METHOD_VERSION_MISMATCH"}
    profiles: dict[str, Any] = {}
    finalists: list[Any] = []
    for payload in payloads:
        for name, entry in dict(payload.get("profiles", {})).items():
            if name in profiles:
                return {"status": "BLOCKED", "blocker_code": "DUPLICATE_FINALIST_PROFILE", "profile": name}
            profiles[name] = entry
        finalists.extend(payload.get("finalists", []))
    receipt = {"schema_version": 1, "receipt_type": "dense2moe-development-finalists", "status": "DEV_FINALISTS", "method_version": next(iter(versions)), "profiles": profiles, "finalists": finalists, "topology_receipts": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths], "opened_evaluation_tiers": [], "external_tuning_forbidden": True, "code_commit": current_git_commit()}
    output = run_dir / "development" / "finalists.json"
    write_immutable_json(output, receipt)
    return {"status": receipt["status"], "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "profiles": sorted(profiles)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--p16-receipt", type=Path)
    parser.add_argument("--p32-receipt", type=Path)
    parser.add_argument("--method-version")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = merge_finalists(run_dir=args.run_dir, p16_receipt=args.p16_receipt, p32_receipt=args.p32_receipt, method_version=args.method_version)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "DEV_FINALISTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
