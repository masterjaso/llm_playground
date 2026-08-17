#!/usr/bin/env python3
"""Validate locked-method representative transfer inputs and receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json


def run_transfer(*, run_dir: Path, layers: str, profiles: list[str], seeds: list[int]) -> dict[str, Any]:
    missing = [profile for profile in profiles if not (run_dir / "method-locks" / f"{profile}.json").exists()]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "METHOD_LOCKS_REQUIRED", "missing_profiles": missing, "message": "representative transfer cannot fit an unlocked method"}
    layer_ids: list[int] = []
    for part in layers.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            layer_ids.extend(range(start, end + 1))
        else:
            layer_ids.append(int(part))
    if sorted(set(layer_ids)) != list(range(4)) + list(range(28, 32)) + list(range(60, 64)):
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_LAYER_MATRIX_INVALID", "message": "exact representative layers 0-3, 28-31, and 60-63 are required"}
    payload = {"schema_version": 1, "receipt_type": "dense2moe-representative-transfer", "profiles": profiles, "layers": sorted(layer_ids), "seeds": seeds, "development_data": "FIT-DEV", "external_data": ["G1", "G2"], "status": "TRANSFER_INPUTS_READY", "evaluation_tiers_opened": []}
    path = run_dir / "representative" / "matrix.json"
    write_immutable_json(path, payload)
    return payload | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_transfer(run_dir=args.run_dir, layers=args.layers, profiles=args.profiles, seeds=[int(value) for value in args.seeds.split(",") if value])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "TRANSFER_INPUTS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

