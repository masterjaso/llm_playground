#!/usr/bin/env python3
"""Run blocking HF sparse-dispatch runtime checks for active profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json


def validate(*, run_dir: Path, profiles: list[str]) -> dict[str, Any]:
    missing_assemblies: list[str] = []
    for profile in profiles:
        receipt = run_dir / "BF16_SPARSE_MASTER" / profile / "assembly-receipt.json"
        if not receipt.exists():
            missing_assemblies.append(profile)
            continue
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            missing_assemblies.append(profile)
            continue
        if payload.get("status") != "BF16_SPARSE_MASTER_ASSEMBLED" or payload.get("layer_checkpoint_application", {}).get("status") != "FULL64_LAYER_CHECKPOINTS_APPLIED":
            missing_assemblies.append(profile)
    if missing_assemblies:
        payload = {
            "schema_version": 1,
            "receipt_type": "dense2moe-hf-sparse-runtime",
            "status": "BLOCKED",
            "blocker_code": "BF16_FULL64_ASSEMBLIES_REQUIRED",
            "profiles": profiles,
            "missing_profiles": missing_assemblies,
            "masked_reference_equivalence": False,
            "gradient_equivalence": False,
            "dispatch_count_evidence": False,
            "message": "HF runtime equivalence is blocked until every requested profile has a checkpoint-applied BF16 full64 assembly",
        }
        path = run_dir / "runtime" / "hf-sparse-receipt.json"
        write_immutable_json(path, payload)
        return payload | {"path": str(path)}
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_sparse_dispatch.py", "tests/test_torch_target.py"]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
    payload = {"schema_version": 1, "receipt_type": "dense2moe-hf-sparse-runtime", "status": "HF_SPARSE_RUNTIME_GREEN" if completed.returncode == 0 else "HF_SPARSE_RUNTIME_REJECTED", "profiles": profiles, "command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:], "masked_reference_equivalence": completed.returncode == 0, "gradient_equivalence": completed.returncode == 0, "dispatch_count_evidence": completed.returncode == 0}
    path = run_dir / "runtime" / "hf-sparse-receipt.json"
    write_immutable_json(path, payload)
    return payload | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate(run_dir=args.run_dir, profiles=args.profiles)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "HF_SPARSE_RUNTIME_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
