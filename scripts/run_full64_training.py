#!/usr/bin/env python3
"""Create a resumable, hash-addressed 64-layer training queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_active_config
from dense2moe.data import write_immutable_json


def run_full64(*, run_dir: Path, profile: str, layers: str) -> dict[str, Any]:
    lock = run_dir / "method-locks" / f"{profile}.json"
    if not lock.exists():
        return {"status": "BLOCKED", "blocker_code": "METHOD_LOCK_REQUIRED", "message": f"no method lock for {profile}"}
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{profile}.yaml"
    config = load_active_config(config_path)[0]
    if layers != "0-63":
        return {"status": "BLOCKED", "blocker_code": "FULL64_LAYER_RANGE_REQUIRED", "message": "full64 requires the explicit 0-63 layer range"}
    payload = {"schema_version": 1, "receipt_type": "dense2moe-full64-training-queue", "status": "QUEUE_READY", "profile": profile, "topology": config.topology_id, "layers": list(range(config.num_hidden_layers)), "resumable": True, "one_active_full64_build": True, "method_lock": str(lock), "checkpoints": [], "external_evaluation_tuning": False}
    path = run_dir / "full64" / "layer-queue.json"
    write_immutable_json(path, payload)
    return payload | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_full64(run_dir=args.run_dir, profile=args.profile, layers=args.layers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "QUEUE_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

