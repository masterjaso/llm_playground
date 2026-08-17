#!/usr/bin/env python3
"""Build a deterministic reuse/invalidation index for a science run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import sha256_file, write_immutable_json
from dense2moe.lineage import build_lineage_index, lineage_identity


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def build_index(*, run_id: str, method_version: str, runtime_lock: Path, artifact_specs: list[Path], output: Path) -> dict[str, Any]:
    lock_hash = sha256_file(runtime_lock)
    artifacts: list[dict[str, Any]] = []
    for spec_path in artifact_specs:
        spec = _read_json(spec_path)
        identity_fields = spec.get("identity_fields")
        if not isinstance(identity_fields, dict):
            raise TypeError(f"{spec_path} requires identity_fields")
        identity = lineage_identity(artifact_kind=str(spec.get("artifact_kind", "receipt")), **identity_fields)
        existing_identity = spec.get("existing_identity") if isinstance(spec.get("existing_identity"), dict) else None
        artifacts.append({
            "name": str(spec.get("name", spec_path.stem)),
            "path": str(spec.get("path", spec_path)),
            "identity": identity,
            "existing_identity": existing_identity,
            "historical": bool(spec.get("historical", False)),
            "blocked_reason": spec.get("blocked_reason"),
        })
    result = build_lineage_index(run_id=run_id, method_version=method_version, runtime_lock_sha256=lock_hash, artifacts=artifacts)
    result["runtime_lock"] = {"path": str(runtime_lock), "sha256": lock_hash}
    result["artifact_specs"] = [str(path) for path in artifact_specs]
    write_immutable_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--artifact-spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_index(run_id=args.run_id, method_version=args.method_version, runtime_lock=args.runtime_lock, artifact_specs=args.artifact_spec, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["receipt_type"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
