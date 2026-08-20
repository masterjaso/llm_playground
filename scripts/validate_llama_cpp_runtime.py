#!/usr/bin/env python3
"""Validate native-Windows llama.cpp loading and sparse dispatch evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import write_immutable_json


def validate(*, run_dir: Path, profile: str) -> dict[str, Any]:
    executable = shutil.which("llama-cli") or shutil.which("llama-cli.exe")
    gguf = run_dir / "artifacts" / f"{profile}-f16.gguf"
    if executable is None or not gguf.exists():
        result = {"status": "BLOCKED", "blocker_code": "LLAMA_CPP_RUNTIME_REQUIRED", "profile": profile, "executable": executable, "gguf": str(gguf), "message": "native-Windows llama-cli and winner GGUF are required; no runtime claim is made"}
    else:
        result = {"status": "BLOCKED", "blocker_code": "LLAMA_CPP_SPARSE_PATCH_REQUIRED", "profile": profile, "executable": executable, "message": "the pinned llama.cpp Qwen35MoE sparse-dispatch patch must be exercised before runtime green"}
    write_immutable_json(run_dir / "quant" / "runtime-discovery.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate(run_dir=args.run_dir, profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "LLAMA_CPP_RUNTIME_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())

