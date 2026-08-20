#!/usr/bin/env python3
"""Build an expert-covering imatrix or fail closed when llama.cpp is absent."""

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


def build(*, run_dir: Path, profile: str) -> dict[str, Any]:
    gguf = run_dir / "artifacts" / f"{profile}-f16.gguf"
    corpus_candidates = [
        run_dir / "corpus-v2.2-receipt.json",
        run_dir / "corpus-v2.2" / "corpus-v2.2-receipt.json",
    ]
    corpus = next((candidate for candidate in corpus_candidates if candidate.exists()), corpus_candidates[0])
    llama = shutil.which("llama-imatrix") or shutil.which("llama-imatrix.exe")
    if not gguf.exists() or not corpus.exists():
        result = {"status": "BLOCKED", "blocker_code": "IMATRIX_INPUTS_REQUIRED", "profile": profile, "message": "validated winner GGUF and sealed Corpus V2.2 receipt are required"}
    elif llama is None:
        result = {"status": "BLOCKED", "blocker_code": "LLAMA_CPP_IMATRIX_REQUIRED", "profile": profile, "message": "pinned native-Windows llama-imatrix executable was not found"}
    else:
        result = {"status": "BLOCKED", "blocker_code": "IMATRIX_EXECUTION_NOT_CONFIGURED", "profile": profile, "executable": llama, "message": "expert-covering imatrix command must be bound to the pinned runtime and corpus manifest"}
    write_immutable_json(run_dir / "quant" / "imatrix-receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(run_dir=args.run_dir, profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "IMATRIX_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
