#!/usr/bin/env python3
"""Export a validated winner through the tensor-bearing GGUF contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.export.gguf import export_gguf, validate_gguf


def export(*, run_dir: Path, profiles: list[str], llama_cpp_revision: str) -> dict[str, Any]:
    if not llama_cpp_revision or llama_cpp_revision.startswith("<"):
        return {"status": "BLOCKED", "blocker_code": "LLAMA_CPP_REVISION_REQUIRED", "message": "pin a tested llama.cpp revision before GGUF export"}
    winner = run_dir / "representative" / "decision.json"
    if not winner.exists():
        return {"status": "BLOCKED", "blocker_code": "WINNER_DECISION_REQUIRED", "message": "p16/p32 winner decision is required before GGUF export"}
    payload = json.loads(winner.read_text(encoding="utf-8"))
    profile = str(payload.get("winner_profile", ""))
    if profile not in profiles:
        return {"status": "BLOCKED", "blocker_code": "WINNER_PROFILE_INVALID", "message": "winner profile is not one of the active profiles"}
    manifest = run_dir / "BF16_SPARSE_MASTER" / profile / "manifest.json"
    if not manifest.exists():
        return {"status": "BLOCKED", "blocker_code": "BF16_MANIFEST_REQUIRED", "message": "validated BF16 sparse master manifest is required"}
    destination = run_dir / "artifacts" / f"{profile}-f16.gguf"
    try:
        result = export_gguf(manifest, destination, metadata={"profile": profile, "topology": payload.get("winner_topology"), "llama_cpp_revision": llama_cpp_revision, "sparse_dispatch_required": True})
        validation = validate_gguf(destination, require_receipt=True)
        return {"status": "GGUF_READY", "profile": profile, "path": str(destination), "export": result, "validation": validation, "llama_cpp_revision": llama_cpp_revision}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "BLOCKED", "blocker_code": "GGUF_EXPORT_FAILED", "message": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--llama-cpp-revision", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = export(run_dir=args.run_dir, profiles=args.profiles, llama_cpp_revision=args.llama_cpp_revision)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "GGUF_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

