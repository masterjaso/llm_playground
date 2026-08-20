#!/usr/bin/env python3
"""Assemble a real Qwen3.5 BF16 sparse checkpoint from a locked winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.models.qwen35_full import Qwen35DenseToMoE, apply_layer_checkpoints


def assemble(*, run_dir: Path, profile: str, source_dir: Path | None, strict: bool) -> dict[str, Any]:
    queue = run_dir / "full64" / "layer-queue.json"
    if not queue.exists():
        return {"status": "BLOCKED", "blocker_code": "FULL64_QUEUE_REQUIRED", "message": "full64 queue is required before Qwen assembly"}
    queue_payload = json.loads(queue.read_text(encoding="utf-8"))
    if queue_payload.get("status") != "QUEUE_READY" or len(queue_payload.get("layers", [])) != 64:
        return {"status": "BLOCKED", "blocker_code": "FULL64_CHECKPOINTS_REQUIRED", "message": "all 64 validated layer checkpoints are required; queue metadata alone is not assembly evidence"}
    checkpoint_manifest = run_dir / "full64" / "checkpoints-manifest.json"
    if not checkpoint_manifest.exists():
        return {"status": "BLOCKED", "blocker_code": "FULL64_CHECKPOINT_MANIFEST_REQUIRED", "message": "checkpoint hashes and profile identity are required before assembly"}
    try:
        manifest_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED", "blocker_code": "FULL64_CHECKPOINT_MANIFEST_INVALID", "message": str(exc)}
    if manifest_payload.get("complete") is not True or len(manifest_payload.get("layers", [])) != 64:
        return {"status": "BLOCKED", "blocker_code": "FULL64_TRAINED_CHECKPOINTS_REQUIRED", "message": "a complete manifest with 64 validated trained layer tensors is required; dense slicing is not an assembly fallback"}
    source = source_dir or (run_dir / "source")
    if not source.exists():
        return {"status": "BLOCKED", "blocker_code": "PINNED_QWEN_SOURCE_REQUIRED", "message": "local pinned Qwen3.5 source snapshot is unavailable"}
    try:
        model = Qwen35DenseToMoE.from_pretrained(source, topology=profile, local_files_only=True, strict_layer_count=64)
        layer_receipt = apply_layer_checkpoints(model, checkpoint_manifest, expected_profile=profile, strict_layer_count=64)
        destination = run_dir / "BF16_SPARSE_MASTER" / profile
        model.save_pretrained(destination)
        receipt = dict(model.receipt)
        receipt.update({"status": "BF16_SPARSE_MASTER_ASSEMBLED", "profile": profile, "strict": strict, "destination": str(destination), "full64_checkpoint_manifest": str(checkpoint_manifest), "layer_checkpoint_application": layer_receipt})
        (destination / "assembly-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return receipt
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return {"status": "BLOCKED", "blocker_code": "QWEN_FULL_ASSEMBLY_FAILED", "profile": profile, "message": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = assemble(run_dir=args.run_dir, profile=args.profile, source_dir=args.source_dir, strict=args.strict)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "BF16_SPARSE_MASTER_ASSEMBLED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
