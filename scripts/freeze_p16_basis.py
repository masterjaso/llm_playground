#!/usr/bin/env python3
"""Verify and receipt the frozen p16/top4 expert/shared basis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.provenance import current_git_commit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="layer-checkpoints/clean-validation/p16-top4-refined-course-correction-continue",
    )
    parser.add_argument("--partition", default="partitions/high-sparsity-p16-top4.json")
    parser.add_argument("--output", default="reports/p16-top4-frozen-basis-20260816.json")
    args = parser.parse_args()
    run = args.run_dir
    checkpoint_dir = run / args.checkpoint
    metadata_path = checkpoint_dir / "layer-0000.json"
    tensor_path = checkpoint_dir / "layer-0000.safetensors"
    partition_path = run / args.partition
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("profile") != "qwen38_p16s1_top4":
        raise ValueError(f"unexpected basis profile: {metadata.get('profile')!r}")
    observed_tensor_hash = _sha256(tensor_path)
    recorded_tensor_hash = str(metadata.get("tensor_sha256", ""))
    if observed_tensor_hash != recorded_tensor_hash:
        raise ValueError("basis tensor SHA-256 does not match checkpoint metadata")
    partition_payload = json.loads(partition_path.read_text(encoding="utf-8"))
    canonical_plan = partition_payload.get("plan", partition_payload)
    # Layer checkpoints historically hash ``plan.as_dict()`` with sorted JSON
    # keys and default separators.  Reproduce that exact recipe so the freeze
    # receipt can be checked against the checkpoint metadata byte-for-byte.
    plan_hash = hashlib.sha256(json.dumps(canonical_plan, sort_keys=True).encode()).hexdigest()
    recorded_plan_hash = str(metadata.get("partition_hash", ""))
    if plan_hash != recorded_plan_hash:
        raise ValueError("basis partition canonical hash does not match checkpoint metadata")
    inventory = metadata.get("tensor_inventory", {})
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("basis checkpoint has no tensor inventory")
    basis_keys = sorted(
        key
        for key in inventory
        if any(token in key for token in ("expert_", "shared_", "expert_scales"))
    )
    router_keys = sorted(key for key in inventory if "router" in key or "amplitude" in key)
    receipt = {
        "schema_version": 1,
        "status": "P16_TOP4_BASIS_FROZEN",
        "classification": "FROZEN_EXPERT_SHARED_BASIS_SELECTOR_ONLY_RESEARCH",
        "profile": metadata["profile"],
        "checkpoint": str(metadata_path),
        "tensor_file": str(tensor_path),
        "tensor_sha256": observed_tensor_hash,
        "partition_file": str(partition_path),
        "partition_file_sha256": _sha256(partition_path),
        "partition_canonical_sha256": plan_hash,
        "basis_checkpoint_code_sha": metadata.get("code_commit"),
        "receipt_writer_code_sha": current_git_commit(),
        "source_revision": metadata.get("source_revision"),
        "source_config_hash": metadata.get("source_config_hash"),
        "source_index_hash": metadata.get("source_index_hash"),
        "dataset_hash": metadata.get("dataset_hash"),
        "basis_tensor_count": len(basis_keys),
        "basis_tensor_keys": basis_keys,
        "router_tensor_count": len(router_keys),
        "router_tensor_keys": router_keys,
        "trainable_scope_for_next_experiment": [
            "selection_router",
            "positive_amplitude_router",
        ],
        "frozen_scope": [
            "shared_basis",
            "routed_expert_basis",
            "expert_scales",
        ],
        "validation_metrics_at_basis_checkpoint": metadata.get("quality_gate", {}).get("metrics"),
        "holdout_opened": False,
        "holdout_tuning": False,
    }
    output = run / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output), "tensor_sha256": observed_tensor_hash, "partition_canonical_sha256": plan_hash, "code_sha": receipt["receipt_writer_code_sha"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
