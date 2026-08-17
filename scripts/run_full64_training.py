#!/usr/bin/env python3
"""Create and execute a resumable, hash-addressed 64-layer training queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_active_config, load_config
from dense2moe.data import sha256_file, write_immutable_json


def _find_manifest(root: Path, layer: int, role: str) -> Path | None:
    names = [
        f"layer-{layer:04d}-{role}.json",
        f"layer-{layer:04d}-{role.upper()}.json",
        f"layer-{layer:04d}-train.json" if role == "FIT-TRAIN" else f"layer-{layer:04d}-dev.json",
        f"layer-{layer:04d}-FIT-DEV.json" if role == "FIT-DEV" else f"layer-{layer:04d}.json",
    ]
    return next((root / name for name in names if (root / name).exists()), None)


def _find_partition(run_dir: Path, profile: str, layer: int, development_run_dir: Path | None) -> Path | None:
    candidates = [
        run_dir / "partitions" / f"layer-{layer:04d}.json",
        run_dir / "partitions" / f"{profile}.json",
    ]
    if development_run_dir is not None:
        candidates.extend(sorted((development_run_dir / "development" / "partitions").glob(f"{'p16' if 'p16' in profile else 'p32'}-top*.json")))
    return next((path for path in candidates if path.exists()), None)


def _layer_lineage(*, method_lock_sha256: str, train_manifest: Path, dev_manifest: Path, profile: Any, partition: Path, layer: int, seed: int, device: str, epochs: int, microbatch: int, learning_rate: float) -> dict[str, Any]:
    profile_hash = hashlib.sha256(json.dumps(profile.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "layer": layer,
        "method_lock_sha256": method_lock_sha256,
        "profile": profile.name,
        "profile_hash": profile_hash,
        "partition_sha256": sha256_file(partition),
        "train_manifest_sha256": sha256_file(train_manifest),
        "dev_manifest_sha256": sha256_file(dev_manifest),
        "seed": seed,
        "optimizer": {"device": device, "epochs": epochs, "microbatch": microbatch, "learning_rate": learning_rate},
    }


def _train_layer(*, method_lock_sha256: str, source_dir: Path, train_manifest: Path, dev_manifest: Path, output_dir: Path, layer: int, profile: Any, partition: Path, seed: int, device: str, epochs: int, microbatch: int, learning_rate: float) -> dict[str, Any]:
    from dense2moe.training import train_torch_layer

    metadata_path = output_dir / f"layer-{layer:04d}.json"
    tensor_path = output_dir / f"layer-{layer:04d}.safetensors"
    lineage_path = output_dir / f"layer-{layer:04d}.lineage.json"
    expected_lineage = _layer_lineage(method_lock_sha256=method_lock_sha256, train_manifest=train_manifest, dev_manifest=dev_manifest, profile=profile, partition=partition, layer=layer, seed=seed, device=device, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate)
    if metadata_path.exists() and tensor_path.exists() and lineage_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage == expected_lineage and int(payload.get("training_seed", -1)) == seed and payload.get("partition_hash"):
            return {"status": "REUSED", "layer": layer, "metadata": str(metadata_path), "lineage": str(lineage_path), "train_manifest_sha256": expected_lineage["train_manifest_sha256"], "dev_manifest_sha256": expected_lineage["dev_manifest_sha256"]}
    result = train_torch_layer(source_dir=source_dir, activation_manifest=train_manifest, selection_manifest=dev_manifest, output_dir=output_dir, layer=layer, profile=profile, partition_path=partition, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate, device=device, seed=seed, source_revision=profile.revision, evaluate_holdout=False)
    write_immutable_json(lineage_path, expected_lineage)
    result.update({"layer": layer, "train_manifest_sha256": sha256_file(train_manifest), "dev_manifest_sha256": sha256_file(dev_manifest)})
    return result


def run_full64(*, run_dir: Path, profile: str, layers: str, execute: bool = False, source_dir: Path | None = None, activation_root: Path | None = None, dev_activation_root: Path | None = None, development_run_dir: Path | None = None, device: str = "cpu", epochs: int = 1, microbatch: int = 8, learning_rate: float = 1e-3, resume: bool = False) -> dict[str, Any]:
    lock = run_dir / "method-locks" / f"{profile}.json"
    if not lock.exists():
        return {"status": "BLOCKED", "blocker_code": "METHOD_LOCK_REQUIRED", "message": f"no method lock for {profile}"}
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{profile}.yaml"
    config = load_active_config(config_path)[0]
    if layers != "0-63":
        return {"status": "BLOCKED", "blocker_code": "FULL64_LAYER_RANGE_REQUIRED", "message": "full64 requires the explicit 0-63 layer range"}
    queue_payload = {"schema_version": 2, "receipt_type": "dense2moe-full64-training-queue", "status": "QUEUE_READY", "profile": profile, "topology": config.topology_id, "layers": list(range(config.num_hidden_layers)), "resumable": True, "one_active_full64_build": True, "method_lock": str(lock), "checkpoints": [], "external_evaluation_tuning": False}
    queue_path = run_dir / "full64" / "layer-queue.json"
    write_immutable_json(queue_path, queue_payload)
    if not execute:
        return queue_payload | {"path": str(queue_path)}
    if source_dir is None or activation_root is None or dev_activation_root is None:
        return {"status": "BLOCKED", "blocker_code": "FULL64_EXECUTION_INPUTS_REQUIRED", "message": "--source-dir, --activation-root, and --dev-activation-root are required with --execute", "queue": str(queue_path)}
    marker = run_dir / "full64" / "active-build.json"
    if marker.exists():
        active = json.loads(marker.read_text(encoding="utf-8"))
        if active.get("profile") != profile:
            return {"status": "BLOCKED", "blocker_code": "FULL64_BUILD_ALREADY_ACTIVE", "active_profile": active.get("profile")}
    method_lock_sha256 = sha256_file(lock)
    write_immutable_json(marker, {"profile": profile, "status": "ACTIVE", "method_lock_sha256": method_lock_sha256})
    results: list[dict[str, Any]] = []
    for layer in range(config.num_hidden_layers):
        train_manifest = _find_manifest(activation_root, layer, "FIT-TRAIN")
        dev_manifest = _find_manifest(dev_activation_root, layer, "FIT-DEV")
        partition = _find_partition(run_dir, profile, layer, development_run_dir)
        if train_manifest is None or dev_manifest is None:
            return {"status": "BLOCKED", "blocker_code": "FULL64_ACTIVATIONS_REQUIRED", "layer": layer, "queue": str(queue_path)}
        if partition is None:
            return {"status": "BLOCKED", "blocker_code": "FULL64_PARTITION_REQUIRED", "layer": layer, "queue": str(queue_path)}
        output_dir = run_dir / "full64" / profile / "layer-checkpoints"
        results.append(_train_layer(method_lock_sha256=method_lock_sha256, source_dir=source_dir, train_manifest=train_manifest, dev_manifest=dev_manifest, output_dir=output_dir, layer=layer, profile=load_config(config_path), partition=partition, seed=17, device=device, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate))
    receipt = {"schema_version": 2, "receipt_type": "dense2moe-full64-training-execution", "status": "FULL64_TRAINING_COMPLETE", "profile": profile, "topology": config.topology_id, "layers": list(range(config.num_hidden_layers)), "resumed": resume, "results": results, "checkpoint_count": len(results), "one_active_full64_build": True, "external_evaluation_tuning": False, "method_lock_sha256": sha256_file(lock)}
    path = run_dir / "full64" / f"{profile}-execution.json"
    write_immutable_json(path, receipt)
    marker.unlink(missing_ok=True)
    return receipt | {"path": str(path), "queue": str(queue_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--activation-root", type=Path)
    parser.add_argument("--dev-activation-root", type=Path)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_full64(run_dir=args.run_dir, profile=args.profile, layers=args.layers, execute=args.execute, source_dir=args.source_dir, activation_root=args.activation_root, dev_activation_root=args.dev_activation_root, development_run_dir=args.development_run_dir, device=args.device, epochs=args.epochs, microbatch=args.microbatch, learning_rate=args.learning_rate, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] in {"QUEUE_READY", "FULL64_TRAINING_COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
