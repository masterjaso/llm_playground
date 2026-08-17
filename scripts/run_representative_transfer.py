#!/usr/bin/env python3
"""Run the locked representative layer matrix with resumable checkpoints.

Without ``--execute`` this command only validates the matrix contract. The
execution path consumes one immutable X/Y manifest per layer and shares those
manifests between p16 and p32; no external corpus is used to tune a method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_config
from dense2moe.data import sha256_file, write_immutable_json

try:
    from scripts.run_full64_training import _layer_lineage
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_full64_training import _layer_lineage  # type: ignore

REPRESENTATIVE_LAYERS = tuple(list(range(4)) + list(range(28, 32)) + list(range(60, 64)))
SENTINEL_LAYERS = (3, 31, 63)


def _parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def _find_manifest(root: Path, layer: int, role: str) -> Path | None:
    candidates = (
        root / f"layer-{layer:04d}-{role}.json",
        root / f"layer-{layer:04d}-{role.upper()}.json",
        root / f"layer-{layer:04d}-train.json" if role == "FIT-TRAIN" else root / f"layer-{layer:04d}-dev.json",
        root / f"layer-{layer:04d}-FIT-DEV.json" if role == "FIT-DEV" else root / f"layer-{layer:04d}.json",
    )
    return next((path for path in candidates if path.exists()), None)


def _partition_for(lock: dict[str, Any], *, profile: str, development_run_dir: Path | None) -> Path | None:
    finalists = lock.get("finalists")
    if isinstance(finalists, list):
        for entry in finalists:
            if isinstance(entry, dict) and entry.get("partition_path"):
                path = Path(str(entry["partition_path"]))
                if path.exists():
                    return path
    direct = lock.get("partition_path")
    if direct:
        path = Path(str(direct))
        if path.exists():
            return path
    if development_run_dir is None:
        return None
    stem = "p16" if "p16" in profile else "p32"
    matches = sorted((development_run_dir / "development" / "partitions").glob(f"{stem}-top*.json"))
    return matches[0] if matches else None


def _run_one(*, method_lock_sha256: str, source_dir: Path, train_manifest: Path, dev_manifest: Path, output_dir: Path, layer: int, profile: Any, partition: Path, seed: int, device: str, epochs: int, microbatch: int, learning_rate: float) -> dict[str, Any]:
    from dense2moe.training import train_torch_layer

    metadata_path = output_dir / f"layer-{layer:04d}.json"
    lineage_path = output_dir / f"layer-{layer:04d}.lineage.json"
    expected_lineage = _layer_lineage(method_lock_sha256=method_lock_sha256, train_manifest=train_manifest, dev_manifest=dev_manifest, profile=profile, partition=partition, layer=layer, seed=seed, device=device, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate)
    if metadata_path.exists() and (output_dir / f"layer-{layer:04d}.safetensors").exists() and lineage_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage == expected_lineage and int(payload.get("training_seed", -1)) == seed and payload.get("partition_hash"):
            return {"status": "REUSED", "layer": layer, "seed": seed, "metadata": str(metadata_path), "lineage": str(lineage_path), "manifest_hashes": {"train": expected_lineage["train_manifest_sha256"], "dev": expected_lineage["dev_manifest_sha256"]}}
    result = train_torch_layer(source_dir=source_dir, activation_manifest=train_manifest, selection_manifest=dev_manifest, output_dir=output_dir, layer=layer, profile=profile, partition_path=partition, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate, device=device, seed=seed, source_revision=profile.revision, evaluate_holdout=False)
    write_immutable_json(lineage_path, expected_lineage)
    result.update({"seed": seed, "manifest_hashes": {"train": sha256_file(train_manifest), "dev": sha256_file(dev_manifest)}})
    return result


def run_transfer(*, run_dir: Path, layers: str, profiles: list[str], seeds: list[int], execute: bool = False, source_dir: Path | None = None, activation_root: Path | None = None, development_run_dir: Path | None = None, device: str = "cpu", epochs: int = 1, microbatch: int = 8, learning_rate: float = 1e-3) -> dict[str, Any]:
    missing = [profile for profile in profiles if not (run_dir / "method-locks" / f"{profile}.json").exists()]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "METHOD_LOCKS_REQUIRED", "missing_profiles": missing, "message": "representative transfer cannot fit an unlocked method"}
    layer_ids = _parse_layers(layers)
    if layer_ids != list(REPRESENTATIVE_LAYERS):
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_LAYER_MATRIX_INVALID", "message": "exact representative layers 0-3, 28-31, and 60-63 are required"}
    payload: dict[str, Any] = {"schema_version": 2, "receipt_type": "dense2moe-representative-transfer", "profiles": profiles, "layers": layer_ids, "seeds": seeds, "sentinel_layers": list(SENTINEL_LAYERS), "development_data": "FIT-DEV", "external_data": ["R1", "R2"], "evaluation_tiers_opened": [], "shared_activation_store": True, "external_tuning_forbidden": True}
    if not execute:
        payload["status"] = "TRANSFER_INPUTS_READY"
        path = run_dir / "representative" / "matrix.json"
        write_immutable_json(path, payload)
        return payload | {"path": str(path)}
    if source_dir is None or activation_root is None:
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_EXECUTION_INPUTS_REQUIRED", "message": "--source-dir and --activation-root are required with --execute"}
    results: list[dict[str, Any]] = []
    for profile_name in profiles:
        lock = json.loads((run_dir / "method-locks" / f"{profile_name}.json").read_text(encoding="utf-8"))
        if lock.get("external_tuning_forbidden") is not True:
            return {"status": "BLOCKED", "blocker_code": "EXTERNAL_TUNING_NOT_FORBIDDEN", "profile": profile_name}
        partition = _partition_for(lock, profile=profile_name, development_run_dir=development_run_dir)
        if partition is None:
            return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_PARTITION_REQUIRED", "profile": profile_name}
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{profile_name}.yaml")
        method_lock_sha256 = sha256_file(run_dir / "method-locks" / f"{profile_name}.json")
        for layer in layer_ids:
            train_manifest = _find_manifest(activation_root, layer, "FIT-TRAIN")
            dev_manifest = _find_manifest(activation_root, layer, "FIT-DEV")
            if train_manifest is None or dev_manifest is None:
                return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_ACTIVATIONS_REQUIRED", "profile": profile_name, "layer": layer}
            layer_seeds = seeds if layer in SENTINEL_LAYERS else [seeds[0]]
            for seed in layer_seeds:
                output = run_dir / "representative" / profile_name / f"layer-{layer:04d}" / f"seed-{seed:02d}"
                results.append(_run_one(method_lock_sha256=method_lock_sha256, source_dir=source_dir, train_manifest=train_manifest, dev_manifest=dev_manifest, output_dir=output, layer=layer, profile=config, partition=partition, seed=seed, device=device, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate))
    payload.update({"status": "TRANSFER_COMPLETE", "results": results, "checkpoint_count": len(results), "shared_activation_hashes": {str(layer): {"train": sha256_file(_find_manifest(activation_root, layer, "FIT-TRAIN")), "dev": sha256_file(_find_manifest(activation_root, layer, "FIT-DEV"))} for layer in layer_ids}})
    path = run_dir / "representative" / "matrix-execution.json"
    write_immutable_json(path, payload)
    return payload | {"path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--activation-root", type=Path)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_transfer(run_dir=args.run_dir, layers=args.layers, profiles=args.profiles, seeds=[int(value) for value in args.seeds.split(",") if value], execute=args.execute, source_dir=args.source_dir, activation_root=args.activation_root, development_run_dir=args.development_run_dir, device=args.device, epochs=args.epochs, microbatch=args.microbatch, learning_rate=args.learning_rate)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] in {"TRANSFER_INPUTS_READY", "TRANSFER_COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
