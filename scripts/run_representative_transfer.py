#!/usr/bin/env python3
"""Run the locked representative layer matrix with resumable checkpoints.

Without ``--execute`` this command only validates the matrix contract. The
execution path consumes one immutable X/Y manifest per layer and shares those
manifests between p16 and p32.  When executing, separate R1/R2 activation
manifests are mandatory and are evaluated only after each frozen checkpoint is
reloaded; they are never passed to the optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from dense2moe.config import load_config
from dense2moe.data import sha256_file, write_immutable_json
from dense2moe.evaluation import PROMOTION_THRESHOLDS

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


def _find_external_manifest(root: Path, tier: str, layer: int) -> Path | None:
    """Resolve one explicit R1/R2 layer manifest without guessing a split.

    Capture tooling has used both per-tier directories and tier-qualified file
    names.  Supporting both layouts keeps the data contract stable while the
    tier name remains part of the path identity.  No fallback to the FIT store
    is permitted.
    """

    names = (
        root / tier / f"layer-{layer:04d}.json",
        root / tier / f"layer-{layer:04d}-FIT-DEV.json",
        root / tier / f"layer-{layer:04d}-train.json",
        root / f"{tier}-layer-{layer:04d}.json",
        root / f"layer-{layer:04d}-{tier}.json",
        root / f"layer-{layer:04d}-{tier.upper()}.json",
    )
    return next((path for path in names if path.exists()), None)


def _existing_partition(value: Any, roots: tuple[Path, ...] = ()) -> Path | None:
    if isinstance(value, dict):
        direct = value.get("partition_path")
        if direct:
            candidate = Path(str(direct))
            for path in (candidate, *(root / candidate for root in roots if not candidate.is_absolute())):
                if path.exists():
                    return path
        for nested in value.values():
            found = _existing_partition(nested, roots)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _existing_partition(nested, roots)
            if found is not None:
                return found
    return None


def _partition_for(lock: dict[str, Any], *, profile: str, development_run_dir: Path | None) -> Path | None:
    roots = (development_run_dir.parent,) if development_run_dir is not None else ()
    found = _existing_partition(lock, roots)
    if found is not None:
        return found
    if development_run_dir is None:
        return None
    stem = "p16" if "p16" in profile else "p32"
    matches = sorted((development_run_dir / "development" / "partitions").glob(f"{stem}-top*.json"))
    return matches[0] if matches else None


def _load_external_values(manifest: Path, *, max_tokens: int | None) -> tuple[Any, dict[str, Any], str]:
    """Materialize a bounded immutable external evaluation view.

    The manifest must identify exactly one split.  Aggregate train/holdout
    wrappers are rejected so an external tier cannot silently become a mixed
    or positional split.
    """

    import numpy as np

    from dense2moe.capture import iter_activation_shards

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    split = payload.get("split")
    if split not in {"train", "holdout"}:
        raise ValueError(f"external manifest must declare one split: {manifest}")
    values: list[Any] = []
    count = 0
    for shard in iter_activation_shards(manifest, expected_split=split):
        if max_tokens is not None and count >= max_tokens:
            break
        remaining = int(shard.shape[0]) if max_tokens is None else max_tokens - count
        values.append(np.asarray(shard[:remaining], dtype=np.float32))
        count += min(int(shard.shape[0]), remaining)
    if not values:
        raise ValueError(f"external activation manifest is empty: {manifest}")
    return np.concatenate(values, axis=0)[:count], payload, str(payload.get("dataset_hash", ""))


def _external_metrics(
    *,
    source_dir: Path,
    manifest: Path,
    checkpoint_dir: Path,
    layer: int,
    profile: Any,
    partition: Any,
    device: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Reload one checkpoint and measure it on an untouched R1/R2 corpus."""

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from dense2moe.models import TorchQwen35SwiGLUMoE
    from dense2moe.training.torch_distill import _dense_target_torch
    from scripts.run_high_sparsity_search import _evaluate_positive_oracle
    from scripts.run_topk_architecture_search import _load_mlp

    inputs, manifest_payload, dataset_hash = _load_external_values(manifest, max_tokens=max_tokens)
    weights = _load_mlp(source_dir, layer)
    model = TorchQwen35SwiGLUMoE.from_dense(
        weights["gate_proj.weight"],
        weights["up_proj.weight"],
        weights["down_proj.weight"],
        routed_experts=profile.routed_experts,
        shared_intermediate_size=profile.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=partition,
        learnable_scales=True,
    )
    tensor_path = checkpoint_dir / f"layer-{layer:04d}.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"representative checkpoint tensor is missing: {tensor_path}")
    raw_state = load_file(str(tensor_path), device="cpu")
    prefix = f"model.layers.{layer}."
    state = {key[len(prefix) :]: value for key, value in raw_state.items() if key.startswith(prefix)}
    if len(state) != len(raw_state):
        raise ValueError(f"representative checkpoint tensor namespace mismatch: {tensor_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict representative checkpoint reload failed: {missing}, {unexpected}")
    model.eval().to(device)
    gate = torch.as_tensor(weights["gate_proj.weight"], dtype=torch.float32, device=device)
    up = torch.as_tensor(weights["up_proj.weight"], dtype=torch.float32, device=device)
    down = torch.as_tensor(weights["down_proj.weight"], dtype=torch.float32, device=device)
    errors = 0.0
    target_norm = 0.0
    cosine_sum = 0.0
    token_count = 0
    ratios: list[float] = []
    loads = np.zeros(profile.routed_experts, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, int(inputs.shape[0]), 32):
            x = torch.as_tensor(inputs[start : start + 32], dtype=torch.float32, device=device)
            target = _dense_target_torch(x, gate, up, down)
            prediction, info = model(x, return_router=True)
            errors += float(torch.sum((prediction - target).square()).item())
            target_norm += float(torch.sum(target.square()).item())
            cosine_sum += float(
                torch.sum(
                    torch.sum(prediction * target, dim=-1)
                    / (torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12)
                ).item()
            )
            ratios.extend(
                (torch.linalg.vector_norm(prediction, dim=-1) / (torch.linalg.vector_norm(target, dim=-1) + 1e-12))
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            loads += np.bincount(info["indices"].detach().cpu().numpy().reshape(-1), minlength=profile.routed_experts)
            token_count += int(target.shape[0])
    if token_count <= 0:
        raise ValueError(f"external manifest is empty: {manifest}")
    ratio_array = np.asarray(ratios, dtype=np.float64)
    oracle_profile = {
        "name": "p16" if profile.routed_experts == 16 else "p32",
        "routed_experts": profile.routed_experts,
        "expert_intermediate_size": profile.expert_intermediate_size,
        "shared_intermediate_size": profile.shared_intermediate_size,
        "dense_intermediate_size": profile.dense_intermediate_size,
        "hidden_size": profile.hidden_size,
    }
    # The representative store is intentionally bounded by the caller.  The
    # oracle uses the same immutable rows and is explicitly labelled so its
    # regret cannot be mistaken for a separate corpus.
    oracle = _evaluate_positive_oracle(
        oracle_profile,
        inputs,
        weights,
        partition,
        top_k=profile.top_k,
        device=device,
        batch_size=32,
        exact=profile.routed_experts == 16,
        pool_size=min(profile.routed_experts, 20),
    )
    return {
        "manifest": str(manifest),
        "dataset_hash": dataset_hash,
        "split": manifest_payload["split"],
        "tokens": token_count,
        "nmse": errors / max(target_norm, 1e-12),
        "cosine": cosine_sum / token_count,
        "loadcv": float(loads.std() / max(loads.mean(), 1e-12)),
        "dead_experts": int(np.sum(loads == 0)),
        "oracle_regret": max(0.0, errors / max(target_norm, 1e-12) - float(oracle["normalized_mse"])),
        "oracle_tokens": int(oracle["tokens"]),
        "median_norm_ratio_error": abs(float(np.median(ratio_array)) - 1.0),
        "p95_relative_norm_error": float(np.quantile(np.abs(ratio_array - 1.0), 0.95)),
        "external_data_untouched": True,
    }


def _representative_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    required = {
        "nmse": float(metrics["nmse"]),
        "cosine": float(metrics["cosine"]),
        "loadcv": float(metrics["loadcv"]),
        "dead_experts": float(metrics["dead_experts"]),
        "oracle_regret": float(metrics["oracle_regret"]),
        "repeat_variation": float(metrics.get("repeat_variation", 0.0)),
        "median_norm_ratio_error": float(metrics["median_norm_ratio_error"]),
        "p95_relative_norm_error": float(metrics["p95_relative_norm_error"]),
    }
    statuses: dict[str, str] = {}
    for name, value in required.items():
        rule = PROMOTION_THRESHOLDS[name]
        green = value <= float(rule["green"]) if rule["lower_is_better"] else value >= float(rule["green"])
        statuses[name] = "green" if green else "red"
    return {"metrics": required, "statuses": statuses, "overall": "green" if all(value == "green" for value in statuses.values()) else "red"}


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


def run_transfer(*, run_dir: Path, layers: str, profiles: list[str], seeds: list[int], execute: bool = False, source_dir: Path | None = None, activation_root: Path | None = None, external_activation_root: Path | None = None, development_run_dir: Path | None = None, device: str = "cpu", epochs: int = 1, microbatch: int = 8, learning_rate: float = 1e-3, external_max_tokens: int | None = 8192) -> dict[str, Any]:
    missing = [profile for profile in profiles if not (run_dir / "method-locks" / f"{profile}.json").exists()]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "METHOD_LOCKS_REQUIRED", "missing_profiles": missing, "message": "representative transfer cannot fit an unlocked method"}
    layer_ids = _parse_layers(layers)
    if layer_ids != list(REPRESENTATIVE_LAYERS):
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_LAYER_MATRIX_INVALID", "message": "exact representative layers 0-3, 28-31, and 60-63 are required"}
    payload: dict[str, Any] = {"schema_version": 3, "receipt_type": "dense2moe-representative-transfer", "profiles": profiles, "layers": layer_ids, "seeds": seeds, "sentinel_layers": list(SENTINEL_LAYERS), "development_data": "FIT-DEV", "external_data": ["R1", "R2"], "evaluation_tiers_opened": [], "shared_activation_store": True, "external_tuning_forbidden": True, "external_generalization_required": True}
    if not execute:
        payload["status"] = "TRANSFER_INPUTS_READY"
        path = run_dir / "representative" / "matrix.json"
        write_immutable_json(path, payload)
        return payload | {"path": str(path)}
    if source_dir is None or activation_root is None or external_activation_root is None:
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_EXTERNAL_INPUTS_REQUIRED", "message": "--source-dir, --activation-root, and a distinct --external-activation-root containing R1/R2 are required with --execute"}
    if not external_activation_root.exists():
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_EXTERNAL_ROOT_MISSING", "path": str(external_activation_root)}
    if external_activation_root.resolve() == activation_root.resolve():
        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_EXTERNAL_ROOT_NOT_DISTINCT", "message": "R1/R2 must be stored outside the development activation root"}
    results: list[dict[str, Any]] = []
    external_results: list[dict[str, Any]] = []
    external_manifest_hashes: dict[str, dict[str, str]] = {}
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
                trained = _run_one(method_lock_sha256=method_lock_sha256, source_dir=source_dir, train_manifest=train_manifest, dev_manifest=dev_manifest, output_dir=output, layer=layer, profile=config, partition=partition, seed=seed, device=device, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate)
                results.append(trained)
                for tier in ("R1", "R2"):
                    external_manifest = _find_external_manifest(external_activation_root, tier, layer)
                    if external_manifest is None:
                        return {"status": "BLOCKED", "blocker_code": "REPRESENTATIVE_EXTERNAL_ACTIVATIONS_REQUIRED", "profile": profile_name, "layer": layer, "tier": tier}
                    external_manifest_hashes.setdefault(str(layer), {})[tier] = sha256_file(external_manifest)
                    from dense2moe.training.torch_distill import _plan_from_path

                    metric = _external_metrics(source_dir=source_dir, manifest=external_manifest, checkpoint_dir=output, layer=layer, profile=config, partition=_plan_from_path(partition), device=device, max_tokens=external_max_tokens)
                    metric.update({"profile": profile_name, "layer": layer, "seed": seed, "tier": tier, "checkpoint_dir": str(output), "development_manifest_sha256": sha256_file(dev_manifest), "external_data_untouched": True})
                    external_results.append(metric)
    by_profile: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    for item in external_results:
        by_profile[item["profile"]].append(item)
    profile_decisions: dict[str, dict[str, Any]] = {}
    for profile_name, rows in by_profile.items():
        if not rows:
            profile_decisions[profile_name] = {"status": "BLOCKED", "gate": {"overall": "red"}}
            continue
        mean_by_repeat: dict[tuple[int, str], list[float]] = {}
        for row in rows:
            mean_by_repeat.setdefault((int(row["layer"]), str(row["tier"])), []).append(float(row["nmse"]))
        for row in rows:
            repeats = mean_by_repeat[(int(row["layer"]), str(row["tier"]))]
            mean = sum(repeats) / max(len(repeats), 1)
            row["repeat_variation"] = max(abs(value - mean) for value in repeats) / max(mean, 1e-12)
        aggregate = {
            key: (max(float(row[key]) for row in rows) if key != "cosine" else min(float(row[key]) for row in rows))
            for key in ("nmse", "cosine", "loadcv", "dead_experts", "oracle_regret", "repeat_variation", "median_norm_ratio_error", "p95_relative_norm_error")
        }
        gate = _representative_gate(aggregate)
        profile_decisions[profile_name] = {"status": "GREEN" if gate["overall"] == "green" else "REJECTED", "gate": gate, "rows": len(rows), "layers": sorted({int(row["layer"]) for row in rows}), "tiers": sorted({str(row["tier"]) for row in rows})}
    winner_profile = None
    for preferred in ("qwen38_p32s1_top5", "qwen38_p16s1_top4"):
        if preferred in profile_decisions and profile_decisions[preferred]["status"] == "GREEN":
            winner_profile = preferred
            break
    decision = {"schema_version": 1, "receipt_type": "dense2moe-representative-winner-decision", "status": "WINNER_SELECTED" if winner_profile else "NO_PRODUCT_CANDIDATE", "winner_profile": winner_profile, "winner_topology": "p32/top5" if winner_profile and "p32" in winner_profile else "p16/top4" if winner_profile else None, "profile_decisions": profile_decisions, "external_data": ["R1", "R2"], "external_activation_root": str(external_activation_root), "external_tuning_forbidden": True, "method_lock_hashes": {profile: sha256_file(run_dir / "method-locks" / f"{profile}.json") for profile in profiles}}
    decision_path = run_dir / "representative" / "decision.json"
    write_immutable_json(decision_path, decision)
    payload.update({"status": "WINNER_SELECTED" if winner_profile else "NO_PRODUCT_CANDIDATE", "results": results, "external_results": external_results, "profile_decisions": profile_decisions, "winner_profile": winner_profile, "checkpoint_count": len(results), "shared_activation_hashes": {str(layer): {"train": sha256_file(_find_manifest(activation_root, layer, "FIT-TRAIN")), "dev": sha256_file(_find_manifest(activation_root, layer, "FIT-DEV"))} for layer in layer_ids}, "external_manifest_hashes": external_manifest_hashes, "external_max_tokens": external_max_tokens, "winner_decision": str(decision_path)})
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
    parser.add_argument("--external-activation-root", type=Path)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--external-max-tokens", type=int, default=8192, help="bounded rows per R1/R2 evaluation; use 0 for all rows")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    external_max_tokens = None if args.external_max_tokens == 0 else args.external_max_tokens
    result = run_transfer(run_dir=args.run_dir, layers=args.layers, profiles=args.profiles, seeds=[int(value) for value in args.seeds.split(",") if value], execute=args.execute, source_dir=args.source_dir, activation_root=args.activation_root, external_activation_root=args.external_activation_root, development_run_dir=args.development_run_dir, device=args.device, epochs=args.epochs, microbatch=args.microbatch, learning_rate=args.learning_rate, external_max_tokens=external_max_tokens)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] in {"TRANSFER_INPUTS_READY", "TRANSFER_COMPLETE", "WINNER_SELECTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
