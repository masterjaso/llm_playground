"""CPU-friendly one-layer oracle initialization and bounded router fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..checkpoint.layer import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from ..config import MoEProfile
from ..partition import oracle_topk, swiglu_contributions


def _source_mlp(source_dir: Path, layer: int) -> tuple[Any, Any, Any, str, str]:
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index.get("weight_map", {}).items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"layer {layer} source MLP inventory mismatch: {sorted(names)}")
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        raise RuntimeError("safetensors is required for layer distillation") from exc
    values: dict[str, Any] = {}
    for short, shard in names.items():
        try:
            with safe_open(str(source_dir / shard), framework="numpy") as handle:
                values[short] = handle.get_tensor(prefix + short)
        except (TypeError, ValueError, RuntimeError):
            # NumPy safetensors cannot decode BF16 on some versions; PyTorch
            # provides the portable conversion path without touching source.
            try:
                __import__("torch")
                with safe_open(str(source_dir / shard), framework="pt", device="cpu") as handle:
                    values[short] = handle.get_tensor(prefix + short).float().numpy()
            except ImportError as exc:
                raise RuntimeError("BF16 source tensors require PyTorch for layer distillation") from exc
    config_hash = hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest()
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    return values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"], config_hash, index_hash


def _fit_router(inputs: Any, oracle_indices: Any, experts: int) -> Any:
    """Fit a stable linear router against contribution-derived assignments."""

    import numpy as np  # type: ignore

    x = np.asarray(inputs, dtype=np.float64)
    ids = np.asarray(oracle_indices)
    labels = np.zeros((x.shape[0], experts), dtype=np.float64)
    for row in range(x.shape[0]):
        labels[row, ids[row]] = 1.0
    # Centering improves numerical stability for large hidden dimensions.  A
    # bounded sample keeps this solve memory-feasible on CPU-only machines.
    sample = min(4096, x.shape[0])
    xs, ys = x[:sample], labels[:sample]
    gram = xs.T @ xs + np.eye(xs.shape[1]) * 1e-3
    return np.linalg.solve(gram, xs.T @ ys).astype(np.float32)


def train_real_layer(
    *,
    source_dir: str | Path,
    activation_manifest: str | Path,
    output_dir: str | Path,
    layer: int,
    profile: MoEProfile,
    seed: int = 17,
    code_commit: str = "unknown",
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Produce a real partitioned layer artifact and fixed holdout metrics."""

    import numpy as np  # type: ignore

    from ..capture import iter_activation_shards

    source = Path(source_dir)
    gate, up, down, source_config_hash, source_index_hash = _source_mlp(source, layer)
    activations = np.concatenate(list(iter_activation_shards(activation_manifest)), axis=0)
    if activations.shape[0] < 2:
        raise ValueError("at least two captured tokens are required for train/holdout validation")
    profile.validate()
    if profile.dense_intermediate_size != int(gate.shape[0]) or profile.hidden_size != int(gate.shape[1]):
        raise ValueError("profile geometry does not match source layer")
    plan = __import__("dense2moe.partition", fromlist=["partition_indices"]).partition_indices(
        profile.dense_intermediate_size,
        profile.routed_experts,
        profile.expert_intermediate_size,
        profile.shared_intermediate_size,
    )
    shared, routed = swiglu_contributions(activations, gate, up, down, plan)
    dense = shared + routed.sum(axis=1)
    split = max(1, int(activations.shape[0] * 0.8))
    train_x, holdout_x = activations[:split], activations[split:]
    train_shared, train_routed = shared[:split], routed[:split]
    holdout_shared, holdout_routed = shared[split:], routed[split:]
    train_target, holdout_target = dense[:split], dense[split:]
    oracle_train = oracle_topk(train_shared, train_routed, train_target, top_k=profile.top_k)
    oracle_holdout = oracle_topk(holdout_shared, holdout_routed, holdout_target, top_k=profile.top_k) if len(holdout_x) else oracle_train
    router = _fit_router(train_x, oracle_train["indices"], profile.routed_experts)
    # Evaluate the fitted router using the model's sparse semantics.
    from ..models.qwen_moe import Qwen35SwiGLUMoE

    moe = Qwen35SwiGLUMoE.from_dense(
        __import__("dense2moe.models", fromlist=["DenseSwiGLU"]).DenseSwiGLU(gate, up, down),
        routed_experts=profile.routed_experts,
        shared_intermediate_size=profile.shared_intermediate_size,
        top_k=profile.top_k,
        partition=plan,
        router=router,
    )
    prediction, router_info = moe(holdout_x if len(holdout_x) else train_x, return_router=True)
    target = holdout_target if len(holdout_x) else train_target
    norm = float(np.mean(target**2)) + 1e-12
    mse = float(np.mean((prediction - target) ** 2))
    cosine = float(np.mean(np.sum(prediction * target, axis=-1) / (np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-12)))
    selected = np.asarray(router_info["indices"])
    loads = np.bincount(selected.reshape(-1), minlength=profile.routed_experts).astype(np.float64)
    load_cv = float(loads.std() / (loads.mean() + 1e-12))
    dead = int(np.sum(loads == 0))
    oracle_norm = float(oracle_holdout["normalized_mse"])
    gate_overall = "green" if mse / norm <= 0.05 and cosine >= 0.98 and dead == 0 and load_cv <= 0.50 else "research-candidate" if mse / norm <= 0.10 and cosine >= 0.95 else "red"
    dataset_hash = str(json.loads(Path(activation_manifest).read_text(encoding="utf-8")).get("dataset_hash", ""))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(
        {f"model.layers.{layer}.{name}": value for name, value in moe.state_dict().items()},
        output / f"layer-{layer:04d}.safetensors",
    )
    metadata_path = output / f"layer-{layer:04d}.json"
    checkpoint = LayerCheckpoint(
        layer=layer,
        profile=profile.name,
        status="TRAINED_VALIDATED" if gate_overall != "red" else "VALIDATION_FAILED",
        profile_hash=profile_fingerprint(profile.as_dict()),
        source_revision=source_revision or profile.revision,
        source_config_hash=source_config_hash,
        source_index_hash=source_index_hash,
        dataset_hash=dataset_hash,
        partition_strategy="contiguous",
        partition_hash=hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest(),
        router_architecture="linear-oracle-warmstart-topk-normalized",
        training_seed=seed,
        training_config={"epochs": 0, "initialization": "dense-slice", "device": "cpu", "note": "oracle warm-start; joint optimization is a bounded follow-up"},
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"oracle_normalized_mse": float(oracle_train["normalized_mse"])},
        holdout_metrics={"normalized_mse": mse / norm, "cosine": cosine, "all_expert_mse": float(np.mean((((shared[split:] if len(holdout_x) else shared[:split]) + (routed[split:] if len(holdout_x) else routed[:split]).sum(axis=1)) - target) ** 2)), "oracle_normalized_mse": oracle_norm},
        router_metrics={"load_cv": load_cv, "dead_experts": dead, "oracle_regret": max(0.0, (mse / norm) - oracle_norm), "selected_counts": loads.tolist()},
        quality_gate={"overall": gate_overall, "thresholds_version": "initial-2026-08-15", "metrics": {"normalized_mse": mse / norm, "cosine": cosine, "load_cv": load_cv, "dead_experts": dead}},
        code_commit=code_commit,
    )
    save_layer_checkpoint(checkpoint, metadata_path)
    return {"status": "TRAINED_VALIDATED" if gate_overall != "red" else "VALIDATION_FAILED", "layer": layer, "metadata": str(metadata_path), "tensor_file": str(tensor_path), "holdout_metrics": checkpoint.holdout_metrics, "router_metrics": checkpoint.router_metrics, "quality_gate": checkpoint.quality_gate}


__all__ = ["train_real_layer"]
