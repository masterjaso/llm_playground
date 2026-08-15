"""Actual PyTorch one-layer distillation with fixed train/holdout splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..capture import iter_activation_shards
from ..checkpoint.layer import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from ..models.torch_moe import TorchQwen35SwiGLUMoE
from ..partition import PartitionPlan, oracle_topk, swiglu_contributions
from ..provenance import current_git_commit


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"activation manifest must be an object: {path}")
    return payload


def _resolve(path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else path.parent / candidate


def _load_split(path: Path, expected: str) -> Any:
    """Load one split and refuse manifests with implicit positional splits."""

    payload = _load_manifest(path)
    if payload.get("split") not in {expected, "both"}:
        raise ValueError(f"activation split mismatch for {path}: expected {expected!r}")
    values = list(iter_activation_shards(path, expected_split=expected if payload.get("split") else None))
    if not values:
        raise ValueError(f"activation split is empty: {path}")
    import numpy as np  # type: ignore

    return np.concatenate(values, axis=0)


def load_fixed_activation_splits(manifest_path: str | Path) -> tuple[Any, Any, str]:
    """Return train/holdout arrays from explicit split manifests only.

    A previous implementation silently used the first 80% of one combined
    array.  This loader requires split-labelled manifests or sibling
    ``*-train.json``/``*-holdout.json`` files so holdout provenance cannot be
    inferred from array position.
    """

    path = Path(manifest_path)
    payload = _load_manifest(path)
    train_ref = payload.get("train_manifest")
    holdout_ref = payload.get("holdout_manifest")
    if isinstance(payload.get("splits"), dict):
        train_ref = train_ref or payload["splits"].get("train")
        holdout_ref = holdout_ref or payload["splits"].get("holdout")
    if train_ref and holdout_ref:
        train_path = _resolve(path, str(train_ref))
        holdout_path = _resolve(path, str(holdout_ref))
    elif path.stem.endswith("-train"):
        train_path = path
        holdout_path = path.with_name(path.name.replace("-train.json", "-holdout.json"))
    elif path.stem.endswith("-holdout"):
        holdout_path = path
        train_path = path.with_name(path.name.replace("-holdout.json", "-train.json"))
    else:
        raise ValueError(
            "activation manifest has no explicit train_manifest/holdout_manifest; "
            "positional 80/20 splitting is prohibited"
        )
    if not train_path.exists() or not holdout_path.exists():
        raise FileNotFoundError(f"fixed split manifests are required: {train_path}, {holdout_path}")
    train = _load_split(train_path, "train")
    holdout = _load_split(holdout_path, "holdout")
    dataset_hash = str(payload.get("dataset_hash") or _load_manifest(train_path).get("dataset_hash") or "")
    if not dataset_hash:
        raise ValueError("fixed split manifest is missing dataset_hash")
    return train, holdout, dataset_hash


def _dense_target(inputs: Any, gate: Any, up: Any, down: Any) -> Any:
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(inputs, dtype=torch.float32)
    gate_tensor = torch.as_tensor(gate, dtype=torch.float32)
    up_tensor = torch.as_tensor(up, dtype=torch.float32)
    down_tensor = torch.as_tensor(down, dtype=torch.float32)
    return (F.silu(x @ gate_tensor.T) * (x @ up_tensor.T)) @ down_tensor.T


def _plan_from_path(path: Path) -> PartitionPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "plan" in payload and isinstance(payload["plan"], dict):
        payload = payload["plan"]
    plan = PartitionPlan(
        int(payload["dense_intermediate_size"]),
        int(payload["routed_experts"]),
        int(payload["expert_intermediate_size"]),
        int(payload["shared_intermediate_size"]),
        tuple(int(value) for value in payload["shared_indices"]),
        tuple(tuple(int(value) for value in group) for group in payload["expert_indices"]),
    )
    plan.validate()
    return plan


def _metrics(model: TorchQwen35SwiGLUMoE, inputs: Any, target: Any, microbatch: int) -> dict[str, Any]:
    import numpy as np  # type: ignore
    import torch

    model.eval()
    model_device = next(model.parameters()).device
    predictions: list[Any] = []
    routing: list[Any] = []
    with torch.inference_mode():
        for start in range(0, int(inputs.shape[0]), microbatch):
            output, info = model(
                torch.as_tensor(inputs[start : start + microbatch], dtype=torch.float32, device=model_device),
                return_router=True,
            )
            predictions.append(output.cpu())
            routing.append({key: value.cpu() for key, value in info.items()})
    predicted = torch.cat(predictions, dim=0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    norm = float(torch.mean(target_tensor.square()).item()) + 1e-12
    mse = float(torch.mean((predicted - target_tensor).square()).item())
    cosine = float(
        torch.mean(
            torch.sum(predicted * target_tensor, dim=-1)
            / (torch.linalg.vector_norm(predicted, dim=-1) * torch.linalg.vector_norm(target_tensor, dim=-1) + 1e-12)
        ).item()
    )
    indices = torch.cat([item["indices"].reshape(-1) for item in routing], dim=0).numpy()
    loads = np.bincount(indices, minlength=model.routed_experts).astype(np.float64)
    return {
        "normalized_mse": mse / norm,
        "mse": mse,
        "cosine": cosine,
        "selected_counts": loads.tolist(),
        "dead_experts": int(np.sum(loads == 0)),
        "load_cv": float(loads.std() / (loads.mean() + 1e-12)),
    }


def _train_stage(
    model: TorchQwen35SwiGLUMoE,
    inputs: Any,
    target: Any,
    *,
    epochs: int,
    microbatch: int,
    learning_rate: float,
    train_scales: bool,
    train_experts: bool,
    device: str,
    stage: str,
    oracle_indices: Any | None = None,
) -> dict[str, Any]:
    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.router.weight.requires_grad = True
    if train_scales:
        model.expert_scales.requires_grad = True
    if train_experts:
        for module in (*model.expert_gate_proj, *model.expert_up_proj, *model.expert_down_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if epochs <= 0 or not parameters:
        return {"stage": stage, "epochs": 0, "updates": 0, "loss": None}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    model.train()
    last_loss = 0.0
    updates = 0
    for _epoch in range(epochs):
        for start in range(0, int(x.shape[0]), microbatch):
            prediction, info = model(x[start : start + microbatch], return_router=True)
            teacher = y[start : start + microbatch]
            mse = torch.mean((prediction - teacher).square())
            cosine = 1.0 - torch.mean(
                torch.sum(prediction * teacher, dim=-1)
                / (torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(teacher, dim=-1) + 1e-12)
            )
            probs = torch.zeros((prediction.shape[0], model.routed_experts), device=device)
            probs.scatter_add_(1, info["indices"], info["weights"])
            load_balance = model.routed_experts * torch.mean(probs, dim=0).square().sum()
            z_loss = torch.mean(torch.logsumexp(info["logits"], dim=-1).square())
            oracle_loss = torch.zeros((), device=device)
            if oracle_indices is not None:
                labels = torch.as_tensor(oracle_indices[start : start + microbatch], dtype=torch.long, device=device)
                oracle_loss = torch.stack(
                    [
                        torch.nn.functional.cross_entropy(info["logits"], labels[:, slot])
                        for slot in range(labels.shape[1])
                    ]
                ).mean()
            loss = mse + 0.05 * cosine + 0.01 * load_balance + 0.001 * z_loss + 0.1 * oracle_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            updates += 1
    return {"stage": stage, "epochs": epochs, "updates": updates, "loss": last_loss}


def train_torch_layer(
    *,
    source_dir: str | Path,
    activation_manifest: str | Path,
    output_dir: str | Path,
    layer: int,
    profile: Any,
    partition_path: str | Path,
    epochs: int = 1,
    microbatch: int = 1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 17,
    source_revision: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Run router, scale, and joint stages against fixed split activations."""

    import numpy as np  # type: ignore
    import torch
    from safetensors import safe_open  # type: ignore

    if epochs < 0 or microbatch <= 0 or learning_rate <= 0:
        raise ValueError("epochs must be non-negative, microbatch positive, and learning rate positive")
    recorded_commit = current_git_commit()
    if code_commit is not None and code_commit != recorded_commit:
        raise ValueError("code_commit must match current git HEAD")
    torch.manual_seed(seed)
    source = Path(source_dir)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"layer {layer} source MLP inventory mismatch: {sorted(names)}")
    values: dict[str, Any] = {}
    for short, shard in names.items():
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            values[short] = handle.get_tensor(prefix + short).float().numpy()
    train_x, holdout_x, dataset_hash = load_fixed_activation_splits(activation_manifest)
    plan = _plan_from_path(Path(partition_path))
    if plan.dense_intermediate_size != int(values["gate_proj.weight"].shape[0]):
        raise ValueError("selected partition does not match source dense width")
    profile.validate()
    model = TorchQwen35SwiGLUMoE.from_dense(
        values["gate_proj.weight"],
        values["up_proj.weight"],
        values["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        partition=plan,
        learnable_scales=True,
    )
    train_target = _dense_target(train_x, values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"])
    holdout_target = _dense_target(holdout_x, values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"])
    train_shared, train_routed = swiglu_contributions(
        np.asarray(train_x),
        values["gate_proj.weight"],
        values["up_proj.weight"],
        values["down_proj.weight"],
        plan,
    )
    train_oracle = oracle_topk(train_shared, train_routed, np.asarray(train_target), top_k=profile.top_k)
    initial = _metrics(model, holdout_x, holdout_target, microbatch)
    stages = [
        _train_stage(
            model,
            train_x,
            train_target,
            epochs=epochs,
            microbatch=microbatch,
            learning_rate=learning_rate,
            train_scales=False,
            train_experts=False,
            device=device,
            stage="router_warm_start",
            oracle_indices=train_oracle["indices"],
        ),
        _train_stage(
            model,
            train_x,
            train_target,
            epochs=epochs,
            microbatch=microbatch,
            learning_rate=learning_rate,
            train_scales=True,
            train_experts=False,
            device=device,
            stage="router_plus_scale",
        ),
        _train_stage(
            model,
            train_x,
            train_target,
            epochs=epochs,
            microbatch=microbatch,
            learning_rate=learning_rate,
            train_scales=True,
            train_experts=True,
            device=device,
            stage="joint_expert_router",
        ),
    ]
    trained = _metrics(model, holdout_x, holdout_target, microbatch)
    all_expert = _dense_target(holdout_x, values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"])
    oracle_shared, oracle_routed = swiglu_contributions(
        np.asarray(holdout_x),
        values["gate_proj.weight"],
        values["up_proj.weight"],
        values["down_proj.weight"],
        plan,
    )
    oracle = oracle_topk(oracle_shared, oracle_routed, np.asarray(holdout_target), top_k=profile.top_k)
    gate_overall = (
        "green"
        if trained["normalized_mse"] <= 0.05 and trained["cosine"] >= 0.98 and trained["dead_experts"] == 0 and trained["load_cv"] <= 0.50
        else "research-candidate"
        if trained["normalized_mse"] <= 0.10 and trained["cosine"] >= 0.95
        else "red"
    )
    if epochs == 0:
        status = "INITIALIZED_UNTRAINED"
    else:
        status = "TRAINED_VALIDATED" if gate_overall != "red" else "VALIDATION_FAILED"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.{layer}.{name}": value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output / f"layer-{layer:04d}.safetensors")
    partition_hash = hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest()
    checkpoint = LayerCheckpoint(
        layer=layer,
        profile=profile.name,
        status=status,
        profile_hash=profile_fingerprint(profile.as_dict()),
        source_revision=source_revision or profile.revision,
        source_config_hash=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        source_index_hash=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        dataset_hash=dataset_hash,
        partition_strategy="artifact",
        partition_hash=partition_hash,
        router_architecture="torch-linear-topk-normalized-v1",
        training_seed=seed,
        training_config={
            "epochs": epochs,
            "microbatch": microbatch,
            "learning_rate": learning_rate,
            "device": device,
            "optimizer": "AdamW",
            "loss_version": "torch-distill-v1",
            "loss_coefficients": {"mse": 1.0, "cosine": 0.05, "load_balance": 0.01, "router_z_loss": 0.001},
            "stages": stages,
            "partition_path": str(partition_path),
        },
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"initial_holdout": initial, "oracle_holdout": {"normalized_mse": float(oracle["normalized_mse"]), "cosine": float(oracle["cosine"])}, "all_expert_reconstruction_mse": float(torch.mean((all_expert - holdout_target).square()).item())},
        holdout_metrics=trained,
        router_metrics={"load_cv": trained["load_cv"], "dead_experts": trained["dead_experts"], "selected_counts": trained["selected_counts"], "actual_improvement": float(initial["normalized_mse"] - trained["normalized_mse"])},
        quality_gate={"overall": gate_overall if epochs > 0 else "untrained", "thresholds_version": "initial-2026-08-15", "metrics": trained},
        code_commit=recorded_commit,
    )
    metadata_path = output / f"layer-{layer:04d}.json"
    save_layer_checkpoint(checkpoint, metadata_path)
    return {"status": status, "layer": layer, "metadata": str(metadata_path), "tensor_file": str(tensor_path), "holdout_metrics": trained, "initial_holdout": initial, "oracle_holdout": {"normalized_mse": float(oracle["normalized_mse"]), "cosine": float(oracle["cosine"])}, "training_config": checkpoint.training_config, "code_commit": recorded_commit}


__all__ = ["load_fixed_activation_splits", "train_torch_layer"]
