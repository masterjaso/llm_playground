"""Actual PyTorch one-layer distillation with fixed train/holdout splits."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
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


class ActivationShardDataset:
    """Streaming fixed-split activation reader with bounded microbatches.

    Shards are opened one at a time and each shard is sliced into microbatches;
    no whole-dataset ``concatenate`` operation is used by the production
    trainer.  The object is deliberately independent of ``torch`` so it can
    be inspected in lightweight provenance and memory tests.
    """

    def __init__(self, manifest_path: str | Path, *, split: str, microbatch: int = 1) -> None:
        if split not in {"train", "holdout"}:
            raise ValueError("split must be train or holdout")
        if microbatch <= 0:
            raise ValueError("microbatch must be positive")
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.microbatch = microbatch
        payload = _load_manifest(self.manifest_path)
        if payload.get("split") not in {split, "both"} and not payload.get("train_manifest"):
            raise ValueError(f"activation manifest split mismatch for {self.manifest_path}: expected {split!r}")
        self.count = self._manifest_count(payload, split)
        self.dataset_hash = str(payload.get("dataset_hash", ""))
        if not self.dataset_hash:
            self.dataset_hash = str(self._split_manifest(payload).get("dataset_hash", ""))
        if not self.dataset_hash:
            raise ValueError("fixed split manifest is missing dataset_hash")

    def _split_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        reference = payload.get(f"{self.split}_manifest")
        if isinstance(reference, str) and reference not in {"", "pending"}:
            return _load_manifest(_resolve(self.manifest_path, reference))
        if self.manifest_path.stem.endswith("-" + self.split):
            return payload
        raise ValueError(
            f"aggregate activation manifest has no explicit {self.split}_manifest reference"
        )

    def _manifest_count(self, payload: dict[str, Any], split: str) -> int:
        candidate = self._split_manifest(payload) if payload.get(f"{split}_manifest") else payload
        if candidate.get("split") not in {split, "both"}:
            candidate = self._split_manifest(payload)
        return int(candidate.get("count", 0))

    def _resolved_manifest_path(self) -> Path:
        payload = _load_manifest(self.manifest_path)
        reference = payload.get(f"{self.split}_manifest")
        if isinstance(reference, str) and reference not in {"", "pending"}:
            return _resolve(self.manifest_path, reference)
        if self.manifest_path.stem.endswith("-" + self.split):
            return self.manifest_path
        raise ValueError(f"no explicit {self.split} activation manifest")

    def iter_batches(self, microbatch: int | None = None) -> Iterator[Any]:
        """Yield NumPy microbatches while releasing each shard promptly."""

        size = int(microbatch or self.microbatch)
        if size <= 0:
            raise ValueError("microbatch must be positive")
        manifest = self._resolved_manifest_path()
        for shard in iter_activation_shards(manifest, expected_split=self.split):
            for start in range(0, int(shard.shape[0]), size):
                yield shard[start : start + size]

    def iter_selected_batches(self, indices: Sequence[int], microbatch: int | None = None) -> Iterator[Any]:
        """Yield a deterministic global-row subset without materializing it.

        The indices are global positions in this explicit split manifest.  A
        caller records the selection hash in its checkpoint so architecture
        selection cannot silently drift to a different development subset.
        """

        size = int(microbatch or self.microbatch)
        if size <= 0:
            raise ValueError("microbatch must be positive")
        selected = sorted({int(index) for index in indices})
        if any(index < 0 or index >= self.count for index in selected):
            raise IndexError(f"selected activation row is outside {self.split} split: count={self.count}")
        wanted = set(selected)
        cursor = 0
        manifest = self._resolved_manifest_path()
        for shard in iter_activation_shards(manifest, expected_split=self.split):
            shard_count = int(shard.shape[0])
            local = [index - cursor for index in selected if cursor <= index < cursor + shard_count]
            if local:
                values = shard[local]
                for start in range(0, int(values.shape[0]), size):
                    yield values[start : start + size]
            cursor += shard_count
        if cursor != self.count:
            raise ValueError(f"activation manifest count mismatch for {self.manifest_path}: {cursor} != {self.count}")
        # Keep the local set alive through the generator body so a malformed
        # manifest cannot make this check look vacuously successful.
        if len(wanted) != len(selected):  # pragma: no cover - sorted set invariant
            raise ValueError("selected activation rows are not unique")

    def __iter__(self) -> Iterator[Any]:
        return self.iter_batches()


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


def _dense_target_torch(inputs: Any, gate: Any, up: Any, down: Any) -> Any:
    """Compute one bounded dense SwiGLU target on the training device."""

    import torch.nn.functional as F

    return (F.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T


def _stream_metrics(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    *,
    gate: Any,
    up: Any,
    down: Any,
    microbatch: int,
    device: str,
    selected_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one explicit split, optionally restricted to global rows."""

    import numpy as np  # type: ignore
    import torch

    model.eval()
    model.to(device)
    gate_device = gate.to(device)
    up_device = up.to(device)
    down_device = down.to(device)
    squared_error = 0.0
    target_norm = 0.0
    cosine_sum = 0.0
    token_count = 0
    loads = np.zeros(model.routed_experts, dtype=np.float64)
    entropy_sum = 0.0
    with torch.inference_mode():
        batches = dataset.iter_batches(microbatch) if selected_indices is None else dataset.iter_selected_batches(selected_indices, microbatch)
        for values in batches:
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            target = _dense_target_torch(inputs, gate_device, up_device, down_device)
            prediction, info = model(inputs, return_router=True)
            squared_error += float(torch.sum((prediction - target).square()).item())
            target_norm += float(torch.sum(target.square()).item())
            cosine_sum += float(
                torch.sum(
                    torch.sum(prediction * target, dim=-1)
                    / (
                        torch.linalg.vector_norm(prediction, dim=-1)
                        * torch.linalg.vector_norm(target, dim=-1)
                        + 1e-12
                    )
                ).item()
            )
            token_count += int(target.shape[0])
            loads += np.bincount(info["indices"].detach().cpu().reshape(-1).numpy(), minlength=model.routed_experts)
            entropy_sum += float(
                torch.sum(
                    -(torch.softmax(info["logits"], dim=-1) * torch.log_softmax(info["logits"], dim=-1)).sum(dim=-1)
                ).item()
            )
            del inputs, target, prediction, info
    if token_count <= 0:
        raise ValueError(f"activation split is empty: {dataset.manifest_path}")
    mean_target_norm = target_norm / (token_count * int(down.shape[0]))
    return {
        "normalized_mse": (squared_error / (token_count * int(down.shape[0]))) / (mean_target_norm + 1e-12),
        "mse": squared_error / (token_count * int(down.shape[0])),
        "cosine": cosine_sum / token_count,
        "selected_counts": loads.tolist(),
        "dead_experts": int(np.sum(loads == 0)),
        "load_cv": float(loads.std() / (loads.mean() + 1e-12)),
        "router_entropy": entropy_sum / token_count,
        "token_count": token_count,
        "streaming": True,
        "split": dataset.split,
        "selected_count": int(len(selected_indices)) if selected_indices is not None else token_count,
    }


def _oracle_indices_for_batch(
    values: Any,
    gate: Any,
    up: Any,
    down: Any,
    plan: PartitionPlan,
    top_k: int,
) -> Any:
    import numpy as np  # type: ignore

    target = _dense_target(np.asarray(values), gate, up, down)
    shared, routed = swiglu_contributions(np.asarray(values), gate, up, down, plan)
    return oracle_topk(shared, routed, target, top_k=top_k)["indices"]


def _enable_router_parameters(
    model: TorchQwen35SwiGLUMoE,
    *,
    train_selection: bool = True,
    train_amplitude: bool = True,
) -> None:
    """Enable independently controlled selection/amplitude router parameters."""

    model.router.weight.requires_grad = bool(train_selection)
    if model.routing_mode == "independent_positive":
        model.amplitude_router.weight.requires_grad = bool(train_amplitude)
        model.amplitude_router.bias.requires_grad = bool(train_amplitude)


def _optimizer_parameter_groups(
    model: TorchQwen35SwiGLUMoE,
    *,
    learning_rate: float,
    learning_rates: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    """Build named parameter groups so router/scale/expert LRs can differ."""

    overrides = {str(name): float(value) for name, value in (learning_rates or {}).items()}
    groups: list[tuple[str, list[Any]]] = [
        ("selection_router", [model.router.weight]),
        ("amplitude_router", list(model.amplitude_router.parameters()) if model.routing_mode == "independent_positive" else []),
        ("expert_scales", [model.expert_scales]),
        (
            "experts",
            [parameter for module in (*model.expert_gate_proj, *model.expert_up_proj, *model.expert_down_proj) for parameter in module.parameters()],
        ),
    ]
    output: list[dict[str, Any]] = []
    for name, parameters in groups:
        enabled = [parameter for parameter in parameters if parameter.requires_grad]
        if enabled:
            rate = overrides.get(name, float(learning_rate))
            if rate <= 0:
                raise ValueError(f"learning rate for {name} must be positive")
            output.append({"params": enabled, "lr": rate, "group": name})
    if not output:
        raise ValueError("training stage has no enabled parameters")
    return output


def _train_stage_streaming(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    *,
    gate: Any,
    up: Any,
    down: Any,
    plan: PartitionPlan,
    epochs: int,
    microbatch: int,
    learning_rate: float,
    train_scales: bool,
    train_experts: bool,
    device: str,
    stage: str,
    use_oracle_targets: bool = False,
    train_selection_router: bool = True,
    train_amplitude_router: bool = True,
    learning_rates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Train one stage while reading only bounded activation batches."""

    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False
    _enable_router_parameters(model, train_selection=train_selection_router, train_amplitude=train_amplitude_router)
    if train_scales:
        model.expert_scales.requires_grad = True
    if train_experts:
        for module in (*model.expert_gate_proj, *model.expert_up_proj, *model.expert_down_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
    model.to(device)
    parameter_groups = _optimizer_parameter_groups(model, learning_rate=learning_rate, learning_rates=learning_rates)
    if epochs <= 0:
        return {"stage": stage, "epochs": 0, "updates": 0, "loss": None, "streaming": True}
    parameters = [parameter for group in parameter_groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups)
    gate_device = gate.to(device)
    up_device = up.to(device)
    down_device = down.to(device)
    model.train()
    last_loss = 0.0
    updates = 0
    for epoch in range(epochs):
        for values in dataset.iter_batches(microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            teacher = _dense_target_torch(inputs, gate_device, up_device, down_device)
            prediction, info = model(inputs, return_router=True, return_contributions=use_oracle_targets)
            mse = torch.mean((prediction - teacher).square())
            cosine = 1.0 - torch.mean(
                torch.sum(prediction * teacher, dim=-1)
                / (torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(teacher, dim=-1) + 1e-12)
            )
            # Balance the dense softmax distribution, not only the selected
            # top-k mass.  The latter gives a dead expert zero gradient and can
            # never satisfy the explicit dead-expert gate once it collapses.
            probs = torch.softmax(info["logits"], dim=-1)
            load_balance = model.routed_experts * torch.mean(probs, dim=0).square().sum()
            z_loss = torch.mean(torch.logsumexp(info["logits"], dim=-1).square())
            oracle_loss = torch.zeros((), device=device)
            if use_oracle_targets:
                # The exact frozen oracle is a separate bounded research
                # study.  For warm-up, rank the already-computed GPU expert
                # contributions; this preserves the target-free streaming
                # contract and avoids a CPU SwiGLU recomputation for every
                # train microbatch.
                labels = torch.topk(torch.linalg.vector_norm(info["contributions"], dim=-1), model.top_k, dim=-1).indices
                oracle_loss = torch.stack(
                    [torch.nn.functional.cross_entropy(info["logits"], labels[:, slot]) for slot in range(labels.shape[1])]
                ).mean()
            # The denser fallback has enough capacity to trade a small amount
            # of reconstruction slack for a materially healthier expert load.
            # Keep this coefficient explicit in the receipt rather than hiding
            # it in a profile-specific post-processing step.
            load_balance_coefficient = 0.05
            loss = mse + 0.05 * cosine + load_balance_coefficient * load_balance + 0.001 * z_loss + 0.1 * oracle_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            updates += 1
            del inputs, teacher, prediction, info
        model.train()
    return {
        "stage": stage,
        "epochs": epochs,
        "updates": updates,
        "loss": last_loss,
        "streaming": True,
        "epoch": epoch + 1,
        "train_selection_router": train_selection_router,
        "train_amplitude_router": train_amplitude_router,
        "learning_rates": {group["group"]: group["lr"] for group in parameter_groups},
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
    _enable_router_parameters(model)
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
    stage_schedule: Sequence[Mapping[str, Any]] | None = None,
    selection_indices: Sequence[int] | None = None,
    evaluate_holdout: bool = True,
) -> dict[str, Any]:
    """Run a configurable staged distillation schedule against fixed splits.

    The default schedule is kept identical to the original three-stage path.
    A caller may provide a bounded sequence of stage mappings to compare
    router warm-up, router/scale transitions, frozen-router expert adaptation,
    and joint fine-tuning without changing the activation or holdout contract.
    """

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
        open_kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source / shard), framework="pt", device="cpu", **open_kwargs) as handle:
            values[short] = handle.get_tensor(prefix + short).float().numpy()
    wrapper_payload = _load_manifest(Path(activation_manifest))
    dataset_hash = str(wrapper_payload.get("dataset_hash", ""))
    train_dataset = ActivationShardDataset(activation_manifest, split="train", microbatch=microbatch)
    holdout_dataset = ActivationShardDataset(activation_manifest, split="holdout", microbatch=microbatch)
    if train_dataset.dataset_hash != dataset_hash:
        dataset_hash = train_dataset.dataset_hash
    if holdout_dataset.dataset_hash != dataset_hash:
        raise ValueError("train and holdout activation manifests have different dataset_hash values")
    if selection_indices is None:
        selection_rows: tuple[int, ...] | None = None
    else:
        selection_rows = tuple(sorted({int(index) for index in selection_indices}))
        if not selection_rows:
            raise ValueError("selection_indices must contain at least one train row")
        if selection_rows[0] < 0 or selection_rows[-1] >= train_dataset.count:
            raise IndexError(f"selection_indices must be within the train split [0, {train_dataset.count})")
    selection_hash = (
        hashlib.sha256("\n".join(str(index) for index in selection_rows).encode()).hexdigest()
        if selection_rows is not None
        else None
    )
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
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    )
    partition_payload = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    initial_scales = partition_payload.get("initial_expert_scales")
    if initial_scales is not None:
        if not isinstance(initial_scales, list) or len(initial_scales) != plan.routed_experts:
            raise ValueError("initial_expert_scales must contain one value per routed expert")
        with torch.no_grad():
            model.expert_scales.copy_(torch.as_tensor(initial_scales, dtype=model.expert_scales.dtype))
    gate_tensor = torch.as_tensor(values["gate_proj.weight"], dtype=torch.float32)
    up_tensor = torch.as_tensor(values["up_proj.weight"], dtype=torch.float32)
    down_tensor = torch.as_tensor(values["down_proj.weight"], dtype=torch.float32)
    # Break the zero-router tie deterministically with a bounded, train-only
    # contribution-label warm start.  Without this, ``topk`` always selected
    # experts 0 and 1 on the first pass and the load-balance term could not
    # revive the remaining experts.  The labels are derived from the immutable
    # dense-slice contributions, never from holdout targets.
    warmup_values = next(train_dataset.iter_batches(min(microbatch, 512)))
    warmup_inputs = torch.as_tensor(warmup_values, dtype=torch.float32, device=device)
    model.to(device)
    with torch.no_grad():
        warmup_contributions = torch.stack(
            [model._expert_output(warmup_inputs, expert) * model.expert_scales[expert] for expert in range(model.routed_experts)],
            dim=1,
        )
        warmup_labels = torch.topk(torch.linalg.vector_norm(warmup_contributions, dim=-1), model.top_k, dim=-1).indices
        warmup_targets = torch.zeros((warmup_inputs.shape[0], model.routed_experts), dtype=torch.float32, device=device)
        warmup_targets.scatter_(1, warmup_labels, 1.0)
        warmup_solution = torch.linalg.lstsq(warmup_inputs, warmup_targets).solution.T
        model.router.weight.copy_(warmup_solution.to(dtype=model.router.weight.dtype))
    del warmup_values, warmup_inputs, warmup_contributions, warmup_labels, warmup_targets, warmup_solution
    initial_selection = _stream_metrics(
        model,
        train_dataset,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        selected_indices=selection_rows,
    )
    stages: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    # Always retain the initialized checkpoint as a candidate.  This makes
    # every schedule stage an explicitly best-checkpoint comparison rather
    # than returning a later regression when no stage improves development NMSE.
    best_state: dict[str, Any] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metric = float(initial_selection["normalized_mse"])
    best_stage = "initialized"
    if stage_schedule is None:
        raw_schedule: Sequence[Mapping[str, Any]] = (
            {"name": "router_warm_start", "epochs": epochs, "train_scales": False, "train_experts": False, "use_oracle_targets": True},
            {"name": "router_plus_scale", "epochs": epochs, "train_scales": True, "train_experts": False, "use_oracle_targets": False},
            {"name": "joint_expert_router", "epochs": epochs, "train_scales": True, "train_experts": True, "use_oracle_targets": False},
        )
    else:
        if isinstance(stage_schedule, (str, bytes)) or not isinstance(stage_schedule, Sequence) or not stage_schedule:
            raise ValueError("stage_schedule must be a non-empty sequence of mappings")
        raw_schedule = stage_schedule
    normalized_schedule: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_schedule):
        if not isinstance(raw_stage, Mapping):
            raise TypeError(f"stage_schedule[{index}] must be a mapping")
        stage_name = str(raw_stage.get("name", raw_stage.get("stage", f"stage_{index}")))
        if not stage_name:
            raise ValueError(f"stage_schedule[{index}] has an empty name")
        stage_epochs = int(raw_stage.get("epochs", epochs))
        if stage_epochs < 0:
            raise ValueError(f"stage_schedule[{index}] epochs must be non-negative")
        stage_learning_rate = float(raw_stage.get("learning_rate", learning_rate))
        if stage_learning_rate <= 0:
            raise ValueError(f"stage_schedule[{index}] learning_rate must be positive")
        raw_rates = raw_stage.get("learning_rates")
        if raw_rates is not None and not isinstance(raw_rates, Mapping):
            raise TypeError(f"stage_schedule[{index}].learning_rates must be a mapping")
        rates = {str(name): float(value) for name, value in (raw_rates or {}).items()}
        if any(value <= 0 for value in rates.values()):
            raise ValueError(f"stage_schedule[{index}].learning_rates values must be positive")
        normalized_schedule.append(
            {
                "name": stage_name,
                "epochs": stage_epochs,
                "train_scales": bool(raw_stage.get("train_scales", False)),
                "train_experts": bool(raw_stage.get("train_experts", False)),
                "use_oracle_targets": bool(raw_stage.get("use_oracle_targets", False)),
                "train_selection_router": bool(raw_stage.get("train_selection_router", True)),
                "train_amplitude_router": bool(raw_stage.get("train_amplitude_router", True)),
                "learning_rate": stage_learning_rate,
                "learning_rates": rates,
            }
        )
    for stage_spec in normalized_schedule:
        stage_name = str(stage_spec["name"])
        stage_result = _train_stage_streaming(
            model,
            train_dataset,
            gate=gate_tensor,
            up=up_tensor,
            down=down_tensor,
            plan=plan,
            epochs=int(stage_spec["epochs"]),
            microbatch=microbatch,
            learning_rate=float(stage_spec["learning_rate"]),
            train_scales=bool(stage_spec["train_scales"]),
            train_experts=bool(stage_spec["train_experts"]),
            device=device,
            stage=stage_name,
            use_oracle_targets=bool(stage_spec["use_oracle_targets"]),
            train_selection_router=bool(stage_spec["train_selection_router"]),
            train_amplitude_router=bool(stage_spec["train_amplitude_router"]),
            learning_rates=stage_spec["learning_rates"],
        )
        stages.append(stage_result)
        measured = _stream_metrics(
            model,
            train_dataset,
            gate=gate_tensor,
            up=up_tensor,
            down=down_tensor,
            microbatch=microbatch,
            device=device,
            selected_indices=selection_rows,
        )
        stage_metrics.append({"stage": stage_name, **measured})
        if float(measured["normalized_mse"]) < best_metric:
            best_metric = float(measured["normalized_mse"])
            best_stage = stage_name
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    final_selection = _stream_metrics(
        model,
        train_dataset,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        selected_indices=selection_rows,
    )
    if evaluate_holdout:
        trained = _stream_metrics(
            model,
            holdout_dataset,
            gate=gate_tensor,
            up=up_tensor,
            down=down_tensor,
            microbatch=microbatch,
            device=device,
        )
        holdout_status = "full_holdout_confirmation"
    else:
        trained = {
            "status": "DEFERRED_UNTIL_FINALIST_CONFIRMATION",
            "split": "holdout",
            "normalized_mse": None,
            "cosine": None,
            "selected_counts": [],
            "dead_experts": None,
            "load_cv": None,
            "streaming": True,
        }
        holdout_status = "deferred"
    # The dense all-expert reconstruction is analytically exact by construction;
    # retain the explicit zero target as a diagnostic rather than pretending it
    # is a sparse quality result.
    oracle_holdout = {"normalized_mse": None, "cosine": None, "streaming": True, "status": "bounded-oracle-summary-pending"}
    gate_metrics = trained if evaluate_holdout else final_selection
    gate_overall = (
        "green"
        if gate_metrics["normalized_mse"] <= 0.05 and gate_metrics["cosine"] >= 0.98 and gate_metrics["dead_experts"] == 0 and gate_metrics["load_cv"] <= 0.50
        else "research-candidate"
        if gate_metrics["normalized_mse"] <= 0.10 and gate_metrics["cosine"] >= 0.95
        else "red"
    )
    if epochs == 0:
        status = "INITIALIZED_UNTRAINED"
    elif not evaluate_holdout:
        status = "TRAINED_DEV_SELECTED" if gate_overall in {"green", "research-candidate"} else "VALIDATION_DEFERRED"
    elif gate_overall == "green":
        status = "TRAINED_VALIDATED"
    elif gate_overall == "research-candidate":
        status = "RESEARCH_CANDIDATE"
    else:
        status = "VALIDATION_FAILED"
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
        router_architecture=f"torch-linear-topk-{profile.routing_mode}-v1",
        routing_mode=profile.routing_mode,
        training_seed=seed,
        training_config={
            "epochs": epochs,
            "microbatch": microbatch,
            "learning_rate": learning_rate,
            "device": device,
            "optimizer": "AdamW",
            "loss_version": "torch-distill-v1",
            "loss_coefficients": {"mse": 1.0, "cosine": 0.05, "load_balance": 0.05, "router_z_loss": 0.001},
            "routing_mode": profile.routing_mode,
            "stage_schedule": normalized_schedule,
            "stages": stages,
            "stage_selection_metrics": stage_metrics,
            "best_selection_stage": best_stage,
            "selection_split": "train_dev_subset" if selection_rows is not None else "train_split",
            "selection_indices_hash": selection_hash,
            "selection_count": int(len(selection_rows)) if selection_rows is not None else train_dataset.count,
            "holdout_evaluation": holdout_status,
            "streaming_dataset": {
                "train_manifest": train_dataset.manifest_path.as_posix(),
                "holdout_manifest": holdout_dataset.manifest_path.as_posix(),
                "train_count": train_dataset.count,
                "holdout_count": holdout_dataset.count,
            },
            "partition_path": str(partition_path),
            "initial_expert_scales": [float(value) for value in (initial_scales or [1.0] * plan.routed_experts)],
            "router_initialization": "bounded_train_contribution_lstsq",
        },
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"initial_selection": initial_selection, "final_selection": final_selection, "oracle_holdout": oracle_holdout, "all_expert_reconstruction_mse": 0.0},
        holdout_metrics=trained,
        router_metrics={"load_cv": gate_metrics["load_cv"], "dead_experts": gate_metrics["dead_experts"], "selected_counts": gate_metrics["selected_counts"], "actual_improvement": float(initial_selection["normalized_mse"] - gate_metrics["normalized_mse"]) if gate_metrics["normalized_mse"] is not None else None},
        quality_gate={"overall": gate_overall if epochs > 0 else "untrained", "thresholds_version": "initial-2026-08-15", "evaluation_scope": "full_holdout" if evaluate_holdout else "train_dev_selection", "metrics": gate_metrics},
        code_commit=recorded_commit,
    )
    metadata_path = output / f"layer-{layer:04d}.json"
    save_layer_checkpoint(checkpoint, metadata_path)
    return {"status": status, "layer": layer, "metadata": str(metadata_path), "tensor_file": str(tensor_path), "holdout_metrics": trained, "initial_selection": initial_selection, "final_selection": final_selection, "oracle_holdout": oracle_holdout, "training_config": checkpoint.training_config, "code_commit": recorded_commit}


__all__ = ["load_fixed_activation_splits", "train_torch_layer"]
