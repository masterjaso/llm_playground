"""Actual PyTorch one-layer distillation with fixed train/holdout splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
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


def _hash_indices(indices: Sequence[int]) -> str:
    """Stable receipt hash for global row positions in an explicit split."""

    return hashlib.sha256("\n".join(str(int(index)) for index in indices).encode()).hexdigest()


def deterministic_shadow_validation_indices(
    count: int,
    *,
    excluded_indices: Sequence[int],
    shadow_count: int,
    seed: int = 20260816,
) -> tuple[tuple[int, ...], str]:
    """Choose a reproducible selector-only validation-B subset from FIT rows.

    Rows returned by this helper are *not* an untouched end-to-end test for a
    basis that has already seen the historical FIT corpus.  They are suitable
    for a future selector-only run when the basis is frozen and both validation
    A and B are excluded from optimizer updates.  The helper refuses overlap,
    never derives rows from the holdout split, and returns a stable identity
    hash for the receipt.
    """

    if count <= 0 or shadow_count <= 0:
        raise ValueError("count and shadow_count must be positive")
    excluded = {int(index) for index in excluded_indices}
    if any(index < 0 or index >= count for index in excluded):
        raise IndexError("excluded validation index is outside the train split")
    candidates = [index for index in range(count) if index not in excluded]
    if shadow_count > len(candidates):
        raise ValueError("shadow_count exceeds FIT rows available after validation-A")
    ranked = sorted(candidates, key=lambda index: hashlib.sha256(f"{int(seed)}:{index}".encode()).hexdigest())
    selected = tuple(sorted(ranked[:shadow_count]))
    if set(selected) & excluded:
        raise AssertionError("shadow validation overlaps validation-A")
    return selected, _hash_indices(selected)


def validate_split_contract(
    count: int,
    *,
    selection_indices: Sequence[int] | None,
    validation_b_indices: Sequence[int] | None = None,
    fit_exclude_indices: Sequence[int] | None = None,
    selection_union_indices: Sequence[int] | None = None,
) -> dict[str, tuple[int, ...] | None]:
    """Validate and normalize the selector FIT/A/B split contract.

    Validation-A is the *only* split that may influence checkpoint selection.
    Validation-B is an independent confirmation split and must be excluded
    from optimizer updates without entering the selection metric.  The old
    ``selection_union_indices`` argument is retained as a compatibility
    guard, but it may only repeat A exactly; a union containing B is rejected
    instead of silently changing the protocol.

    The returned global row positions are tuples so callers can persist their
    hashes without depending on input ordering or duplicate entries.
    """

    if count <= 0:
        raise ValueError("count must be positive")

    def normalize(name: str, values: Sequence[int] | None) -> tuple[int, ...]:
        if values is None:
            return ()
        rows = tuple(sorted({int(value) for value in values}))
        if any(index < 0 or index >= count for index in rows):
            raise IndexError(f"{name} must be within the train split [0, {count})")
        return rows

    selection_rows = normalize("selection_indices", selection_indices)
    validation_b_rows = normalize("validation_b_indices", validation_b_indices)
    if selection_rows and set(selection_rows).intersection(validation_b_rows):
        raise ValueError("validation-A and validation-B rows must be disjoint")

    if selection_union_indices is not None:
        union_rows = normalize("selection_union_indices", selection_union_indices)
        if not selection_rows:
            raise ValueError("selection_union_indices require selection_indices")
        if union_rows != selection_rows:
            raise ValueError(
                "selection_union_indices are disabled: checkpoint selection must use "
                "validation-A only; validation-B cannot participate"
            )

    if fit_exclude_indices is None:
        if selection_rows or validation_b_rows:
            raise ValueError(
                "selection_indices/validation_b_indices require explicit fit_exclude_indices"
            )
        fit_excluded_rows: tuple[int, ...] = ()
    else:
        fit_excluded_rows = normalize("fit_exclude_indices", fit_exclude_indices)
    fit_excluded_set = set(fit_excluded_rows)
    if not set(selection_rows).issubset(fit_excluded_set):
        raise ValueError("selection_indices must be excluded from optimizer updates")
    if not set(validation_b_rows).issubset(fit_excluded_set):
        raise ValueError("validation_b_indices must be excluded from optimizer updates")

    fit_rows = tuple(index for index in range(count) if index not in fit_excluded_set)
    return {
        "selection_indices": selection_rows or None,
        "validation_b_indices": validation_b_rows,
        "fit_exclude_indices": fit_excluded_rows,
        "fit_indices": fit_rows,
    }


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
        if payload.get("split") == self.split or self.manifest_path.stem.endswith("-" + self.split):
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
        if payload.get("split") == self.split or self.manifest_path.stem.endswith("-" + self.split):
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

    def iter_excluding_batches(
        self,
        excluded_indices: Sequence[int],
        microbatch: int | None = None,
    ) -> Iterator[Any]:
        """Yield every row except an explicit global-index exclusion set.

        This is the FIT-side primitive for a true train/validation contract:
        validation rows remain addressable for metrics, but never enter an
        optimizer batch.
        """

        size = int(microbatch or self.microbatch)
        if size <= 0:
            raise ValueError("microbatch must be positive")
        excluded = sorted({int(index) for index in excluded_indices})
        if any(index < 0 or index >= self.count for index in excluded):
            raise IndexError(f"excluded activation row is outside {self.split} split: count={self.count}")
        excluded_set = set(excluded)
        cursor = 0
        manifest = self._resolved_manifest_path()
        for shard in iter_activation_shards(manifest, expected_split=self.split):
            shard_count = int(shard.shape[0])
            local = [
                offset
                for offset in range(shard_count)
                if cursor + offset not in excluded_set
            ]
            if local:
                values = shard[local]
                for start in range(0, int(values.shape[0]), size):
                    yield values[start : start + size]
            cursor += shard_count
        if cursor != self.count:
            raise ValueError(f"activation manifest count mismatch for {self.manifest_path}: {cursor} != {self.count}")

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
    excluded_indices: Sequence[int] | None = None,
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
    soft_loads = np.zeros(model.routed_experts, dtype=np.float64)
    entropy_sum = 0.0
    margin_sum = 0.0
    margin_count = 0
    with torch.inference_mode():
        if selected_indices is not None and excluded_indices is not None:
            raise ValueError("selected_indices and excluded_indices are mutually exclusive")
        if selected_indices is not None:
            batches = dataset.iter_selected_batches(selected_indices, microbatch)
        elif excluded_indices is not None:
            batches = dataset.iter_excluding_batches(excluded_indices, microbatch)
        else:
            batches = dataset.iter_batches(microbatch)
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
            soft_loads += torch.softmax(info["logits"], dim=-1).sum(dim=0).detach().cpu().numpy()
            entropy_sum += float(
                torch.sum(
                    -(torch.softmax(info["logits"], dim=-1) * torch.log_softmax(info["logits"], dim=-1)).sum(dim=-1)
                ).item()
            )
            if model.top_k < model.routed_experts:
                top_values = torch.topk(info["logits"], model.top_k + 1, dim=-1).values
                margin_sum += float(torch.sum(top_values[:, model.top_k - 1] - top_values[:, model.top_k]).item())
                margin_count += int(target.shape[0])
            del inputs, target, prediction, info
    if token_count <= 0:
        raise ValueError(f"activation split is empty: {dataset.manifest_path}")
    mean_target_norm = target_norm / (token_count * int(down.shape[0]))
    hard_distribution = loads / max(float(token_count * model.top_k), 1.0)
    soft_distribution = soft_loads / max(float(token_count), 1.0)
    return {
        "normalized_mse": (squared_error / (token_count * int(down.shape[0]))) / (mean_target_norm + 1e-12),
        "mse": squared_error / (token_count * int(down.shape[0])),
        "cosine": cosine_sum / token_count,
        "selected_counts": loads.tolist(),
        "dead_experts": int(np.sum(loads == 0)),
        "load_cv": float(loads.std() / (loads.mean() + 1e-12)),
        "hard_load_balance": float(model.routed_experts * np.square(hard_distribution).sum()),
        "soft_load_balance": float(model.routed_experts * np.square(soft_distribution).sum()),
        "soft_loads": soft_loads.tolist(),
        "soft_load_cv": float(soft_loads.std() / (soft_loads.mean() + 1e-12)),
        "router_entropy": entropy_sum / token_count,
        "topk_logit_margin": (margin_sum / margin_count) if margin_count else None,
        "token_count": token_count,
        "streaming": True,
        "split": dataset.split,
        "selected_count": len(selected_indices) if selected_indices is not None else token_count,
        "excluded_count": len(excluded_indices) if excluded_indices is not None else 0,
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


def _hard_dispatch_straight_through(logits: Any, indices: Any, top_k: int) -> Any:
    """Return top-k dispatch mass with hard forward values and soft gradients.

    The forward value is the actual selected-expert membership normalized by
    ``top_k``.  The backward path follows the full softmax over router logits,
    so a hard-load penalty can revive experts that are not currently selected.
    """

    import torch

    if logits.ndim != 2 or indices.ndim != 2:
        raise ValueError("logits and indices must be rank-2 tensors")
    if indices.shape[0] != logits.shape[0] or indices.shape[1] != top_k:
        raise ValueError("indices shape must be [batch, top_k] for logits")
    if top_k <= 0 or top_k > logits.shape[1]:
        raise ValueError("top_k must be in [1, experts]")
    hard = torch.zeros_like(logits)
    hard.scatter_(1, indices, 1.0 / float(top_k))
    soft = torch.softmax(logits, dim=-1)
    return hard + soft - soft.detach()


def _enable_router_parameters(
    model: TorchQwen35SwiGLUMoE,
    *,
    train_selection: bool = True,
    train_amplitude: bool = True,
) -> None:
    """Enable independently controlled selection/amplitude router parameters."""

    for parameter in model.router.parameters():
        parameter.requires_grad = bool(train_selection)
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
        ("selection_router", list(model.router.parameters())),
        ("amplitude_router", list(model.amplitude_router.parameters()) if model.routing_mode == "independent_positive" else []),
        ("expert_scales", [model.expert_scales]),
        (
            "shared",
            [
                parameter
                for module in (model.shared_gate_proj, model.shared_up_proj, model.shared_down_proj)
                for parameter in module.parameters()
            ],
        ),
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
    train_shared: bool = False,
    device: str,
    stage: str,
    use_oracle_targets: bool = False,
    oracle_target_mode: str = "contribution_norm",
    oracle_loss_mode: str = "repeated_cross_entropy",
    oracle_amplitude_mode: str = "student_selected",
    teacher_forcing_ratio: float = 0.0,
    train_selection_router: bool = True,
    train_amplitude_router: bool = True,
    learning_rates: Mapping[str, float] | None = None,
    loss_coefficients: Mapping[str, float] | None = None,
    oracle_regret_weight: float = 0.0,
    expert_use_prices: Sequence[float] | None = None,
    excluded_indices: Sequence[int] | None = None,
    epoch_callback: Callable[[TorchQwen35SwiGLUMoE, str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train one stage while reading only bounded activation batches."""

    import torch
    import torch.nn.functional as F

    if oracle_target_mode not in {"contribution_norm", "residual_correlation"}:
        raise ValueError("oracle_target_mode must be contribution_norm or residual_correlation")
    if oracle_loss_mode not in {"repeated_cross_entropy", "multilabel_bce"}:
        raise ValueError("oracle_loss_mode must be repeated_cross_entropy or multilabel_bce")
    if oracle_amplitude_mode not in {"student_selected", "teacher_forced", "mixed"}:
        raise ValueError("oracle_amplitude_mode must be student_selected, teacher_forced, or mixed")
    if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
        raise ValueError("teacher_forcing_ratio must be between 0 and 1")
    if float(oracle_regret_weight) < 0.0:
        raise ValueError("oracle_regret_weight must be non-negative")
    if expert_use_prices is not None:
        if len(expert_use_prices) != model.routed_experts:
            raise ValueError("expert_use_prices must contain one value per routed expert")
        if any(not math.isfinite(float(price)) for price in expert_use_prices):
            raise ValueError("expert_use_prices must contain finite values")

    for parameter in model.parameters():
        parameter.requires_grad = False
    _enable_router_parameters(model, train_selection=train_selection_router, train_amplitude=train_amplitude_router)
    if train_scales:
        model.expert_scales.requires_grad = True
    if train_experts:
        for module in (*model.expert_gate_proj, *model.expert_up_proj, *model.expert_down_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if train_shared:
        for module in (model.shared_gate_proj, model.shared_up_proj, model.shared_down_proj):
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
    epoch_metrics: list[dict[str, Any]] = []
    regret_sum = 0.0
    regret_weight_sum = 0.0
    regret_token_count = 0
    price_tensor = (
        torch.as_tensor(expert_use_prices, dtype=torch.float32, device=device)
        if expert_use_prices is not None
        else None
    )
    for epoch in range(epochs):
        model.train()
        batches = dataset.iter_batches(microbatch) if excluded_indices is None else dataset.iter_excluding_batches(excluded_indices, microbatch)
        for values in batches:
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
            hard_dispatch = _hard_dispatch_straight_through(info["logits"], info["indices"], model.top_k)
            hard_load_balance = model.routed_experts * torch.mean(hard_dispatch, dim=0).square().sum()
            z_loss = torch.mean(torch.logsumexp(info["logits"], dim=-1).square())
            oracle_loss = torch.zeros((), device=device)
            oracle_amplitude_loss = torch.zeros((), device=device)
            if use_oracle_targets:
                contributions = info["contributions"]
                if oracle_target_mode == "residual_correlation":
                    # These labels use only the current train microbatch and
                    # the dense teacher target.  They provide a differentiable
                    # cross-entropy signal to the otherwise discrete top-k
                    # selector while leaving the selected amplitudes positive.
                    residual = teacher - info["shared"]
                    contribution_norm = torch.linalg.vector_norm(contributions, dim=-1)
                    scores = torch.sum(contributions * residual.unsqueeze(1), dim=-1) / (contribution_norm + 1e-12)
                else:
                    # Preserve the original target-free warm-start baseline.
                    scores = torch.linalg.vector_norm(contributions, dim=-1)
                if price_tensor is not None:
                    scores = scores - price_tensor.reshape(1, -1)
                labels = torch.topk(scores, model.top_k, dim=-1).indices
                if float(oracle_regret_weight) > 0.0 and model.top_k < model.routed_experts:
                    ranked_scores = torch.topk(scores, model.top_k + 1, dim=-1).values
                    regret = (ranked_scores[:, model.top_k - 1] - ranked_scores[:, model.top_k]).clamp_min(0.0)
                    normalized_regret = regret.detach() / (torch.mean(regret.detach()) + 1e-12)
                    oracle_row_weights = 1.0 + float(oracle_regret_weight) * normalized_regret
                    regret_sum += float(regret.detach().sum().item())
                    regret_weight_sum += float(oracle_row_weights.detach().sum().item())
                    regret_token_count += int(regret.shape[0])
                else:
                    oracle_row_weights = torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device)
                if oracle_loss_mode == "multilabel_bce":
                    membership = torch.zeros_like(info["logits"])
                    membership.scatter_(1, labels, 1.0)
                    per_token = F.binary_cross_entropy_with_logits(info["logits"], membership, reduction="none").mean(dim=-1)
                    oracle_loss = torch.mean(per_token * oracle_row_weights)
                else:
                    per_token = torch.stack(
                        [F.cross_entropy(info["logits"], labels[:, slot], reduction="none") for slot in range(labels.shape[1])],
                        dim=1,
                    ).mean(dim=1)
                    oracle_loss = torch.mean(per_token * oracle_row_weights)
                amplitude_coefficient = float((loss_coefficients or {}).get("oracle_amplitude", 0.0))
                if amplitude_coefficient > 0.0 and model.routing_mode == "independent_positive":
                    # Fit positive coefficients for the currently selected
                    # experts against the train-only residual.  The tiny
                    # batched k-by-k solve is bounded by top-k, and labels are
                    # detached so it cannot turn into an implicit teacher
                    # gradient path.
                    residual_for_amplitude = teacher - info["shared"]
                    def _positive_coefficients(
                        ids: Any,
                        contribution_values: Any = contributions,
                        residual_values: Any = residual_for_amplitude,
                    ) -> Any:
                        selected = torch.gather(
                            contribution_values,
                            1,
                            ids.unsqueeze(-1).expand(-1, -1, contribution_values.shape[-1]),
                        )
                        gram = torch.einsum("bkh,blh->bkl", selected, selected)
                        rhs = torch.einsum("bkh,bh->bk", selected, residual_values)
                        identity = torch.eye(model.top_k, dtype=gram.dtype, device=gram.device).unsqueeze(0)
                        return torch.linalg.solve(gram + 1e-4 * identity, rhs.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)

                    student_coefficients = _positive_coefficients(info["indices"])
                    oracle_coefficients = _positive_coefficients(labels)
                    predicted_all = F.softplus(info["amplitude_logits"])
                    student_predicted = torch.gather(predicted_all, 1, info["indices"])
                    oracle_predicted = torch.gather(predicted_all, 1, labels)
                    ratio = float(teacher_forcing_ratio)
                    if oracle_amplitude_mode == "teacher_forced":
                        ratio = 1.0
                    elif oracle_amplitude_mode == "student_selected":
                        ratio = 0.0
                    oracle_amplitude_loss = (
                        ratio * F.smooth_l1_loss(oracle_predicted, oracle_coefficients.detach())
                        + (1.0 - ratio) * F.smooth_l1_loss(student_predicted, student_coefficients.detach())
                    )
            # The denser fallback has enough capacity to trade a small amount
            # of reconstruction slack for a materially healthier expert load.
            # Keep this coefficient explicit in the receipt rather than hiding
            # it in a profile-specific post-processing step.
            coefficients = {"mse": 1.0, "cosine": 0.05, "load_balance": 0.05, "router_z_loss": 0.001, "oracle": 0.1}
            coefficients.update({str(name): float(value) for name, value in (loss_coefficients or {}).items()})
            load_balance_coefficient = coefficients["load_balance"]
            loss = (
                coefficients["mse"] * mse
                + coefficients["cosine"] * cosine
                + load_balance_coefficient * load_balance
                + coefficients.get("hard_load_balance", 0.0) * hard_load_balance
                + coefficients["router_z_loss"] * z_loss
                + coefficients["oracle"] * oracle_loss
                + coefficients.get("oracle_amplitude", 0.0) * oracle_amplitude_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            updates += 1
            del inputs, teacher, prediction, info
        model.train()
        if epoch_callback is not None:
            measured = dict(epoch_callback(model, stage, epoch + 1))
            epoch_metrics.append(measured)
    return {
        "stage": stage,
        "epochs": epochs,
        "updates": updates,
        "loss": last_loss,
        "streaming": True,
        "epoch": epoch + 1,
        "train_selection_router": train_selection_router,
        "train_amplitude_router": train_amplitude_router,
        "train_shared": train_shared,
        "learning_rates": {group["group"]: group["lr"] for group in parameter_groups},
        "loss_coefficients": {str(name): float(value) for name, value in (loss_coefficients or {}).items()},
        "hard_load_balance": float(hard_load_balance.detach().cpu().item()) if updates else None,
        "oracle_regret_weight": float(oracle_regret_weight),
        "expert_use_prices": [float(value) for value in expert_use_prices] if expert_use_prices is not None else None,
        "mean_oracle_regret": regret_sum / regret_token_count if regret_token_count else None,
        "mean_oracle_row_weight": regret_weight_sum / regret_token_count if regret_token_count else None,
        "oracle_target_mode": oracle_target_mode,
        "oracle_loss_mode": oracle_loss_mode,
        "oracle_amplitude_mode": oracle_amplitude_mode,
        "teacher_forcing_ratio": float(teacher_forcing_ratio),
        "fit_excluded_count": len(excluded_indices) if excluded_indices is not None else 0,
        "epoch_validation_metrics": epoch_metrics,
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


def _gate_feasible(metrics: Mapping[str, Any]) -> bool:
    """Return whether a validation checkpoint satisfies the product gate."""

    return bool(
        metrics.get("normalized_mse") is not None
        and metrics.get("cosine") is not None
        and metrics.get("dead_experts") is not None
        and metrics.get("load_cv") is not None
        and float(metrics["normalized_mse"]) <= 0.05
        and float(metrics["cosine"]) >= 0.98
        and int(metrics["dead_experts"]) == 0
        and float(metrics["load_cv"]) <= 0.50
    )


def _feasible_rank(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Gate-aware ordering: cosine first, then NMSE and load CV."""

    return (float(metrics["cosine"]), -float(metrics["normalized_mse"]), -float(metrics["load_cv"]))


def _pareto_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Whether ``left`` is no worse on all frontier dimensions and better on one."""

    left_values = (float(left["cosine"]), -float(left["normalized_mse"]), -float(left["load_cv"]))
    right_values = (float(right["cosine"]), -float(right["normalized_mse"]), -float(right["load_cv"]))
    return all(a >= b for a, b in zip(left_values, right_values)) and any(a > b for a, b in zip(left_values, right_values))


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
    selection_union_indices: Sequence[int] | None = None,
    fit_exclude_indices: Sequence[int] | None = None,
    selection_identity_hash: str | None = None,
    validation_b_indices: Sequence[int] | None = None,
    validation_b_identity_hash: str | None = None,
    selection_manifest: str | Path | None = None,
    selection_split: str = "train",
    validation_b_manifest: str | Path | None = None,
    validation_b_split: str = "train",
    evaluate_holdout: bool = True,
    initial_checkpoint_dir: str | Path | None = None,
    router_hidden_size: int | None = None,
    router_feature_mode: str = "none",
) -> dict[str, Any]:
    """Run a configurable staged distillation schedule against fixed splits.

    The default schedule is kept identical to the original three-stage path.
    A caller may provide a bounded sequence of stage mappings to compare
    router warm-up, router/scale transitions, frozen-router expert adaptation,
    and joint fine-tuning without changing the activation or holdout contract.
    ``selection_indices`` identifies validation-A rows used only for checkpoint
    selection.  ``validation_b_indices`` identifies an optional independent
    confirmation set.  Both are excluded from optimizer updates; callers must
    provide the complete ``fit_exclude_indices`` set explicitly and its
    identity is persisted alongside the A/B hashes.  Validation-B is evaluated
    only after the A-selected checkpoint is restored.  The legacy
    ``selection_union_indices`` argument is accepted only when it is exactly
    validation-A; an A+B union raises instead of allowing B to influence
    checkpoint selection.
    ``selection_manifest`` and ``validation_b_manifest`` provide independent
    FIT-DEV/SHADOW manifests. Their dataset identities are intentionally
    allowed to differ from FIT-TRAIN; they are never optimizer inputs.
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
    # Selector-only research may use a train-only fresh capture with a
    # separately frozen confirmation index.  Do not require or open the
    # historical holdout manifest unless this invocation explicitly authorizes
    # final holdout confirmation.
    holdout_dataset = (
        ActivationShardDataset(activation_manifest, split="holdout", microbatch=microbatch)
        if evaluate_holdout
        else None
    )
    if train_dataset.dataset_hash != dataset_hash:
        dataset_hash = train_dataset.dataset_hash
    if holdout_dataset is not None and holdout_dataset.dataset_hash != dataset_hash:
        raise ValueError("train and holdout activation manifests have different dataset_hash values")
    selection_dataset = (
        ActivationShardDataset(selection_manifest, split=selection_split, microbatch=microbatch)
        if selection_manifest is not None
        else None
    )
    validation_b_dataset = (
        ActivationShardDataset(validation_b_manifest, split=validation_b_split, microbatch=microbatch)
        if validation_b_manifest is not None
        else None
    )
    if selection_dataset is not None and any(value is not None for value in (selection_indices, selection_union_indices, validation_b_indices, fit_exclude_indices)):
        raise ValueError("independent selection manifests cannot be combined with positional selection indices")
    if validation_b_dataset is not None and validation_b_indices is not None:
        raise ValueError("independent validation-B manifest cannot be combined with positional validation-B indices")
    split_contract = validate_split_contract(
        train_dataset.count,
        selection_indices=None if selection_dataset is not None else selection_indices,
        validation_b_indices=None if validation_b_dataset is not None else validation_b_indices,
        fit_exclude_indices=fit_exclude_indices,
        selection_union_indices=None if selection_dataset is not None else selection_union_indices,
    )
    selection_rows = split_contract["selection_indices"]
    validation_b_rows = split_contract["validation_b_indices"] or ()
    fit_excluded_rows = split_contract["fit_exclude_indices"] or ()
    fit_indices = split_contract["fit_indices"] or ()
    # Independent manifests are streamed exactly like the training manifest,
    # but they must never be represented as positional rows from FIT-TRAIN.
    # Keep the legacy positional path intact when no independent manifest was
    # supplied so existing checkpoints retain their byte-compatible metadata.
    selection_source = selection_dataset or train_dataset
    selection_indices_for_metrics = None if selection_dataset is not None else selection_rows
    validation_b_source = validation_b_dataset or train_dataset
    validation_b_indices_for_metrics = None if validation_b_dataset is not None else validation_b_rows
    has_selection = selection_dataset is not None or selection_rows is not None
    has_validation_b = validation_b_dataset is not None or bool(validation_b_rows)
    selection_hash = selection_dataset.dataset_hash if selection_dataset is not None else _hash_indices(selection_rows) if selection_rows is not None else None
    # This remains a metadata key for old consumers, but is deliberately null:
    # there is no second split in the checkpoint-selection metric.
    selection_union_hash = None
    fit_hash = _hash_indices(fit_indices)
    fit_exclusion_hash = _hash_indices(fit_excluded_rows) if fit_excluded_rows else None
    validation_a_hash = selection_identity_hash or selection_hash
    computed_validation_b_hash = validation_b_dataset.dataset_hash if validation_b_dataset is not None else _hash_indices(validation_b_rows) if validation_b_rows else None
    validation_b_hash = validation_b_identity_hash or computed_validation_b_hash
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
        router_hidden_size=router_hidden_size,
        router_feature_mode=router_feature_mode,
        partition=plan,
        learnable_scales=True,
    )
    initialized_from_checkpoint = False
    if initial_checkpoint_dir is not None:
        checkpoint_dir = Path(initial_checkpoint_dir)
        tensor_path = checkpoint_dir / f"layer-{layer:04d}.safetensors"
        metadata_path = checkpoint_dir / f"layer-{layer:04d}.json"
        if not tensor_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"initial checkpoint is incomplete: {checkpoint_dir}")
        from safetensors.torch import load_file  # type: ignore

        raw_state = load_file(str(tensor_path), device="cpu")
        tensor_prefix = f"model.layers.{layer}."
        state = {key[len(tensor_prefix) :]: value for key, value in raw_state.items() if key.startswith(tensor_prefix)}
        if len(state) != len(raw_state):
            raise ValueError(f"initial checkpoint tensor namespace mismatch: {sorted(raw_state)[:3]}")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(f"strict initial checkpoint reload failed: missing={missing}, unexpected={unexpected}")
        initialized_from_checkpoint = True
    partition_payload = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    initial_scales = partition_payload.get("initial_expert_scales")
    # A strict checkpoint reload is authoritative for every tensor, including
    # learned expert scales.  Reapplying partition initialization here would
    # silently mutate a frozen basis before a selector-only continuation and
    # would make the reported checkpoint differ from the input checkpoint.
    if initial_scales is not None and not initialized_from_checkpoint:
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
    model.to(device)
    if not initialized_from_checkpoint:
        if router_hidden_size is not None:
            raise ValueError("nonlinear router training requires an explicit initial checkpoint")
        warmup_batches = (
            train_dataset.iter_batches(min(microbatch, 512))
            if not fit_excluded_rows
            else train_dataset.iter_excluding_batches(fit_excluded_rows, min(microbatch, 512))
        )
        warmup_values = next(warmup_batches)
        warmup_inputs = torch.as_tensor(warmup_values, dtype=torch.float32, device=device)
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
        selection_source,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        selected_indices=selection_indices_for_metrics,
    )
    initial_fit = _stream_metrics(
        model,
        train_dataset,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        excluded_indices=fit_excluded_rows if fit_excluded_rows else None,
    )
    stages: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    validation_trajectory: list[dict[str, Any]] = []
    # Always retain the initialized checkpoint as a candidate.  Checkpoint
    # selection is gate-aware: feasible candidates are ranked by cosine first;
    # if none is feasible, the full Pareto frontier is retained in metadata.
    best_state: dict[str, Any] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_record = {"stage": "initialized", "epoch": 0, **initial_selection}
    initial_record["feasible"] = _gate_feasible(initial_selection)
    initial_record["selection_reason"] = "initial_checkpoint"
    validation_trajectory.append(initial_record)
    best_record: dict[str, Any] = initial_record
    best_has_feasible = bool(initial_record["feasible"])

    def consider_checkpoint(current_model: TorchQwen35SwiGLUMoE, stage_name: str, epoch_number: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal best_state, best_record, best_has_feasible
        record = {"stage": stage_name, "epoch": int(epoch_number), **dict(metrics)}
        record["feasible"] = _gate_feasible(metrics)
        record["selection_reason"] = "validation_epoch_observation"
        validation_trajectory.append(record)
        should_select = False
        if record["feasible"]:
            if not best_has_feasible or _feasible_rank(record) > _feasible_rank(best_record):
                should_select = True
                record["selection_reason"] = "gate_feasible_max_cosine"
            best_has_feasible = True
        elif not best_has_feasible and _feasible_rank(record) > _feasible_rank(best_record):
            should_select = True
            record["selection_reason"] = "pareto_frontier_max_cosine_fallback"
        if should_select:
            best_state = {key: value.detach().cpu().clone() for key, value in current_model.state_dict().items()}
            best_record = record
        return record
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
        raw_loss_coefficients = raw_stage.get("loss_coefficients")
        if raw_loss_coefficients is not None and not isinstance(raw_loss_coefficients, Mapping):
            raise TypeError(f"stage_schedule[{index}].loss_coefficients must be a mapping")
        loss_coefficients = {str(name): float(value) for name, value in (raw_loss_coefficients or {}).items()}
        if any(value < 0 for value in loss_coefficients.values()):
            raise ValueError(f"stage_schedule[{index}].loss_coefficients values must be non-negative")
        oracle_regret_weight = float(raw_stage.get("oracle_regret_weight", 0.0))
        if oracle_regret_weight < 0:
            raise ValueError(f"stage_schedule[{index}].oracle_regret_weight must be non-negative")
        raw_prices = raw_stage.get("expert_use_prices")
        if raw_prices is not None:
            if isinstance(raw_prices, (str, bytes)) or not isinstance(raw_prices, Sequence):
                raise TypeError(f"stage_schedule[{index}].expert_use_prices must be a sequence")
            expert_use_prices = [float(value) for value in raw_prices]
            if len(expert_use_prices) != plan.routed_experts:
                raise ValueError(
                    f"stage_schedule[{index}].expert_use_prices must contain {plan.routed_experts} values"
                )
            if any(not math.isfinite(value) for value in expert_use_prices):
                raise ValueError(f"stage_schedule[{index}].expert_use_prices must contain finite values")
        else:
            expert_use_prices = None
        oracle_target_mode = str(raw_stage.get("oracle_target_mode", "contribution_norm"))
        if oracle_target_mode not in {"contribution_norm", "residual_correlation"}:
            raise ValueError(
                f"stage_schedule[{index}].oracle_target_mode must be contribution_norm or residual_correlation"
            )
        oracle_loss_mode = str(raw_stage.get("oracle_loss_mode", "repeated_cross_entropy"))
        if oracle_loss_mode not in {"repeated_cross_entropy", "multilabel_bce"}:
            raise ValueError(
                f"stage_schedule[{index}].oracle_loss_mode must be repeated_cross_entropy or multilabel_bce"
            )
        oracle_amplitude_mode = str(raw_stage.get("oracle_amplitude_mode", "student_selected"))
        if oracle_amplitude_mode not in {"student_selected", "teacher_forced", "mixed"}:
            raise ValueError(
                f"stage_schedule[{index}].oracle_amplitude_mode must be student_selected, teacher_forced, or mixed"
            )
        teacher_forcing_ratio = float(raw_stage.get("teacher_forcing_ratio", 0.0))
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError(f"stage_schedule[{index}].teacher_forcing_ratio must be between 0 and 1")
        normalized_schedule.append(
            {
                "name": stage_name,
                "epochs": stage_epochs,
                "train_scales": bool(raw_stage.get("train_scales", False)),
                "train_experts": bool(raw_stage.get("train_experts", False)),
                "train_shared": bool(raw_stage.get("train_shared", False)),
                "use_oracle_targets": bool(raw_stage.get("use_oracle_targets", False)),
                "oracle_target_mode": oracle_target_mode,
                "oracle_loss_mode": oracle_loss_mode,
                "oracle_amplitude_mode": oracle_amplitude_mode,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "train_selection_router": bool(raw_stage.get("train_selection_router", True)),
                "train_amplitude_router": bool(raw_stage.get("train_amplitude_router", True)),
                "learning_rate": stage_learning_rate,
                "learning_rates": rates,
                "loss_coefficients": loss_coefficients,
                "oracle_regret_weight": oracle_regret_weight,
                "expert_use_prices": expert_use_prices,
            }
        )
    for stage_spec in normalized_schedule:
        stage_name = str(stage_spec["name"])

        def validation_epoch_callback(current_model: TorchQwen35SwiGLUMoE, callback_stage: str, callback_epoch: int) -> Mapping[str, Any]:
            return consider_checkpoint(
                current_model,
                callback_stage,
                callback_epoch,
                _stream_metrics(
                    current_model,
                    selection_source,
                    gate=gate_tensor,
                    up=up_tensor,
                    down=down_tensor,
                    microbatch=microbatch,
                    device=device,
                    selected_indices=selection_indices_for_metrics,
                ),
            )

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
            train_shared=bool(stage_spec["train_shared"]),
            device=device,
            stage=stage_name,
            use_oracle_targets=bool(stage_spec["use_oracle_targets"]),
            oracle_target_mode=str(stage_spec["oracle_target_mode"]),
            oracle_loss_mode=str(stage_spec["oracle_loss_mode"]),
            oracle_amplitude_mode=str(stage_spec["oracle_amplitude_mode"]),
            teacher_forcing_ratio=float(stage_spec["teacher_forcing_ratio"]),
            train_selection_router=bool(stage_spec["train_selection_router"]),
            train_amplitude_router=bool(stage_spec["train_amplitude_router"]),
            learning_rates=stage_spec["learning_rates"],
            loss_coefficients=stage_spec["loss_coefficients"],
            oracle_regret_weight=float(stage_spec["oracle_regret_weight"]),
            expert_use_prices=stage_spec["expert_use_prices"],
            excluded_indices=fit_excluded_rows if fit_excluded_rows else None,
            epoch_callback=validation_epoch_callback if has_selection else None,
        )
        stages.append(stage_result)
        if stage_result.get("epoch_validation_metrics"):
            stage_metrics.append({"stage": stage_name, **stage_result["epoch_validation_metrics"][-1]})
        elif has_selection:
            measured = _stream_metrics(
                model,
                selection_source,
                gate=gate_tensor,
                up=up_tensor,
                down=down_tensor,
                microbatch=microbatch,
                device=device,
                selected_indices=selection_indices_for_metrics,
            )
            stage_metrics.append({"stage": stage_name, "epoch": 0, **measured, "feasible": _gate_feasible(measured)})
    model.load_state_dict(best_state, strict=True)
    final_selection = _stream_metrics(
        model,
        selection_source,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        selected_indices=selection_indices_for_metrics,
    )
    final_fit = _stream_metrics(
        model,
        train_dataset,
        gate=gate_tensor,
        up=up_tensor,
        down=down_tensor,
        microbatch=microbatch,
        device=device,
        excluded_indices=fit_excluded_rows if fit_excluded_rows else None,
    )
    if has_validation_b:
        validation_b_metrics = _stream_metrics(
            model,
            validation_b_source,
            gate=gate_tensor,
            up=up_tensor,
            down=down_tensor,
            microbatch=microbatch,
            device=device,
            selected_indices=validation_b_indices_for_metrics,
        )
        validation_b_metrics = {
            **validation_b_metrics,
            "split": "validation-b",
            "checkpoint_selection": False,
        }
    else:
        validation_b_metrics = {
            "split": "validation-b",
            "status": "NOT_CONFIGURED",
            "checkpoint_selection": False,
            "normalized_mse": None,
            "cosine": None,
            "selected_counts": [],
            "dead_experts": None,
            "load_cv": None,
            "streaming": True,
        }
    pareto_frontier = [
        record
        for record in validation_trajectory
        if not any(
            other is not record and _pareto_dominates(other, record)
            for other in validation_trajectory
        )
    ]
    best_stage = str(best_record["stage"])
    best_epoch = int(best_record.get("epoch", 0))
    if evaluate_holdout:
        if holdout_dataset is None:  # pragma: no cover - guarded at construction
            raise RuntimeError("holdout dataset is required for final confirmation")
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
        router_architecture=(f"torch-low-rank-silu-topk-{profile.routing_mode}-v1" if router_hidden_size is not None else f"torch-linear-topk-{profile.routing_mode}-v1"),
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
            "validation_trajectory": validation_trajectory,
            "pareto_frontier": pareto_frontier,
            "validation_b_metrics": validation_b_metrics,
            "checkpoint_selection_rule": {
                "feasibility": "normalized_mse<=0.05 and cosine>=0.98 and dead_experts==0 and load_cv<=0.50",
                "ordering": "maximize cosine, then minimize normalized_mse, then minimize load_cv",
                "fallback": "retain Pareto frontier and select highest-cosine frontier point",
                "selection_split": "validation-A only",
                "validation_b_role": "independent confirmation after checkpoint selection",
                "selection_union": "disabled",
            },
            "best_selection_stage": best_stage,
            "best_selection_epoch": best_epoch,
            "best_selection_reason": best_record.get("selection_reason"),
            "selection_split": "independent-manifest" if selection_dataset is not None else "validation-A" if selection_rows is not None else "train_split",
            "selection_indices_hash": selection_hash,
            "selection_union_indices_hash": selection_union_hash,
            "selection_identity_hash": validation_a_hash,
            "selection_count": selection_dataset.count if selection_dataset is not None else len(selection_rows) if selection_rows is not None else train_dataset.count,
            "validation_count": selection_dataset.count if selection_dataset is not None else len(selection_rows) if selection_rows is not None else None,
            "validation_hash": validation_a_hash,
            "validation_a_indices_hash": selection_hash,
            "validation_a_identity_hash": validation_a_hash,
            "validation_a_count": selection_dataset.count if selection_dataset is not None else len(selection_rows) if selection_rows is not None else 0,
            "selection_union_count": 0,
            "validation_b_indices_hash": computed_validation_b_hash,
            "validation_b_identity_hash": validation_b_hash,
            "validation_b_count": validation_b_dataset.count if validation_b_dataset is not None else len(validation_b_rows),
            "fit_count": len(fit_indices),
            "fit_index_hash": fit_hash,
            "fit_excluded_indices_hash": fit_exclusion_hash,
            "fit_excluded_count": len(fit_excluded_rows),
            "fit_exclusion_contract": (
                "train rows excluding validation-A and validation-B"
                if has_validation_b and has_selection
                else "train rows excluding validation-A"
                if has_selection
                else "train rows excluding validation-B"
                if has_validation_b
                else "no exclusion"
            ),
            "holdout_evaluation": holdout_status,
            "split_opened_for": {
                "fit": {"gradient_updates": True, "checkpoint_selection": False, "final_confirmation": False},
                "validation": {"gradient_updates": False, "checkpoint_selection": has_selection, "final_confirmation": False},
                "validation_a": {"gradient_updates": False, "checkpoint_selection": has_selection, "final_confirmation": False},
                "validation_b": {"gradient_updates": False, "checkpoint_selection": False, "final_confirmation": has_validation_b},
                "holdout": {"gradient_updates": False, "checkpoint_selection": False, "final_confirmation": evaluate_holdout},
            },
            "streaming_dataset": {
                "train_manifest": train_dataset.manifest_path.as_posix(),
                "selection_manifest": selection_dataset.manifest_path.as_posix() if selection_dataset is not None else None,
                "validation_b_manifest": validation_b_dataset.manifest_path.as_posix() if validation_b_dataset is not None else None,
                "holdout_manifest": holdout_dataset.manifest_path.as_posix() if holdout_dataset is not None else None,
                "train_count": train_dataset.count,
                "selection_count": selection_dataset.count if selection_dataset is not None else None,
                "validation_b_count": validation_b_dataset.count if validation_b_dataset is not None else None,
                "holdout_count": holdout_dataset.count if holdout_dataset is not None else None,
            },
            "partition_path": str(partition_path),
            "initial_expert_scales": [float(value) for value in (initial_scales or [1.0] * plan.routed_experts)],
            "router_initialization": "initial_checkpoint" if initialized_from_checkpoint else "bounded_train_contribution_lstsq",
            "initial_checkpoint_dir": str(initial_checkpoint_dir) if initial_checkpoint_dir is not None else None,
            "router_hidden_size": router_hidden_size,
            "router_feature_mode": router_feature_mode,
        },
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"initial_fit": initial_fit, "final_fit": final_fit, "initial_selection": initial_selection, "final_selection": final_selection, "oracle_holdout": oracle_holdout, "all_expert_reconstruction_mse": 0.0},
        holdout_metrics=trained,
        router_metrics={"load_cv": gate_metrics["load_cv"], "dead_experts": gate_metrics["dead_experts"], "selected_counts": gate_metrics["selected_counts"], "actual_improvement": float(initial_selection["normalized_mse"] - gate_metrics["normalized_mse"]) if gate_metrics["normalized_mse"] is not None else None},
        quality_gate={"overall": gate_overall if epochs > 0 else "untrained", "thresholds_version": "gate-aware-2026-08-16", "evaluation_scope": "full_holdout" if evaluate_holdout else "validation-a" if has_selection else "fit", "metrics": gate_metrics},
        code_commit=recorded_commit,
    )
    metadata_path = output / f"layer-{layer:04d}.json"
    save_layer_checkpoint(checkpoint, metadata_path)
    return {"status": status, "layer": layer, "metadata": str(metadata_path), "tensor_file": str(tensor_path), "holdout_metrics": trained, "validation_b_metrics": validation_b_metrics, "initial_fit": initial_fit, "final_fit": final_fit, "initial_selection": initial_selection, "final_selection": final_selection, "oracle_holdout": oracle_holdout, "training_config": checkpoint.training_config, "code_commit": recorded_commit}


__all__ = ["load_fixed_activation_splits", "train_torch_layer"]
