"""Checkpoint-aware contribution stores for frozen MoE bases.

The original contribution path in this repository evaluates the dense teacher
and then slices its intermediate neurons according to a partition plan.  That
mode is useful for raw partition diagnostics, but it is not evidence about a
trained checkpoint.  This module keeps the two paths deliberately explicit:

``basis_source=raw_dense_partition``
    reconstructs slices from the immutable dense MLP tensors;
``basis_source=trained_checkpoint``
    evaluates the shared and routed tensors loaded from the supplied frozen
    checkpoint.  There is no fallback from the latter to the former.

The numerical implementation is NumPy based so stores can be materialized on
Windows without constructing a full trainable model.  The same formulas are
used by :class:`TorchQwen35SwiGLUMoE`, and ``basis_outputs_from_state`` is
intentionally small enough to be used in an independent equivalence receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ffn import PartitionPlan


def _np() -> Any:
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("numpy is required for contribution evaluation") from exc


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without reading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_partition_sha256(plan: PartitionPlan) -> str:
    """Hash the canonical partition object used by checkpoint receipts."""

    return hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest()


def load_partition_plan(path: str | Path) -> PartitionPlan:
    """Load a partition JSON that may be wrapped in a ``plan`` object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = payload.get("plan", payload)
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


def resolve_checkpoint_tensor(path: str | Path) -> Path:
    """Resolve a checkpoint directory, metadata JSON, or tensor file.

    A caller that supplies ``--checkpoint`` is required to identify an actual
    tensor artifact.  Metadata is used only to locate that artifact; an absent
    file is an error rather than an invitation to reconstruct raw dense slices.
    """

    candidate = Path(path)
    if candidate.is_file() and candidate.suffix == ".safetensors":
        return candidate
    if candidate.is_file() and candidate.suffix == ".json":
        metadata = json.loads(candidate.read_text(encoding="utf-8"))
        value = metadata.get("tensor_file")
        if value:
            tensor = Path(str(value))
            if not tensor.is_absolute():
                tensor = candidate.parent / tensor
            if tensor.is_file():
                return tensor
        tensor = candidate.with_suffix(".safetensors")
        if tensor.is_file():
            return tensor
        raise FileNotFoundError(f"checkpoint metadata has no readable tensor artifact: {candidate}")
    if candidate.is_dir():
        for name in ("layer-0000.safetensors", "model.safetensors"):
            tensor = candidate / name
            if tensor.is_file():
                return tensor
        raise FileNotFoundError(f"checkpoint directory has no layer-0000.safetensors/model.safetensors: {candidate}")
    raise FileNotFoundError(candidate)


def _normalise_key(key: str) -> str:
    for prefix in ("model.layers.0.", "layers.0.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _load_safetensors(path: Path) -> dict[str, Any]:
    """Load tensors through safetensors without importing torch eagerly."""

    try:
        from safetensors.numpy import load_file  # type: ignore
        try:
            values = load_file(str(path), device="cpu")
        except TypeError:  # older safetensors.numpy has no device keyword
            values = load_file(str(path))
    except (ImportError, ValueError):
        try:
            from safetensors.torch import load_file  # type: ignore

            values = load_file(str(path), device="cpu")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required for checkpoint contributions") from exc
    return {_normalise_key(str(key)): value for key, value in values.items()}


def _to_numpy(value: Any) -> Any:
    np = _np()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _required_basis_keys(plan: PartitionPlan) -> set[str]:
    keys = {
        "shared_gate_proj.weight",
        "shared_up_proj.weight",
        "shared_down_proj.weight",
        "expert_scales",
    }
    for expert in range(plan.routed_experts):
        keys.update(
            {
                f"expert_gate_proj.{expert}.weight",
                f"expert_up_proj.{expert}.weight",
                f"expert_down_proj.{expert}.weight",
            }
        )
    return keys


def load_trained_basis_state(checkpoint: str | Path, plan: PartitionPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the frozen basis tensors from ``checkpoint``.

    Router tensors are permitted in the artifact but ignored: a contribution
    store represents the shared branch and every routed expert, before a route
    assignment is selected.  Missing basis tensors are fatal.
    """

    plan.validate()
    tensor_path = resolve_checkpoint_tensor(checkpoint)
    raw = _load_safetensors(tensor_path)
    required = _required_basis_keys(plan)
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"trained checkpoint is missing basis tensors: {missing[:8]}")
    state = {key: _to_numpy(raw[key]) for key in required}
    hidden = int(state["shared_down_proj.weight"].shape[0])
    if state["shared_gate_proj.weight"].shape != (plan.shared_intermediate_size, hidden):
        raise ValueError("trained shared gate projection shape does not match partition")
    if state["shared_up_proj.weight"].shape != state["shared_gate_proj.weight"].shape:
        raise ValueError("trained shared gate/up projection shapes differ")
    if state["shared_down_proj.weight"].shape != (hidden, plan.shared_intermediate_size):
        raise ValueError("trained shared down projection shape does not match partition")
    if tuple(state["expert_scales"].shape) != (plan.routed_experts,):
        raise ValueError("trained expert scale shape does not match expert count")
    for expert in range(plan.routed_experts):
        gate = state[f"expert_gate_proj.{expert}.weight"]
        up = state[f"expert_up_proj.{expert}.weight"]
        down = state[f"expert_down_proj.{expert}.weight"]
        expected = (plan.expert_intermediate_size, hidden)
        if tuple(gate.shape) != expected or tuple(up.shape) != expected:
            raise ValueError(f"trained expert {expert} gate/up shape does not match partition")
        if tuple(down.shape) != (hidden, plan.expert_intermediate_size):
            raise ValueError(f"trained expert {expert} down shape does not match partition")
    metadata_path = tensor_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            metadata = dict(payload)
    metadata.update(
        {
            "checkpoint_path": str(tensor_path),
            "checkpoint_tensor_sha256": sha256_file(tensor_path),
            "checkpoint_metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        }
    )
    return state, metadata


def _silu(value: Any) -> Any:
    np = _np()
    values = np.asarray(value)
    # The clipped form avoids overflow when a tiny deterministic fixture uses
    # unusually large activations while matching torch.nn.functional.silu.
    return values / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def basis_outputs_from_state(inputs: Any, state: Mapping[str, Any], plan: PartitionPlan, *, batch_size: int = 256) -> tuple[Any, Any]:
    """Evaluate shared and every routed expert output from learned tensors."""

    np = _np()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    plan.validate()
    x = np.asarray(_to_numpy(inputs))
    if x.ndim < 2:
        raise ValueError("inputs must have a hidden dimension")
    original = x.shape
    x = x.reshape(-1, original[-1])
    gate = np.asarray(state["shared_gate_proj.weight"])
    up = np.asarray(state["shared_up_proj.weight"])
    down = np.asarray(state["shared_down_proj.weight"])
    scales = np.asarray(state["expert_scales"]).reshape(plan.routed_experts)
    shared_parts: list[Any] = []
    routed_parts: list[Any] = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        shared_hidden = _silu(batch @ gate.T) * (batch @ up.T)
        shared_parts.append(shared_hidden @ down.T)
        expert_outputs: list[Any] = []
        for expert in range(plan.routed_experts):
            e_gate = np.asarray(state[f"expert_gate_proj.{expert}.weight"])
            e_up = np.asarray(state[f"expert_up_proj.{expert}.weight"])
            e_down = np.asarray(state[f"expert_down_proj.{expert}.weight"])
            hidden = _silu(batch @ e_gate.T) * (batch @ e_up.T)
            expert_outputs.append((hidden @ e_down.T) * scales[expert])
        routed_parts.append(np.stack(expert_outputs, axis=1))
    shared = np.concatenate(shared_parts, axis=0).reshape(*original[:-1], -1)
    routed = np.concatenate(routed_parts, axis=0).reshape(*original[:-1], plan.routed_experts, -1)
    return shared, routed


def trained_checkpoint_contributions(
    inputs: Any,
    checkpoint: str | Path,
    plan: PartitionPlan,
    *,
    batch_size: int = 256,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a checkpoint and return its actual shared/expert contributions."""

    state, metadata = load_trained_basis_state(checkpoint, plan)
    shared, routed = basis_outputs_from_state(inputs, state, plan, batch_size=batch_size)
    metadata = dict(metadata)
    metadata.update(
        {
            "basis_source": "trained_checkpoint",
            "dtype": str(next(iter(state.values())).dtype),
            "expert_count": plan.routed_experts,
            "expert_width": plan.expert_intermediate_size,
            "shared_width": plan.shared_intermediate_size,
            "topology_basis_output": "shared_plus_scaled_routed_expert_outputs",
        }
    )
    return shared, routed, metadata


def raw_dense_partition_contributions(inputs: Any, dense_state: Mapping[str, Any], plan: PartitionPlan) -> tuple[Any, Any]:
    """Evaluate the explicitly labelled raw dense-partition contribution mode."""

    np = _np()
    gate = np.asarray(_to_numpy(dense_state["gate_proj.weight"]))
    up = np.asarray(_to_numpy(dense_state["up_proj.weight"]))
    down = np.asarray(_to_numpy(dense_state["down_proj.weight"]))
    x = np.asarray(_to_numpy(inputs))
    if gate.shape != up.shape or tuple(down.shape) != (gate.shape[1], gate.shape[0]):
        raise ValueError("raw dense projection shapes do not match")
    plan.validate()
    x2 = x.reshape(-1, x.shape[-1])
    hidden = _silu(x2 @ gate.T) * (x2 @ up.T)
    shared = hidden[:, plan.shared_indices] @ down[:, plan.shared_indices].T
    routed = np.stack(
        [hidden[:, group] @ down[:, group].T for group in plan.expert_indices],
        axis=1,
    )
    return shared.reshape(*x.shape[:-1], -1), routed.reshape(*x.shape[:-1], plan.routed_experts, -1)


def reconstruct_selected(shared: Any, routed: Any, ids: Any, weights: Any) -> Any:
    """Reconstruct arbitrary selected top-k sums from a contribution store."""

    np = _np()
    shared_values = np.asarray(shared)
    routed_values = np.asarray(routed)
    ids_values = np.asarray(ids, dtype=np.int64)
    weights_values = np.asarray(weights)
    if ids_values.ndim != 2 or weights_values.shape != ids_values.shape:
        raise ValueError("ids and weights must both be [tokens, top_k]")
    selected = np.take_along_axis(routed_values, ids_values[:, :, None], axis=1)
    return shared_values + np.sum(selected * weights_values[:, :, None], axis=1)


def contribution_manifest(
    *,
    basis_source: str,
    plan: PartitionPlan,
    row_count: int,
    split: str,
    dataset_hash: str,
    partition_path: str | Path,
    top_k: int | None = None,
    checkpoint: str | Path | None = None,
    source_revision: str | None = None,
    capture_identity: Mapping[str, Any] | None = None,
    dtype: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build the unambiguous provenance block shared by stores and reports."""

    if basis_source not in {"trained_checkpoint", "raw_dense_partition"}:
        raise ValueError(f"unsupported basis_source: {basis_source!r}")
    if basis_source == "trained_checkpoint" and checkpoint is None:
        raise ValueError("trained_checkpoint manifests require checkpoint")
    partition_file = Path(partition_path)
    payload: dict[str, Any] = {
        "basis_source": basis_source,
        "checkpoint_path": str(resolve_checkpoint_tensor(checkpoint)) if checkpoint is not None else None,
        "checkpoint_tensor_sha256": sha256_file(resolve_checkpoint_tensor(checkpoint)) if checkpoint is not None else None,
        "partition_path": str(partition_file),
        "partition_sha256": sha256_file(partition_file),
        "partition_canonical_sha256": canonical_partition_sha256(plan),
        "topology": {
            "expert_count": int(plan.routed_experts),
            "expert_width": int(plan.expert_intermediate_size),
            "shared_width": int(plan.shared_intermediate_size),
            "top_k": int(top_k) if top_k is not None else None,
        },
        "source_revision": source_revision,
        "dataset_hash": dataset_hash,
        "capture_identity": dict(capture_identity or {}),
        "split": split,
        "code_commit": code_commit,
        "dtype": dtype,
        "row_count": int(row_count),
    }
    return payload


__all__ = [
    "basis_outputs_from_state",
    "canonical_partition_sha256",
    "contribution_manifest",
    "load_partition_plan",
    "load_trained_basis_state",
    "raw_dense_partition_contributions",
    "reconstruct_selected",
    "resolve_checkpoint_tensor",
    "sha256_file",
    "trained_checkpoint_contributions",
]
