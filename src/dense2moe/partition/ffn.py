"""Capacity-preserving dense FFN partitioning."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


def _as_list(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values]


@dataclass(frozen=True)
class PartitionPlan:
    dense_intermediate_size: int
    routed_experts: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    shared_indices: tuple[int, ...]
    expert_indices: tuple[tuple[int, ...], ...]

    @property
    def total_capacity(self) -> int:
        return self.shared_intermediate_size + self.routed_experts * self.expert_intermediate_size

    @property
    def all_indices(self) -> tuple[int, ...]:
        return self.shared_indices + tuple(index for group in self.expert_indices for index in group)

    def validate(self) -> None:
        if self.total_capacity != self.dense_intermediate_size:
            raise ValueError("partition capacity does not equal dense intermediate size")
        if len(self.shared_indices) != self.shared_intermediate_size:
            raise ValueError("shared index count mismatch")
        if len(self.expert_indices) != self.routed_experts:
            raise ValueError("expert count mismatch")
        if any(len(group) != self.expert_intermediate_size for group in self.expert_indices):
            raise ValueError("expert width mismatch")
        if len(set(self.all_indices)) != self.dense_intermediate_size:
            raise ValueError("partition is not disjoint")
        if set(self.all_indices) != set(range(self.dense_intermediate_size)):
            raise ValueError("partition is not exhaustive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "dense_intermediate_size": self.dense_intermediate_size,
            "routed_experts": self.routed_experts,
            "expert_intermediate_size": self.expert_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
            "shared_indices": list(self.shared_indices),
            "expert_indices": [list(group) for group in self.expert_indices],
        }


def partition_indices(
    dense_intermediate_size: int,
    routed_experts: int,
    expert_intermediate_size: int,
    shared_intermediate_size: int,
    *,
    strategy: str = "contiguous",
    scores: Any | None = None,
    seed: int = 0,
) -> PartitionPlan:
    if dense_intermediate_size <= 0 or routed_experts <= 0 or expert_intermediate_size <= 0 or shared_intermediate_size <= 0:
        raise ValueError("partition dimensions must be positive")
    if shared_intermediate_size + routed_experts * expert_intermediate_size != dense_intermediate_size:
        raise ValueError("partition capacity must equal dense_intermediate_size")
    indices = list(range(dense_intermediate_size))
    if strategy not in {"contiguous", "interleave", "activation_magnitude", "output_contribution", "balanced_signature", "random"}:
        raise ValueError(f"unknown partition strategy: {strategy}")
    if strategy in {"activation_magnitude", "output_contribution", "balanced_signature"}:
        if scores is None:
            raise ValueError(f"{strategy} requires deterministic neuron scores")
        try:
            import numpy as np  # type: ignore

            values = np.asarray(scores)
            if values.ndim > 1:
                values = np.mean(np.abs(values), axis=tuple(range(values.ndim - 1)))
            if values.shape[0] != dense_intermediate_size:
                raise ValueError("partition scores do not match dense width")
            order = sorted(range(dense_intermediate_size), key=lambda i: (-float(values[i]), i))
        except ImportError:
            order = list(range(dense_intermediate_size))
        shared = tuple(order[:shared_intermediate_size])
        remaining = order[shared_intermediate_size:]
        groups = tuple(tuple(remaining[i * expert_intermediate_size : (i + 1) * expert_intermediate_size]) for i in range(routed_experts))
    elif strategy == "random":
        import random

        order = list(range(dense_intermediate_size))
        random.Random(seed).shuffle(order)
        shared = tuple(order[:shared_intermediate_size])
        remaining = order[shared_intermediate_size:]
        groups = tuple(tuple(remaining[i * expert_intermediate_size : (i + 1) * expert_intermediate_size]) for i in range(routed_experts))
    elif strategy == "interleave":
        # A deterministic permutation gives each expert a spread of source
        # neurons while keeping the shared block stable.
        shared = tuple(indices[:shared_intermediate_size])
        remaining = indices[shared_intermediate_size:]
        groups = tuple(tuple(remaining[offset::routed_experts][:expert_intermediate_size]) for offset in range(routed_experts))
    else:
        shared = tuple(indices[:shared_intermediate_size])
        remaining = indices[shared_intermediate_size:]
        groups = tuple(tuple(remaining[i * expert_intermediate_size : (i + 1) * expert_intermediate_size]) for i in range(routed_experts))
    plan = PartitionPlan(dense_intermediate_size, routed_experts, expert_intermediate_size, shared_intermediate_size, shared, tuple(tuple(group) for group in groups))
    plan.validate()
    return plan


def _take_rows(value: Any, indices: Sequence[int]) -> Any:
    try:
        return value[list(indices)]
    except (TypeError, IndexError):
        return [value[index] for index in indices]


def _concat(values: Sequence[Any], axis: int = 0) -> Any:
    try:
        import numpy as np  # type: ignore

        if all(hasattr(value, "shape") for value in values):
            return np.concatenate(values, axis=axis)
    except ImportError:
        pass
    first = values[0]
    if isinstance(first, list):
        result: list[Any] = []
        for value in values:
            result.extend(value)
        return result
    return values


def partition_ffn_weights(weight: Any, plan: PartitionPlan, bias: Any | None = None) -> dict[str, Any]:
    plan.validate()
    shared = _take_rows(weight, plan.shared_indices)
    experts = [_take_rows(weight, group) for group in plan.expert_indices]
    output: dict[str, Any] = {"shared": shared, "experts": experts, "plan": plan}
    if bias is not None:
        output["shared_bias"] = _take_rows(bias, plan.shared_indices)
        output["expert_bias"] = [_take_rows(bias, group) for group in plan.expert_indices]
    return output


def reconstruct_ffn_weights(parts: dict[str, Any], plan: PartitionPlan) -> Any:
    plan.validate()
    rows: list[tuple[int, Any]] = list(zip(plan.shared_indices, parts["shared"]))
    for group, values in zip(plan.expert_indices, parts["experts"]):
        rows.extend(zip(group, values))
    rows.sort(key=lambda item: item[0])
    first_value = rows[0][1] if rows else None
    if getattr(first_value.__class__, "__module__", "").startswith("torch"):
        import torch  # type: ignore

        return torch.stack([value for _, value in rows], dim=0)
    try:
        import numpy as np  # type: ignore

        return np.stack([value for _, value in rows], axis=0)
    except ImportError:
        return [value for _, value in rows]


def pack_experts(experts: Sequence[Any]) -> Any:
    if not experts:
        raise ValueError("at least one expert is required")
    if getattr(experts[0].__class__, "__module__", "").startswith("torch"):
        import torch  # type: ignore

        return torch.stack(list(experts), dim=0)
    try:
        import numpy as np  # type: ignore

        return np.stack(experts, axis=0)
    except ImportError:
        return [list(expert) for expert in experts]


def unpack_experts(packed: Any) -> list[Any]:
    try:
        return [packed[index] for index in range(len(packed))]
    except TypeError:
        return list(packed)


class ActivationPartitioner:
    """Assign dense neurons to groups using deterministic activation scores."""

    def __init__(self, plan: PartitionPlan, strategy: str = "balanced"):
        self.plan = plan
        self.strategy = strategy

    def assign(self, activations: Any | None = None) -> PartitionPlan:
        if activations is None or self.strategy == "contiguous":
            return self.plan
        try:
            import numpy as np  # type: ignore

            values = np.asarray(activations)
            if values.ndim == 1:
                scores = np.abs(values)
            else:
                scores = np.mean(np.abs(values), axis=tuple(range(values.ndim - 1)))
            if scores.shape[0] != self.plan.dense_intermediate_size:
                raise ValueError("activation neuron dimension does not match plan")
            order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        except ImportError:
            order = list(range(self.plan.dense_intermediate_size))
        shared = tuple(order[: self.plan.shared_intermediate_size])
        remaining = order[self.plan.shared_intermediate_size :]
        groups = tuple(tuple(remaining[i * self.plan.expert_intermediate_size : (i + 1) * self.plan.expert_intermediate_size]) for i in range(self.plan.routed_experts))
        result = PartitionPlan(self.plan.dense_intermediate_size, self.plan.routed_experts, self.plan.expert_intermediate_size, self.plan.shared_intermediate_size, shared, groups)
        result.validate()
        return result

    @staticmethod
    def fingerprint(plan: PartitionPlan) -> str:
        digest = hashlib.sha256(repr(plan.as_dict()).encode("utf-8"))
        return digest.hexdigest()
