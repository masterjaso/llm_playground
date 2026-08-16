"""Trainable PyTorch SwiGLU MoE blocks and reloadable layer artifacts.

The NumPy target in :mod:`dense2moe.models.qwen_moe` is intentionally kept as
the dependency-light numerical oracle.  This module is the trainable path used
by one-layer distillation and by the tiny full-model save/reload spike.  It
uses the same Hugging Face weight orientation (``[out, in]``), preserves the
selected dense partition, and performs normalized top-k routing without token
dropping.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..partition import PartitionPlan, partition_indices

try:  # Optional ML dependency: the control plane remains importable without it.
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    torch = None  # type: ignore[assignment]

    class _MissingModule:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for TorchQwen35SwiGLUMoE")

    class _MissingNN:
        Module = _MissingModule

    nn = _MissingNN()  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the trainable MoE path")
    return torch


def _as_tensor(value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Tensor:
    runtime = _require_torch()
    if isinstance(value, runtime.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = runtime.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _partition_from_payload(payload: Mapping[str, Any]) -> PartitionPlan:
    return PartitionPlan(
        int(payload["dense_intermediate_size"]),
        int(payload["routed_experts"]),
        int(payload["expert_intermediate_size"]),
        int(payload["shared_intermediate_size"]),
        tuple(int(value) for value in payload["shared_indices"]),
        tuple(tuple(int(value) for value in group) for group in payload["expert_indices"]),
    )


class TorchQwen35SwiGLUMoE(nn.Module):
    """Trainable shared+routed SwiGLU layer with normalized top-k routing.

    ``from_dense`` initializes every routed and shared parameter from one
    immutable dense layer according to the supplied :class:`PartitionPlan`.
    The forward pass computes every expert and masks the unselected outputs;
    this is deliberately token-safe and easy to audit.  A production kernel
    can later replace the loop without changing checkpoint semantics.
    """

    architecture = "qwen3_5_text_torch_swiglu_moe_v1"

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        routed_experts: int,
        expert_intermediate_size: int,
        shared_intermediate_size: int,
        top_k: int = 2,
        partition: PartitionPlan | None = None,
        learnable_scales: bool = False,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> None:
        runtime = _require_torch()
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0 or routed_experts <= 0:
            raise ValueError("MoE dimensions must be positive")
        if top_k <= 0 or top_k > routed_experts:
            raise ValueError("top_k must be in [1, routed_experts]")
        if partition is None:
            if intermediate_size != shared_intermediate_size + routed_experts * expert_intermediate_size:
                raise ValueError("partition capacity must equal intermediate_size")
            partition = partition_indices(
                intermediate_size,
                routed_experts,
                expert_intermediate_size,
                shared_intermediate_size,
            )
        partition.validate()
        if partition.dense_intermediate_size != intermediate_size:
            raise ValueError("partition and intermediate_size mismatch")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.routed_experts = int(routed_experts)
        self.expert_intermediate_size = int(expert_intermediate_size)
        self.shared_intermediate_size = int(shared_intermediate_size)
        self.top_k = int(top_k)
        self.partition = partition
        linear_kwargs: dict[str, Any] = {"bias": False}
        if dtype is not None:
            linear_kwargs["dtype"] = dtype
        if device is not None:
            linear_kwargs["device"] = device
        self.shared_gate_proj = nn.Linear(hidden_size, shared_intermediate_size, **linear_kwargs)
        self.shared_up_proj = nn.Linear(hidden_size, shared_intermediate_size, **linear_kwargs)
        self.shared_down_proj = nn.Linear(shared_intermediate_size, hidden_size, **linear_kwargs)
        self.expert_gate_proj = nn.ModuleList(
            [nn.Linear(hidden_size, expert_intermediate_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        self.expert_up_proj = nn.ModuleList(
            [nn.Linear(hidden_size, expert_intermediate_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        self.expert_down_proj = nn.ModuleList(
            [nn.Linear(expert_intermediate_size, hidden_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        self.router = nn.Linear(hidden_size, routed_experts, bias=False, **{key: value for key, value in linear_kwargs.items() if key != "bias"})
        if learnable_scales:
            self.expert_scales = nn.Parameter(runtime.ones(routed_experts, **{key: value for key, value in linear_kwargs.items() if key in {"dtype", "device"}}))
        else:
            self.register_buffer("expert_scales", runtime.ones(routed_experts, **{key: value for key, value in linear_kwargs.items() if key in {"dtype", "device"}}), persistent=True)
        self.learnable_scales = bool(learnable_scales)

    @classmethod
    def from_dense(
        cls,
        gate_proj: Any,
        up_proj: Any,
        down_proj: Any,
        *,
        routed_experts: int,
        shared_intermediate_size: int,
        top_k: int = 2,
        partition: PartitionPlan | None = None,
        router: Any | None = None,
        learnable_scales: bool = False,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> TorchQwen35SwiGLUMoE:
        runtime = _require_torch()
        gate = _as_tensor(gate_proj, dtype=dtype, device=device)
        up = _as_tensor(up_proj, dtype=dtype, device=device)
        down = _as_tensor(down_proj, dtype=dtype, device=device)
        if gate.ndim != 2 or up.shape != gate.shape or tuple(down.shape) != (gate.shape[1], gate.shape[0]):
            raise ValueError("dense SwiGLU projection shapes do not match")
        intermediate_size, hidden_size = map(int, gate.shape)
        if partition is None:
            expert_width = (intermediate_size - shared_intermediate_size) // routed_experts
            partition = partition_indices(intermediate_size, routed_experts, expert_width, shared_intermediate_size)
        model = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            routed_experts=routed_experts,
            expert_intermediate_size=partition.expert_intermediate_size,
            shared_intermediate_size=partition.shared_intermediate_size,
            top_k=top_k,
            partition=partition,
            learnable_scales=learnable_scales,
            dtype=dtype or gate.dtype,
            device=device,
        )
        with runtime.no_grad():
            model.shared_gate_proj.weight.copy_(gate[list(partition.shared_indices)])
            model.shared_up_proj.weight.copy_(up[list(partition.shared_indices)])
            model.shared_down_proj.weight.copy_(down[:, list(partition.shared_indices)])
            for expert, group in enumerate(partition.expert_indices):
                expert_gate: Any = model.expert_gate_proj[expert]
                expert_up: Any = model.expert_up_proj[expert]
                expert_down: Any = model.expert_down_proj[expert]
                expert_gate.weight.copy_(gate[list(group)])
                expert_up.weight.copy_(up[list(group)])
                expert_down.weight.copy_(down[:, list(group)])
            if router is not None:
                value = _as_tensor(router, dtype=model.router.weight.dtype, device=model.router.weight.device)
                if tuple(value.shape) == (hidden_size, routed_experts):
                    value = value.T
                if tuple(value.shape) != tuple(model.router.weight.shape):
                    raise ValueError("router shape mismatch")
                model.router.weight.copy_(value)
            else:
                model.router.weight.zero_()
        return model

    def _expert_output(self, x: Tensor, expert: int) -> Tensor:
        hidden = F.silu(self.expert_gate_proj[expert](x)) * self.expert_up_proj[expert](x)
        return self.expert_down_proj[expert](hidden)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_router: bool = False,
        return_contributions: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        runtime = _require_torch()
        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("input hidden dimension does not match MoE layer")
        original_shape = inputs.shape
        x = inputs.reshape(-1, self.hidden_size)
        shared = self.shared_down_proj(F.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
        logits = self.router(x)
        values, indices = runtime.topk(logits, self.top_k, dim=-1)
        weights = runtime.softmax(values, dim=-1)
        routed = runtime.zeros_like(shared)
        contributions: list[Tensor] = []
        for expert in range(self.routed_experts):
            contribution = self._expert_output(x, expert) * self.expert_scales[expert]
            if return_contributions:
                contributions.append(contribution.detach())
            selected = (indices == expert).to(contribution.dtype)
            coefficient = (weights * selected).sum(dim=-1, keepdim=True)
            routed = routed + contribution * coefficient
        result = (shared + routed).reshape(*original_shape[:-1], self.hidden_size)
        if not return_router:
            return result
        info = {
            "indices": indices.reshape(*original_shape[:-1], self.top_k),
            "weights": weights.reshape(*original_shape[:-1], self.top_k),
            "logits": logits.reshape(*original_shape[:-1], self.routed_experts),
        }
        if return_contributions:
            info["contributions"] = runtime.stack(contributions, dim=1).reshape(
                *original_shape[:-1], self.routed_experts, self.hidden_size
            )
        return result, info

    def state_dict_inventory(self) -> dict[str, Any]:
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in self.state_dict().items()}

    def partition_hash(self) -> str:
        encoded = json.dumps(self.partition.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save_pretrained(self, destination: str | Path) -> Path:
        runtime = _require_torch()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required to save the trainable MoE") from exc
        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(state, str(destination / "model.safetensors"))
        config = {
            "architectures": [self.architecture],
            "model_type": "qwen3_5_text",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "routed_experts": self.routed_experts,
            "expert_intermediate_size": self.expert_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
            "top_k": self.top_k,
            "learnable_scales": self.learnable_scales,
            "partition": self.partition.as_dict(),
            "partition_hash": self.partition_hash(),
            "state_dict_inventory": self.state_dict_inventory(),
            "torch_version": runtime.__version__,
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_pretrained(cls, source: str | Path, *, strict: bool = True, device: Any | None = None) -> TorchQwen35SwiGLUMoE:
        _require_torch()
        source = Path(source)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        partition = _partition_from_payload(config["partition"])
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required to reload the trainable MoE") from exc
        state = load_file(str(source / "model.safetensors"), device="cpu")
        expected = set(config.get("state_dict_inventory", {}))
        if strict and set(state) != expected:
            raise ValueError(f"strict state-dict mismatch: missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}")
        model = cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            routed_experts=int(config["routed_experts"]),
            expert_intermediate_size=int(config["expert_intermediate_size"]),
            shared_intermediate_size=int(config["shared_intermediate_size"]),
            top_k=int(config["top_k"]),
            partition=partition,
            learnable_scales=bool(config.get("learnable_scales", False)),
            dtype=next(iter(state.values())).dtype,
            device=device,
        )
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise ValueError(f"strict state-dict mismatch: missing={missing}, unexpected={unexpected}")
        return model


__all__ = ["TorchQwen35SwiGLUMoE"]
