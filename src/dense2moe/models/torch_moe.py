"""Trainable PyTorch SwiGLU MoE blocks and reloadable layer artifacts.

The NumPy target in :mod:`dense2moe.models.qwen_moe` is intentionally kept as
the dependency-light numerical oracle.  This module is the trainable path used
by one-layer distillation and by the tiny full-model save/reload spike.  It
uses the same Hugging Face weight orientation (``[out, in]``), preserves the
selected dense partition, and supports normalized-simplex or independent-
positive top-k routing without token dropping.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..partition import PartitionPlan, partition_indices
from ..science.hard_tail import (
    MAX_FALLBACK_RATE,
    summarize_active_widths,
    validate_fallback_rate,
)

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


class LowRankSiLURouter(nn.Module):
    """Small nonlinear selector used for the bounded router A/B test."""

    def __init__(self, hidden_size: int, router_hidden_size: int, routed_experts: int, *, dtype: Any | None = None, device: Any | None = None) -> None:
        runtime = _require_torch()
        if router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive")
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, router_hidden_size, bias=True, **kwargs)
        self.out_proj = nn.Linear(router_hidden_size, routed_experts, bias=False, **kwargs)
        with runtime.no_grad():
            self.out_proj.weight.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.out_proj(F.silu(self.in_proj(inputs)))


class SharedOutputFeatureRouter(nn.Module):
    """Small selector that sees ``x`` and the already-computed shared output.

    The shared SwiGLU branch is part of every sparse FFN invocation, so using
    its output here adds only a compact router projection rather than a second
    expert-sized feed-forward network.  This is intentionally an opt-in A/B
    experiment; the default linear/low-rank router state is unchanged.
    """

    def __init__(self, hidden_size: int, router_hidden_size: int, routed_experts: int, *, dtype: Any | None = None, device: Any | None = None) -> None:
        runtime = _require_torch()
        if router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive")
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        super().__init__()
        self.in_proj = nn.Linear(hidden_size * 2, router_hidden_size, bias=True, **kwargs)
        self.out_proj = nn.Linear(router_hidden_size, routed_experts, bias=False, **kwargs)
        with runtime.no_grad():
            self.out_proj.weight.zero_()

    def forward(self, inputs: Tensor, shared_output: Tensor) -> Tensor:
        if inputs.shape[:-1] != shared_output.shape[:-1]:
            raise ValueError("router input and shared output shapes do not match")
        return self.out_proj(F.silu(self.in_proj(torch.cat((inputs, shared_output), dim=-1))))


class LowRankResidualCorrector(nn.Module):
    """A real, checkpointed low-rank residual FFN correction branch.

    The branch is intentionally linear: its intermediate width is the
    conservative active-width budget, while both projections are ordinary
    trainable parameters.  The output projection starts at zero so adding the
    optional seam does not perturb a converted dense checkpoint until the
    branch is trained or explicitly initialized by an experiment.
    """

    def __init__(self, hidden_size: int, residual_width: int, *, dtype: Any | None = None, device: Any | None = None) -> None:
        runtime = _require_torch()
        if residual_width <= 0:
            raise ValueError("residual_width must be positive")
        kwargs: dict[str, Any] = {"bias": False}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, residual_width, **kwargs)
        self.output_proj = nn.Linear(residual_width, hidden_size, **kwargs)
        with runtime.no_grad():
            self.output_proj.weight.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_proj(self.input_proj(inputs))


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
    """Trainable shared+routed SwiGLU layer with explicit routing semantics.

    ``from_dense`` initializes every routed and shared parameter from one
    immutable dense layer according to the supplied :class:`PartitionPlan`.
    Ordinary forward uses token-bucket dispatch: each routed expert sees only
    the rows selected for it and the result is scatter-added back into the
    flattened token output.  The explicit ``return_contributions=True`` path
    remains an all-expert diagnostic mode because oracle refinement needs the
    complete contribution tensor; its telemetry marks that dense fallback.
    """

    architecture = "qwen3_5_text_torch_swiglu_moe_v1"
    ROUTING_MODES: ClassVar[set[str]] = {"normalized_softmax", "independent_positive"}

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        routed_experts: int,
        expert_intermediate_size: int,
        shared_intermediate_size: int,
        top_k: int = 2,
        routing_mode: str = "normalized_softmax",
        router_hidden_size: int | None = None,
        router_feature_mode: str = "none",
        partition: PartitionPlan | None = None,
        learnable_scales: bool = False,
        residual_intermediate_size: int = 0,
        residual_scope: str = "static",
        fallback_mode: str = "none",
        fallback_rate_budget: float = MAX_FALLBACK_RATE,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> None:
        runtime = _require_torch()
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0 or routed_experts <= 0:
            raise ValueError("MoE dimensions must be positive")
        if top_k <= 0 or top_k > routed_experts:
            raise ValueError("top_k must be in [1, routed_experts]")
        if routing_mode not in self.ROUTING_MODES:
            raise ValueError("routing_mode must be normalized_softmax or independent_positive")
        if router_hidden_size is not None and router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive when provided")
        if router_feature_mode not in {"none", "shared_output"}:
            raise ValueError("router_feature_mode must be none or shared_output")
        if router_feature_mode == "shared_output" and router_hidden_size is None:
            raise ValueError("router_hidden_size is required for shared_output router features")
        if residual_intermediate_size < 0:
            raise ValueError("residual_intermediate_size must be non-negative")
        if residual_scope not in {"static", "selected"}:
            raise ValueError("residual_scope must be static or selected")
        if fallback_mode not in {"none", "top8", "top10", "residual"}:
            raise ValueError("fallback_mode must be none, top8, top10, or residual")
        if fallback_mode == "top8" and routed_experts < 8:
            raise ValueError("top8 fallback requires at least eight routed experts")
        if fallback_mode == "top10" and routed_experts < 10:
            raise ValueError("top10 fallback requires at least ten routed experts")
        if fallback_mode == "residual" and (residual_scope != "selected" or residual_intermediate_size <= 0):
            raise ValueError("residual fallback requires a positive selected residual branch")
        if residual_scope == "selected" and fallback_mode != "residual":
            raise ValueError("selected residual scope is only valid for residual fallback")
        validate_fallback_rate(float(fallback_rate_budget), maximum=MAX_FALLBACK_RATE)
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
        self.routing_mode = str(routing_mode)
        self.router_hidden_size = int(router_hidden_size) if router_hidden_size is not None else None
        self.router_feature_mode = str(router_feature_mode)
        self.residual_intermediate_size = int(residual_intermediate_size)
        self.residual_scope = str(residual_scope)
        self.fallback_mode = str(fallback_mode)
        self.fallback_rate_budget = float(fallback_rate_budget)
        self.partition = partition
        linear_kwargs: dict[str, Any] = {"bias": False}
        if dtype is not None:
            linear_kwargs["dtype"] = dtype
        if device is not None:
            linear_kwargs["device"] = device
        self.shared_gate_proj = nn.Linear(hidden_size, shared_intermediate_size, **linear_kwargs)
        self.shared_up_proj = nn.Linear(hidden_size, shared_intermediate_size, **linear_kwargs)
        self.shared_down_proj = nn.Linear(shared_intermediate_size, hidden_size, **linear_kwargs)
        if self.residual_intermediate_size > 0:
            self.residual_corrector: LowRankResidualCorrector | None = LowRankResidualCorrector(
                hidden_size,
                self.residual_intermediate_size,
                dtype=dtype,
                device=device,
            )
        else:
            self.residual_corrector = None
        self.expert_gate_proj = nn.ModuleList(
            [nn.Linear(hidden_size, expert_intermediate_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        self.expert_up_proj = nn.ModuleList(
            [nn.Linear(hidden_size, expert_intermediate_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        self.expert_down_proj = nn.ModuleList(
            [nn.Linear(expert_intermediate_size, hidden_size, **linear_kwargs) for _ in range(routed_experts)]
        )
        if router_feature_mode == "shared_output":
            self.router = SharedOutputFeatureRouter(
                hidden_size,
                int(router_hidden_size),
                routed_experts,
                dtype=dtype,
                device=device,
            )
        elif router_hidden_size is None:
            self.router = nn.Linear(hidden_size, routed_experts, bias=False, **{key: value for key, value in linear_kwargs.items() if key != "bias"})
        else:
            self.router = LowRankSiLURouter(
                hidden_size,
                router_hidden_size,
                routed_experts,
                dtype=dtype,
                device=device,
            )
        if self.routing_mode == "independent_positive":
            # A bias gives the positive router a stable amplitude-one starting
            # point while retaining an unconstrained, token-dependent scale.
            amplitude_kwargs = {key: value for key, value in linear_kwargs.items() if key != "bias"}
            amplitude_kwargs["bias"] = True
            self.amplitude_router = nn.Linear(hidden_size, routed_experts, **amplitude_kwargs)
            inverse_softplus_one = float(runtime.log(runtime.expm1(runtime.tensor(1.0))).item())
            with runtime.no_grad():
                self.amplitude_router.weight.zero_()
                self.amplitude_router.bias.fill_(inverse_softplus_one)
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
        routing_mode: str = "normalized_softmax",
        router_hidden_size: int | None = None,
        router_feature_mode: str = "none",
        partition: PartitionPlan | None = None,
        router: Any | None = None,
        amplitude_router: Any | None = None,
        learnable_scales: bool = False,
        residual_intermediate_size: int = 0,
        residual_scope: str = "static",
        fallback_mode: str = "none",
        fallback_rate_budget: float = MAX_FALLBACK_RATE,
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
            routing_mode=routing_mode,
            router_hidden_size=router_hidden_size,
            router_feature_mode=router_feature_mode,
            partition=partition,
            learnable_scales=learnable_scales,
            residual_intermediate_size=residual_intermediate_size,
            residual_scope=residual_scope,
            fallback_mode=fallback_mode,
            fallback_rate_budget=fallback_rate_budget,
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
                if router_hidden_size is not None or router_feature_mode != "none":
                    raise ValueError("direct router tensor initialization is only supported for a linear router")
                value = _as_tensor(router, dtype=model.router.weight.dtype, device=model.router.weight.device)
                if tuple(value.shape) == (hidden_size, routed_experts):
                    value = value.T
                if tuple(value.shape) != tuple(model.router.weight.shape):
                    raise ValueError("router shape mismatch")
                model.router.weight.copy_(value)
            else:
                if router_feature_mode == "shared_output":
                    model.router.in_proj.weight.zero_()
                    model.router.in_proj.bias.zero_()
                    model.router.out_proj.weight.zero_()
                elif router_hidden_size is None:
                    model.router.weight.zero_()
                else:
                    model.router.out_proj.weight.zero_()
            if routing_mode == "independent_positive" and amplitude_router is not None:
                value = _as_tensor(amplitude_router, dtype=model.amplitude_router.weight.dtype, device=model.amplitude_router.weight.device)
                if tuple(value.shape) == (hidden_size, routed_experts):
                    value = value.T
                if tuple(value.shape) != tuple(model.amplitude_router.weight.shape):
                    raise ValueError("amplitude router weight shape mismatch")
                model.amplitude_router.weight.copy_(value)
        return model

    def _expert_output(self, x: Tensor, expert: int) -> Tensor:
        hidden = F.silu(self.expert_gate_proj[expert](x)) * self.expert_up_proj[expert](x)
        return self.expert_down_proj[expert](hidden)

    def _routed_dense(
        self,
        x: Tensor,
        indices: Tensor,
        weights: Tensor,
        *,
        return_contributions: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        """Reference all-expert route used only by contribution diagnostics."""

        runtime = _require_torch()
        routed = runtime.zeros((x.shape[0], self.hidden_size), dtype=x.dtype, device=x.device)
        contributions: list[Tensor] = []
        for expert in range(self.routed_experts):
            contribution = self._expert_output(x, expert) * self.expert_scales[expert]
            if return_contributions:
                contributions.append(contribution.detach())
            selected = (indices == expert).to(contribution.dtype)
            coefficient = (weights * selected).sum(dim=-1, keepdim=True)
            routed = routed + contribution * coefficient
        return routed, contributions

    def _routed_sparse(
        self,
        x: Tensor,
        indices: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, list[int]]:
        """Dispatch only selected token rows to each routed expert.

        ``index_add`` is out-of-place so gradients flow through the selected
        expert outputs and coefficients while token indices remain a pure
        routing decision.  Empty expert buckets are skipped entirely.
        """

        runtime = _require_torch()
        routed = runtime.zeros((x.shape[0], self.hidden_size), dtype=x.dtype, device=x.device)
        expert_token_counts: list[int] = []
        for expert in range(self.routed_experts):
            selected_mask = ((indices == expert) & (weights != 0)).any(dim=-1)
            token_ids = runtime.nonzero(selected_mask, as_tuple=False).flatten()
            count = int(token_ids.numel())
            expert_token_counts.append(count)
            if count == 0:
                continue
            expert_inputs = x.index_select(0, token_ids)
            contribution = self._expert_output(expert_inputs, expert) * self.expert_scales[expert]
            selected_slots = (indices.index_select(0, token_ids) == expert).to(weights.dtype)
            coefficient = (weights.index_select(0, token_ids) * selected_slots).sum(dim=-1, keepdim=True)
            routed = routed.index_add(0, token_ids, contribution * coefficient)
        return routed, expert_token_counts

    @property
    def router_parameter_count(self) -> int:
        """Number of selector/amplitude parameters added by the router."""

        count = sum(int(parameter.numel()) for parameter in self.router.parameters())
        if self.routing_mode == "independent_positive":
            count += sum(int(parameter.numel()) for parameter in self.amplitude_router.parameters())
        return count

    @property
    def router_parameter_fraction(self) -> float:
        """Router parameters as a fraction of the active FFN projection size."""

        active_width = self.static_active_intermediate_width
        active_parameters = max(3 * self.hidden_size * active_width, 1)
        return float(self.router_parameter_count / active_parameters)

    @property
    def fallback_top_k(self) -> int:
        if self.fallback_mode == "top8":
            return 8
        if self.fallback_mode == "top10":
            return 10
        return self.top_k

    @property
    def static_active_intermediate_width(self) -> int:
        residual = self.residual_intermediate_size if self.residual_scope == "static" else 0
        return int(self.shared_intermediate_size + self.top_k * self.expert_intermediate_size + residual)

    @property
    def fallback_active_intermediate_width(self) -> int:
        residual = self.residual_intermediate_size if (
            self.residual_scope == "static" or self.fallback_mode == "residual"
        ) else 0
        return int(self.shared_intermediate_size + self.fallback_top_k * self.expert_intermediate_size + residual)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_router: bool = False,
        return_contributions: bool = False,
        fallback_mask: Tensor | None = None,
        fallback_mode: str | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        runtime = _require_torch()
        if inputs.shape[-1] != self.hidden_size:
            raise ValueError("input hidden dimension does not match MoE layer")
        if not bool(runtime.isfinite(inputs).all().item()):
            raise ValueError("inputs must be finite")
        original_shape = inputs.shape
        x = inputs.reshape(-1, self.hidden_size)
        active_fallback_mode = self.fallback_mode if fallback_mode is None else str(fallback_mode)
        if active_fallback_mode not in {"none", "top8", "top10", "residual"}:
            raise ValueError("fallback_mode must be none, top8, top10, or residual")
        if active_fallback_mode == "top8" and self.routed_experts < 8:
            raise ValueError("top8 fallback requires at least eight routed experts")
        if active_fallback_mode == "top10" and self.routed_experts < 10:
            raise ValueError("top10 fallback requires at least ten routed experts")
        if active_fallback_mode == "residual" and self.residual_corrector is None:
            raise ValueError("residual fallback requires a residual corrector")
        if fallback_mask is None:
            mask = runtime.zeros((x.shape[0],), dtype=runtime.bool, device=x.device)
        else:
            mask = runtime.as_tensor(fallback_mask, device=x.device, dtype=runtime.bool)
            if int(mask.numel()) != int(x.shape[0]):
                raise ValueError("fallback_mask must cover exactly one boolean per input token")
            mask = mask.reshape(-1)
        fallback_tokens = int(mask.sum().item())
        fallback_rate = float(fallback_tokens / max(int(x.shape[0]), 1))
        if fallback_tokens and active_fallback_mode == "none":
            raise ValueError("fallback_mask is non-empty while fallback_mode is none")
        validate_fallback_rate(fallback_rate, maximum=self.fallback_rate_budget)
        shared = self.shared_down_proj(F.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
        if self.router_feature_mode == "shared_output":
            logits = self.router(x, shared)
        else:
            logits = self.router(x)
        requested_top_k = self.top_k
        if active_fallback_mode in {"top8", "top10"} and fallback_tokens:
            requested_top_k = self.fallback_top_k if active_fallback_mode == self.fallback_mode else (8 if active_fallback_mode == "top8" else 10)
        values, indices = runtime.topk(logits, requested_top_k, dim=-1)
        amplitude_logits = None
        if self.routing_mode == "normalized_softmax":
            expanded_weights = runtime.softmax(values, dim=-1)
            if requested_top_k > self.top_k and fallback_tokens:
                base_weights = runtime.softmax(values[:, : self.top_k], dim=-1)
                weights = runtime.zeros_like(expanded_weights)
                weights[:, : self.top_k] = base_weights
                weights = runtime.where(mask[:, None], expanded_weights, weights)
            else:
                weights = expanded_weights
        else:
            amplitude_logits = self.amplitude_router(x)
            weights = F.softplus(runtime.gather(amplitude_logits, dim=-1, index=indices))
            if requested_top_k > self.top_k and fallback_tokens:
                base_weights = F.softplus(runtime.gather(amplitude_logits, dim=-1, index=indices[:, : self.top_k]))
                padded = runtime.zeros_like(weights)
                padded[:, : self.top_k] = base_weights
                weights = runtime.where(mask[:, None], weights, padded)
        if return_contributions:
            routed, contributions = self._routed_dense(x, indices, weights, return_contributions=True)
            expert_token_counts = [int(x.shape[0])] * self.routed_experts
            dispatch_mode = "dense_contributions"
            dense_fallback_used = True
        else:
            routed, expert_token_counts = self._routed_sparse(x, indices, weights)
            contributions = []
            dispatch_mode = "sparse_token_dispatch"
            dense_fallback_used = False
        residual_output = runtime.zeros_like(shared)
        if self.residual_corrector is not None:
            if self.residual_scope == "static":
                residual_output = self.residual_corrector(x)
            elif fallback_tokens:
                token_ids = runtime.nonzero(mask, as_tuple=False).flatten()
                residual_values = self.residual_corrector(x.index_select(0, token_ids))
                residual_output = residual_output.index_add(0, token_ids, residual_values)
        result = (shared + routed + residual_output).reshape(*original_shape[:-1], self.hidden_size)
        if not bool(runtime.isfinite(result).all().item()):
            raise ValueError("outputs must be finite")
        static_residual_width = self.residual_intermediate_size if self.residual_scope == "static" else 0
        fallback_residual_width = self.residual_intermediate_size if (
            self.residual_scope == "static" or active_fallback_mode == "residual"
        ) else 0
        static_width = int(self.shared_intermediate_size + self.top_k * self.expert_intermediate_size + static_residual_width)
        fallback_width = int(self.shared_intermediate_size + requested_top_k * self.expert_intermediate_size + fallback_residual_width)
        width_tensor = runtime.where(
            mask,
            runtime.full((x.shape[0],), fallback_width, dtype=runtime.int64, device=x.device),
            runtime.full((x.shape[0],), static_width, dtype=runtime.int64, device=x.device),
        )
        width_summary = summarize_active_widths(width_tensor.detach().cpu().tolist())
        if not return_router:
            return result
        selected_dispatches = int(x.shape[0] * self.top_k + fallback_tokens * max(requested_top_k - self.top_k, 0))
        info = {
            "indices": indices.reshape(*original_shape[:-1], requested_top_k),
            "weights": weights.reshape(*original_shape[:-1], requested_top_k),
            "logits": logits.reshape(*original_shape[:-1], self.routed_experts),
            "routing_mode": self.routing_mode,
            "router_hidden_size": self.router_hidden_size,
            "router_architecture": (
                "shared_output_feature_low_rank_silu"
                if self.router_feature_mode == "shared_output"
                else "linear" if self.router_hidden_size is None else "low_rank_silu"
            ),
            "router_feature_mode": self.router_feature_mode,
            "router_parameter_count": self.router_parameter_count,
            "router_parameter_fraction": self.router_parameter_fraction,
            "dispatch_mode": dispatch_mode,
            "dense_fallback_used": dense_fallback_used,
            "dispatch_token_count": int(x.shape[0]),
            "expert_token_counts": expert_token_counts,
            "selected_dispatches": selected_dispatches,
            "nonempty_experts": int(sum(count > 0 for count in expert_token_counts)),
            "dense_intermediate_width": int(self.intermediate_size),
            # The public active-width field is the actual per-token width,
            # including the permanent residual branch.  Earlier revisions
            # reported only shared+routed width here and silently understated
            # the p16/top6 residual candidate by 256 units.
            "active_intermediate_width": static_width,
            "static_active_intermediate_width": static_width,
            "fallback_active_intermediate_width": fallback_width,
            "residual_intermediate_size": int(self.residual_intermediate_size),
            "residual_scope": self.residual_scope,
            "fallback_mode": active_fallback_mode,
            "fallback_token_count": fallback_tokens,
            "fallback_rate": fallback_rate,
            "residual_executed": bool(self.residual_corrector is not None and (
                self.residual_scope == "static" or fallback_tokens > 0
            )),
            "residual_parameter_count": int(
                sum(parameter.numel() for parameter in self.residual_corrector.parameters())
                if self.residual_corrector is not None else 0
            ),
            "active_intermediate_widths": width_tensor.detach().reshape(*original_shape[:-1]),
            "active_intermediate_width_mean": width_summary.mean,
            "active_intermediate_width_p50": width_summary.p50,
            "active_intermediate_width_p95": width_summary.p95,
            "active_intermediate_width_max": width_summary.maximum,
            "average_ffn_reduction": float(1.0 - width_summary.mean / max(float(self.intermediate_size), 1.0)),
            "dropped_token_count": 0,
            "invalid_token_count": 0,
            "non_finite_token_count": 0,
            "estimated_ffn_reduction": float(
                1.0
                - static_width
                / max(self.intermediate_size, 1)
            ),
        }
        if return_contributions:
            # The detached shared branch is used only for optional
            # train-only oracle-label diagnostics.  Keeping it out of the
            # ordinary router receipt avoids enlarging inference metadata.
            info["shared"] = shared.detach().reshape(*original_shape[:-1], self.hidden_size)
        if amplitude_logits is not None:
            info["amplitude_logits"] = amplitude_logits.reshape(*original_shape[:-1], self.routed_experts)
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
            "routing_mode": self.routing_mode,
            "router_hidden_size": self.router_hidden_size,
            "router_feature_mode": self.router_feature_mode,
            "router_architecture": (
                "shared_output_feature_low_rank_silu"
                if self.router_feature_mode == "shared_output"
                else "linear" if self.router_hidden_size is None else "low_rank_silu"
            ),
            "router_parameter_count": self.router_parameter_count,
            "router_parameter_fraction": self.router_parameter_fraction,
            "learnable_scales": self.learnable_scales,
            "residual_intermediate_size": self.residual_intermediate_size,
            "residual_scope": self.residual_scope,
            "residual_architecture": "low_rank_linear" if self.residual_corrector is not None else "none",
            "fallback_mode": self.fallback_mode,
            "fallback_rate_budget": self.fallback_rate_budget,
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
            routing_mode=str(config.get("routing_mode", "normalized_softmax")),
            router_hidden_size=(int(config["router_hidden_size"]) if config.get("router_hidden_size") is not None else None),
            router_feature_mode=str(config.get("router_feature_mode", "none")),
            partition=partition,
            learnable_scales=bool(config.get("learnable_scales", False)),
            residual_intermediate_size=int(config.get("residual_intermediate_size", 0)),
            residual_scope=str(config.get("residual_scope", "static")),
            fallback_mode=str(config.get("fallback_mode", "none")),
            fallback_rate_budget=float(config.get("fallback_rate_budget", MAX_FALLBACK_RATE)),
            dtype=next(iter(state.values())).dtype,
            device=device,
        )
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise ValueError(f"strict state-dict mismatch: missing={missing}, unexpected={unexpected}")
        return model


__all__ = [
    "LowRankResidualCorrector",
    "LowRankSiLURouter",
    "SharedOutputFeatureRouter",
    "TorchQwen35SwiGLUMoE",
]
