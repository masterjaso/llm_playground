"""Bounded hard-tail experiment contracts and compute accounting.

This module is deliberately independent of the PyTorch model so configuration
identity, arithmetic, predictor-feature policy, and percentile definitions can
be tested without a GPU.  It is the source of truth for the phase-01 gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DENSE_INTERMEDIATE_SIZE = 17_408
ROUTED_EXPERTS = 16
BASE_TOP_K = 6
MAX_FALLBACK_RATE = 0.30
MIN_AVERAGE_REDUCTION = 0.50
FALLBACK_MODES = frozenset({"none", "top8", "top10", "residual"})
FORBIDDEN_INFERENCE_FEATURE_TOKENS = frozenset(
    {
        "teacher",
        "teacher_ffn_output",
        "realized_reconstruction_error",
        "reconstruction_error",
        "target_norm",
        "posthoc_dense_output",
        "dense_moe_output_difference",
        "dense_teacher_difference",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def derive_expert_width(
    shared_width: int,
    *,
    dense_width: int = DENSE_INTERMEDIATE_SIZE,
    routed_experts: int = ROUTED_EXPERTS,
) -> int:
    """Derive an exact dense-capacity partition width for a shared path."""

    dense = int(dense_width)
    shared = int(shared_width)
    experts = int(routed_experts)
    if dense <= 0 or shared <= 0 or experts <= 0:
        raise ValueError("dense_width, shared_width, and routed_experts must be positive")
    remaining = dense - shared
    if remaining <= 0 or remaining % experts:
        raise ValueError("shared width does not produce an exact expert partition")
    return remaining // experts


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentiles require at least one value")
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class ActiveWidthSummary:
    """Nearest-rank active-width summary for one token population."""

    count: int
    mean: float
    p50: float
    p95: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "p50": float(self.p50),
            "p95": float(self.p95),
            "max": float(self.maximum),
        }


def summarize_active_widths(widths: Sequence[int | float]) -> ActiveWidthSummary:
    values = [float(value) for value in widths]
    if not values:
        raise ValueError("active-width summaries require at least one token")
    return ActiveWidthSummary(
        count=len(values),
        mean=sum(values) / len(values),
        p50=_nearest_rank(values, 0.50),
        p95=_nearest_rank(values, 0.95),
        maximum=max(values),
    )


def validate_fallback_rate(rate: float, *, maximum: float = MAX_FALLBACK_RATE) -> float:
    value = float(rate)
    if not math.isfinite(value) or value < 0.0 or value > float(maximum) + 1e-12:
        raise ValueError(f"fallback rate must be finite and in [0, {float(maximum):.3f}]")
    return value


def validate_inference_features(features: Mapping[str, Any] | Sequence[str]) -> tuple[str, ...]:
    """Reject features that require a dense teacher at deployment."""

    names = tuple(str(key) for key in (features.keys() if isinstance(features, Mapping) else features))
    leaked = []
    for name in names:
        normalized = name.strip().lower().replace("-", "_")
        if any(token in normalized for token in FORBIDDEN_INFERENCE_FEATURE_TOKENS):
            leaked.append(name)
    if leaked:
        raise ValueError("teacher-dependent inference features are forbidden: " + ", ".join(sorted(leaked)))
    return tuple(sorted(names))


@dataclass(frozen=True)
class HardTailConfig:
    """One frozen static/fallback geometry in the preregistered frontier."""

    shared_width: int
    expert_width: int
    residual_width: int = 0
    routed_experts: int = ROUTED_EXPERTS
    top_k: int = BASE_TOP_K
    dense_width: int = DENSE_INTERMEDIATE_SIZE
    fallback_mode: str = "none"
    residual_scope: str = "static"
    fallback_rate_budget: float = MAX_FALLBACK_RATE

    def __post_init__(self) -> None:
        if min(int(self.shared_width), int(self.expert_width), int(self.routed_experts), int(self.top_k)) <= 0:
            raise ValueError("hard-tail geometry dimensions must be positive")
        if int(self.residual_width) < 0:
            raise ValueError("residual width must be non-negative")
        if int(self.shared_width) + int(self.routed_experts) * int(self.expert_width) != int(self.dense_width):
            raise ValueError("shared plus all routed expert widths must equal dense width")
        if int(self.top_k) > int(self.routed_experts):
            raise ValueError("top_k cannot exceed routed experts")
        if self.fallback_mode not in FALLBACK_MODES:
            raise ValueError(f"unsupported fallback mode: {self.fallback_mode}")
        if self.fallback_mode == "top8" and int(self.routed_experts) < 8:
            raise ValueError("top8 fallback requires at least eight routed experts")
        if self.fallback_mode == "top10" and int(self.routed_experts) < 10:
            raise ValueError("top10 fallback requires at least ten routed experts")
        if self.residual_scope not in {"static", "selected"}:
            raise ValueError("residual_scope must be static or selected")
        if self.fallback_mode == "residual" and (self.residual_scope != "selected" or int(self.residual_width) <= 0):
            raise ValueError("residual fallback requires a positive selected residual branch")
        if self.residual_scope == "selected" and self.fallback_mode != "residual":
            raise ValueError("selected residual scope is only valid for residual fallback")
        if self.fallback_mode == "none" and float(self.fallback_rate_budget) != 0.0 and float(self.fallback_rate_budget) != MAX_FALLBACK_RATE:
            raise ValueError("no-fallback configurations cannot declare a custom fallback budget")
        validate_fallback_rate(float(self.fallback_rate_budget), maximum=MAX_FALLBACK_RATE)

    @property
    def fallback_top_k(self) -> int:
        if self.fallback_mode == "top8":
            return 8
        if self.fallback_mode == "top10":
            return 10
        return int(self.top_k)

    @property
    def static_active_width(self) -> int:
        residual = int(self.residual_width) if self.residual_scope == "static" else 0
        return int(self.shared_width) + int(self.top_k) * int(self.expert_width) + residual

    @property
    def fallback_active_width(self) -> int:
        residual = int(self.residual_width) if self.residual_scope == "static" or self.fallback_mode == "residual" else 0
        return int(self.shared_width) + int(self.fallback_top_k) * int(self.expert_width) + residual

    @property
    def static_reduction(self) -> float:
        return 1.0 - float(self.static_active_width) / float(self.dense_width)

    def mean_active_width(self, fallback_rate: float) -> float:
        rate = validate_fallback_rate(fallback_rate)
        return float(self.static_active_width) + rate * float(self.fallback_active_width - self.static_active_width)

    def average_reduction(self, fallback_rate: float) -> float:
        return 1.0 - self.mean_active_width(fallback_rate) / float(self.dense_width)

    def require_compute_budget(self, fallback_rate: float = 0.0) -> None:
        reduction = self.average_reduction(fallback_rate)
        if reduction + 1e-12 < MIN_AVERAGE_REDUCTION:
            raise ValueError(
                f"average active FFN reduction {reduction:.6f} is below {MIN_AVERAGE_REDUCTION:.6f}"
            )

    def widths_for_mask(self, fallback_mask: Sequence[bool]) -> tuple[int, ...]:
        mask = tuple(bool(value) for value in fallback_mask)
        rate = sum(mask) / len(mask) if mask else 0.0
        validate_fallback_rate(rate)
        return tuple(self.fallback_active_width if value else self.static_active_width for value in mask)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "shared_width": int(self.shared_width),
            "expert_width": int(self.expert_width),
            "residual_width": int(self.residual_width),
            "routed_experts": int(self.routed_experts),
            "top_k": int(self.top_k),
            "dense_width": int(self.dense_width),
            "fallback_mode": self.fallback_mode,
            "residual_scope": self.residual_scope,
            "fallback_rate_budget": float(self.fallback_rate_budget),
        }

    @property
    def configuration_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.identity_payload()).encode("utf-8")).hexdigest()[:16]
        return "ht-" + digest

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.identity_payload())
        result.update(
            {
                "configuration_id": self.configuration_id,
                "fallback_top_k": self.fallback_top_k,
                "static_active_width": self.static_active_width,
                "fallback_active_width": self.fallback_active_width,
                "static_reduction": self.static_reduction,
            }
        )
        return result


def make_static_config(
    shared_width: int,
    *,
    residual_width: int = 0,
    dense_width: int = DENSE_INTERMEDIATE_SIZE,
    routed_experts: int = ROUTED_EXPERTS,
    top_k: int = BASE_TOP_K,
    fallback_mode: str = "none",
    residual_scope: str = "static",
    fallback_rate_budget: float = MAX_FALLBACK_RATE,
) -> HardTailConfig:
    return HardTailConfig(
        shared_width=int(shared_width),
        expert_width=derive_expert_width(shared_width, dense_width=dense_width, routed_experts=routed_experts),
        residual_width=int(residual_width),
        dense_width=int(dense_width),
        routed_experts=int(routed_experts),
        top_k=int(top_k),
        fallback_mode=str(fallback_mode),
        residual_scope=str(residual_scope),
        fallback_rate_budget=float(fallback_rate_budget),
    )


__all__ = [
    "BASE_TOP_K",
    "DENSE_INTERMEDIATE_SIZE",
    "FALLBACK_MODES",
    "MAX_FALLBACK_RATE",
    "MIN_AVERAGE_REDUCTION",
    "ROUTED_EXPERTS",
    "ActiveWidthSummary",
    "HardTailConfig",
    "derive_expert_width",
    "make_static_config",
    "summarize_active_widths",
    "validate_fallback_rate",
    "validate_inference_features",
]
