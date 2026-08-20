"""Versioned Dense2MoE V2.3 parameter and active-compute accounting.

This module is deliberately independent of the experiment runners.  It gives
the V2.3 candidate-search code one small, typed surface for comparing the
always-on shared branch, selected routed experts, optional residual branch,
router, and scale/calibration overhead.  Product topology selection remains a
fail-closed operation delegated to :func:`dense2moe.config.active_topology_contract`.

The arithmetic uses the usual two-FLOPs-per-multiply-add convention for a
linear projection.  It is an estimate, not a hardware benchmark: activation,
top-k, and elementwise costs can be supplied explicitly when a candidate needs
to account for them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

from ..config import (
    ACTIVE_TOPOLOGY_IDS,
    FORBIDDEN_TOPOLOGY_IDS,
    MoEProfile,
    TopologyContract,
    active_topology_contract,
)

V23_ACCOUNTING_VERSION = "dense2moe-v2.3"
V23_SCHEMA_VERSION = 1
FLOP_MULTIPLIER = 2

TopologySelector: TypeAlias = str | Path | MoEProfile | TopologyContract


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    """Return an integer dimension, rejecting ambiguous/negative values."""

    converted = int(value)
    minimum = 0 if allow_zero else 1
    if converted < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return converted


def _canonical_architecture(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear",
        "linear_router": "linear",
        "low_rank_silu": "low_rank_silu",
        "lowrank_silu": "low_rank_silu",
        "low_rank": "low_rank_silu",
        "shared_output": "shared_output",
        "shared_output_feature": "shared_output",
        "shared_output_feature_low_rank_silu": "shared_output",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "router_architecture must be linear, low_rank_silu, or shared_output"
        ) from exc


def _resolve_contract(selector: TopologySelector) -> TopologyContract:
    """Resolve an active topology without weakening the product allow-list."""

    if isinstance(selector, TopologyContract):
        # ``active_topology_contract`` intentionally accepts selectors rather
        # than arbitrary contracts.  Resolve by ID so a forged contract cannot
        # silently introduce a V2.3 topology or alter active geometry.
        contract = active_topology_contract(selector.topology_id)
        if selector != contract:
            raise ValueError(
                f"topology contract {selector.topology_id!r} does not match the "
                "canonical active topology contract"
            )
        return contract
    return active_topology_contract(selector)


def _check_geometry(
    contract: TopologyContract,
    *,
    dense_intermediate_size: int | None,
    routed_experts: int | None,
    expert_intermediate_size: int | None,
    shared_intermediate_size: int | None,
    top_k: int | None,
) -> None:
    expected = {
        "dense_intermediate_size": contract.dense_intermediate_size,
        "routed_experts": contract.routed_experts,
        "expert_intermediate_size": contract.expert_intermediate_size,
        "shared_intermediate_size": contract.shared_intermediate_size,
        "top_k": contract.top_k,
    }
    supplied = {
        "dense_intermediate_size": dense_intermediate_size,
        "routed_experts": routed_experts,
        "expert_intermediate_size": expert_intermediate_size,
        "shared_intermediate_size": shared_intermediate_size,
        "top_k": top_k,
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, got {value!r}"
        for name, value in supplied.items()
        if value is not None and int(value) != expected[name]
    ]
    if mismatches:
        raise ValueError(
            f"V2.3 accounting geometry does not satisfy active topology "
            f"{contract.topology_id!r}: " + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class RouterAccountingSpec:
    """Router shape and routing-mode options.

    The defaults match :class:`TorchQwen35SwiGLUMoE`: a bias-free linear
    selector and normalized top-k weights.  Independent-positive routing adds
    the existing hidden-to-expert amplitude projection with an expert bias.
    """

    architecture: str = "linear"
    hidden_size: int | None = None
    routed_experts: int | None = None
    hidden_size_router: int | None = None
    router_hidden_size: int | None = None
    routing_mode: str = "normalized_softmax"
    selector_bias: bool = False
    amplitude_bias: bool = True
    include_amplitude_router: bool | None = None
    parameter_count: int | None = None
    flops_per_token: int | None = None

    def validate(self) -> None:
        _canonical_architecture(self.architecture)
        if self.hidden_size is not None:
            _positive_int(self.hidden_size, "router hidden_size")
        if self.routed_experts is not None:
            _positive_int(self.routed_experts, "router routed_experts")
        if self.hidden_size_router is not None:
            _positive_int(self.hidden_size_router, "router hidden_size_router")
        if self.router_hidden_size is not None:
            _positive_int(self.router_hidden_size, "router router_hidden_size")
        if (
            self.hidden_size_router is not None
            and self.router_hidden_size is not None
            and self.hidden_size_router != self.router_hidden_size
        ):
            raise ValueError("router hidden_size_router and router_hidden_size disagree")
        if self.routing_mode not in {"normalized_softmax", "independent_positive"}:
            raise ValueError(
                "routing_mode must be normalized_softmax or independent_positive"
            )
        if self.parameter_count is not None:
            _positive_int(self.parameter_count, "router parameter_count", allow_zero=True)
        if self.flops_per_token is not None:
            _positive_int(self.flops_per_token, "router flops_per_token", allow_zero=True)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualAccountingSpec:
    """Optional residual branch accounting.

    ``swiglu`` derives three projection matrices from the residual width.  A
    caller may instead provide explicit parameter/FLOP counts for a residual
    implementation whose tensors are not a standard SwiGLU branch.
    """

    intermediate_size: int = 0
    width: int | None = None
    mode: str = "swiglu"
    parameter_count: int | None = None
    flops_per_token: int | None = None

    def validate(self) -> None:
        _positive_int(self.intermediate_size, "residual intermediate_size", allow_zero=True)
        if self.width is not None:
            _positive_int(self.width, "residual width", allow_zero=True)
        if self.width is not None and self.intermediate_size not in {0, self.width}:
            raise ValueError("residual intermediate_size and width disagree")
        if self.mode not in {"swiglu", "linear"}:
            raise ValueError("residual mode must be swiglu or linear")
        if self.parameter_count is not None:
            _positive_int(self.parameter_count, "residual parameter_count", allow_zero=True)
        if self.flops_per_token is not None:
            _positive_int(self.flops_per_token, "residual flops_per_token", allow_zero=True)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleCalibrationSpec:
    """Scale and calibration state associated with routed contributions."""

    learnable_scales: bool = False
    parameter_count: int | None = None
    scale_parameters: int | None = None
    calibration_parameter_count: int = 0
    calibration_parameters: int | None = None
    apply_scales: bool = True
    flops_per_token: int | None = None
    calibration_flops_per_token: int = 0

    def validate(self) -> None:
        if self.parameter_count is not None:
            _positive_int(self.parameter_count, "scale parameter_count", allow_zero=True)
        if self.scale_parameters is not None:
            _positive_int(self.scale_parameters, "scale scale_parameters", allow_zero=True)
        if self.parameter_count is not None and self.scale_parameters is not None and self.parameter_count != self.scale_parameters:
            raise ValueError("scale parameter_count and scale_parameters disagree")
        _positive_int(
            self.calibration_parameter_count,
            "scale calibration_parameter_count",
            allow_zero=True,
        )
        if self.calibration_parameters is not None:
            _positive_int(self.calibration_parameters, "scale calibration_parameters", allow_zero=True)
        if (
            self.calibration_parameters is not None
            and self.calibration_parameter_count not in {0, self.calibration_parameters}
        ):
            raise ValueError(
                "scale calibration_parameter_count and calibration_parameters disagree"
            )
        if self.flops_per_token is not None:
            _positive_int(self.flops_per_token, "scale flops_per_token", allow_zero=True)
        _positive_int(
            self.calibration_flops_per_token,
            "scale calibration_flops_per_token",
            allow_zero=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V23AccountingSpec:
    """Inputs for :func:`account_v23`.

    Topology geometry is optional because it is derived from the active
    topology contract.  If a geometry value is supplied, it must exactly match
    that contract; this makes accidental p32/top4 or legacy-profile accounting
    fail closed before any arithmetic is emitted.
    """

    topology: TopologySelector = "p16/top4"
    hidden_size: int | None = None
    dense_intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    routed_experts: int | None = None
    expert_intermediate_size: int | None = None
    shared_intermediate_size: int | None = None
    top_k: int | None = None

    router_architecture: str = "linear"
    router_hidden_size: int | None = None
    routing_mode: str = "normalized_softmax"
    router_bias: bool = False
    amplitude_router: bool | None = None
    router_parameter_count: int | None = None
    router_flops_per_token: int | None = None

    residual_intermediate_size: int = 0
    residual_mode: str = "swiglu"
    residual_parameter_count: int | None = None
    residual_flops_per_token: int | None = None

    normalization_parameters: int = 0
    normalization_flops_per_token: int = 0

    learnable_scales: bool = False
    scale_parameter_count: int | None = None
    scale_calibration_parameters: int = 0
    apply_scales: bool = True
    scale_flops_per_token: int | None = None
    scale_calibration_flops_per_token: int = 0

    # ``calibration_*`` names are accepted separately because calibration can
    # include state beyond the routed scale vector.
    calibration_parameters: int | None = None
    calibration_flops_per_token: int = 0

    tokens: int = 1
    flop_multiplier: int = FLOP_MULTIPLIER

    # Structured specs are optional conveniences for callers that prefer a
    # nested JSON-shaped input.  When supplied, their component values are
    # used for that component; direct fields remain the flat JSON form.
    router: RouterAccountingSpec | None = field(default=None, repr=False, compare=False)
    residual: ResidualAccountingSpec | None = field(default=None, repr=False, compare=False)
    scale_calibration: ScaleCalibrationSpec | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def for_topology(cls, topology: TopologySelector, **kwargs: Any) -> V23AccountingSpec:
        return cls(topology=topology, **kwargs)

    def validate(self) -> TopologyContract:
        contract = _resolve_contract(self.topology)
        _check_geometry(
            contract,
            dense_intermediate_size=self.dense_intermediate_size,
            routed_experts=self.routed_experts,
            expert_intermediate_size=self.expert_intermediate_size,
            shared_intermediate_size=self.shared_intermediate_size,
            top_k=self.top_k,
        )
        hidden_size = self.hidden_size
        if hidden_size is not None:
            _positive_int(hidden_size, "hidden_size")
        layers = self.num_hidden_layers
        if layers is not None:
            _positive_int(layers, "num_hidden_layers")
        _positive_int(self.normalization_parameters, "normalization_parameters", allow_zero=True)
        _positive_int(
            self.normalization_flops_per_token,
            "normalization_flops_per_token",
            allow_zero=True,
        )
        _positive_int(self.tokens, "tokens")
        _positive_int(self.flop_multiplier, "flop_multiplier")
        direct_router = RouterAccountingSpec(
            architecture=self.router_architecture,
            hidden_size=hidden_size,
            routed_experts=contract.routed_experts,
            hidden_size_router=self.router_hidden_size,
            routing_mode=self.routing_mode,
            selector_bias=self.router_bias,
            include_amplitude_router=self.amplitude_router,
            parameter_count=self.router_parameter_count,
            flops_per_token=self.router_flops_per_token,
        )
        direct_router.validate()
        if self.router is not None:
            self.router.validate()
            if self.router.hidden_size is not None and hidden_size is not None and self.router.hidden_size != hidden_size:
                raise ValueError("router hidden_size does not match accounting hidden_size")
            if self.router.routed_experts is not None and self.router.routed_experts != contract.routed_experts:
                raise ValueError("router routed_experts does not match active topology")
        direct_residual = ResidualAccountingSpec(
            intermediate_size=self.residual_intermediate_size,
            mode=self.residual_mode,
            parameter_count=self.residual_parameter_count,
            flops_per_token=self.residual_flops_per_token,
        )
        direct_residual.validate()
        if self.residual is not None:
            self.residual.validate()
        direct_scale = ScaleCalibrationSpec(
            learnable_scales=self.learnable_scales,
            parameter_count=self.scale_parameter_count,
            calibration_parameter_count=self.scale_calibration_parameters,
            apply_scales=self.apply_scales,
            flops_per_token=self.scale_flops_per_token,
            calibration_flops_per_token=self.scale_calibration_flops_per_token,
        )
        direct_scale.validate()
        if self.scale_calibration is not None:
            self.scale_calibration.validate()
        if self.calibration_parameters is not None:
            _positive_int(self.calibration_parameters, "calibration_parameters", allow_zero=True)
        _positive_int(
            self.calibration_flops_per_token,
            "calibration_flops_per_token",
            allow_zero=True,
        )
        return contract

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly, stable representation of the input spec."""

        payload = asdict(self)
        # The selector may be a Path/profile/contract.  JSON should expose its
        # stable topology identity rather than a Python object representation.
        if isinstance(self.topology, (TopologyContract, MoEProfile)):
            payload["topology"] = self.topology.topology_id
        else:
            payload["topology"] = str(self.topology)
        return payload


@dataclass(frozen=True)
class ParameterAccounting:
    """Per-layer parameter counts for a validated V2.3 candidate."""

    dense_ffn_parameters: int
    shared_parameters: int
    routed_parameters: int
    active_routed_parameters: int
    residual_parameters: int
    normalization_parameters: int
    router_parameters: int
    scale_parameters: int
    calibration_parameters: int
    active_ffn_parameters: int
    active_parameters: int
    total_parameters: int

    @property
    def dense_parameters(self) -> int:
        return self.dense_ffn_parameters

    @property
    def routed_active_parameters(self) -> int:
        return self.active_routed_parameters

    @property
    def shared(self) -> int:
        return self.shared_parameters

    @property
    def routed(self) -> int:
        return self.routed_parameters

    @property
    def residual(self) -> int:
        return self.residual_parameters

    @property
    def router(self) -> int:
        return self.router_parameters

    @property
    def scale_calibration(self) -> int:
        return self.scale_parameters + self.calibration_parameters

    @property
    def active_parameter_count(self) -> int:
        return self.active_parameters

    @property
    def total_parameter_count(self) -> int:
        return self.total_parameters

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlopAccounting:
    """Per-token and requested-window active FLOP estimates.

    ``*_per_token`` values are for one layer.  ``*_flops`` values include the
    requested token count and layer count, which makes receipts directly
    comparable when branches use a common budget.
    """

    dense_ffn_flops_per_token: int
    shared_flops_per_token: int
    routed_flops_per_token: int
    residual_flops_per_token: int
    normalization_flops_per_token: int
    router_flops_per_token: int
    scale_flops_per_token: int
    calibration_flops_per_token: int
    active_ffn_flops_per_token: int
    active_flops_per_token: int
    dense_ffn_flops: int
    active_ffn_flops: int
    active_flops: int
    tokens: int
    num_hidden_layers: int

    @property
    def dense_flops(self) -> int:
        return self.dense_ffn_flops

    @property
    def active_flop_estimate(self) -> int:
        return self.active_flops

    @property
    def active_flops_per_layer(self) -> int:
        return self.active_flops_per_token

    @property
    def active_routed_flops_per_token(self) -> int:
        return self.routed_flops_per_token

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V23Accounting:
    """Complete deterministic accounting receipt for one V2.3 candidate."""

    version: str
    schema_version: int
    topology_id: str
    profile_name: str
    hidden_size: int
    dense_intermediate_size: int
    routed_experts: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    top_k: int
    active_intermediate_size: int
    active_width_reduction: float
    parameters: ParameterAccounting
    flops: FlopAccounting
    spec: V23AccountingSpec

    @classmethod
    def from_spec(cls, spec: V23AccountingSpec) -> V23Accounting:
        return account_v23(spec)

    @property
    def dense_ffn_parameters(self) -> int:
        return self.parameters.dense_ffn_parameters

    @property
    def shared_parameters(self) -> int:
        return self.parameters.shared_parameters

    @property
    def routed_parameters(self) -> int:
        return self.parameters.routed_parameters

    @property
    def active_routed_parameters(self) -> int:
        return self.parameters.active_routed_parameters

    @property
    def residual_parameters(self) -> int:
        return self.parameters.residual_parameters

    @property
    def router_parameters(self) -> int:
        return self.parameters.router_parameters

    @property
    def scale_parameters(self) -> int:
        return self.parameters.scale_parameters

    @property
    def calibration_parameters(self) -> int:
        return self.parameters.calibration_parameters

    @property
    def active_parameters(self) -> int:
        return self.parameters.active_parameters

    @property
    def total_parameters(self) -> int:
        return self.parameters.total_parameters

    @property
    def active_flops(self) -> int:
        return self.flops.active_flops

    @property
    def active_ffn_flops(self) -> int:
        return self.flops.active_ffn_flops

    @property
    def sparsity(self) -> float:
        return self.active_width_reduction

    @property
    def active_flop_reduction(self) -> float:
        return 1.0 - self.flops.active_flops / max(self.flops.dense_ffn_flops, 1)

    @property
    def active_ffn_flop_reduction(self) -> float:
        return 1.0 - self.flops.active_ffn_flops / max(self.flops.dense_ffn_flops, 1)

    @property
    def active_parameters_total(self) -> int:
        return self.parameters.active_parameters * self.flops.num_hidden_layers

    @property
    def total_parameters_total(self) -> int:
        return self.parameters.total_parameters * self.flops.num_hidden_layers

    @property
    def model_parameter_count(self) -> int:
        return self.total_parameters_total

    def as_dict(self) -> dict[str, Any]:
        """Return only JSON primitives with stable nested field names."""

        return {
            "version": self.version,
            "schema_version": self.schema_version,
            "topology": {
                "id": self.topology_id,
                "profile_name": self.profile_name,
                "hidden_size": self.hidden_size,
                "dense_intermediate_size": self.dense_intermediate_size,
                "routed_experts": self.routed_experts,
                "expert_intermediate_size": self.expert_intermediate_size,
                "shared_intermediate_size": self.shared_intermediate_size,
                "top_k": self.top_k,
                "active_intermediate_size": self.active_intermediate_size,
                "active_width_reduction": self.active_width_reduction,
            },
            "active_flop_reduction": self.active_flop_reduction,
            "active_ffn_flop_reduction": self.active_ffn_flop_reduction,
            "active_parameters": self.active_parameters,
            "active_parameters_total": self.active_parameters_total,
            "total_parameters": self.total_parameters,
            "total_parameters_total": self.total_parameters_total,
            "parameters": self.parameters.as_dict(),
            "flops": self.flops.as_dict(),
            "spec": self.spec.as_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    def as_json(self, *, indent: int | None = 2) -> str:
        return self.to_json(indent=indent)


_OVERRIDE_ALIASES = {
    "residual_width": "residual_intermediate_size",
    "residual_parameters": "residual_parameter_count",
    "router_hidden": "router_hidden_size",
    "router_parameters": "router_parameter_count",
    "scale_parameters": "scale_parameter_count",
    "scale_calibration_parameter_count": "scale_calibration_parameters",
    "calibration_parameter_count": "calibration_parameters",
    "residual_flops": "residual_flops_per_token",
    "router_flops": "router_flops_per_token",
    "scale_flops": "scale_flops_per_token",
    "scale_calibration_flops": "scale_calibration_flops_per_token",
    "calibration_flops": "calibration_flops_per_token",
    "normalization_parameter_count": "normalization_parameters",
    "normalization_flops": "normalization_flops_per_token",
    "layers": "num_hidden_layers",
}


def _normalise_overrides(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(kwargs)
    for alias, canonical in _OVERRIDE_ALIASES.items():
        if alias not in values:
            continue
        if canonical in values and values[canonical] != values[alias]:
            raise TypeError(f"{alias} and {canonical} overrides disagree")
        values[canonical] = values.pop(alias)
    return values


def _coerce_spec(
    spec: V23AccountingSpec | TopologySelector | None,
    *,
    topology: TopologySelector | None,
    kwargs: Mapping[str, Any],
) -> V23AccountingSpec:
    if isinstance(spec, V23AccountingSpec):
        if topology is not None or kwargs:
            raise TypeError("cannot combine a V23AccountingSpec with topology or keyword overrides")
        return spec
    selected = topology if topology is not None else spec
    if selected is None:
        selected = "p16/top4"
    return V23AccountingSpec(topology=selected, **_normalise_overrides(kwargs))


def _router_counts(
    spec: V23AccountingSpec,
    *,
    hidden_size: int,
    routed_experts: int,
    top_k: int,
) -> tuple[int, int]:
    """Return router parameters and per-token FLOPs for one layer."""

    nested = spec.router
    architecture = _canonical_architecture(
        nested.architecture if nested is not None else spec.router_architecture
    )
    router_hidden_size = (
        (
            nested.router_hidden_size
            if nested.router_hidden_size is not None
            else nested.hidden_size_router
        )
        if nested is not None
        else spec.router_hidden_size
    )
    routing_mode = nested.routing_mode if nested is not None else spec.routing_mode
    selector_bias = nested.selector_bias if nested is not None else spec.router_bias
    amplitude_router = (
        nested.include_amplitude_router
        if nested is not None and nested.include_amplitude_router is not None
        else spec.amplitude_router
    )
    if amplitude_router is None:
        amplitude_router = routing_mode == "independent_positive"
    if nested is not None and nested.parameter_count is not None:
        parameters = int(nested.parameter_count)
    elif spec.router_parameter_count is not None:
        parameters = int(spec.router_parameter_count)
    elif architecture == "linear":
        parameters = hidden_size * routed_experts + (routed_experts if selector_bias else 0)
    else:
        if router_hidden_size is None:
            raise ValueError("router_hidden_size is required for a non-linear router")
        input_width = hidden_size * 2 if architecture == "shared_output" else hidden_size
        # The trainable router has a bias on its input projection and no bias
        # on its expert projection.
        parameters = input_width * router_hidden_size + router_hidden_size
        parameters += router_hidden_size * routed_experts
    if amplitude_router:
        parameters += hidden_size * routed_experts
        if (nested.amplitude_bias if nested is not None else True):
            parameters += routed_experts

    if nested is not None and nested.flops_per_token is not None:
        flops = int(nested.flops_per_token)
    elif spec.router_flops_per_token is not None:
        flops = int(spec.router_flops_per_token)
    elif architecture == "linear":
        flops = FLOP_MULTIPLIER * hidden_size * routed_experts
    else:
        if router_hidden_size is None:
            raise ValueError("router_hidden_size is required for a non-linear router")
        input_width = hidden_size * 2 if architecture == "shared_output" else hidden_size
        flops = FLOP_MULTIPLIER * (input_width * router_hidden_size + router_hidden_size * routed_experts)
        # One inexpensive activation estimate per hidden unit keeps this
        # monotonic without pretending to model a particular kernel.
        flops += router_hidden_size
    if amplitude_router:
        flops += FLOP_MULTIPLIER * hidden_size * routed_experts
        # Positive amplitude applies softplus to each expert logit.  Counting
        # one operation per expert is intentionally conservative and stable.
        flops += routed_experts
    # Top-k selection and normalisation are small but active router work.
    flops += routed_experts + top_k
    return parameters, flops


def _residual_counts(
    spec: V23AccountingSpec,
    *,
    hidden_size: int,
) -> tuple[int, int]:
    nested = spec.residual
    width = (
        (nested.width if nested.width is not None else nested.intermediate_size)
        if nested is not None
        else spec.residual_intermediate_size
    )
    mode = nested.mode if nested is not None else spec.residual_mode
    explicit_parameters = (
        nested.parameter_count
        if nested is not None and nested.parameter_count is not None
        else spec.residual_parameter_count
    )
    explicit_flops = (
        nested.flops_per_token
        if nested is not None and nested.flops_per_token is not None
        else spec.residual_flops_per_token
    )
    if explicit_parameters is not None:
        parameters = int(explicit_parameters)
    elif mode == "linear":
        parameters = hidden_size * int(width)
    else:
        parameters = 3 * hidden_size * int(width)
    if explicit_flops is not None:
        flops = int(explicit_flops)
    elif mode == "linear":
        flops = FLOP_MULTIPLIER * parameters
    else:
        flops = FLOP_MULTIPLIER * parameters
    return parameters, flops


def _scale_counts(
    spec: V23AccountingSpec,
    *,
    routed_experts: int,
    hidden_size: int,
    top_k: int,
) -> tuple[int, int, int, int]:
    nested = spec.scale_calibration
    learnable = nested.learnable_scales if nested is not None else spec.learnable_scales
    explicit_parameters = (
        nested.parameter_count
        if nested is not None and nested.parameter_count is not None
        else spec.scale_parameter_count
    )
    if nested is not None and nested.scale_parameters is not None:
        explicit_parameters = nested.scale_parameters
    if explicit_parameters is not None:
        scale_parameters = int(explicit_parameters)
    else:
        # The model keeps one routed scale per expert; only learnable scales
        # contribute parameters, while the multiplication remains active for
        # either a learned or fixed scale vector.
        scale_parameters = routed_experts if learnable else 0
    calibration_parameters = (
        (
            nested.calibration_parameters
            if nested.calibration_parameters is not None
            else nested.calibration_parameter_count
        )
        if nested is not None
        else spec.scale_calibration_parameters
    )
    if spec.calibration_parameters is not None:
        calibration_parameters = int(spec.calibration_parameters)
    apply_scales = nested.apply_scales if nested is not None else spec.apply_scales
    explicit_scale_flops = (
        nested.flops_per_token
        if nested is not None and nested.flops_per_token is not None
        else spec.scale_flops_per_token
    )
    if explicit_scale_flops is not None:
        scale_flops = int(explicit_scale_flops)
    elif apply_scales:
        scale_flops = top_k * hidden_size
    else:
        scale_flops = 0
    calibration_flops = (
        nested.calibration_flops_per_token
        if nested is not None
        else spec.scale_calibration_flops_per_token
    )
    calibration_flops += int(spec.calibration_flops_per_token)
    return int(scale_parameters), int(calibration_parameters), int(scale_flops), int(calibration_flops)


def account_v23(
    spec: V23AccountingSpec | TopologySelector | None = None,
    *,
    topology: TopologySelector | None = None,
    **overrides: Any,
) -> V23Accounting:
    """Build a deterministic V2.3 accounting receipt.

    ``spec`` may be a :class:`V23AccountingSpec` or a topology selector.  For
    convenience, callers may pass the spec fields directly as keyword
    overrides, for example ``account_v23("p16/top4", learnable_scales=True)``.
    """

    selected = _coerce_spec(spec, topology=topology, kwargs=_normalise_overrides(overrides))
    contract = selected.validate()
    profile = selected.topology if isinstance(selected.topology, MoEProfile) else None
    hidden_size = int(selected.hidden_size or (profile.hidden_size if profile else 5120))
    dense_intermediate_size = int(
        selected.dense_intermediate_size or contract.dense_intermediate_size
    )
    num_hidden_layers = int(
        selected.num_hidden_layers or (profile.num_hidden_layers if profile else 64)
    )
    routed_experts = contract.routed_experts
    expert_width = contract.expert_intermediate_size
    shared_width = contract.shared_intermediate_size
    top_k = contract.top_k

    dense_ffn_parameters = 3 * hidden_size * dense_intermediate_size
    shared_parameters = 3 * hidden_size * shared_width
    routed_parameters = 3 * hidden_size * expert_width * routed_experts
    active_routed_parameters = 3 * hidden_size * expert_width * top_k
    residual_parameters, residual_flops = _residual_counts(selected, hidden_size=hidden_size)
    router_parameters, router_flops = _router_counts(
        selected,
        hidden_size=hidden_size,
        routed_experts=routed_experts,
        top_k=top_k,
    )
    (
        scale_parameters,
        calibration_parameters,
        scale_flops,
        calibration_flops,
    ) = _scale_counts(
        selected,
        routed_experts=routed_experts,
        hidden_size=hidden_size,
        top_k=top_k,
    )
    normalization_parameters = int(selected.normalization_parameters)
    normalization_flops = int(selected.normalization_flops_per_token)
    active_ffn_parameters = (
        shared_parameters
        + active_routed_parameters
        + residual_parameters
        + normalization_parameters
        + calibration_parameters
    )
    active_parameters = active_ffn_parameters + router_parameters + scale_parameters
    total_parameters = (
        shared_parameters
        + routed_parameters
        + residual_parameters
        + normalization_parameters
        + router_parameters
        + scale_parameters
        + calibration_parameters
    )
    parameters = ParameterAccounting(
        dense_ffn_parameters=dense_ffn_parameters,
        shared_parameters=shared_parameters,
        routed_parameters=routed_parameters,
        active_routed_parameters=active_routed_parameters,
        residual_parameters=residual_parameters,
        normalization_parameters=normalization_parameters,
        router_parameters=router_parameters,
        scale_parameters=scale_parameters,
        calibration_parameters=calibration_parameters,
        active_ffn_parameters=active_ffn_parameters,
        active_parameters=active_parameters,
        total_parameters=total_parameters,
    )

    multiplier = int(selected.flop_multiplier)
    dense_ffn_flops_per_token = multiplier * 3 * hidden_size * dense_intermediate_size
    shared_flops = multiplier * 3 * hidden_size * shared_width
    routed_flops = multiplier * 3 * hidden_size * expert_width * top_k
    active_ffn_flops = (
        shared_flops
        + routed_flops
        + residual_flops
        + normalization_flops
    )
    active_flops_per_token = active_ffn_flops + router_flops + scale_flops + calibration_flops
    window = int(selected.tokens) * num_hidden_layers
    flops = FlopAccounting(
        dense_ffn_flops_per_token=dense_ffn_flops_per_token,
        shared_flops_per_token=shared_flops,
        routed_flops_per_token=routed_flops,
        residual_flops_per_token=residual_flops,
        normalization_flops_per_token=normalization_flops,
        router_flops_per_token=router_flops,
        scale_flops_per_token=scale_flops,
        calibration_flops_per_token=calibration_flops,
        active_ffn_flops_per_token=active_ffn_flops,
        active_flops_per_token=active_flops_per_token,
        dense_ffn_flops=dense_ffn_flops_per_token * window,
        active_ffn_flops=active_ffn_flops * window,
        active_flops=active_flops_per_token * window,
        tokens=int(selected.tokens),
        num_hidden_layers=num_hidden_layers,
    )
    active_width = shared_width + top_k * expert_width
    return V23Accounting(
        version=V23_ACCOUNTING_VERSION,
        schema_version=V23_SCHEMA_VERSION,
        topology_id=contract.topology_id,
        profile_name=contract.profile_name,
        hidden_size=hidden_size,
        dense_intermediate_size=dense_intermediate_size,
        routed_experts=routed_experts,
        expert_intermediate_size=expert_width,
        shared_intermediate_size=shared_width,
        top_k=top_k,
        active_intermediate_size=active_width,
        active_width_reduction=1.0 - active_width / dense_intermediate_size,
        parameters=parameters,
        flops=flops,
        spec=selected,
    )


def validate_v23_topology(selector: TopologySelector) -> TopologyContract:
    """Resolve one of the two active product topologies, failing closed."""

    return _resolve_contract(selector)


validate_active_topology = validate_v23_topology


def estimate_active_flops(
    accounting_or_spec: V23Accounting | V23AccountingSpec | TopologySelector = "p16/top4",
    *,
    tokens: int | None = None,
    num_hidden_layers: int | None = None,
    **overrides: Any,
) -> int:
    """Return total active FLOPs for a token/layer window.

    Passing an existing receipt avoids recomputation; token/layer overrides
    rebuild a receipt from its spec so the result remains internally coherent.
    """

    if isinstance(accounting_or_spec, V23Accounting):
        if tokens is None and num_hidden_layers is None and not overrides:
            return accounting_or_spec.flops.active_flops
        base = accounting_or_spec.spec
        values = dict(overrides)
        if tokens is not None:
            values["tokens"] = tokens
        if num_hidden_layers is not None:
            values["num_hidden_layers"] = num_hidden_layers
        return account_v23(replace(base, **values)).flops.active_flops
    if isinstance(accounting_or_spec, V23AccountingSpec):
        values = dict(overrides)
        if tokens is not None:
            values["tokens"] = tokens
        if num_hidden_layers is not None:
            values["num_hidden_layers"] = num_hidden_layers
        return account_v23(replace(accounting_or_spec, **_normalise_overrides(values))).flops.active_flops
    values = dict(overrides)
    if tokens is not None:
        values["tokens"] = tokens
    if num_hidden_layers is not None:
        values["num_hidden_layers"] = num_hidden_layers
    return account_v23(accounting_or_spec, **values).flops.active_flops


def build_v23_accounting(
    topology: TopologySelector = "p16/top4", **kwargs: Any
) -> V23Accounting:
    """Readable alias for callers constructing a receipt from a topology."""

    return account_v23(topology, **kwargs)


# Names used by early V2.3 notebooks are kept as aliases, not separate
# implementations, so arithmetic and topology validation cannot diverge.
Dense2MoEV23Accounting = V23Accounting
V23AccountingReceipt = V23Accounting
account_dense2moe_v23 = account_v23
estimate_v23_accounting = account_v23

__all__ = [
    "ACTIVE_TOPOLOGY_IDS",
    "FLOP_MULTIPLIER",
    "FORBIDDEN_TOPOLOGY_IDS",
    "V23_ACCOUNTING_VERSION",
    "V23_SCHEMA_VERSION",
    "Dense2MoEV23Accounting",
    "FlopAccounting",
    "ParameterAccounting",
    "ResidualAccountingSpec",
    "RouterAccountingSpec",
    "ScaleCalibrationSpec",
    "V23Accounting",
    "V23AccountingReceipt",
    "V23AccountingSpec",
    "account_dense2moe_v23",
    "account_v23",
    "build_v23_accounting",
    "estimate_active_flops",
    "estimate_v23_accounting",
    "validate_active_topology",
    "validate_v23_topology",
]
