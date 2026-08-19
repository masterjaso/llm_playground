"""Versioned V2.3 candidate-design registry.

The registry is the research-side source of truth for the five designs in the
V2.3 candidate search.  It intentionally does not replace the product
topology allow-list in :mod:`dense2moe.config`: ``p16/top6`` and ``p32/top8``
are valid *research candidates* even though only the two locked product
topologies are active at the product boundary.

Every design keeps its dense-capacity arithmetic explicit.  Shared width plus
all routed expert widths must equal the dense intermediate width; the active
FFN reduction is computed from shared width, selected experts, and any
declared residual correction.  This makes an accidental below-threshold
candidate fail before it can enter a capture or training runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

V23_DESIGN_REGISTRY_VERSION = "dense2moe-v2.3-design-registry"
V23_DESIGN_SCHEMA_VERSION = 1
DENSE_INTERMEDIATE_SIZE = 17_408
MIN_ACTIVE_FFN_REDUCTION = 0.50


def derive_expert_width(
    dense_intermediate_size: int,
    routed_experts: int,
    shared_intermediate_size: int,
) -> int:
    """Derive an integer expert width from a capacity-preserving partition."""

    dense = int(dense_intermediate_size)
    experts = int(routed_experts)
    shared = int(shared_intermediate_size)
    if dense <= 0 or experts <= 0 or shared <= 0:
        raise ValueError("dense width, routed experts, and shared width must be positive")
    remaining = dense - shared
    if remaining <= 0:
        raise ValueError("shared width must be smaller than the dense width")
    if remaining % experts:
        raise ValueError(
            "capacity is not divisible by routed experts: "
            f"{dense} - {shared} is not divisible by {experts}"
        )
    return remaining // experts


def is_capacity_valid(
    dense_intermediate_size: int,
    routed_experts: int,
    shared_intermediate_size: int,
    expert_intermediate_size: int,
) -> bool:
    """Return whether a geometry exactly preserves dense FFN capacity."""

    return (
        int(shared_intermediate_size)
        + int(routed_experts) * int(expert_intermediate_size)
        == int(dense_intermediate_size)
    )


@dataclass(frozen=True)
class ResidualDesignSpec:
    """Metadata for an optional residual correction branch.

    ``width`` is an effective active-width budget used for the conservative
    reduction check.  A low-rank branch may additionally expose its matrix
    rank; the effective-width budget remains explicit rather than pretending
    that low-rank parameter count alone is a dense FFN width.
    """

    kind: str = "none"
    width: int = 0
    rank: int | None = None
    initialization: str = "none"

    def validate(self) -> None:
        if self.kind not in {"none", "swiglu", "low_rank_silu"}:
            raise ValueError(
                "residual kind must be none, swiglu, or low_rank_silu"
            )
        if int(self.width) < 0:
            raise ValueError("residual width must be non-negative")
        if self.kind == "none" and int(self.width) != 0:
            raise ValueError("a disabled residual branch must have zero width")
        if self.kind != "none" and int(self.width) <= 0:
            raise ValueError("an enabled residual branch must have positive width")
        if self.rank is not None:
            if int(self.rank) <= 0:
                raise ValueError("residual rank must be positive")
            if int(self.rank) > int(self.width):
                raise ValueError("residual rank cannot exceed residual width")
        if not str(self.initialization).strip():
            raise ValueError("residual initialization must be non-empty")

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def intermediate_size(self) -> int:
        """Alias used by accounting and runner adapters."""

        return int(self.width)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterDesignSpec:
    """Declared selector family and supervision strategy."""

    architecture: str = "linear"
    selection: str = "topk"
    supervision: str = "router_cross_entropy"
    load_penalty: str = "none"
    load_penalty_weight: float = 0.0
    amplitude_supervision: bool = False

    def validate(self) -> None:
        if self.architecture not in {
            "linear",
            "load_priced_linear",
            "nonlinear_listwise",
            "multilabel_set",
        }:
            raise ValueError(
                "router architecture must be linear, load_priced_linear, "
                "nonlinear_listwise, or multilabel_set"
            )
        if self.selection not in {"topk", "listwise", "multilabel_set"}:
            raise ValueError("router selection must be topk, listwise, or multilabel_set")
        if self.load_penalty not in {"none", "hard_load", "load_aware", "load_priced"}:
            raise ValueError(
                "router load_penalty must be none, hard_load, load_aware, or load_priced"
            )
        if float(self.load_penalty_weight) < 0:
            raise ValueError("router load penalty weight must be non-negative")
        if self.architecture == "nonlinear_listwise" and self.selection != "listwise":
            raise ValueError("nonlinear_listwise routers must declare listwise selection")
        if self.architecture == "multilabel_set" and self.selection != "multilabel_set":
            raise ValueError("multilabel_set routers must declare multilabel_set selection")

    @property
    def nonlinear(self) -> bool:
        return self.architecture in {"nonlinear_listwise", "multilabel_set"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LossDesignSpec:
    """Loss and oracle supervision metadata for one candidate design."""

    mode: str
    components: tuple[str, ...]
    norm_aware: bool = False
    covariance_partition: bool = False
    hard_load_penalty: bool = False
    load_aware: bool = False
    mixed_amplitude_supervision: bool = False

    def validate(self) -> None:
        if not str(self.mode).strip():
            raise ValueError("loss mode must be non-empty")
        if not self.components or any(not str(item).strip() for item in self.components):
            raise ValueError("loss components must be a non-empty sequence")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"components": list(self.components)}


@dataclass(frozen=True)
class V23Design:
    """One arithmetic-validated V2.3 research candidate."""

    design_id: str
    name: str
    topology_family: str
    routed_experts: int
    top_k: int
    shared_intermediate_size: int
    expert_intermediate_size: int
    residual: ResidualDesignSpec
    router: RouterDesignSpec
    loss: LossDesignSpec
    dense_intermediate_size: int = DENSE_INTERMEDIATE_SIZE
    description: str = ""

    @property
    def id(self) -> str:
        """Short stable design identity (A through E)."""

        return self.design_id

    @property
    def family(self) -> str:
        return self.topology_family

    @property
    def topology_id(self) -> str:
        return f"p{self.routed_experts}/top{self.top_k}"

    @property
    def routed_capacity(self) -> int:
        return int(self.routed_experts) * int(self.expert_intermediate_size)

    @property
    def total_capacity(self) -> int:
        return int(self.shared_intermediate_size) + self.routed_capacity

    @property
    def active_intermediate_size(self) -> int:
        return int(self.shared_intermediate_size) + int(self.top_k) * int(
            self.expert_intermediate_size
        )

    @property
    def residual_active_width(self) -> int:
        return int(self.residual.width)

    @property
    def active_width_for_reduction(self) -> int:
        return self.active_intermediate_size + self.residual_active_width

    @property
    def active_ffn_reduction(self) -> float:
        return 1.0 - self.active_width_for_reduction / int(self.dense_intermediate_size)

    @property
    def active_width_reduction(self) -> float:
        """Compatibility alias for reports that use width terminology."""

        return self.active_ffn_reduction

    def validate(self, *, minimum_reduction: float = MIN_ACTIVE_FFN_REDUCTION) -> None:
        design_id = str(self.design_id).strip().upper()
        if design_id not in {"A", "B", "C", "D", "E"}:
            raise ValueError("V2.3 design_id must be one of A, B, C, D, or E")
        if not str(self.name).strip():
            raise ValueError("design name must be non-empty")
        family = str(self.topology_family).strip().lower()
        if family != f"p{int(self.routed_experts)}":
            raise ValueError(
                "topology family must match routed experts: "
                f"expected p{self.routed_experts!s}, got {self.topology_family!r}"
            )
        positive = {
            "dense_intermediate_size": self.dense_intermediate_size,
            "routed_experts": self.routed_experts,
            "top_k": self.top_k,
            "shared_intermediate_size": self.shared_intermediate_size,
            "expert_intermediate_size": self.expert_intermediate_size,
        }
        invalid = [key for key, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"design dimensions must be positive: {', '.join(invalid)}")
        if int(self.top_k) > int(self.routed_experts):
            raise ValueError("top_k cannot exceed routed experts")
        if not is_capacity_valid(
            self.dense_intermediate_size,
            self.routed_experts,
            self.shared_intermediate_size,
            self.expert_intermediate_size,
        ):
            raise ValueError(
                "capacity mismatch: shared width plus all routed expert widths "
                f"equals {self.total_capacity}, expected {self.dense_intermediate_size}"
            )
        self.residual.validate()
        self.router.validate()
        self.loss.validate()
        if not 0 <= float(minimum_reduction) < 1:
            raise ValueError("minimum reduction must be in [0, 1)")
        if self.active_ffn_reduction < float(minimum_reduction):
            raise ValueError(
                f"active FFN reduction {self.active_ffn_reduction:.6f} is below "
                f"the required {float(minimum_reduction):.6f}"
            )

    def with_geometry(
        self,
        *,
        shared_intermediate_size: int | None = None,
        top_k: int | None = None,
        expert_intermediate_size: int | None = None,
    ) -> V23Design:
        """Return a candidate with a capacity-valid geometry.

        When ``expert_intermediate_size`` is omitted it is derived from the
        dense width, routed expert count, and requested shared width.  This is
        useful for bounded arithmetic retries within a declared design family.
        """

        shared = (
            int(shared_intermediate_size)
            if shared_intermediate_size is not None
            else int(self.shared_intermediate_size)
        )
        expert = (
            int(expert_intermediate_size)
            if expert_intermediate_size is not None
            else derive_expert_width(self.dense_intermediate_size, self.routed_experts, shared)
        )
        updated = replace(
            self,
            shared_intermediate_size=shared,
            expert_intermediate_size=expert,
            top_k=int(top_k) if top_k is not None else self.top_k,
        )
        updated.validate()
        return updated

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["registry_version"] = V23_DESIGN_REGISTRY_VERSION
        result["schema_version"] = V23_DESIGN_SCHEMA_VERSION
        result["topology_id"] = self.topology_id
        result["routed_capacity"] = self.routed_capacity
        result["total_capacity"] = self.total_capacity
        result["active_intermediate_size"] = self.active_intermediate_size
        result["residual_active_width"] = self.residual_active_width
        result["active_width_for_reduction"] = self.active_width_for_reduction
        result["active_ffn_reduction"] = self.active_ffn_reduction
        result["active_width_reduction"] = self.active_width_reduction
        result["loss"] = self.loss.as_dict()
        return result

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def _designs() -> tuple[V23Design, ...]:
    """Build the immutable default registry."""

    p16_expert_width = derive_expert_width(DENSE_INTERMEDIATE_SIZE, 16, 2_048)
    p32_expert_width = derive_expert_width(DENSE_INTERMEDIATE_SIZE, 32, 2_048)
    return (
        V23Design(
            design_id="A",
            name="p16-top6-fidelity-first",
            topology_family="p16",
            routed_experts=16,
            top_k=6,
            shared_intermediate_size=2_048,
            expert_intermediate_size=p16_expert_width,
            residual=ResidualDesignSpec(),
            router=RouterDesignSpec(
                architecture="linear",
                selection="topk",
                supervision="fidelity_first",
            ),
            loss=LossDesignSpec(
                mode="norm_aware",
                components=("normalized_mse", "cosine"),
                norm_aware=True,
            ),
            description="Expanded shared branch with fidelity-first, norm-aware training.",
        ),
        V23Design(
            design_id="B",
            name="p16-top5-residual-covariance",
            topology_family="p16",
            routed_experts=16,
            top_k=5,
            shared_intermediate_size=2_048,
            expert_intermediate_size=p16_expert_width,
            residual=ResidualDesignSpec(
                kind="swiglu",
                width=256,
                initialization="residual_covariance_partition",
            ),
            router=RouterDesignSpec(
                architecture="load_priced_linear",
                selection="topk",
                supervision="load_priced_oracle",
                load_penalty="hard_load",
                load_penalty_weight=0.25,
            ),
            loss=LossDesignSpec(
                mode="residual_covariance_partition",
                components=("residual_mse", "covariance", "load_penalty"),
                covariance_partition=True,
                hard_load_penalty=True,
            ),
            description="Residual/covariance partition with load-priced oracle selection.",
        ),
        V23Design(
            design_id="C",
            name="p16-top4-low-rank-residual",
            topology_family="p16",
            routed_experts=16,
            top_k=4,
            shared_intermediate_size=2_048,
            expert_intermediate_size=p16_expert_width,
            residual=ResidualDesignSpec(
                kind="low_rank_silu",
                width=256,
                rank=64,
                initialization="bounded_low_rank_correction",
            ),
            router=RouterDesignSpec(
                architecture="linear",
                selection="topk",
                supervision="residual_correlation",
            ),
            loss=LossDesignSpec(
                mode="bounded_low_rank_residual",
                components=("normalized_mse", "residual_cosine", "low_rank_penalty"),
                norm_aware=True,
            ),
            description="P16/top4 with a bounded low-rank residual correction.",
        ),
        V23Design(
            design_id="D",
            name="p32-top8-load-aware",
            topology_family="p32",
            routed_experts=32,
            top_k=8,
            shared_intermediate_size=2_048,
            expert_intermediate_size=p32_expert_width,
            residual=ResidualDesignSpec(),
            router=RouterDesignSpec(
                architecture="load_priced_linear",
                selection="topk",
                supervision="load_aware_oracle",
                load_penalty="load_aware",
                load_penalty_weight=0.20,
            ),
            loss=LossDesignSpec(
                mode="load_aware",
                components=("normalized_mse", "load_cv_penalty"),
                load_aware=True,
            ),
            description="P32/top8 with expanded shared width and load-aware training.",
        ),
        V23Design(
            design_id="E",
            name="p16-top5-nonlinear-set-router",
            topology_family="p16",
            routed_experts=16,
            top_k=5,
            shared_intermediate_size=2_048,
            expert_intermediate_size=p16_expert_width,
            residual=ResidualDesignSpec(),
            router=RouterDesignSpec(
                architecture="nonlinear_listwise",
                selection="listwise",
                supervision="mixed_amplitude_supervision",
                amplitude_supervision=True,
            ),
            loss=LossDesignSpec(
                mode="mixed_amplitude_supervision",
                components=("set_ranking", "amplitude_bce", "normalized_mse"),
                mixed_amplitude_supervision=True,
            ),
            description="Nonlinear/listwise set router with mixed amplitude supervision.",
        ),
    )


V23_DESIGNS: dict[str, V23Design] = {design.design_id: design for design in _designs()}
DESIGN_REGISTRY = V23_DESIGNS
DESIGN_A, DESIGN_B, DESIGN_C, DESIGN_D, DESIGN_E = tuple(V23_DESIGNS.values())


def get_design(design_id: str) -> V23Design:
    """Return a design by its stable A-E identity."""

    key = str(design_id).strip().upper()
    try:
        return V23_DESIGNS[key]
    except KeyError as exc:
        raise ValueError(f"unknown V2.3 design {design_id!r}; expected A, B, C, D, or E") from exc


def iter_designs() -> tuple[V23Design, ...]:
    """Return designs in deterministic A-to-E order."""

    return tuple(V23_DESIGNS[key] for key in ("A", "B", "C", "D", "E"))


def validate_design_registry(
    *, minimum_reduction: float = MIN_ACTIVE_FFN_REDUCTION
) -> tuple[V23Design, ...]:
    """Validate and return every registered design in deterministic order."""

    designs = iter_designs()
    if len({design.design_id for design in designs}) != 5:
        raise ValueError("V2.3 registry must contain exactly five unique designs")
    for design in designs:
        design.validate(minimum_reduction=minimum_reduction)
    return designs


def registry_as_dict() -> dict[str, Any]:
    """Return a deterministic, receipt-ready representation of the registry."""

    validate_design_registry()
    return {
        "version": V23_DESIGN_REGISTRY_VERSION,
        "schema_version": V23_DESIGN_SCHEMA_VERSION,
        "dense_intermediate_size": DENSE_INTERMEDIATE_SIZE,
        "minimum_active_ffn_reduction": MIN_ACTIVE_FFN_REDUCTION,
        "designs": [design.as_dict() for design in iter_designs()],
    }


def registry_to_json() -> str:
    """Serialize the registry with stable key and separator ordering."""

    return json.dumps(registry_as_dict(), sort_keys=True, separators=(",", ":"))


def design_from_mapping(values: Mapping[str, Any]) -> V23Design:
    """Load one receipt design while retaining strict validation."""

    residual_values = values.get("residual", {})
    router_values = values.get("router", {})
    loss_values = values.get("loss", {})
    if not isinstance(residual_values, Mapping):
        raise TypeError("residual design metadata must be a mapping")
    if not isinstance(router_values, Mapping):
        raise TypeError("router design metadata must be a mapping")
    if not isinstance(loss_values, Mapping):
        raise TypeError("loss design metadata must be a mapping")
    design = V23Design(
        design_id=str(values["design_id"]),
        name=str(values["name"]),
        topology_family=str(values["topology_family"]),
        routed_experts=int(values["routed_experts"]),
        top_k=int(values["top_k"]),
        shared_intermediate_size=int(values["shared_intermediate_size"]),
        expert_intermediate_size=int(values["expert_intermediate_size"]),
        residual=ResidualDesignSpec(
            kind=str(residual_values.get("kind", "none")),
            width=int(residual_values.get("width", residual_values.get("intermediate_size", 0))),
            rank=(
                int(residual_values["rank"])
                if residual_values.get("rank") is not None
                else None
            ),
            initialization=str(residual_values.get("initialization", "none")),
        ),
        router=RouterDesignSpec(**{
            key: router_values[key]
            for key in (
                "architecture",
                "selection",
                "supervision",
                "load_penalty",
                "load_penalty_weight",
                "amplitude_supervision",
            )
            if key in router_values
        }),
        loss=LossDesignSpec(
            mode=str(loss_values["mode"]),
            components=tuple(str(item) for item in loss_values["components"]),
            norm_aware=bool(loss_values.get("norm_aware", False)),
            covariance_partition=bool(loss_values.get("covariance_partition", False)),
            hard_load_penalty=bool(loss_values.get("hard_load_penalty", False)),
            load_aware=bool(loss_values.get("load_aware", False)),
            mixed_amplitude_supervision=bool(
                loss_values.get("mixed_amplitude_supervision", False)
            ),
        ),
        dense_intermediate_size=int(
            values.get("dense_intermediate_size", DENSE_INTERMEDIATE_SIZE)
        ),
        description=str(values.get("description", "")),
    )
    design.validate()
    return design


# Friendly aliases for adapters and notebooks.
V23DesignConfig = V23Design
V23DesignRegistry = dict[str, V23Design]
validate_registry = validate_design_registry


__all__ = [
    "DENSE_INTERMEDIATE_SIZE",
    "DESIGN_A",
    "DESIGN_B",
    "DESIGN_C",
    "DESIGN_D",
    "DESIGN_E",
    "DESIGN_REGISTRY",
    "MIN_ACTIVE_FFN_REDUCTION",
    "V23_DESIGNS",
    "V23_DESIGN_REGISTRY_VERSION",
    "V23_DESIGN_SCHEMA_VERSION",
    "LossDesignSpec",
    "ResidualDesignSpec",
    "RouterDesignSpec",
    "V23Design",
    "V23DesignConfig",
    "V23DesignRegistry",
    "derive_expert_width",
    "design_from_mapping",
    "get_design",
    "is_capacity_valid",
    "iter_designs",
    "registry_as_dict",
    "registry_to_json",
    "validate_design_registry",
    "validate_registry",
]
