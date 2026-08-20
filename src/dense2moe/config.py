"""Configuration and geometry validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The feature-completion epic deliberately has only two active product
# topologies.  ``MoEProfile`` remains a general geometry value object because
# historical fixtures and diagnostics still need to load old profiles, but
# callers that select a product topology must go through the explicit
# contract below.  Keeping the active allow-list here gives orchestration and
# validation one source of truth without silently turning every legacy config
# into a current product target.
ACTIVE_TOPOLOGY_IDS = ("p16/top4", "p32/top5")
ACTIVE_PROFILE_NAMES = ("qwen38_p16s1_top4", "qwen38_p32s1_top5")
FORBIDDEN_TOPOLOGY_IDS = frozenset({"p32/top4"})
FORBIDDEN_PROFILE_NAMES = frozenset({"qwen38_p32s1_top4"})


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except ImportError:
        pass
    result: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = _scalar(value)
    return result


@dataclass(frozen=True)
class MoEProfile:
    """A dense-FFN preserving MoE geometry.

    The capacity invariant is deliberately strict: the shared neurons plus all
    routed expert neurons must equal the source dense intermediate dimension.
    """

    name: str
    hidden_size: int
    dense_intermediate_size: int
    num_hidden_layers: int
    routed_experts: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    top_k: int
    model: str = "Qwen/Qwen3.8-27B"
    revision: str = "main"
    dtype: str = "bfloat16"
    routing_mode: str = "normalized_softmax"

    @property
    def routed_capacity(self) -> int:
        return self.routed_experts * self.expert_intermediate_size

    @property
    def topology_id(self) -> str:
        """Return the stable topology identity used by phase contracts."""

        return f"p{self.routed_experts}/top{self.top_k}"

    @property
    def total_capacity(self) -> int:
        return self.routed_capacity + self.shared_intermediate_size

    @property
    def active_intermediate_size(self) -> int:
        return self.shared_intermediate_size + self.top_k * self.expert_intermediate_size

    @property
    def sparsity(self) -> float:
        return 1.0 - self.active_intermediate_size / self.dense_intermediate_size

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "dense_intermediate_size": self.dense_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "routed_experts": self.routed_experts,
            "expert_intermediate_size": self.expert_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
            "top_k": self.top_k,
        }
        invalid = [key for key, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"profile values must be positive: {', '.join(invalid)}")
        if self.top_k > self.routed_experts:
            raise ValueError("top_k cannot exceed routed_experts")
        if self.routing_mode not in {"normalized_softmax", "independent_positive"}:
            raise ValueError("routing_mode must be normalized_softmax or independent_positive")
        if self.total_capacity != self.dense_intermediate_size:
            raise ValueError(
                "capacity mismatch: routed_experts * expert_intermediate_size "
                f"+ shared_intermediate_size = {self.total_capacity}, expected "
                f"{self.dense_intermediate_size}"
            )
        if not self.revision or self.revision in {"latest", "main", "master"}:
            # Mutable revisions are permitted for exploratory configs but are
            # surfaced by source-manifest validation as an unpinned fact.
            return

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "routed_capacity": self.routed_capacity,
            "total_capacity": self.total_capacity,
            "active_intermediate_size": self.active_intermediate_size,
            "sparsity": self.sparsity,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MoEProfile:
        aliases = {
            "num_experts": "routed_experts",
            "num_routed_experts": "routed_experts",
            "moe_intermediate_size": "expert_intermediate_size",
            "shared_expert_intermediate_size": "shared_intermediate_size",
            "num_layers": "num_hidden_layers",
        }
        normalized = {aliases.get(str(k), str(k)): v for k, v in values.items()}
        required = (
            "name",
            "hidden_size",
            "dense_intermediate_size",
            "num_hidden_layers",
            "routed_experts",
            "expert_intermediate_size",
            "shared_intermediate_size",
            "top_k",
        )
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError(f"profile is missing required keys: {', '.join(missing)}")
        profile = cls(
            name=str(normalized["name"]),
            model=str(normalized.get("model", cls.model)),
            revision=str(normalized.get("revision", cls.revision)),
            dtype=str(normalized.get("dtype", cls.dtype)),
            hidden_size=int(normalized["hidden_size"]),
            dense_intermediate_size=int(normalized["dense_intermediate_size"]),
            num_hidden_layers=int(normalized["num_hidden_layers"]),
            routed_experts=int(normalized["routed_experts"]),
            expert_intermediate_size=int(normalized["expert_intermediate_size"]),
            shared_intermediate_size=int(normalized["shared_intermediate_size"]),
            top_k=int(normalized["top_k"]),
            routing_mode=str(normalized.get("routing_mode", "normalized_softmax")),
        )
        profile.validate()
        return profile


def load_config(path: str | Path) -> MoEProfile:
    return MoEProfile.from_mapping(_read_mapping(Path(path)))


@dataclass(frozen=True)
class TopologyContract:
    """An active product topology and its non-negotiable geometry.

    A profile can still be loaded for historical diagnostics with
    :func:`load_config`.  ``validate_active_profile`` is the fail-closed
    boundary for epic orchestration: it rejects inactive profiles (including
    the explicitly forbidden p32/top4 experiment) and checks every geometry
    field rather than relying on a name alone.
    """

    topology_id: str
    profile_name: str
    role: str
    routed_experts: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    top_k: int
    dense_intermediate_size: int = 17_408

    @property
    def active_intermediate_size(self) -> int:
        return self.shared_intermediate_size + self.top_k * self.expert_intermediate_size

    @property
    def sparsity(self) -> float:
        return 1.0 - self.active_intermediate_size / self.dense_intermediate_size

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "active_intermediate_size": self.active_intermediate_size,
            "sparsity": self.sparsity,
        }

    def validate_profile(self, profile: MoEProfile) -> None:
        """Validate a loaded profile against this exact active contract."""

        profile.validate()
        expected = {
            "topology_id": self.topology_id,
            "profile_name": self.profile_name,
            "routed_experts": self.routed_experts,
            "expert_intermediate_size": self.expert_intermediate_size,
            "shared_intermediate_size": self.shared_intermediate_size,
            "top_k": self.top_k,
            "dense_intermediate_size": self.dense_intermediate_size,
        }
        actual = {
            "topology_id": profile.topology_id,
            "profile_name": profile.name,
            "routed_experts": profile.routed_experts,
            "expert_intermediate_size": profile.expert_intermediate_size,
            "shared_intermediate_size": profile.shared_intermediate_size,
            "top_k": profile.top_k,
            "dense_intermediate_size": profile.dense_intermediate_size,
        }
        mismatches = [
            f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
            for key in expected
            if actual[key] != expected[key]
        ]
        if mismatches:
            raise ValueError(
                f"profile {profile.name!r} does not satisfy active topology "
                f"{self.topology_id!r}: " + "; ".join(mismatches)
            )


SAFE_FALLBACK_TOPOLOGY = TopologyContract(
    topology_id="p16/top4",
    profile_name="qwen38_p16s1_top4",
    role="safe_fallback",
    routed_experts=16,
    expert_intermediate_size=1024,
    shared_intermediate_size=1024,
    top_k=4,
)
PRIMARY_PRODUCT_TOPOLOGY = TopologyContract(
    topology_id="p32/top5",
    profile_name="qwen38_p32s1_top5",
    role="primary_product",
    routed_experts=32,
    expert_intermediate_size=512,
    shared_intermediate_size=1024,
    top_k=5,
)
ACTIVE_TOPOLOGIES: dict[str, TopologyContract] = {
    SAFE_FALLBACK_TOPOLOGY.topology_id: SAFE_FALLBACK_TOPOLOGY,
    PRIMARY_PRODUCT_TOPOLOGY.topology_id: PRIMARY_PRODUCT_TOPOLOGY,
}
ACTIVE_TOPOLOGIES_BY_PROFILE: dict[str, TopologyContract] = {
    contract.profile_name: contract for contract in ACTIVE_TOPOLOGIES.values()
}


def _topology_key(value: str | Path) -> str:
    """Normalize a topology/profile/path selector to a lookup key."""

    raw = str(value).strip().lower().replace("\\", "/")
    if not raw:
        return raw
    # Preserve slash-separated topology IDs before applying Path semantics;
    # on POSIX, Path("p16/top4").name would otherwise collapse to "top4".
    if raw in ACTIVE_TOPOLOGY_IDS or raw in FORBIDDEN_TOPOLOGY_IDS:
        return raw
    name = raw.rsplit("/", 1)[-1]
    if name.endswith((".yaml", ".yml", ".json")):
        name = Path(name).stem
    return name


def active_topology_contract(selector: str | Path | MoEProfile) -> TopologyContract:
    """Resolve an active topology selector, failing closed for legacy ones."""

    if isinstance(selector, MoEProfile):
        profile = selector
        key = profile.name.lower()
    else:
        key = _topology_key(selector)
    contract = ACTIVE_TOPOLOGIES.get(key) or ACTIVE_TOPOLOGIES_BY_PROFILE.get(key)
    if contract is None:
        topology_id = "p32/top4" if key in FORBIDDEN_PROFILE_NAMES else key if key in FORBIDDEN_TOPOLOGY_IDS else None
        if topology_id:
            raise ValueError(f"topology {topology_id!r} is explicitly forbidden")
        raise ValueError(
            f"inactive topology/profile {str(selector)!r}; active choices are "
            + ", ".join(ACTIVE_TOPOLOGY_IDS)
        )
    if isinstance(selector, MoEProfile):
        contract.validate_profile(selector)
    return contract


def validate_active_profile(
    profile: MoEProfile,
    *,
    topology: str | Path | TopologyContract | None = None,
) -> TopologyContract:
    """Validate and return the active contract for ``profile``.

    ``topology`` can pin the expected active choice.  This is useful at phase
    boundaries where a run must not silently switch from the safe fallback to
    the primary candidate (or vice versa) during resume.
    """

    contract = (
        topology
        if isinstance(topology, TopologyContract)
        else active_topology_contract(topology)
        if topology is not None
        else active_topology_contract(profile)
    )
    contract.validate_profile(profile)
    return contract


def load_active_config(path: str | Path) -> tuple[MoEProfile, TopologyContract]:
    """Load a config and enforce the two-topology product allow-list."""

    profile = load_config(path)
    return profile, validate_active_profile(profile)


def write_config_json(profile: MoEProfile, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
