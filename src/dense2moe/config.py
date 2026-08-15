"""Configuration and geometry validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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

    @property
    def routed_capacity(self) -> int:
        return self.routed_experts * self.expert_intermediate_size

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
        )
        profile.validate()
        return profile


def load_config(path: str | Path) -> MoEProfile:
    return MoEProfile.from_mapping(_read_mapping(Path(path)))


def write_config_json(profile: MoEProfile, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

