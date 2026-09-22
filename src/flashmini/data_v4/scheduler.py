"""Token-target scheduler for deficit-aware corpus ingestion."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class DeficitTokenScheduler:
    """Choose the domain furthest behind its exact token target.

    The scheduler is deliberately independent of source iteration.  A source
    can be exhausted or blocked without making another domain consume the
    build merely because it appeared first in YAML.
    """

    targets: dict[str, int]
    seed: int = 0
    actual: dict[str, int] = field(default_factory=dict)
    attempted: dict[str, int] = field(default_factory=dict)
    exhausted_sources: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.targets = {str(k): int(v) for k, v in self.targets.items() if int(v) > 0}
        if not self.targets:
            raise ValueError("scheduler requires at least one positive token target")
        self.actual = {name: int(self.actual.get(name, 0)) for name in self.targets}
        self.attempted = {name: int(self.attempted.get(name, 0)) for name in self.targets}

    @property
    def total_target(self) -> int:
        return sum(self.targets.values())

    @property
    def total_actual(self) -> int:
        return sum(self.actual.values())

    @property
    def deficits(self) -> dict[str, int]:
        return {name: max(0, self.targets[name] - self.actual.get(name, 0))
                for name in self.targets}

    def choose_domain(self, available: Iterable[str] | None = None) -> str | None:
        """Return the highest normalized deficit among available domains."""
        names = set(available) if available is not None else set(self.targets)
        candidates = [name for name in self.targets if name in names and self.deficits[name] > 0]
        if not candidates:
            return None

        def tie_key(name: str) -> tuple[float, int, str]:
            ratio = self.deficits[name] / max(self.targets[name], 1)
            digest = hashlib.sha256(f"{self.seed}\0{name}".encode()).digest()
            return ratio, int.from_bytes(digest[:8], "big"), name

        return max(candidates, key=tie_key)

    def record(self, domain: str, exact_tokens: int, *, attempted_tokens: int | None = None) -> None:
        if domain not in self.targets:
            raise KeyError(f"unknown domain {domain!r}")
        if exact_tokens < 0:
            raise ValueError("exact_tokens cannot be negative")
        self.actual[domain] += int(exact_tokens)
        self.attempted[domain] += int(exact_tokens if attempted_tokens is None else attempted_tokens)

    def mark_source_exhausted(self, source_id: str) -> None:
        self.exhausted_sources.add(str(source_id))

    def complete(self) -> bool:
        return all(value <= 0 for value in self.deficits.values())

    def unresolved(self) -> dict[str, int]:
        return {name: value for name, value in self.deficits.items() if value > 0}

    def snapshot(self) -> dict:
        return {
            "version": 1,
            "targets": dict(sorted(self.targets.items())),
            "actual": dict(sorted(self.actual.items())),
            "attempted": dict(sorted(self.attempted.items())),
            "seed": int(self.seed),
            "exhausted_sources": sorted(self.exhausted_sources),
        }

    @classmethod
    def restore(cls, state: dict) -> DeficitTokenScheduler:
        return cls(
            targets=dict(state.get("targets", {})),
            seed=int(state.get("seed", 0)),
            actual=dict(state.get("actual", {})),
            attempted=dict(state.get("attempted", {})),
            exhausted_sources=set(state.get("exhausted_sources", [])),
        )


# Short compatibility name for callers that describe this as a scheduler.
DeficitScheduler = DeficitTokenScheduler
