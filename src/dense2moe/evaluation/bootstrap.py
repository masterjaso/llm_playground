"""Deterministic grouped bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .registry import INSUFFICIENT_EVIDENCE


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def grouped_bootstrap(
    values: Sequence[float] | Mapping[str, Sequence[float]],
    groups: Sequence[str] | None = None,
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    seed: int = 0,
    repetitions: int = 2000,
    confidence_level: float = 0.95,
    minimum_group_count: int = 2,
    minimum_sample_count: int = 16,
) -> dict[str, Any]:
    """Bootstrap independent groups, not correlated raw tokens.

    ``values`` may be a flat record vector plus a parallel ``groups`` vector,
    or a mapping from group identity to record values.  Each resample draws
    group identities with replacement and includes all records in the drawn
    groups.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if isinstance(values, Mapping):
        grouped = {str(key): [float(item) for item in items] for key, items in values.items()}
    else:
        if groups is None:
            raise ValueError("groups are required when values is not a mapping")
        if len(values) != len(groups):
            raise ValueError("values and groups must have equal length")
        grouped = defaultdict(list)
        for value, group in zip(values, groups):
            grouped[str(group)].append(float(value))
        grouped = dict(grouped)
    grouped = {key: items for key, items in grouped.items() if items}
    group_ids = sorted(grouped)
    sample_count = sum(len(items) for items in grouped.values())
    point_values = [value for group in group_ids for value in grouped[group]]
    if len(group_ids) < minimum_group_count or sample_count < minimum_sample_count:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "point_estimate": statistic(point_values) if point_values else None,
            "lower": None,
            "upper": None,
            "grouping_identity": "group_ids",
            "group_ids": group_ids,
            "seed": int(seed),
            "repetitions": int(repetitions),
            "confidence_level": float(confidence_level),
            "independent_group_count": len(group_ids),
            "scored_token_count": sample_count,
            "minimum_group_count": int(minimum_group_count),
            "minimum_sample_count": int(minimum_sample_count),
            "reason": "insufficient independent groups or records; diagnostic-only",
        }
    rng = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(int(repetitions)):
        sampled_groups = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
        sampled_values = [value for group in sampled_groups for value in grouped[group]]
        estimate = float(statistic(sampled_values))
        if not math.isfinite(estimate):
            continue
        estimates.append(estimate)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "COMPUTED" if estimates else INSUFFICIENT_EVIDENCE,
        "point_estimate": float(statistic(point_values)),
        "lower": _quantile(estimates, alpha) if estimates else None,
        "upper": _quantile(estimates, 1.0 - alpha) if estimates else None,
        "grouping_identity": "group_ids",
        "group_ids": group_ids,
        "seed": int(seed),
        "repetitions": int(repetitions),
        "confidence_level": float(confidence_level),
        "independent_group_count": len(group_ids),
        "scored_token_count": sample_count,
        "minimum_group_count": int(minimum_group_count),
        "minimum_sample_count": int(minimum_sample_count),
        "bootstrap_unit": "independent_group",
    }


def evaluate_threshold_with_confidence(
    interval: Mapping[str, Any],
    *,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    """Apply a conservative confidence-bound rule near a gate."""

    if interval.get("status") != "COMPUTED" or interval.get("lower") is None or interval.get("upper") is None:
        return {"status": INSUFFICIENT_EVIDENCE, "classification": "INSUFFICIENT_EVIDENCE", "threshold": threshold}
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if direction == "lower-is-better":
        classification = "GREEN" if upper <= threshold else "RED"
    elif direction == "higher-is-better":
        classification = "GREEN" if lower >= threshold else "RED"
    else:
        raise ValueError(f"unsupported threshold direction: {direction}")
    return {
        "status": "COMPUTED",
        "classification": classification,
        "threshold": float(threshold),
        "conservative_bound": upper if direction == "lower-is-better" else lower,
        "direction": direction,
        "confidence_interval": {"lower": lower, "upper": upper},
    }


bootstrap_independent_groups = grouped_bootstrap


__all__ = ["bootstrap_independent_groups", "evaluate_threshold_with_confidence", "grouped_bootstrap"]
