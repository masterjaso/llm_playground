"""Explicit metric classification; thresholds are never silently weakened."""

from __future__ import annotations

from typing import Any


def classify_metric(value: float, *, green: float, yellow: float, lower_is_better: bool = True) -> str:
    if lower_is_better:
        if value <= green:
            return "green"
        if value <= yellow:
            return "yellow"
    else:
        if value >= green:
            return "green"
        if value >= yellow:
            return "yellow"
    return "red"


def quality_gate(metrics: dict[str, float], thresholds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    for name, value in metrics.items():
        rule = thresholds.get(name, {})
        statuses[name] = classify_metric(float(value), green=float(rule.get("green", 0.0)), yellow=float(rule.get("yellow", 0.0)), lower_is_better=bool(rule.get("lower_is_better", True)))
    overall = "green" if statuses and all(status == "green" for status in statuses.values()) else "yellow" if statuses and all(status in {"green", "yellow"} for status in statuses.values()) else "red"
    return {"metrics": metrics, "statuses": statuses, "overall": overall}

