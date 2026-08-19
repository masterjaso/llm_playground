"""Cohort-level generalization and correlation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .generalization import classify_generalization


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lookup(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        number = _number(value)
        if number is not None:
            return number
    return None


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average = (index + end - 1) / 2.0 + 1.0
        for position in range(index, end):
            ranks[indexed[position][0]] = average
        index = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, spearman: bool = False) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x = _rank(left) if spearman else list(left)
    y = _rank(right) if spearman else list(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator > 0 else None


def analyze_cohort(rows: Sequence[Mapping[str, Any]], *, minimum_correlation_groups: int = 3) -> dict[str, Any]:
    scatter: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for row in rows:
        fit = row.get("fit_train", row.get("fit", {}))
        dev = row.get("fit_dev", row.get("dev", {}))
        generalization = row.get("generalization")
        if not isinstance(generalization, Mapping) and isinstance(fit, Mapping) and isinstance(dev, Mapping):
            generalization = classify_generalization(fit, dev, trajectory=row.get("trajectory"))
        generalization = dict(generalization or {})
        lm = row.get("lm_metrics", row.get("lm", {}))
        item = {
            "candidate_id": row.get("candidate_id"),
            "design_id": row.get("design_id"),
            "topology": row.get("topology"),
            "seed": row.get("seed"),
            "stage": row.get("stage"),
            "checkpoint": row.get("checkpoint"),
            "fit_cosine": _lookup(fit, "cosine_similarity", "cosine", "structural.cosine_similarity"),
            "dev_cosine": _lookup(dev, "cosine_similarity", "cosine", "structural.cosine_similarity"),
            "fit_nmse": _lookup(fit, "normalized_mse", "nmse", "structural.normalized_mse"),
            "dev_nmse": _lookup(dev, "normalized_mse", "nmse", "structural.normalized_mse"),
            "fit_norm_error": _lookup(fit, "target_relative_norm_error", "relative_norm_error"),
            "dev_norm_error": _lookup(dev, "target_relative_norm_error", "relative_norm_error"),
            "fit_load_cv": _lookup(fit, "learned_load_cv", "load_cv", "routing.learned_load_cv"),
            "dev_load_cv": _lookup(dev, "learned_load_cv", "load_cv", "routing.learned_load_cv"),
            "lm_mean_forward_kl": _lookup(lm, "mean_forward_kl", "lm.mean_forward_kl"),
            "lm_excess_mean_forward_kl": _lookup(lm, "excess_mean_forward_kl", "lm.excess_mean_forward_kl"),
            "lm_high_margin_flip": _lookup(lm, "high_margin_top1_flip_rate", "lm.high_margin_top1_flip_rate"),
            "generalization": generalization,
            "slice_summaries": {key: value for key, value in dev.items() if isinstance(key, str) and key.endswith("_slices")} if isinstance(dev, Mapping) else {},
            "rerun_status": row.get("rerun_status", row.get("status")),
            "metric_policy_version": row.get("metric_policy_version"),
            "policy_hash": row.get("policy_hash"),
        }
        enriched.append(item)
        scatter.append(dict(item))
    def pairs(left_key: str, right_key: str) -> tuple[list[float], list[float]]:
        paired = [(item[left_key], item[right_key]) for item in enriched if item.get(left_key) is not None and item.get(right_key) is not None]
        return [pair[0] for pair in paired], [pair[1] for pair in paired]

    relationships = {
        "spearman_fit_cosine_vs_dev_cosine": ("fit_cosine", "dev_cosine"),
        "spearman_fit_nmse_vs_dev_nmse": ("fit_nmse", "dev_nmse"),
        "spearman_dev_cosine_vs_lm_kl": ("dev_cosine", "lm_mean_forward_kl"),
        "spearman_dev_nmse_vs_lm_kl": ("dev_nmse", "lm_mean_forward_kl"),
        "spearman_dev_norm_error_vs_lm_kl": ("dev_norm_error", "lm_mean_forward_kl"),
        "spearman_learned_loadcv_vs_lm_kl": ("dev_load_cv", "lm_mean_forward_kl"),
        "pearson_fit_cosine_vs_dev_cosine": ("fit_cosine", "dev_cosine"),
        "pearson_fit_nmse_vs_dev_nmse": ("fit_nmse", "dev_nmse"),
        "pearson_dev_cosine_vs_lm_kl": ("dev_cosine", "lm_mean_forward_kl"),
    }
    correlations: dict[str, Any] = {}
    for name, (left_key, right_key) in relationships.items():
        left, right = pairs(left_key, right_key)
        if len(left) < minimum_correlation_groups:
            correlations[name] = {"status": "INSUFFICIENT_EVIDENCE", "independent_group_count": len(left), "value": None}
        else:
            correlations[name] = {
                "status": "COMPUTED",
                "independent_group_count": len(left),
                "value": _correlation(left, right, spearman=name.startswith("spearman")),
                "method": "spearman" if name.startswith("spearman") else "pearson",
            }
    best_fit = max((item for item in enriched if item.get("fit_cosine") is not None), key=lambda item: item["fit_cosine"], default=None)
    best_dev = max((item for item in enriched if item.get("dev_cosine") is not None), key=lambda item: item["dev_cosine"], default=None)
    return {
        "row_count": len(enriched),
        "scatter_ready_rows": scatter,
        "correlations": correlations,
        "best_fit_checkpoint": best_fit,
        "best_dev_checkpoint": best_dev,
        "overfit_or_generalization_classifications": [
            {"candidate_id": item.get("candidate_id"), "classification": item.get("generalization", {}).get("classification")}
            for item in enriched
        ],
        "causation_claimed": False,
    }


cohort_analysis = analyze_cohort


__all__ = ["analyze_cohort", "cohort_analysis"]
