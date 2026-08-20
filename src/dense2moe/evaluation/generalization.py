"""FIT-TRAIN/FIT-DEV generalization arithmetic and classifications."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .registry import (
    DECISION_POLICY_V2,
    INSUFFICIENT_EVIDENCE,
    NOT_AVAILABLE,
    canonical_metric_id,
    get_metric,
)


_ALIASES = {
    "cosine": "cosine_similarity",
    "cos": "cosine_similarity",
    "normalized_mse": "normalized_mse",
    "nmse": "normalized_mse",
    "load_cv": "learned_load_cv",
    "loadcv": "learned_load_cv",
    "dead_experts": "dead_expert_count",
    "p95_relative_norm_error": "p95_abs_relative_norm_error",
    "relative_norm_error": "target_relative_norm_error",
}


def _metric_value(metrics: Mapping[str, Any], name: str) -> float | None:
    values = metrics.get("metrics") if isinstance(metrics.get("metrics"), Mapping) else metrics
    if not isinstance(values, Mapping):
        return None
    candidates = (name, _ALIASES.get(name, name))
    for key in candidates:
        value = values.get(key)
        if value is None or isinstance(value, str):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _metric_status(metric_id: str, metrics: Mapping[str, Any]) -> str:
    spec = get_metric(metric_id)
    value = _metric_value(metrics, spec.serialization_field)
    if value is None:
        # Accept the unprefixed aliases used by direct structural metrics.
        value = _metric_value(metrics, spec.metric_id.rsplit(".", 1)[-1])
    if value is None:
        return INSUFFICIENT_EVIDENCE
    if spec.green is None:
        return "diagnostic"
    if spec.direction == "higher-is-better":
        if value >= spec.green:
            return "GREEN"
        if spec.yellow is not None and value >= spec.yellow:
            return "YELLOW"
        return "RED"
    if value <= spec.green:
        return "GREEN"
    if spec.yellow is not None and value <= spec.yellow:
        return "YELLOW"
    return "RED"


def evaluate_structural_split(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a split using only its absolute structural metrics."""

    gate_metric_ids = (
        "structural.cosine_similarity",
        "structural.normalized_mse",
        "structural.target_relative_norm_error",
        "structural.mean_prediction_to_target_norm_ratio",
        "structural.p95_abs_relative_norm_error",
        "routing.learned_load_cv",
        "routing.dead_expert_count",
        "quality.dropped_token_count",
        "quality.invalid_token_count",
        "quality.non_finite_token_count",
    )
    statuses = {metric_id: _metric_status(metric_id, metrics) for metric_id in gate_metric_ids}
    missing = [metric_id for metric_id, status in statuses.items() if status == INSUFFICIENT_EVIDENCE]
    red = [metric_id for metric_id, status in statuses.items() if status == "RED"]
    yellow = [metric_id for metric_id, status in statuses.items() if status == "YELLOW"]
    if missing or red:
        overall = "RED"
    elif yellow:
        overall = "YELLOW"
    else:
        overall = "GREEN"
    return {"overall": overall, "statuses": statuses, "missing_metrics": missing, "red_metrics": red, "yellow_metrics": yellow}


def compute_generalization_gaps(
    fit_metrics: Mapping[str, Any],
    dev_metrics: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute diagnostics without replacing absolute DEV metrics."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    fit_cosine = _metric_value(fit_metrics, "cosine_similarity")
    dev_cosine = _metric_value(dev_metrics, "cosine_similarity")
    fit_nmse = _metric_value(fit_metrics, "normalized_mse")
    dev_nmse = _metric_value(dev_metrics, "normalized_mse")
    fit_norm = _metric_value(fit_metrics, "target_relative_norm_error")
    dev_norm = _metric_value(dev_metrics, "target_relative_norm_error")
    fit_load = _metric_value(fit_metrics, "learned_load_cv")
    dev_load = _metric_value(dev_metrics, "learned_load_cv")
    fit_dead = _metric_value(fit_metrics, "dead_expert_count")
    dev_dead = _metric_value(dev_metrics, "dead_expert_count")
    fit_health = _metric_value(fit_metrics, "routing_health")
    dev_health = _metric_value(dev_metrics, "routing_health")

    def delta(dev: float | None, fit: float | None) -> float | None:
        return None if dev is None or fit is None else float(dev - fit)

    result: dict[str, Any] = {
        "epsilon": float(epsilon),
        "denominator_policy": "nmse_ratio=DEV_NMSE/max(FIT_NMSE,epsilon); additive gaps are DEV-FIT except cosine gap FIT-DEV",
        "cosine_gap": None if fit_cosine is None or dev_cosine is None else float(fit_cosine - dev_cosine),
        "absolute_nmse_increase": delta(dev_nmse, fit_nmse),
        "nmse_ratio": None if fit_nmse is None or dev_nmse is None else float(dev_nmse / max(fit_nmse, epsilon)),
        "relative_norm_error_increase": delta(dev_norm, fit_norm),
        "load_cv_change": delta(dev_load, fit_load),
        "dead_expert_change": delta(dev_dead, fit_dead),
        "routing_health_change": delta(dev_health, fit_health),
        "fit_class": evaluate_structural_split(fit_metrics)["overall"],
        "dev_class": evaluate_structural_split(dev_metrics)["overall"],
        "fit_metrics": dict(fit_metrics),
        "dev_metrics": dict(dev_metrics),
    }
    fit_slices = {key: value for key, value in fit_metrics.items() if key.endswith("_slices") and isinstance(value, Mapping)}
    dev_slices = {key: value for key, value in dev_metrics.items() if key.endswith("_slices") and isinstance(value, Mapping)}
    result["slice_deltas"] = {}
    for slice_key in sorted(set(fit_slices) & set(dev_slices)):
        result["slice_deltas"][slice_key] = {}
        for identity in sorted(set(fit_slices[slice_key]) & set(dev_slices[slice_key])):
            fit_slice = fit_slices[slice_key][identity]
            dev_slice = dev_slices[slice_key][identity]
            if not isinstance(fit_slice, Mapping) or not isinstance(dev_slice, Mapping):
                continue
            fit_cos = _metric_value(fit_slice, "cosine_similarity")
            dev_cos = _metric_value(dev_slice, "cosine_similarity")
            values: dict[str, Any] = {}
            if fit_cos is not None and dev_cos is not None:
                values["cosine_gap"] = float(fit_cos - dev_cos)
            for name in ("normalized_mse", "target_relative_norm_error", "learned_load_cv", "dead_expert_count"):
                fit_value = _metric_value(fit_slice, name)
                dev_value = _metric_value(dev_slice, name)
                if fit_value is not None and dev_value is not None:
                    values[f"{name}_change"] = float(dev_value - fit_value)
            values["fit_gate_eligible"] = bool(fit_slice.get("gate_eligible", False))
            values["dev_gate_eligible"] = bool(dev_slice.get("gate_eligible", False))
            values["gate_eligible"] = values["fit_gate_eligible"] and values["dev_gate_eligible"]
            result["slice_deltas"][slice_key][str(identity)] = values
    return result


def _trajectory_supports_overfit(trajectory: Sequence[Mapping[str, Any]] | None) -> bool:
    if not trajectory or len(trajectory) < 2:
        return False
    # The evidence must show a continuing FIT improvement, a DEV stall/reversal,
    # and a widening gap.  A worse final checkpoint alone is insufficient.
    fit_values = [_metric_value(item.get("fit", item), "cosine_similarity") for item in trajectory]
    dev_values = [_metric_value(item.get("dev", item), "cosine_similarity") for item in trajectory]
    fit_values = [value for value in fit_values if value is not None]
    dev_values = [value for value in dev_values if value is not None]
    if len(fit_values) < 2 or len(dev_values) < 2:
        return False
    fit_improves = fit_values[-1] > fit_values[0]
    dev_stalls_or_reverses = dev_values[-1] <= dev_values[-2]
    gap_start = fit_values[0] - dev_values[0]
    gap_end = fit_values[-1] - dev_values[-1]
    return bool(fit_improves and dev_stalls_or_reverses and gap_end > gap_start)


def classify_generalization(
    fit_metrics: Mapping[str, Any],
    dev_metrics: Mapping[str, Any],
    *,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
    source_domain_slices: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_slice_groups: int = 2,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Return explicit FIT/DEV classes and a conservative interpretation."""

    gaps = compute_generalization_gaps(fit_metrics, dev_metrics, epsilon=epsilon)
    fit_class = gaps["fit_class"]
    dev_class = gaps["dev_class"]
    if fit_class == "GREEN" and dev_class == "GREEN":
        label = "STABLE_CANDIDATE"
    elif fit_class == "GREEN" and dev_class == "YELLOW":
        label = "DATA_SPECIFIC_FIT_OR_DISTRIBUTION_SHIFT_SENSITIVITY"
    elif fit_class == "GREEN" and dev_class == "RED":
        label = "GENERALIZATION_REJECT"
    elif fit_class == "YELLOW" and dev_class == "GREEN":
        label = "INSPECT_TRAINING_METRIC"
    elif fit_class == "YELLOW" and dev_class == "YELLOW":
        label = "RESEARCH_ONLY"
    elif dev_class == "RED":
        label = "DISTRIBUTION_GENERALIZATION_FAILURE"
    else:
        label = "INSUFFICIENT_EVIDENCE"
    if _trajectory_supports_overfit(trajectory):
        label = "TRAINING_OVERFIT_SIGNAL"
    source_slice_collapse: list[str] = []
    if source_domain_slices:
        for identity, values in source_domain_slices.items():
            if not isinstance(values, Mapping):
                continue
            groups = int(values.get("independent_group_count", values.get("group_count", 0)) or 0)
            if groups < minimum_slice_groups or values.get("gate_eligible") is False:
                continue
            status = values.get("overall")
            if status == "RED" or values.get("source_slice_collapse") is True:
                source_slice_collapse.append(str(identity))
    if source_slice_collapse:
        label = "SOURCE_SLICE_COLLAPSE"
    return {
        **gaps,
        "fit_class": fit_class,
        "dev_class": dev_class,
        "classification": label,
        "source_slice_collapse": sorted(source_slice_collapse),
        "absolute_dev_governs": True,
        "fit_dev_averaged": False,
        "trajectory_supports_training_overfit": _trajectory_supports_overfit(trajectory),
    }


def build_generalization_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact FIT/DEV interpretation matrix without ranking scores."""

    matrix: list[dict[str, Any]] = []
    for row in rows:
        fit = row.get("fit_train", row.get("fit", {}))
        dev = row.get("fit_dev", row.get("dev", {}))
        if not isinstance(fit, Mapping) or not isinstance(dev, Mapping):
            matrix.append({"candidate_id": row.get("candidate_id"), "status": INSUFFICIENT_EVIDENCE})
            continue
        classification = classify_generalization(fit, dev, trajectory=row.get("trajectory"))
        matrix.append(
            {
                "candidate_id": row.get("candidate_id"),
                "design_id": row.get("design_id"),
                "topology": row.get("topology"),
                "fit_train_class": classification["fit_class"],
                "fit_dev_class": classification["dev_class"],
                "interpretation": classification["classification"],
                "cosine_gap": classification.get("cosine_gap"),
                "absolute_nmse_increase": classification.get("absolute_nmse_increase"),
                "nmse_ratio": classification.get("nmse_ratio"),
            }
        )
    return matrix


generalization_gaps = compute_generalization_gaps
generalization_classification = classify_generalization


__all__ = [
    "build_generalization_matrix",
    "classify_generalization",
    "compute_generalization_gaps",
    "evaluate_structural_split",
    "generalization_classification",
    "generalization_gaps",
]
