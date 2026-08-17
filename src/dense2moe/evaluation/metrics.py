"""Explicit metric classification; thresholds are never silently weakened."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROMOTION_THRESHOLDS: dict[str, dict[str, Any]] = {
    "nmse": {"green": 0.05, "yellow": 0.08, "lower_is_better": True},
    "cosine": {"green": 0.98, "yellow": 0.96, "lower_is_better": False},
    "loadcv": {"green": 0.50, "yellow": 0.65, "lower_is_better": True},
    "dead_experts": {"green": 0.0, "yellow": 0.0, "lower_is_better": True},
    "oracle_regret": {"green": 0.10, "yellow": 0.15, "lower_is_better": True},
    "repeat_variation": {"green": 0.05, "yellow": 0.08, "lower_is_better": True},
    "median_norm_ratio_error": {"green": 0.05, "yellow": 0.10, "lower_is_better": True},
    "p95_relative_norm_error": {"green": 0.15, "yellow": 0.20, "lower_is_better": True},
}


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


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric must be numeric, got {value!r}") from exc


def amplitude_metrics(
    student: Any,
    teacher: Any,
    *,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Measure output amplitude preservation without changing tensor dtype.

    The returned norm ratio is ``||student|| / ||teacher||`` per row.  Both
    PyTorch tensors and NumPy-like arrays are supported; no gradients are
    retained by this diagnostic.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    try:
        import torch  # type: ignore

        student_tensor = student if isinstance(student, torch.Tensor) else torch.as_tensor(student)
        teacher_tensor = teacher if isinstance(teacher, torch.Tensor) else torch.as_tensor(teacher)
        if student_tensor.shape != teacher_tensor.shape:
            raise ValueError(f"student/teacher shape mismatch: {tuple(student_tensor.shape)} != {tuple(teacher_tensor.shape)}")
        student_tensor = student_tensor.detach().to(dtype=torch.float64)
        teacher_tensor = teacher_tensor.detach().to(dtype=torch.float64)
        if student_tensor.ndim == 0:
            student_tensor = student_tensor.reshape(1, 1)
            teacher_tensor = teacher_tensor.reshape(1, 1)
        else:
            student_tensor = student_tensor.reshape(student_tensor.shape[0], -1)
            teacher_tensor = teacher_tensor.reshape(teacher_tensor.shape[0], -1)
        student_norm = torch.linalg.vector_norm(student_tensor, dim=1)
        teacher_norm = torch.linalg.vector_norm(teacher_tensor, dim=1)
        ratio = student_norm / teacher_norm.clamp_min(epsilon)
        relative_error = (student_norm - teacher_norm).abs() / teacher_norm.clamp_min(epsilon)
        ratios = ratio.cpu().tolist()
        relative = relative_error.cpu().tolist()
    except ImportError:  # pragma: no cover - project ML environments provide torch
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("amplitude_metrics requires torch or numpy") from exc
        student_array = np.asarray(student, dtype=np.float64)
        teacher_array = np.asarray(teacher, dtype=np.float64)
        if student_array.shape != teacher_array.shape:
            raise ValueError(f"student/teacher shape mismatch: {student_array.shape} != {teacher_array.shape}")
        if student_array.ndim == 0:
            student_array = student_array.reshape(1, 1)
            teacher_array = teacher_array.reshape(1, 1)
        else:
            student_array = student_array.reshape(student_array.shape[0], -1)
            teacher_array = teacher_array.reshape(teacher_array.shape[0], -1)
        student_norm = np.linalg.norm(student_array, axis=1)
        teacher_norm = np.linalg.norm(teacher_array, axis=1)
        ratios = (student_norm / np.maximum(teacher_norm, epsilon)).tolist()
        relative = (np.abs(student_norm - teacher_norm) / np.maximum(teacher_norm, epsilon)).tolist()

    def percentile(values: Sequence[float], fraction: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    median_ratio = percentile(ratios, 0.5)
    p95_error = percentile(relative, 0.95)
    return {
        "samples": len(ratios),
        "median_norm_ratio": median_ratio,
        "median_norm_ratio_error": abs(median_ratio - 1.0),
        "p95_relative_norm_error": p95_error,
        "norm_ratio_range": [min(ratios) if ratios else 0.0, max(ratios) if ratios else 0.0],
        "per_sample_norm_ratio": ratios,
        "per_sample_relative_norm_error": relative,
        "epsilon": float(epsilon),
    }


def _canonical_metric_name(name: str) -> str:
    return str(name).casefold().replace(" ", "_").replace("-", "_")


def evaluate_promotion_metrics(
    metrics: Mapping[str, Any],
    *,
    domain_slices: Mapping[str, Mapping[str, Any]] | None = None,
    development_metrics: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the sealed external-promotion contract to one corpus.

    Missing metrics, dead experts, failed domain slices, or material
    development-to-external collapse make promotion non-green.  Domain slices
    are intentionally evaluated independently so a strong aggregate cannot
    conceal an agentic/code failure.
    """

    rules = {key: dict(value) for key, value in (thresholds or PROMOTION_THRESHOLDS).items()}
    canonical = {_canonical_metric_name(key): _as_float(value) for key, value in metrics.items()}
    required = tuple(rules)
    missing = [key for key in required if key not in canonical]
    base_metrics = {key: canonical[key] for key in required if key in canonical}
    base_gate = quality_gate(base_metrics, rules) if base_metrics else {"metrics": {}, "statuses": {}, "overall": "red"}
    domain_results: dict[str, Any] = {}
    for domain, values in (domain_slices or {}).items():
        domain_results[str(domain)] = evaluate_promotion_metrics(values, thresholds=rules)
    failed_domains = [domain for domain, result in domain_results.items() if result["overall"] == "red"]
    degradation: dict[str, float] = {}
    if development_metrics is not None:
        dev = {_canonical_metric_name(key): _as_float(value) for key, value in development_metrics.items()}
        for key in ("nmse", "loadcv", "oracle_regret", "repeat_variation", "p95_relative_norm_error"):
            if key in canonical and key in dev:
                degradation[key] = canonical[key] - dev[key]
        if "cosine" in canonical and "cosine" in dev:
            degradation["cosine"] = dev["cosine"] - canonical["cosine"]
    material_collapse = any(value > 0.10 for key, value in degradation.items() if key != "cosine") or degradation.get("cosine", 0.0) > 0.05
    overall = "green" if not missing and base_gate["overall"] == "green" and not failed_domains and not material_collapse else "red" if missing or failed_domains or material_collapse or base_gate["overall"] == "red" else "yellow"
    return {
        "metrics": dict(metrics),
        "thresholds": rules,
        "gate": base_gate,
        "domain_slices": domain_results,
        "missing_metrics": missing,
        "failed_domains": failed_domains,
        "development_to_external_degradation": degradation,
        "material_collapse": material_collapse,
        "overall": overall,
        "promotion_decision": "PROMOTE" if overall == "green" else "REJECT",
        "external_data_untouched": True,
    }
