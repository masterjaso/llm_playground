"""Generalization-first structural/LM decision engine."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .generalization import classify_generalization, evaluate_structural_split
from .registry import DECISION_POLICY_V2, get_metric


def _value(metrics: Mapping[str, Any], *names: str) -> float | None:
    nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), Mapping) else metrics
    if not isinstance(nested, Mapping):
        return None
    for name in names:
        for key in (name, name.replace(".", "_"), name.rsplit(".", 1)[-1]):
            raw = nested.get(key)
            if raw is None or isinstance(raw, str):
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _has_nonfinite(metrics: Mapping[str, Any]) -> bool:
    value = _value(metrics, "quality.non_finite_token_count", "non_finite_token_count", "lm.non_finite_token_count")
    return value is not None and value > 0


def _slice_collapses(metrics: Mapping[str, Any]) -> list[str]:
    collapsed: list[str] = []
    for key, values in metrics.items():
        if not key.endswith("_slices") or not isinstance(values, Mapping):
            continue
        for identity, payload in values.items():
            if not isinstance(payload, Mapping) or payload.get("gate_eligible") is not True:
                continue
            overall = payload.get("overall")
            if overall == "RED" or payload.get("source_slice_collapse") is True:
                collapsed.append(f"{key}:{identity}")
                continue
            try:
                split = evaluate_structural_split(payload)
            except (TypeError, ValueError):
                continue
            if split["overall"] == "RED":
                collapsed.append(f"{key}:{identity}")
    return sorted(collapsed)


def _lm_gate(lm: Mapping[str, Any]) -> dict[str, Any]:
    metric_ids = (
        "lm.mean_forward_kl",
        "lm.p95_forward_kl",
        "lm.top1_agreement",
        "lm.top5_set_recall",
        "lm.teacher_top5_mass_retention",
        "lm.high_margin_top1_flip_rate",
        "lm.absolute_nll_delta",
        "lm.relative_nll_increase",
        "lm.non_finite_token_count",
    )
    statuses: dict[str, str] = {}
    missing: list[str] = []
    red: list[str] = []
    yellow: list[str] = []
    for metric_id in metric_ids:
        spec = get_metric(metric_id)
        value = _value(lm, metric_id, spec.serialization_field)
        if value is None:
            missing.append(metric_id)
            continue
        if spec.direction == "higher-is-better":
            status = "GREEN" if value >= float(spec.green) else "YELLOW" if spec.yellow is not None and value >= float(spec.yellow) else "RED"
        else:
            status = "GREEN" if value <= float(spec.green) else "YELLOW" if spec.yellow is not None and value <= float(spec.yellow) else "RED"
        statuses[metric_id] = status
        (red if status == "RED" else yellow if status == "YELLOW" else []).append(metric_id)
    if missing or red:
        overall = "RED"
    elif yellow:
        overall = "YELLOW"
    else:
        overall = "GREEN"
    return {"overall": overall, "statuses": statuses, "missing_metrics": missing, "red_metrics": red, "yellow_metrics": yellow}


def _override_check(dev_metrics: Mapping[str, Any], *, lm: Mapping[str, Any] | None, source_slice_collapse: Sequence[str], protected_confirmation: bool) -> dict[str, Any]:
    cosine = _value(dev_metrics, "structural.cosine_similarity", "cosine_similarity", "cosine")
    nmse = _value(dev_metrics, "structural.normalized_mse", "normalized_mse", "nmse")
    misses: list[str] = []
    envelopes = DECISION_POLICY_V2["override_envelopes"]
    if cosine is not None and envelopes["structural.cosine_similarity"]["lower_exclusive"] <= cosine < envelopes["structural.cosine_similarity"]["upper_exclusive"]:
        misses.append("structural.cosine_similarity")
    if nmse is not None and envelopes["structural.normalized_mse"]["lower_exclusive"] < nmse <= envelopes["structural.normalized_mse"]["upper_inclusive"]:
        misses.append("structural.normalized_mse")
    lm_gate = _lm_gate(lm) if lm is not None else None
    eligible = bool(
        len(misses) == 1
        and not source_slice_collapse
        and lm_gate is not None
        and lm_gate["overall"] == "GREEN"
        and not _has_nonfinite(dev_metrics)
        and (_value(dev_metrics, "routing.learned_load_cv", "learned_load_cv", "load_cv") is not None)
        and (_value(dev_metrics, "routing.dead_expert_count", "dead_expert_count", "dead_experts") == 0)
    )
    return {
        "requested": bool(misses),
        "eligible": eligible,
        "override_metric": misses[0] if len(misses) == 1 else None,
        "misses": misses,
        "envelope": envelopes.get(misses[0]) if len(misses) == 1 else None,
        "protected_confirmation_required": bool(eligible),
        "protected_confirmation_satisfied": bool(eligible and protected_confirmation),
        "lm_gate": lm_gate,
        "reason": None if eligible else ("exactly one narrow overrideable miss plus all structural/LM conditions required" if misses else "no narrow override envelope observed"),
    }


def decide_candidate(
    fit_train: Mapping[str, Any],
    fit_dev: Mapping[str, Any],
    *,
    lm_metrics: Mapping[str, Any] | None = None,
    identity_checks: Mapping[str, Any] | None = None,
    evidence_checks: Mapping[str, Any] | None = None,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
    protected_confirmation: bool = False,
    source_domain_slices: Mapping[str, Mapping[str, Any]] | None = None,
    oracle_metrics: Mapping[str, Any] | None = None,
    override_requested: bool = False,
) -> dict[str, Any]:
    """Apply the required deterministic priority order; never form a score."""

    identity_checks = dict(identity_checks or {})
    evidence_checks = dict(evidence_checks or {})
    failed_identity = sorted(str(key) for key, value in identity_checks.items() if value is False)
    failed_evidence = sorted(str(key) for key, value in evidence_checks.items() if value is False)
    if failed_identity:
        return _decision_base("BLOCKED_IDENTITY", "REJECT", failed_identity, fit_train, fit_dev, lm_metrics, oracle_metrics)
    if failed_evidence:
        return _decision_base("INVALID_EVIDENCE", "REJECT", failed_evidence, fit_train, fit_dev, lm_metrics, oracle_metrics)
    if _has_nonfinite(fit_train) or _has_nonfinite(fit_dev) or (lm_metrics is not None and _has_nonfinite(lm_metrics)):
        return _decision_base("NON_FINITE_EVIDENCE", "REJECT", ["non_finite_token_count"], fit_train, fit_dev, lm_metrics, oracle_metrics)

    fit_gate = evaluate_structural_split(fit_train)
    dev_gate = evaluate_structural_split(fit_dev)
    health_failures: list[str] = []
    learned_load = _value(fit_dev, "routing.learned_load_cv", "learned_load_cv", "load_cv")
    dead = _value(fit_dev, "routing.dead_expert_count", "dead_expert_count", "dead_experts")
    if learned_load is None:
        health_failures.append("routing.learned_load_cv:missing")
    elif learned_load > float(get_metric("routing.learned_load_cv").green):
        health_failures.append("routing.learned_load_cv")
    if dead is None:
        health_failures.append("routing.dead_expert_count:missing")
    elif dead > 0:
        health_failures.append("routing.dead_expert_count")
    if health_failures:
        return _decision_base("ROUTER_HEALTH_REJECT", "REJECT", health_failures, fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate)

    gaps = classify_generalization(fit_train, fit_dev, trajectory=trajectory, source_domain_slices=source_domain_slices)
    source_collapses = sorted(set(gaps.get("source_slice_collapse", [])) | set(_slice_collapses(fit_dev)))
    if source_collapses:
        return _decision_base("SOURCE_SLICE_COLLAPSE", "REJECT", source_collapses, fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate, gaps=gaps, source_collapses=source_collapses)
    if dev_gate["overall"] == "RED":
        return _decision_base("GENERALIZATION_REJECT" if fit_gate["overall"] == "GREEN" else "STRUCTURAL_FRESH_RED", "REJECT", dev_gate["red_metrics"] + dev_gate["missing_metrics"], fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate, gaps=gaps)

    lm_gate = _lm_gate(lm_metrics) if lm_metrics is not None else None
    if lm_gate is not None and lm_gate["overall"] == "RED":
        return _decision_base("LM_VETO", "REJECT", lm_gate["red_metrics"] + lm_gate["missing_metrics"], fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate, gaps=gaps, lm_gate=lm_gate)
    override = _override_check(fit_dev, lm=lm_metrics, source_slice_collapse=source_collapses, protected_confirmation=protected_confirmation)
    if override_requested and not override["eligible"]:
        return _decision_base("OVERRIDE_NOT_ELIGIBLE", "REJECT", [override["reason"]], fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate, gaps=gaps, lm_gate=lm_gate, override=override, source_collapses=source_collapses)
    if override["eligible"] and dev_gate["overall"] == "YELLOW":
        if protected_confirmation:
            decision = "PROMOTE_OVERRIDE"
            status = "PROMOTION_OVERRIDE_CONFIRMED"
            reason = []
        else:
            decision = "OVERRIDE_PENDING_PROTECTED_CONFIRMATION"
            status = "PROTECTED_CONFIRMATION_REQUIRED"
            reason = ["authorized untouched protected confirmation required"]
    else:
        decision = "PROMOTE" if dev_gate["overall"] == "GREEN" and (lm_gate is None or lm_gate["overall"] == "GREEN") else "RESEARCH_ONLY"
        status = "STRUCTURAL_LM_GREEN" if decision == "PROMOTE" else "STRUCTURAL_YELLOW"
        reason = []
    return _decision_base(status, decision, reason, fit_train, fit_dev, lm_metrics, oracle_metrics, fit_gate=fit_gate, dev_gate=dev_gate, gaps=gaps, lm_gate=lm_gate, override=override, source_collapses=source_collapses)


def _decision_base(
    status: str,
    decision: str,
    veto_reason: Sequence[str],
    fit_train: Mapping[str, Any],
    fit_dev: Mapping[str, Any],
    lm_metrics: Mapping[str, Any] | None,
    oracle_metrics: Mapping[str, Any] | None,
    *,
    fit_gate: Mapping[str, Any] | None = None,
    dev_gate: Mapping[str, Any] | None = None,
    gaps: Mapping[str, Any] | None = None,
    lm_gate: Mapping[str, Any] | None = None,
    override: Mapping[str, Any] | None = None,
    source_collapses: Sequence[str] = (),
) -> dict[str, Any]:
    fit_gate = dict(fit_gate or evaluate_structural_split(fit_train))
    dev_gate = dict(dev_gate or evaluate_structural_split(fit_dev))
    gaps = dict(gaps or classify_generalization(fit_train, fit_dev))
    return {
        "status": status,
        "decision": decision,
        "veto_reason": list(veto_reason),
        "fit_train_class": fit_gate["overall"],
        "fit_dev_class": dev_gate["overall"],
        "generalization_classification": gaps.get("classification"),
        "fit_train_gate": fit_gate,
        "fit_dev_gate": dev_gate,
        "generalization": gaps,
        "lm_evaluation_eligibility": "ELIGIBLE" if dev_gate["overall"] == "GREEN" and not source_collapses else "NOT_ELIGIBLE",
        "lm_gate": dict(lm_gate) if lm_gate is not None else {"overall": "NOT_EVALUATED"},
        "override_requested": bool((override or {}).get("requested", False)),
        "override_eligible": bool((override or {}).get("eligible", False)),
        "override_metric": (override or {}).get("override_metric"),
        "override_envelope": (override or {}).get("envelope"),
        "protected_confirmation_required": bool((override or {}).get("protected_confirmation_required", False)),
        "protected_confirmation_satisfied": bool((override or {}).get("protected_confirmation_satisfied", False)),
        "source_slice_collapses": list(source_collapses),
        "oracle_evidence_present": oracle_metrics is not None,
        "oracle_evidence_authorizes_advancement": False,
        "fit_dev_averaged": False,
        "weighted_aggregate_score": None,
        "ranking_key": ranking_key({"decision": decision, "fit_dev_class": dev_gate["overall"], "lm_gate": lm_gate or {}, "fit_dev": fit_dev}),
    }


def ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the requested lexicographic FIT-DEV-first ranking key."""

    structural_rank = {"GREEN": 0, "YELLOW": 1, "RED": 2, "INSUFFICIENT_EVIDENCE": 3}
    lm_rank = {"GREEN": 0, "YELLOW": 1, "RED": 2, "NOT_EVALUATED": 3}
    fit_dev = candidate.get("fit_dev", candidate.get("fit_dev_metrics", {}))
    lm = candidate.get("lm_gate", candidate.get("lm_metrics", {}))
    if not isinstance(fit_dev, Mapping):
        fit_dev = {}
    if not isinstance(lm, Mapping):
        lm = {}
    decision = str(candidate.get("decision", ""))
    return (
        bool(candidate.get("source_slice_collapses")),
        structural_rank.get(str(candidate.get("fit_dev_class", "INSUFFICIENT_EVIDENCE")), 3),
        lm_rank.get(str(lm.get("overall", "NOT_EVALUATED")), 3),
        float(_value(candidate.get("lm_metrics", lm), "lm.excess_mean_forward_kl", "excess_mean_forward_kl") if _value(candidate.get("lm_metrics", lm), "lm.excess_mean_forward_kl", "excess_mean_forward_kl") is not None else float("inf")),
        float(_value(candidate.get("lm_metrics", lm), "lm.high_margin_top1_flip_rate", "high_margin_top1_flip_rate") if _value(candidate.get("lm_metrics", lm), "lm.high_margin_top1_flip_rate", "high_margin_top1_flip_rate") is not None else float("inf")),
        float(_value(candidate.get("lm_metrics", lm), "lm.relative_nll_increase", "relative_nll_increase") if _value(candidate.get("lm_metrics", lm), "lm.relative_nll_increase", "relative_nll_increase") is not None else float("inf")),
        -float(_value(candidate.get("lm_metrics", lm), "lm.teacher_top5_mass_retention", "teacher_top5_mass_retention") if _value(candidate.get("lm_metrics", lm), "lm.teacher_top5_mass_retention", "teacher_top5_mass_retention") is not None else float("-inf")),
        -float(_value(candidate.get("lm_metrics", lm), "lm.top1_agreement", "top1_agreement") if _value(candidate.get("lm_metrics", lm), "lm.top1_agreement", "top1_agreement") is not None else float("-inf")),
        -float(_value(fit_dev, "structural.cosine_similarity", "cosine_similarity", "cosine") if _value(fit_dev, "structural.cosine_similarity", "cosine_similarity", "cosine") is not None else float("-inf")),
        float(_value(fit_dev, "structural.normalized_mse", "normalized_mse", "nmse") if _value(fit_dev, "structural.normalized_mse", "normalized_mse", "nmse") is not None else float("inf")),
        float(_value(fit_dev, "routing.learned_load_cv", "learned_load_cv", "load_cv") if _value(fit_dev, "routing.learned_load_cv", "learned_load_cv", "load_cv") is not None else float("inf")),
        decision,
    )


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in sorted(candidates, key=ranking_key)]


evaluate_candidate_decision = decide_candidate


__all__ = ["decide_candidate", "evaluate_candidate_decision", "rank_candidates", "ranking_key"]
