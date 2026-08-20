"""Shared V2 evaluation seams used by candidate runners.

The historical V2.3/V2.4 experiment receipts remain immutable and retain
their legacy schemas.  New runners should call these adapters so metric
computation, decisions, and receipt serialization all use the same registry
and policy hash rather than maintaining per-script copies of the contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .decision import decide_candidate
from .generalization import classify_generalization
from .lm import compute_lm_output_metrics
from .receipts import build_lm_output_receipt, build_structural_generalization_receipt
from .registry import POLICY_HASH, STRUCTURAL_POLICY_VERSION
from .structural import compute_structural_metrics


def evaluate_structural_pair_v2(
    *,
    teacher_fit_train: Any,
    candidate_fit_train: Any,
    teacher_fit_dev: Any,
    candidate_fit_dev: Any,
    source_model: Mapping[str, Any],
    candidate: Mapping[str, Any],
    layer: int,
    fit_train_options: Mapping[str, Any] | None = None,
    fit_dev_options: Mapping[str, Any] | None = None,
    fit_train_identity: Mapping[str, Any] | None = None,
    fit_dev_identity: Mapping[str, Any] | None = None,
    identity_checks: Mapping[str, Any] | None = None,
    evidence_checks: Mapping[str, Any] | None = None,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
    source_domain_slices: Mapping[str, Mapping[str, Any]] | None = None,
    oracle_metrics: Mapping[str, Any] | None = None,
    protected_confirmation: bool = False,
    override_requested: bool = False,
    source_receipt_lineage: Mapping[str, Any] | None = None,
    code_science_identity: Mapping[str, Any] | None = None,
    runtime_lock_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate independent FIT-TRAIN and FIT-DEV arrays under V2.

    The function intentionally accepts separate option mappings for the two
    splits.  It never concatenates, averages, or otherwise turns FIT and DEV
    into one promotion metric.  A caller can pass masking, metadata, and
    routing evidence through each option mapping accepted by
    :func:`compute_structural_metrics`.
    """

    fit_train = compute_structural_metrics(
        teacher_fit_train,
        candidate_fit_train,
        **dict(fit_train_options or {}),
    )
    fit_dev = compute_structural_metrics(
        teacher_fit_dev,
        candidate_fit_dev,
        **dict(fit_dev_options or {}),
    )
    generalization = classify_generalization(
        fit_train,
        fit_dev,
        trajectory=trajectory,
        source_domain_slices=source_domain_slices,
    )
    decision = decide_candidate(
        fit_train,
        fit_dev,
        identity_checks=identity_checks,
        evidence_checks=evidence_checks,
        trajectory=trajectory,
        protected_confirmation=protected_confirmation,
        source_domain_slices=source_domain_slices,
        oracle_metrics=oracle_metrics,
        override_requested=override_requested,
    )
    receipt = build_structural_generalization_receipt(
        source_model=source_model,
        layer=layer,
        candidate=candidate,
        fit_train=fit_train,
        fit_dev=fit_dev,
        generalization=generalization,
        source_receipt_lineage=source_receipt_lineage,
        code_science_identity=code_science_identity,
        runtime_lock_identity=runtime_lock_identity,
        fit_train_identity=fit_train_identity,
        fit_dev_identity=fit_dev_identity,
        lm_evaluation_eligibility={
            "status": decision.get("lm_evaluation_eligibility"),
            "reason": decision.get("veto_reason", []),
        },
    )
    return {
        "metric_policy_version": STRUCTURAL_POLICY_VERSION,
        "policy_hash": POLICY_HASH,
        "fit_train": fit_train,
        "fit_dev": fit_dev,
        "generalization": generalization,
        "decision": decision,
        "receipt": receipt,
    }


def evaluate_lm_pair_v2(
    *,
    teacher_logits: Any,
    candidate_logits: Any,
    source_model: Mapping[str, Any],
    candidate: Mapping[str, Any],
    layer: int,
    dataset: Mapping[str, Any],
    baseline_logits: Any | None = None,
    metric_options: Mapping[str, Any] | None = None,
    confidence_intervals: Mapping[str, Any] | None = None,
    threshold_classification: str = "NOT_CLASSIFIED",
    combined_decision: Mapping[str, Any] | None = None,
    veto_reason: str | None = None,
    override: Mapping[str, Any] | None = None,
    protected_confirmation_requirement: Mapping[str, Any] | None = None,
    source_receipt_lineage: Mapping[str, Any] | None = None,
    code_science_identity: Mapping[str, Any] | None = None,
    runtime_lock_identity: Mapping[str, Any] | None = None,
    resource_measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate paired teacher-forced logits and emit an LM V2 receipt.

    ``baseline_logits`` should be the dense-repeat output produced on the
    same frozen sequences/runtime.  When omitted, teacher logits are used as
    the numerical floor, which is useful for an exact identity fixture but
    should not replace a measured dense-repeat run in production.
    """

    options = dict(metric_options or {})
    if baseline_logits is not None:
        options["baseline_logits"] = baseline_logits
    elif "baseline_logits" not in options:
        options["baseline_logits"] = teacher_logits
    metrics = compute_lm_output_metrics(teacher_logits, candidate_logits, **options)
    receipt = build_lm_output_receipt(
        source_model=source_model,
        candidate=candidate,
        layer=layer,
        dataset=dataset,
        raw_metrics=metrics,
        dense_numerical_baseline={
            key: metrics[key]
            for key in metrics
            if key.startswith("dense_repeat_") or key.startswith("excess_")
        },
        confidence_intervals=confidence_intervals,
        threshold_classification=threshold_classification,
        combined_decision=combined_decision,
        veto_reason=veto_reason,
        override=override,
        protected_confirmation_requirement=protected_confirmation_requirement,
        source_receipt_lineage=source_receipt_lineage,
        code_science_identity=code_science_identity,
        runtime_lock_identity=runtime_lock_identity,
        resource_measurements=resource_measurements,
    )
    return {
        "metrics": metrics,
        "receipt": receipt,
    }


# Concise aliases for runner integrations that already use the ``evaluate_*``
# naming convention.
evaluate_structural_candidate_v2 = evaluate_structural_pair_v2
evaluate_lm_candidate_v2 = evaluate_lm_pair_v2


__all__ = [
    "evaluate_lm_candidate_v2",
    "evaluate_lm_pair_v2",
    "evaluate_structural_candidate_v2",
    "evaluate_structural_pair_v2",
]
