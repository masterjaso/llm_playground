"""Deterministic contract/meta-validation helpers for V2 evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .registry import METRIC_REGISTRY, TEST_MATRIX, canonical_metric_id, get_metric, validate_metric_registry


STRUCTURAL_RECEIPT_FIELDS = frozenset(
    {
        "receipt_type", "schema_version", "metric_policy_version", "policy_hash", "status", "evidence_class", "source_model", "source_revision", "layer", "attention_and_cycle_identity", "candidate", "fit_train", "fit_dev", "generalization", "target_norm_slices", "hard_token_slices", "source_domain_slices", "routing", "sample_counts", "generalization_classification", "lm_evaluation_eligibility", "source_receipt_lineage", "code_science_identity", "runtime_lock_identity", "receipt_sha256", "receipt_id",
    }
)
LM_RECEIPT_FIELDS = frozenset(
    {
        "receipt_type", "schema_version", "metric_policy_version", "policy_hash", "status", "source_model", "source_revision", "candidate", "layer", "routing_evidence_class", "dataset", "dense_numerical_baseline", "raw_metrics", "baseline_adjusted_metrics", "confidence_intervals", "per_source_metrics", "per_domain_metrics", "threshold_classification", "combined_structural_lm_decision", "veto_reason", "override_requested", "override_eligible", "override_metric", "override_envelope", "protected_confirmation_requirement", "resource_measurements", "source_receipt_lineage", "code_science_identity", "runtime_lock_identity", "receipt_sha256", "receipt_id",
    }
)


def validate_emitted_metric_ids(metric_ids: Iterable[str]) -> dict[str, Any]:
    unknown: list[str] = []
    for metric_id in metric_ids:
        try:
            get_metric(str(metric_id))
        except KeyError:
            unknown.append(str(metric_id))
    return {"valid": not unknown, "unknown_metric_ids": sorted(set(unknown))}


def validate_decision_metric_references(metric_ids: Iterable[str]) -> dict[str, Any]:
    return validate_emitted_metric_ids(metric_ids)


def validate_receipt_fields(receipt: Mapping[str, Any], *, receipt_kind: str) -> dict[str, Any]:
    expected = STRUCTURAL_RECEIPT_FIELDS if receipt_kind in {"structural", "structural-generalization"} else LM_RECEIPT_FIELDS
    missing = sorted(expected - set(receipt))
    documented_extra = sorted(set(receipt) - expected)
    # Extra fields are allowed only when explicitly namespaced so a new field
    # cannot silently evade the schema/test matrix.
    undocumented = [field for field in documented_extra if not (field.startswith("x_") or field in {"fit_dev_averaged", "weighted_aggregate_score", "exact_forward_kl_direction", "logarithm_base", "full_vocabulary_kl", "training_objective"})]
    return {"valid": not missing and not undocumented, "missing_fields": missing, "undocumented_fields": undocumented, "expected_fields": sorted(expected)}


def validate_threshold_policy(*, registry: Mapping[str, Any] = METRIC_REGISTRY) -> dict[str, Any]:
    registry_result = validate_metric_registry(registry, test_matrix=TEST_MATRIX)
    hashes = {str(spec.policy_version): [] for spec in registry.values()}
    for spec in registry.values():
        hashes.setdefault(str(spec.policy_version), []).append((spec.metric_id, spec.green, spec.yellow, spec.direction, spec.veto_capability, spec.override_eligible))
    return {"valid": bool(registry_result["valid"]), "registry": registry_result, "threshold_groups": {key: sorted(value) for key, value in hashes.items()}}


def validate_evaluation_contracts(
    *,
    emitted_metric_ids: Iterable[str] = (),
    decision_metric_ids: Iterable[str] = (),
    structural_receipt: Mapping[str, Any] | None = None,
    lm_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = validate_metric_registry()
    emitted = validate_emitted_metric_ids(emitted_metric_ids)
    decisions = validate_decision_metric_references(decision_metric_ids)
    receipts = {}
    if structural_receipt is not None:
        receipts["structural"] = validate_receipt_fields(structural_receipt, receipt_kind="structural")
    if lm_receipt is not None:
        receipts["lm"] = validate_receipt_fields(lm_receipt, receipt_kind="lm")
    errors = list(registry["errors"])
    errors.extend(f"unknown emitted metric: {item}" for item in emitted["unknown_metric_ids"])
    errors.extend(f"unknown decision metric: {item}" for item in decisions["unknown_metric_ids"])
    errors.extend(f"receipt {kind}: {item}" for kind, result in receipts.items() for item in result.get("missing_fields", []) + result.get("undocumented_fields", []))
    return {"valid": not errors, "errors": errors, "registry": registry, "emitted_metrics": emitted, "decision_metrics": decisions, "receipts": receipts, "test_matrix": {key: list(value) for key, value in TEST_MATRIX.items()}}


meta_validate = validate_evaluation_contracts


__all__ = ["validate_decision_metric_references", "validate_evaluation_contracts", "validate_emitted_metric_ids", "validate_receipt_fields", "validate_threshold_policy", "meta_validate"]
