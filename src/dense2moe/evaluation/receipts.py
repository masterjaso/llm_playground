"""Versioned immutable structural-generalization and LM-output receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .registry import (
    LM_POLICY_VERSION,
    LM_SCHEMA_VERSION,
    METRIC_REGISTRY,
    NOT_APPLICABLE,
    NOT_AVAILABLE,
    NOT_COMPUTED,
    POLICY_HASH,
    STRUCTURAL_POLICY_VERSION,
    STRUCTURAL_SCHEMA_VERSION,
    metric_policy_hash,
)


class ReceiptValidationError(ValueError):
    """Raised when a new receipt is malformed or policy-incompatible."""


class ImmutableReceiptError(ValueError):
    """Raised when a different receipt would overwrite an existing one."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def receipt_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    body.pop("receipt_id", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    result["receipt_sha256"] = receipt_sha256(result)
    result["receipt_id"] = result["receipt_sha256"]
    return result


def _identity(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _metric_status_fields(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_id, value in (metrics or {}).items():
        key = str(metric_id)
        if isinstance(value, Mapping) and "status" in value:
            result[key] = dict(value)
        elif value is None:
            result[key] = {"status": NOT_AVAILABLE, "value": None, "metric_id": key}
        elif isinstance(value, str) and value in {NOT_COMPUTED, NOT_AVAILABLE, NOT_APPLICABLE}:
            result[key] = {"status": value, "value": None, "metric_id": key}
        else:
            result[key] = {"status": "COMPUTED", "value": value, "metric_id": key}
    return result


def build_structural_generalization_receipt(
    *,
    source_model: Mapping[str, Any],
    layer: int,
    candidate: Mapping[str, Any],
    fit_train: Mapping[str, Any],
    fit_dev: Mapping[str, Any],
    generalization: Mapping[str, Any],
    evidence_class: str = "structural-ffn",
    source_receipt_lineage: Mapping[str, Any] | None = None,
    code_science_identity: Mapping[str, Any] | None = None,
    runtime_lock_identity: Mapping[str, Any] | None = None,
    lm_evaluation_eligibility: Mapping[str, Any] | None = None,
    fit_train_identity: Mapping[str, Any] | None = None,
    fit_dev_identity: Mapping[str, Any] | None = None,
    status: str = "STRUCTURAL_GENERALIZATION_EVALUATED",
) -> dict[str, Any]:
    fit_train_identity = dict(fit_train_identity or {})
    fit_dev_identity = dict(fit_dev_identity or {})
    payload = {
        "receipt_type": "dense2moe-ffn-structural-generalization-v2",
        "schema_version": STRUCTURAL_SCHEMA_VERSION,
        "metric_policy_version": STRUCTURAL_POLICY_VERSION,
        "policy_hash": POLICY_HASH,
        "status": status,
        "evidence_class": evidence_class,
        "source_model": _identity(source_model),
        "source_revision": source_model.get("revision"),
        "layer": int(layer),
        "attention_and_cycle_identity": {"attention_unchanged": True, "cycle_unchanged": True},
        "candidate": _identity(candidate),
        "fit_train": {
            "split": "FIT-TRAIN",
            "metrics": _metric_status_fields(fit_train),
            "manifest": fit_train_identity.get("manifest", fit_train.get("manifest") if isinstance(fit_train, Mapping) else None),
            "manifest_sha256": fit_train_identity.get("manifest_sha256", fit_train.get("manifest_sha256") if isinstance(fit_train, Mapping) else None),
            "data_identity": fit_train_identity,
            "sample_counts": {key: fit_train.get(key) for key in ("scored_token_count", "independent_group_count") if isinstance(fit_train, Mapping) and key in fit_train},
        },
        "fit_dev": {
            "split": "FIT-DEV",
            "metrics": _metric_status_fields(fit_dev),
            "manifest": fit_dev_identity.get("manifest", fit_dev.get("manifest") if isinstance(fit_dev, Mapping) else None),
            "manifest_sha256": fit_dev_identity.get("manifest_sha256", fit_dev.get("manifest_sha256") if isinstance(fit_dev, Mapping) else None),
            "data_identity": fit_dev_identity,
            "sample_counts": {key: fit_dev.get(key) for key in ("scored_token_count", "independent_group_count") if isinstance(fit_dev, Mapping) and key in fit_dev},
        },
        "generalization": dict(generalization),
        "target_norm_slices": dict(fit_dev.get("target_norm_buckets", {})) if isinstance(fit_dev, Mapping) else {},
        "hard_token_slices": dict(fit_dev.get("hard_token_buckets", {})) if isinstance(fit_dev, Mapping) else {},
        "source_domain_slices": {
            "fit_train": {key: value for key, value in fit_train.items() if key.endswith("_slices")} if isinstance(fit_train, Mapping) else {},
            "fit_dev": {key: value for key, value in fit_dev.items() if key.endswith("_slices")} if isinstance(fit_dev, Mapping) else {},
        },
        "routing": {
            "learned_router": {key: fit_dev.get(key) for key in ("learned_load_cv", "dead_expert_count", "expert_utilization", "routing_entropy") if isinstance(fit_dev, Mapping)},
            "oracle": {key: fit_dev.get(key) for key in ("oracle_load_cv", "oracle_dead_expert_count", "oracle_expert_counts") if isinstance(fit_dev, Mapping)},
            "oracle_promotion_eligible": False,
        },
        "sample_counts": {
            "fit_train": {key: fit_train.get(key) for key in ("scored_token_count", "independent_group_count") if isinstance(fit_train, Mapping)},
            "fit_dev": {key: fit_dev.get(key) for key in ("scored_token_count", "independent_group_count") if isinstance(fit_dev, Mapping)},
        },
        "generalization_classification": generalization.get("classification", generalization.get("generalization_classification")),
        "lm_evaluation_eligibility": dict(lm_evaluation_eligibility or {"status": NOT_COMPUTED, "reason": "not yet evaluated"}),
        "source_receipt_lineage": _identity(source_receipt_lineage),
        "code_science_identity": _identity(code_science_identity),
        "runtime_lock_identity": _identity(runtime_lock_identity),
        "fit_dev_averaged": False,
        "weighted_aggregate_score": None,
    }
    return _with_hash(payload)


def build_lm_output_receipt(
    *,
    source_model: Mapping[str, Any],
    candidate: Mapping[str, Any],
    layer: int,
    dataset: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
    dense_numerical_baseline: Mapping[str, Any] | None = None,
    confidence_intervals: Mapping[str, Any] | None = None,
    threshold_classification: str = "NOT_CLASSIFIED",
    combined_decision: Mapping[str, Any] | None = None,
    veto_reason: str | None = None,
    override: Mapping[str, Any] | None = None,
    protected_confirmation_requirement: Mapping[str, Any] | None = None,
    routing_evidence_class: str = "learned-router",
    source_receipt_lineage: Mapping[str, Any] | None = None,
    code_science_identity: Mapping[str, Any] | None = None,
    runtime_lock_identity: Mapping[str, Any] | None = None,
    resource_measurements: Mapping[str, Any] | None = None,
    status: str = "LM_OUTPUT_EVALUATED",
) -> dict[str, Any]:
    payload = {
        "receipt_type": "dense2moe-layer-patch-lm-output-v2",
        "schema_version": LM_SCHEMA_VERSION,
        "metric_policy_version": LM_POLICY_VERSION,
        "policy_hash": POLICY_HASH,
        "status": status,
        "source_model": _identity(source_model),
        "source_revision": source_model.get("revision"),
        "candidate": _identity(candidate),
        "layer": int(layer),
        "routing_evidence_class": routing_evidence_class,
        "dataset": _identity(dataset),
        "dense_numerical_baseline": _identity(dense_numerical_baseline),
        "raw_metrics": _metric_status_fields(raw_metrics),
        "baseline_adjusted_metrics": {
            key: value for key, value in _metric_status_fields(raw_metrics).items() if "excess" in key or "baseline" in key or key.startswith("dense_repeat_")
        },
        "confidence_intervals": dict(confidence_intervals or {}),
        "per_source_metrics": dict(raw_metrics.get("source_slices", {})) if isinstance(raw_metrics, Mapping) else {},
        "per_domain_metrics": dict(raw_metrics.get("domain_slices", {})) if isinstance(raw_metrics, Mapping) else {},
        "threshold_classification": threshold_classification,
        "combined_structural_lm_decision": dict(combined_decision or {}),
        "veto_reason": veto_reason,
        "override_requested": bool((override or {}).get("requested", False)),
        "override_eligible": bool((override or {}).get("eligible", False)),
        "override_metric": (override or {}).get("override_metric"),
        "override_envelope": (override or {}).get("envelope"),
        "protected_confirmation_requirement": dict(protected_confirmation_requirement or {}),
        "resource_measurements": _identity(resource_measurements),
        "source_receipt_lineage": _identity(source_receipt_lineage),
        "code_science_identity": _identity(code_science_identity),
        "runtime_lock_identity": _identity(runtime_lock_identity),
        "exact_forward_kl_direction": "KL(teacher || candidate)",
        "logarithm_base": "natural",
        "full_vocabulary_kl": True,
        "fit_dev_averaged": False,
        "training_objective": None,
    }
    return _with_hash(payload)


def write_immutable_receipt(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Write once; identical replay is idempotent, different content fails."""

    target = Path(path)
    receipt = _with_hash(payload)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ImmutableReceiptError(f"existing receipt is unreadable: {target}") from exc
        if existing.get("receipt_sha256") == receipt.get("receipt_sha256"):
            return existing
        raise ImmutableReceiptError(f"refusing to overwrite immutable receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return receipt


def _validate_common(payload: Mapping[str, Any], *, receipt_type: str, policy_version: str, schema_version: int) -> list[str]:
    errors: list[str] = []
    if payload.get("receipt_type") != receipt_type:
        errors.append("receipt_type mismatch")
    if payload.get("schema_version") != schema_version:
        errors.append("schema_version mismatch")
    if payload.get("metric_policy_version") != policy_version:
        errors.append("metric_policy_version mismatch")
    expected_hash = metric_policy_hash()
    if payload.get("policy_hash") != expected_hash:
        errors.append("policy_hash mismatch")
    if payload.get("receipt_sha256") != receipt_sha256(payload):
        errors.append("receipt_sha256 mismatch")
    if payload.get("source_revision") is None:
        errors.append("source_revision missing")
    return errors


def validate_structural_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = _validate_common(payload, receipt_type="dense2moe-ffn-structural-generalization-v2", policy_version=STRUCTURAL_POLICY_VERSION, schema_version=STRUCTURAL_SCHEMA_VERSION)
    for field in ("source_model", "candidate", "fit_train", "fit_dev", "generalization", "routing", "generalization_classification", "source_receipt_lineage", "runtime_lock_identity"):
        if field not in payload:
            errors.append(f"missing field: {field}")
    if payload.get("fit_dev_averaged") is not False:
        errors.append("FIT/DEV averaging is forbidden")
    if payload.get("routing", {}).get("oracle_promotion_eligible") is not False:
        errors.append("oracle evidence cannot be promotion eligible")
    return {"valid": not errors, "errors": errors, "compatibility": "V2_COMPLETE" if not errors else "INVALID_V2"}


def validate_lm_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = _validate_common(payload, receipt_type="dense2moe-layer-patch-lm-output-v2", policy_version=LM_POLICY_VERSION, schema_version=LM_SCHEMA_VERSION)
    for field in ("source_model", "candidate", "dataset", "dense_numerical_baseline", "raw_metrics", "baseline_adjusted_metrics", "confidence_intervals", "combined_structural_lm_decision", "resource_measurements", "source_receipt_lineage", "runtime_lock_identity"):
        if field not in payload:
            errors.append(f"missing field: {field}")
    if payload.get("exact_forward_kl_direction") != "KL(teacher || candidate)":
        errors.append("forward KL direction must be teacher to candidate")
    if payload.get("logarithm_base") != "natural":
        errors.append("LM logarithms must be natural")
    if payload.get("full_vocabulary_kl") is not True:
        errors.append("gate-bearing LM KL must be full-vocabulary")
    return {"valid": not errors, "errors": errors, "compatibility": "V2_COMPLETE" if not errors else "INVALID_V2"}


def read_receipt(path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    compatibility = classify_receipt_compatibility(payload)
    if strict and compatibility["classification"] != "V2_COMPLETE":
        raise ReceiptValidationError(f"receipt is not V2-complete: {compatibility}")
    return payload


def classify_receipt_compatibility(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt_type = str(payload.get("receipt_type", ""))
    if receipt_type == "dense2moe-ffn-structural-generalization-v2":
        result = validate_structural_receipt(payload)
    elif receipt_type == "dense2moe-layer-patch-lm-output-v2":
        result = validate_lm_receipt(payload)
    else:
        missing = [
            "metric_policy_version",
            "policy_hash",
            "fit_train",
            "fit_dev",
            "generalization",
            "source_receipt_lineage",
        ]
        return {
            "classification": "LEGACY_REQUIRES_RECOMPUTATION",
            "legacy": True,
            "metrics_requiring_recomputation": missing,
            "reason": "legacy receipt lacks the V2 metric-policy contract; it remains readable but is not V2-complete",
        }
    return {"classification": result["compatibility"], "legacy": False, "metrics_requiring_recomputation": [], "validation": result}


validate_structural_generalization_receipt = validate_structural_receipt
validate_lm_output_receipt = validate_lm_receipt
write_receipt_once = write_immutable_receipt


__all__ = [
    "ImmutableReceiptError",
    "ReceiptValidationError",
    "build_lm_output_receipt",
    "build_structural_generalization_receipt",
    "classify_receipt_compatibility",
    "read_receipt",
    "receipt_sha256",
    "validate_lm_output_receipt",
    "validate_lm_receipt",
    "validate_structural_generalization_receipt",
    "validate_structural_receipt",
    "write_immutable_receipt",
    "write_receipt_once",
]
