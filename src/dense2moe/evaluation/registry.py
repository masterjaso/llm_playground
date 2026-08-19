"""Canonical, versioned evaluation metric and decision-policy contracts.

The legacy :mod:`dense2moe.evaluation.metrics` dictionaries remain available
for historical receipts.  New evaluation code uses the registry in this
module so a metric's formula, direction, threshold, evidence class, and
serialization identity have one authoritative definition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

STRUCTURAL_POLICY_VERSION = "dense2moe-ffn-structural-generalization-v2"
LM_POLICY_VERSION = "dense2moe-layer-patch-lm-output-v2"
STRUCTURAL_SCHEMA_VERSION = 2
LM_SCHEMA_VERSION = 2

NOT_COMPUTED = "NOT_COMPUTED"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
MISSING_METRIC_STATUSES = frozenset(
    {NOT_COMPUTED, NOT_AVAILABLE, NOT_APPLICABLE, INSUFFICIENT_EVIDENCE}
)


@dataclass(frozen=True)
class MetricSpec:
    """All semantics needed to compute, gate, and serialize one metric."""

    metric_id: str
    name: str
    policy_version: str
    direction: str
    unit: str
    aggregation: str
    mask_semantics: str
    numerical_dtype: str
    epsilon_policy: str
    evidence_class: str
    routing_modes: tuple[str, ...]
    splits_or_tiers: tuple[str, ...]
    slice_eligible: bool
    minimum_sample_count: int
    minimum_group_count: int
    green: float | None = None
    yellow: float | None = None
    gate_bearing: bool = False
    override_eligible: bool = False
    veto_capability: str = "none"
    serialization_field: str = ""
    schema_version: int = 2
    formula: str = ""
    deprecation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"higher-is-better", "lower-is-better", "diagnostic"}:
            raise ValueError(f"unsupported metric direction: {self.direction!r}")
        if self.minimum_sample_count < 0 or self.minimum_group_count < 0:
            raise ValueError("minimum evidence counts must be non-negative")
        if self.gate_bearing and self.green is None:
            raise ValueError(f"gate-bearing metric {self.metric_id} needs a GREEN threshold")
        if self.veto_capability not in {"none", "hard", "non_overridable"}:
            raise ValueError(f"unsupported veto capability: {self.veto_capability!r}")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["routing_modes"] = list(self.routing_modes)
        value["splits_or_tiers"] = list(self.splits_or_tiers)
        value["deprecation_metadata"] = dict(self.deprecation_metadata)
        return value


def _structural(
    metric_id: str,
    name: str,
    *,
    direction: str,
    unit: str,
    aggregation: str,
    formula: str,
    green: float | None = None,
    yellow: float | None = None,
    gate_bearing: bool = False,
    override_eligible: bool = False,
    veto_capability: str = "none",
    slice_eligible: bool = True,
    minimum_sample_count: int = 16,
    minimum_group_count: int = 2,
    routing_modes: tuple[str, ...] = ("learned-router", "oracle", "dense-repeat"),
) -> MetricSpec:
    return MetricSpec(
        metric_id=metric_id,
        name=name,
        policy_version=STRUCTURAL_POLICY_VERSION,
        direction=direction,
        unit=unit,
        aggregation=aggregation,
        mask_semantics="exclude padding, invalid, and non-finite rows before aggregation",
        numerical_dtype="float32 metrics with float64 accumulators",
        epsilon_policy="denominators use max(target_norm_sq or fit_nmse, 1e-8); epsilon is serialized",
        evidence_class="structural-ffn",
        routing_modes=routing_modes,
        splits_or_tiers=("FIT-TRAIN", "FIT-DEV", "development"),
        slice_eligible=slice_eligible,
        minimum_sample_count=minimum_sample_count,
        minimum_group_count=minimum_group_count,
        green=green,
        yellow=yellow,
        gate_bearing=gate_bearing,
        override_eligible=override_eligible,
        veto_capability=veto_capability,
        serialization_field=metric_id,
        schema_version=STRUCTURAL_SCHEMA_VERSION,
        formula=formula,
    )


def _lm(
    metric_id: str,
    name: str,
    *,
    direction: str,
    unit: str,
    aggregation: str,
    formula: str,
    green: float | None = None,
    yellow: float | None = None,
    gate_bearing: bool = False,
    veto_capability: str = "none",
    slice_eligible: bool = True,
) -> MetricSpec:
    return MetricSpec(
        metric_id=metric_id,
        name=name,
        policy_version=LM_POLICY_VERSION,
        direction=direction,
        unit=unit,
        aggregation=aggregation,
        mask_semantics="exclude padding, masked, invalid, and non-finite positions",
        numerical_dtype="float32 softmax/log-softmax and float64 streaming accumulators",
        epsilon_policy="NLL and relative deltas use max(dense_nll, 1e-8); KL uses log-softmax",
        evidence_class="lm-output-evaluation",
        routing_modes=("learned-router", "dense-repeat"),
        splits_or_tiers=("FIT-DEV", "evaluation-only", "development"),
        slice_eligible=slice_eligible,
        minimum_sample_count=16,
        minimum_group_count=2,
        green=green,
        yellow=yellow,
        gate_bearing=gate_bearing,
        veto_capability=veto_capability,
        serialization_field=metric_id,
        schema_version=LM_SCHEMA_VERSION,
        formula=formula,
    )


METRIC_REGISTRY: dict[str, MetricSpec] = {
    # Structural fidelity and health gates.
    "structural.cosine_similarity": _structural(
        "structural.cosine_similarity", "Cosine similarity", direction="higher-is-better", unit="ratio", aggregation="mean token cosine", formula="mean(dot(pred,target)/(||pred||*||target||))", green=0.98, yellow=0.96, gate_bearing=True, override_eligible=True, veto_capability="hard"
    ),
    "structural.normalized_mse": _structural(
        "structural.normalized_mse", "Normalized mean squared error", direction="lower-is-better", unit="ratio", aggregation="mean token squared-error/target-energy", formula="mean(||pred-target||^2/max(||target||^2,epsilon))", green=0.05, yellow=0.08, gate_bearing=True, override_eligible=True, veto_capability="hard"
    ),
    "structural.target_relative_norm_error": _structural(
        "structural.target_relative_norm_error", "Target-relative norm error", direction="lower-is-better", unit="ratio", aggregation="mean token relative norm error", formula="mean(abs(||pred||-||target||)/max(||target||,epsilon))", green=0.05, yellow=0.10, gate_bearing=True, veto_capability="hard"
    ),
    "structural.mean_prediction_to_target_norm_ratio": _structural(
        "structural.mean_prediction_to_target_norm_ratio", "Mean prediction-to-target norm ratio", direction="higher-is-better", unit="ratio", aggregation="mean token norm ratio", formula="mean(||pred||/max(||target||,epsilon))", green=0.95, yellow=0.90, gate_bearing=True, veto_capability="hard"
    ),
    "structural.p95_abs_relative_norm_error": _structural(
        "structural.p95_abs_relative_norm_error", "P95 absolute relative norm error", direction="lower-is-better", unit="ratio", aggregation="95th percentile token error", formula="p95(abs(||pred||-||target||)/max(||target||,epsilon))", green=0.15, yellow=0.20, gate_bearing=True, veto_capability="hard"
    ),
    "routing.learned_load_cv": _structural(
        "routing.learned_load_cv", "Learned-router load coefficient of variation", direction="lower-is-better", unit="ratio", aggregation="population CV over learned hard assignments", formula="std(counts)/mean(counts)", green=0.50, yellow=0.65, gate_bearing=True, veto_capability="non_overridable", routing_modes=("learned-router",)
    ),
    "routing.oracle_load_cv": _structural(
        "routing.oracle_load_cv", "Oracle load coefficient of variation", direction="lower-is-better", unit="ratio", aggregation="population CV over oracle assignments", formula="std(oracle_counts)/mean(oracle_counts)", slice_eligible=False, routing_modes=("oracle",)
    ),
    "routing.dead_expert_count": _structural(
        "routing.dead_expert_count", "Dead-expert count", direction="lower-is-better", unit="count", aggregation="number of zero-use experts", formula="count(count_i == 0)", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="non_overridable"
    ),
    "routing.dead_expert_rate": _structural(
        "routing.dead_expert_rate", "Dead-expert rate", direction="lower-is-better", unit="ratio", aggregation="dead experts / experts", formula="dead_expert_count/max(num_experts,1)", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="non_overridable"
    ),
    "routing.expert_utilization": _structural(
        "routing.expert_utilization", "Expert utilization", direction="higher-is-better", unit="ratio", aggregation="fraction of experts with positive use", formula="1-dead_expert_rate", green=1.0, yellow=1.0, gate_bearing=True, veto_capability="non_overridable"
    ),
    "routing.entropy": _structural(
        "routing.entropy", "Routing entropy", direction="higher-is-better", unit="nats", aggregation="mean token routing entropy", formula="mean(-sum(p_i*log(p_i)))", green=None, yellow=None, gate_bearing=False
    ),
    "quality.dropped_token_count": _structural(
        "quality.dropped_token_count", "Dropped token count", direction="lower-is-better", unit="count", aggregation="sum", formula="sum(dropped_or_capacity_exceeded)", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="hard"
    ),
    "quality.invalid_token_count": _structural(
        "quality.invalid_token_count", "Invalid token count", direction="lower-is-better", unit="count", aggregation="sum", formula="sum(masked_invalid)", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="hard"
    ),
    "quality.non_finite_token_count": _structural(
        "quality.non_finite_token_count", "Non-finite token count", direction="lower-is-better", unit="count", aggregation="sum", formula="sum(~isfinite(pred or target))", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="hard"
    ),
    "quality.scored_token_count": _structural(
        "quality.scored_token_count", "Scored token count", direction="higher-is-better", unit="count", aggregation="sum", formula="count(rows surviving mask and finiteness)", gate_bearing=False, slice_eligible=False
    ),
    "quality.independent_group_count": _structural(
        "quality.independent_group_count", "Independent group count", direction="higher-is-better", unit="count", aggregation="count unique group identities", formula="|unique(group_identity)|", gate_bearing=False, slice_eligible=False
    ),
    # Bounded distributional diagnostics.
    **{
        f"distribution.{percentile}_token_cosine_error": _structural(
            f"distribution.{percentile}_token_cosine_error", f"{percentile.upper()} token cosine error", direction="lower-is-better", unit="ratio", aggregation=f"{percentile} percentile over bounded token sample", formula=f"quantile(1-cosine,{float(percentile[1:]) / 100.0})", gate_bearing=False
        )
        for percentile in ("p50", "p90", "p95", "p99")
    },
    **{
        f"distribution.{percentile}_token_normalized_error": _structural(
            f"distribution.{percentile}_token_normalized_error", f"{percentile.upper()} token normalized error", direction="lower-is-better", unit="ratio", aggregation=f"{percentile} percentile over bounded token sample", formula=f"quantile(token_nmse,{float(percentile[1:]) / 100.0})", gate_bearing=False
        )
        for percentile in ("p50", "p90", "p95", "p99")
    },
    # Generalization diagnostics (never a replacement for absolute DEV gates).
    "generalization.cosine_gap": _structural(
        "generalization.cosine_gap", "FIT-to-DEV cosine gap", direction="lower-is-better", unit="ratio", aggregation="FIT cosine - DEV cosine", formula="fit.cosine_similarity-dev.cosine_similarity", slice_eligible=True
    ),
    "generalization.absolute_nmse_increase": _structural(
        "generalization.absolute_nmse_increase", "Absolute NMSE increase", direction="lower-is-better", unit="ratio", aggregation="DEV NMSE - FIT NMSE", formula="dev.normalized_mse-fit.normalized_mse", slice_eligible=True
    ),
    "generalization.nmse_ratio": _structural(
        "generalization.nmse_ratio", "NMSE ratio", direction="lower-is-better", unit="ratio", aggregation="DEV NMSE / max(FIT NMSE,epsilon)", formula="dev.normalized_mse/max(fit.normalized_mse,epsilon)", slice_eligible=True
    ),
    "generalization.relative_norm_error_increase": _structural(
        "generalization.relative_norm_error_increase", "Relative norm-error increase", direction="lower-is-better", unit="ratio", aggregation="DEV relative norm error - FIT relative norm error", formula="dev.target_relative_norm_error-fit.target_relative_norm_error", slice_eligible=True
    ),
    "generalization.load_cv_change": _structural(
        "generalization.load_cv_change", "Learned-router loadCV change", direction="lower-is-better", unit="ratio", aggregation="DEV loadCV - FIT loadCV", formula="dev.learned_load_cv-fit.learned_load_cv", slice_eligible=False
    ),
    "generalization.dead_expert_change": _structural(
        "generalization.dead_expert_change", "Dead-expert change", direction="lower-is-better", unit="count", aggregation="DEV dead experts - FIT dead experts", formula="dev.dead_expert_count-fit.dead_expert_count", slice_eligible=False
    ),
    "generalization.routing_health_change": _structural(
        "generalization.routing_health_change", "Routing-health change", direction="lower-is-better", unit="ratio", aggregation="DEV health score - FIT health score", formula="dev.routing_health-fit.routing_health", slice_eligible=False
    ),
    # Paired LM-output metrics.
    "lm.mean_forward_kl": _lm("lm.mean_forward_kl", "Mean teacher-to-candidate forward KL", direction="lower-is-better", unit="nats", aggregation="mean token KL", formula="mean(sum(softmax(t)*(log_softmax(t)-log_softmax(c))))", green=0.10, yellow=0.20, gate_bearing=True, veto_capability="hard"),
    "lm.p95_forward_kl": _lm("lm.p95_forward_kl", "P95 teacher-to-candidate forward KL", direction="lower-is-better", unit="nats", aggregation="p95 token KL", formula="quantile(token_forward_kl,.95)", green=0.25, yellow=0.50, gate_bearing=True, veto_capability="hard"),
    "lm.excess_mean_forward_kl": _lm("lm.excess_mean_forward_kl", "Excess mean KL above dense-repeat floor", direction="lower-is-better", unit="nats", aggregation="max(raw_mean-baseline_mean,0)", formula="max(mean_forward_kl-dense_repeat_mean_forward_kl,0)", gate_bearing=False),
    "lm.excess_p95_forward_kl": _lm("lm.excess_p95_forward_kl", "Excess P95 KL above dense-repeat floor", direction="lower-is-better", unit="nats", aggregation="max(raw_p95-baseline_p95,0)", formula="max(p95_forward_kl-dense_repeat_p95_forward_kl,0)", gate_bearing=False),
    "lm.top1_agreement": _lm("lm.top1_agreement", "Top-1 agreement", direction="higher-is-better", unit="ratio", aggregation="mean exact top-1 equality", formula="mean(argmax(t)==argmax(c))", green=0.85, yellow=0.75, gate_bearing=True, veto_capability="hard"),
    "lm.top5_set_recall": _lm("lm.top5_set_recall", "Top-5 set recall/overlap", direction="higher-is-better", unit="ratio", aggregation="teacher top-5 items present in candidate top-5 / 5", formula="mean(|top5(t)∩top5(c)|/5)", green=0.80, yellow=0.70, gate_bearing=True, veto_capability="hard"),
    "lm.teacher_top5_mass_retention": _lm("lm.teacher_top5_mass_retention", "Teacher Top-5 probability-mass retention", direction="higher-is-better", unit="ratio", aggregation="candidate probability mass on teacher top-5", formula="mean(sum(softmax(c)[top5(t)]))", green=0.80, yellow=0.70, gate_bearing=True, veto_capability="hard"),
    "lm.high_margin_top1_flip_rate": _lm("lm.high_margin_top1_flip_rate", "High-margin Top-1 flip rate", direction="lower-is-better", unit="ratio", aggregation="flips among teacher high-margin positions", formula="mean(flip | teacher_margin>=threshold)", green=0.10, yellow=0.20, gate_bearing=True, veto_capability="hard"),
    "lm.near_tie_top1_flip_rate": _lm("lm.near_tie_top1_flip_rate", "Near-tie Top-1 flip rate", direction="lower-is-better", unit="ratio", aggregation="flips among teacher near-tie positions", formula="mean(flip | teacher_margin<threshold)", gate_bearing=False),
    "lm.candidate_nll": _lm("lm.candidate_nll", "Candidate NLL", direction="lower-is-better", unit="nats/token", aggregation="mean target NLL", formula="mean(-log_softmax(candidate)[target])", gate_bearing=False),
    "lm.dense_teacher_nll": _lm("lm.dense_teacher_nll", "Dense-teacher NLL", direction="lower-is-better", unit="nats/token", aggregation="mean target NLL", formula="mean(-log_softmax(teacher)[target])", gate_bearing=False),
    "lm.absolute_nll_delta": _lm("lm.absolute_nll_delta", "Absolute NLL delta", direction="lower-is-better", unit="nats/token", aggregation="candidate NLL - teacher NLL", formula="candidate_nll-dense_teacher_nll", green=0.05, yellow=0.10, gate_bearing=True, veto_capability="hard"),
    "lm.relative_nll_increase": _lm("lm.relative_nll_increase", "Relative NLL increase", direction="lower-is-better", unit="ratio", aggregation="delta / max(teacher NLL,epsilon)", formula="absolute_nll_delta/max(dense_teacher_nll,epsilon)", green=0.05, yellow=0.10, gate_bearing=True, veto_capability="hard"),
    "lm.perplexity": _lm("lm.perplexity", "Candidate perplexity", direction="lower-is-better", unit="dimensionless", aggregation="exp(candidate NLL)", formula="exp(candidate_nll)", gate_bearing=False),
    "lm.valid_token_count": _lm("lm.valid_token_count", "Valid LM token count", direction="higher-is-better", unit="count", aggregation="sum", formula="count(masked and finite positions)", gate_bearing=False, slice_eligible=False),
    "lm.masked_token_count": _lm("lm.masked_token_count", "Masked LM token count", direction="lower-is-better", unit="count", aggregation="sum", formula="count(~mask)", gate_bearing=False, slice_eligible=False),
    "lm.non_finite_token_count": _lm("lm.non_finite_token_count", "Non-finite LM token count", direction="lower-is-better", unit="count", aggregation="sum", formula="count(~isfinite(logits or metric))", green=0.0, yellow=0.0, gate_bearing=True, veto_capability="hard", slice_eligible=False),
    "lm.approximate_topk_kl": _lm("lm.approximate_topk_kl", "Approximate Top-K KL diagnostic", direction="lower-is-better", unit="nats", aggregation="mean diagnostic token estimate", formula="topk-only KL approximation", gate_bearing=False, slice_eligible=True),
}


METRIC_ALIASES: dict[str, str] = {
    "cosine": "structural.cosine_similarity",
    "cosine_similarity": "structural.cosine_similarity",
    "nmse": "structural.normalized_mse",
    "normalized_mse": "structural.normalized_mse",
    "target_relative_norm_error": "structural.target_relative_norm_error",
    "p95_relative_norm_error": "structural.p95_abs_relative_norm_error",
    "loadcv": "routing.learned_load_cv",
    "load_cv": "routing.learned_load_cv",
    "learned_router_loadcv": "routing.learned_load_cv",
    "oracle_loadcv": "routing.oracle_load_cv",
    "dead_experts": "routing.dead_expert_count",
    "router_entropy": "routing.entropy",
    "mean_forward_kl": "lm.mean_forward_kl",
    "p95_forward_kl": "lm.p95_forward_kl",
    "top1_agreement": "lm.top1_agreement",
    "teacher_top5_mass_retention": "lm.teacher_top5_mass_retention",
}


DECISION_POLICY_V2: dict[str, Any] = {
    "priority": [
        "identity_reload_preflight_policy_evidence",
        "non_finite_invalid_rejection",
        "dead_expert_and_learned_router_health",
        "absolute_fit_dev_structural_gates",
        "gate_bearing_source_domain_vetoes",
        "generalization_gap_classification",
        "lm_eligibility",
        "lm_veto",
        "narrow_single_metric_structural_override",
        "protected_confirmation_requirement",
        "candidate_ranking",
    ],
    "override_envelopes": {
        "structural.cosine_similarity": {"lower_exclusive": 0.975, "upper_exclusive": 0.980},
        "structural.normalized_mse": {"lower_exclusive": 0.050, "upper_inclusive": 0.060},
    },
    "override_rules": {
        "exactly_one_overrideable_metric": True,
        "dual_miss_allowed": False,
        "requires_absolute_dev_green": True,
        "requires_learned_router_health_green": True,
        "requires_no_source_slice_collapse": True,
        "requires_lm_green_when_evaluated": True,
        "requires_protected_confirmation": True,
        "oracle_authorizes_override": False,
        "loadcv_overridable": False,
        "dead_experts_overridable": False,
    },
    "generalization_labels": {
        "fit_green_dev_green": "STABLE_CANDIDATE",
        "fit_green_dev_yellow": "DATA_SPECIFIC_FIT_OR_DISTRIBUTION_SHIFT_SENSITIVITY",
        "fit_green_dev_red": "GENERALIZATION_REJECT",
        "fit_yellow_dev_green": "INSPECT_TRAINING_METRIC",
        "fit_yellow_dev_yellow": "RESEARCH_ONLY",
    },
    "epsilon": 1e-8,
    "confidence_boundary_rule": "For lower-is-better gates, the conservative upper confidence bound must be <= GREEN; for higher-is-better gates, the lower confidence bound must be >= GREEN.",
}


REQUIRED_TEST_COVERAGE = (
    "reference_formula",
    "streaming_reference",
    "masking",
    "slice_aggregation",
    "serialization_schema",
    "decision_boundary",
    "runner_integration",
    "missing_insufficient_evidence",
    "non_finite_input",
    "receipt_round_trip",
    "policy_hash_sensitivity",
)

TEST_MATRIX: dict[str, tuple[str, ...]] = {
    metric_id: REQUIRED_TEST_COVERAGE if spec.gate_bearing else ("reference_formula", "serialization_schema", "runner_integration")
    for metric_id, spec in METRIC_REGISTRY.items()
}


def canonical_metric_id(metric_id: str) -> str:
    """Resolve a canonical ID or a legacy-friendly alias."""

    value = str(metric_id).strip().casefold().replace("-", "_").replace(" ", "_")
    if value in METRIC_REGISTRY:
        return value
    return METRIC_ALIASES.get(value, value)


def get_metric(metric_id: str) -> MetricSpec:
    canonical = canonical_metric_id(metric_id)
    try:
        return METRIC_REGISTRY[canonical]
    except KeyError as exc:
        raise KeyError(f"unknown canonical metric id: {metric_id!r}") from exc


def validate_metric_registry(
    registry: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    *,
    test_matrix: Mapping[str, tuple[str, ...]] = TEST_MATRIX,
) -> dict[str, Any]:
    """Validate IDs, semantics, thresholds, and gate-bearing test coverage."""

    errors: list[str] = []
    seen_semantics: dict[str, tuple[Any, ...]] = {}
    for key, spec in registry.items():
        if key != spec.metric_id:
            errors.append(f"registry key {key!r} disagrees with metric_id {spec.metric_id!r}")
        if spec.metric_id in seen_semantics:
            errors.append(f"duplicate metric id: {spec.metric_id}")
        seen_semantics[spec.metric_id] = (
            spec.direction,
            spec.unit,
            spec.formula,
            spec.green,
            spec.yellow,
            spec.veto_capability,
        )
        if spec.gate_bearing and not set(test_matrix.get(key, ())).issuperset(REQUIRED_TEST_COVERAGE):
            missing = sorted(set(REQUIRED_TEST_COVERAGE) - set(test_matrix.get(key, ())))
            errors.append(f"gate-bearing metric {key} lacks test coverage: {missing}")
        if spec.green is not None and spec.yellow is not None:
            if spec.direction == "lower-is-better" and spec.green > spec.yellow:
                errors.append(f"threshold direction drift for {key}")
            if spec.direction == "higher-is-better" and spec.green < spec.yellow:
                errors.append(f"threshold direction drift for {key}")
    return {"valid": not errors, "errors": errors, "metric_count": len(registry), "gate_metric_count": sum(int(v.gate_bearing) for v in registry.values())}


def metric_policy_payload(
    registry: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    decision_policy: Mapping[str, Any] = DECISION_POLICY_V2,
) -> dict[str, Any]:
    return {
        "structural_policy_version": STRUCTURAL_POLICY_VERSION,
        "lm_policy_version": LM_POLICY_VERSION,
        "structural_schema_version": STRUCTURAL_SCHEMA_VERSION,
        "lm_schema_version": LM_SCHEMA_VERSION,
        "metrics": {key: value.as_dict() for key, value in sorted(registry.items())},
        "decision_policy": json.loads(json.dumps(decision_policy, sort_keys=True)),
    }


def metric_policy_hash(
    registry: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    decision_policy: Mapping[str, Any] = DECISION_POLICY_V2,
) -> str:
    """Hash every gate-bearing formula, threshold, veto, and override rule."""

    payload = json.dumps(metric_policy_payload(registry, decision_policy), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


POLICY_HASH = metric_policy_hash()


def policy_hash() -> str:
    """Return the current canonical policy hash."""

    return metric_policy_hash()


def missing_metric_status(metric_id: str, status: str = NOT_AVAILABLE) -> dict[str, Any]:
    if status not in MISSING_METRIC_STATUSES:
        raise ValueError(f"unknown missing metric status: {status}")
    spec = get_metric(metric_id)
    return {"metric_id": spec.metric_id, "status": status, "value": None, "reason": status}


__all__ = [
    "DECISION_POLICY_V2",
    "INSUFFICIENT_EVIDENCE",
    "LM_POLICY_VERSION",
    "LM_SCHEMA_VERSION",
    "METRIC_REGISTRY",
    "NOT_APPLICABLE",
    "NOT_AVAILABLE",
    "NOT_COMPUTED",
    "POLICY_HASH",
    "REQUIRED_TEST_COVERAGE",
    "STRUCTURAL_POLICY_VERSION",
    "STRUCTURAL_SCHEMA_VERSION",
    "TEST_MATRIX",
    "MetricSpec",
    "canonical_metric_id",
    "get_metric",
    "metric_policy_hash",
    "metric_policy_payload",
    "missing_metric_status",
    "policy_hash",
    "validate_metric_registry",
]
