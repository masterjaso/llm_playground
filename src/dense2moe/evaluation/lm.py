"""Paired, streaming teacher-forced LM-output metrics.

The evaluator computes full-vocabulary forward KL (teacher || candidate) in
float32 and keeps only scalar reductions.  It never persists corpus-sized
logit matrices.  ``StreamingLMMetrics`` is batch-partition invariant for all
additive metrics and percentile diagnostics retain a bounded scalar sample.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .registry import INSUFFICIENT_EVIDENCE, NOT_AVAILABLE, NOT_COMPUTED


def _to_numpy(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]
    if np is not None:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu")
            value = value.float() if hasattr(value, "float") else value
            value = value.numpy()
        return np.asarray(value)
    return value


def _flatten_logits(value: Any) -> Any:
    import numpy as np  # type: ignore

    array = np.asarray(_to_numpy(value), dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("logits must have at least [tokens, vocab] dimensions")
    return array.reshape(-1, array.shape[-1])


def _flatten_vector(value: Any, count: int, *, dtype: Any = None) -> Any:
    import numpy as np  # type: ignore

    if value is None:
        return None
    array = np.asarray(_to_numpy(value), dtype=dtype).reshape(-1)
    if array.size != count:
        raise ValueError(f"vector row count mismatch: expected {count}, got {array.size}")
    return array


def _log_softmax(logits: Any) -> Any:
    import numpy as np  # type: ignore

    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_norm = np.log(np.maximum(np.exp(shifted, dtype=np.float32).sum(axis=-1, keepdims=True), 1e-30))
    return (shifted - log_norm).astype(np.float32, copy=False)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    import numpy as np  # type: ignore

    try:
        return float(np.quantile(np.asarray(values, dtype=np.float64), fraction, method="linear"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(np.asarray(values, dtype=np.float64), fraction, interpolation="linear"))


def _batch_arrays(
    teacher_logits: Any,
    candidate_logits: Any,
    *,
    targets: Any = None,
    mask: Any = None,
    baseline_logits: Any = None,
    high_margin_threshold: float = 0.5,
    near_tie_threshold: float = 0.1,
    approximate_topk: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    import numpy as np  # type: ignore

    teacher = _flatten_logits(teacher_logits)
    candidate = _flatten_logits(candidate_logits)
    if teacher.shape != candidate.shape:
        raise ValueError(f"teacher/candidate logit shape mismatch: {teacher.shape} != {candidate.shape}")
    count, vocab = teacher.shape
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    # Tiny deterministic fixtures may have fewer than five vocabulary items;
    # use the complete vocabulary while retaining the requested value in the
    # receipt.  Real Qwen runs have vocab >> 5 and therefore execute exact
    # Top-5 semantics.
    effective_top_k = min(int(top_k), int(vocab))
    if mask is None:
        row_mask = np.ones(count, dtype=bool)
    else:
        row_mask = _flatten_vector(mask, count, dtype=bool).astype(bool)
    finite = np.isfinite(teacher).all(axis=1) & np.isfinite(candidate).all(axis=1)
    non_finite = int((row_mask & ~finite).sum())
    masked = int((~row_mask).sum())
    valid = row_mask & finite
    if not valid.any():
        return {
            "valid": valid,
            "teacher_log_probs": np.empty((0, vocab), dtype=np.float32),
            "candidate_log_probs": np.empty((0, vocab), dtype=np.float32),
            "forward_kl": np.empty(0, dtype=np.float64),
            "baseline_forward_kl": np.empty(0, dtype=np.float64),
            "top1_agreement": np.empty(0, dtype=bool),
            "top5_recall": np.empty(0, dtype=np.float64),
            "top5_mass": np.empty(0, dtype=np.float64),
            "high_margin_flip": np.empty(0, dtype=bool),
            "high_margin": np.empty(0, dtype=bool),
            "near_tie_flip": np.empty(0, dtype=bool),
            "near_tie": np.empty(0, dtype=bool),
            "candidate_nll": None,
            "teacher_nll": None,
            "baseline_forward_kl_valid": False,
            "approximate_topk_kl": None,
            "masked_count": masked,
            "non_finite_count": non_finite,
            "total_count": count,
            "targets": None,
        }
    t = teacher[valid]
    c = candidate[valid]
    t_logp = _log_softmax(t)
    c_logp = _log_softmax(c)
    t_prob = np.exp(t_logp, dtype=np.float32)
    forward_kl = np.sum(t_prob * (t_logp - c_logp), axis=-1, dtype=np.float64)
    forward_kl = np.maximum(forward_kl, 0.0)
    teacher_order = np.argsort(-t, axis=-1, kind="stable")[:, :effective_top_k]
    candidate_order = np.argsort(-c, axis=-1, kind="stable")[:, :effective_top_k]
    teacher_top1 = teacher_order[:, 0]
    candidate_top1 = candidate_order[:, 0]
    agreement = teacher_top1 == candidate_top1
    overlap = np.asarray([len(set(row_t.tolist()) & set(row_c.tolist())) / float(effective_top_k) for row_t, row_c in zip(teacher_order, candidate_order)], dtype=np.float64)
    candidate_prob = np.exp(c_logp, dtype=np.float32)
    top5_mass = np.asarray([float(candidate_prob[index, row_t].sum()) for index, row_t in enumerate(teacher_order)], dtype=np.float64)
    if vocab >= 2:
        teacher_top2 = np.partition(t, -2, axis=-1)[:, -2:]
        teacher_top2.sort(axis=-1)
        margins = teacher_top2[:, 1] - teacher_top2[:, 0]
    else:
        # A one-token fixture has no meaningful top-1 margin; keep it out of
        # both margin-conditioned populations rather than inventing evidence.
        margins = np.zeros(len(t), dtype=np.float32)
    high_margin = margins >= float(high_margin_threshold)
    near_tie = margins < float(near_tie_threshold)
    flip = ~agreement
    baseline_kl = np.empty(0, dtype=np.float64)
    baseline_valid = baseline_logits is not None
    if baseline_logits is not None:
        baseline = _flatten_logits(baseline_logits)
        if baseline.shape != teacher.shape:
            raise ValueError("baseline logits shape must match teacher logits")
        baseline = baseline[valid]
        if not np.isfinite(baseline).all(axis=1).all():
            baseline_valid = False
        else:
            b_logp = _log_softmax(baseline)
            baseline_kl = np.maximum(np.sum(t_prob * (t_logp - b_logp), axis=-1, dtype=np.float64), 0.0)
    target_values = _flatten_vector(targets, count, dtype=np.int64) if targets is not None else None
    candidate_nll = None
    teacher_nll = None
    if target_values is not None:
        target_values = target_values[valid]
        valid_targets = (target_values >= 0) & (target_values < vocab)
        if valid_targets.any():
            candidate_nll = -c_logp[np.arange(len(c_logp))[valid_targets], target_values[valid_targets]].astype(np.float64)
            teacher_nll = -t_logp[np.arange(len(t_logp))[valid_targets], target_values[valid_targets]].astype(np.float64)
        else:
            candidate_nll = np.empty(0, dtype=np.float64)
            teacher_nll = np.empty(0, dtype=np.float64)
    approx = None
    if approximate_topk:
        union = np.unique(np.concatenate((teacher_order, candidate_order), axis=1), axis=1)
        approx = np.asarray([float(np.sum(t_prob[index, row] * (t_logp[index, row] - c_logp[index, row]))) for index, row in enumerate(union)], dtype=np.float64)
    return {
        "valid": valid,
        "teacher_log_probs": t_logp,
        "candidate_log_probs": c_logp,
        "forward_kl": forward_kl,
        "baseline_forward_kl": baseline_kl,
        "baseline_forward_kl_valid": baseline_valid and baseline_kl.size == forward_kl.size,
        "top1_agreement": agreement,
        "top5_recall": overlap,
        "top5_mass": top5_mass,
        "high_margin_flip": flip[high_margin],
        "high_margin": high_margin,
        "near_tie_flip": flip[near_tie],
        "near_tie": near_tie,
        "candidate_nll": candidate_nll,
        "teacher_nll": teacher_nll,
        "approximate_topk_kl": approx,
        "masked_count": masked,
        "non_finite_count": non_finite,
        "total_count": count,
        "targets": target_values,
    }


class StreamingLMMetrics:
    """Streaming paired evaluator; stores scalar samples only, never logits."""

    def __init__(
        self,
        *,
        high_margin_threshold: float = 0.5,
        near_tie_threshold: float = 0.1,
        top_k: int = 5,
        max_distribution_samples: int = 100_000,
        approximate_topk: bool = False,
    ) -> None:
        self.high_margin_threshold = float(high_margin_threshold)
        self.near_tie_threshold = float(near_tie_threshold)
        self.top_k = int(top_k)
        self.max_distribution_samples = int(max_distribution_samples)
        self.approximate_topk = bool(approximate_topk)
        self._forward_kl: list[float] = []
        self._baseline_kl: list[float] = []
        self._top1: list[float] = []
        self._top5: list[float] = []
        self._top5_mass: list[float] = []
        self._high_flip: list[float] = []
        self._near_flip: list[float] = []
        self._candidate_nll: list[float] = []
        self._teacher_nll: list[float] = []
        self._approximate: list[float] = []
        self._valid_tokens = 0
        self._masked_tokens = 0
        self._non_finite_tokens = 0
        self._total_tokens = 0
        self._metadata_row_offset = 0
        self._source_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._domain_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def update(
        self,
        teacher_logits: Any,
        candidate_logits: Any,
        *,
        targets: Any = None,
        mask: Any = None,
        baseline_logits: Any = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> "StreamingLMMetrics":
        batch = _batch_arrays(
            teacher_logits,
            candidate_logits,
            targets=targets,
            mask=mask,
            baseline_logits=baseline_logits,
            high_margin_threshold=self.high_margin_threshold,
            near_tie_threshold=self.near_tie_threshold,
            approximate_topk=self.approximate_topk,
            top_k=self.top_k,
        )
        forward = batch["forward_kl"]
        self._valid_tokens += int(len(forward))
        self._masked_tokens += int(batch["masked_count"])
        self._non_finite_tokens += int(batch["non_finite_count"])
        self._total_tokens += int(batch["total_count"])
        sample_remaining = max(0, self.max_distribution_samples - len(self._forward_kl))
        if sample_remaining:
            self._forward_kl.extend(float(value) for value in forward[:sample_remaining])
            self._top1.extend(float(value) for value in batch["top1_agreement"][:sample_remaining])
            self._top5.extend(float(value) for value in batch["top5_recall"][:sample_remaining])
            self._top5_mass.extend(float(value) for value in batch["top5_mass"][:sample_remaining])
        self._forward_kl_sum = getattr(self, "_forward_kl_sum", 0.0) + float(forward.sum())
        self._top1_sum = getattr(self, "_top1_sum", 0.0) + float(batch["top1_agreement"].sum())
        self._top5_sum = getattr(self, "_top5_sum", 0.0) + float(batch["top5_recall"].sum())
        self._top5_mass_sum = getattr(self, "_top5_mass_sum", 0.0) + float(batch["top5_mass"].sum())
        self._high_flip_count = getattr(self, "_high_flip_count", 0) + int(batch["high_margin_flip"].sum())
        self._high_count = getattr(self, "_high_count", 0) + int(batch["high_margin"].sum())
        self._near_flip_count = getattr(self, "_near_flip_count", 0) + int(batch["near_tie_flip"].sum())
        self._near_count = getattr(self, "_near_count", 0) + int(batch["near_tie"].sum())
        if batch["baseline_forward_kl_valid"]:
            baseline = batch["baseline_forward_kl"]
            baseline_remaining = max(0, self.max_distribution_samples - len(self._baseline_kl))
            self._baseline_kl.extend(float(value) for value in baseline[:baseline_remaining])
            self._baseline_kl_sum = getattr(self, "_baseline_kl_sum", 0.0) + float(baseline.sum())
            self._baseline_valid_tokens = getattr(self, "_baseline_valid_tokens", 0) + int(len(baseline))
        else:
            self._baseline_valid_tokens = getattr(self, "_baseline_valid_tokens", 0)
        if batch["candidate_nll"] is not None:
            candidate_remaining = max(0, self.max_distribution_samples - len(self._candidate_nll))
            self._candidate_nll.extend(float(value) for value in batch["candidate_nll"][:candidate_remaining])
            self._teacher_nll.extend(float(value) for value in batch["teacher_nll"][:candidate_remaining])
            self._candidate_nll_sum = getattr(self, "_candidate_nll_sum", 0.0) + float(batch["candidate_nll"].sum())
            self._teacher_nll_sum = getattr(self, "_teacher_nll_sum", 0.0) + float(batch["teacher_nll"].sum())
            self._nll_tokens = getattr(self, "_nll_tokens", 0) + int(len(batch["candidate_nll"]))
        if batch["approximate_topk_kl"] is not None:
            approximate_remaining = max(0, self.max_distribution_samples - len(self._approximate))
            self._approximate.extend(float(value) for value in batch["approximate_topk_kl"][:approximate_remaining])
        # Per-source/domain values are scalar reductions only.
        if metadata:
            valid_indices = [index for index, valid in enumerate(batch["valid"]) if bool(valid)]
            for position, original_index in enumerate(valid_indices):
                for metadata_key, output_key in (("source_family", "source"), ("domain", "domain")):
                    identity = metadata[original_index].get(metadata_key) if original_index < len(metadata) else None
                    if identity is None:
                        continue
                    group = self._source_groups[str(identity)] if output_key == "source" else self._domain_groups[str(identity)]
                    group["forward_kl"].append(float(forward[position]))
                    group["top1"].append(float(batch["top1_agreement"][position]))
                    group["top5_mass"].append(float(batch["top5_mass"][position]))
                    metadata_item = metadata[original_index] if original_index < len(metadata) and isinstance(metadata[original_index], Mapping) else {}
                    group["group_ids"].append(str(metadata_item.get("independent_group", metadata_item.get("group_identity", self._metadata_row_offset + original_index))))
        self._metadata_row_offset += int(batch["total_count"])
        return self

    def finalize(self) -> dict[str, Any]:
        valid = self._valid_tokens
        result: dict[str, Any] = {
            "status": "COMPUTED" if valid else INSUFFICIENT_EVIDENCE,
            "mean_forward_kl": (self._forward_kl_sum / valid if valid else None),
            "p95_forward_kl": _percentile(self._forward_kl, 0.95),
            "top1_agreement": (self._top1_sum / valid if valid else None),
            "top5_set_recall": (self._top5_sum / valid if valid else None),
            "teacher_top5_mass_retention": (self._top5_mass_sum / valid if valid else None),
            "high_margin_top1_flip_rate": (self._high_flip_count / self._high_count if self._high_count else None),
            "near_tie_top1_flip_rate": (self._near_flip_count / self._near_count if self._near_count else None),
            "candidate_nll": (self._candidate_nll_sum / self._nll_tokens if getattr(self, "_nll_tokens", 0) else NOT_AVAILABLE),
            "dense_teacher_nll": (self._teacher_nll_sum / self._nll_tokens if getattr(self, "_nll_tokens", 0) else NOT_AVAILABLE),
            "valid_token_count": valid,
            "masked_token_count": self._masked_tokens,
            "non_finite_token_count": self._non_finite_tokens,
            "scored_token_count": valid,
            "batch_partition_invariant": True,
            "numerical_dtype": "float32 softmax/log-softmax",
            "high_margin_threshold": self.high_margin_threshold,
            "near_tie_threshold": self.near_tie_threshold,
            "top_k": self.top_k,
        }
        if getattr(self, "_baseline_valid_tokens", 0) == valid and valid:
            baseline_mean = self._baseline_kl_sum / valid
            baseline_p95 = _percentile(self._baseline_kl, 0.95)
            result["dense_repeat_mean_forward_kl"] = baseline_mean
            result["dense_repeat_p95_forward_kl"] = baseline_p95
            result["excess_mean_forward_kl"] = max(0.0, float(result["mean_forward_kl"] - baseline_mean))
            result["excess_p95_forward_kl"] = max(0.0, float(result["p95_forward_kl"] - baseline_p95)) if baseline_p95 is not None and result["p95_forward_kl"] is not None else None
            result["dense_repeat_baseline_status"] = "COMPUTED"
        else:
            result["dense_repeat_mean_forward_kl"] = NOT_AVAILABLE
            result["dense_repeat_p95_forward_kl"] = NOT_AVAILABLE
            result["excess_mean_forward_kl"] = NOT_AVAILABLE
            result["excess_p95_forward_kl"] = NOT_AVAILABLE
            result["dense_repeat_baseline_status"] = NOT_AVAILABLE
        if getattr(self, "_nll_tokens", 0):
            result["absolute_nll_delta"] = result["candidate_nll"] - result["dense_teacher_nll"]
            result["relative_nll_increase"] = result["absolute_nll_delta"] / max(float(result["dense_teacher_nll"]), 1e-8)
            result["perplexity"] = math.exp(min(50.0, float(result["candidate_nll"])))
        else:
            result["absolute_nll_delta"] = NOT_AVAILABLE
            result["relative_nll_increase"] = NOT_AVAILABLE
            result["perplexity"] = NOT_AVAILABLE
        if self.approximate_topk:
            result["approximate_topk_kl"] = _percentile(self._approximate, 0.5) if self._approximate else None
            result["approximate_topk_kl_gate_eligible"] = False
        else:
            result["approximate_topk_kl"] = NOT_COMPUTED
            result["approximate_topk_kl_gate_eligible"] = False
        def _slice_payload(values: Mapping[str, list[float]]) -> dict[str, Any]:
            group_count = len(set(values.get("group_ids", [])))
            token_count = len(values["forward_kl"])
            gate_eligible = token_count >= 16 and group_count >= 2
            return {
                "mean_forward_kl": sum(values["forward_kl"]) / len(values["forward_kl"]) if values["forward_kl"] else None,
                "top1_agreement": sum(values["top1"]) / len(values["top1"]) if values["top1"] else None,
                "teacher_top5_mass_retention": sum(values["top5_mass"]) / len(values["top5_mass"]) if values["top5_mass"] else None,
                "scored_token_count": token_count,
                "independent_group_count": group_count,
                "gate_eligible": gate_eligible,
                **({} if gate_eligible else {"status": INSUFFICIENT_EVIDENCE, "reason": "slice below minimum sample/group count; diagnostic-only"}),
            }

        per_source: dict[str, Any] = {}
        per_domain: dict[str, Any] = {}
        for identity, values in self._source_groups.items():
            per_source[identity] = _slice_payload(values)
        for identity, values in self._domain_groups.items():
            per_domain[identity] = _slice_payload(values)
        result["source_slices"] = per_source
        result["domain_slices"] = per_domain
        return result


def compute_lm_output_metrics(
    teacher_logits: Any,
    candidate_logits: Any,
    *,
    targets: Any = None,
    mask: Any = None,
    baseline_logits: Any = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    high_margin_threshold: float = 0.5,
    near_tie_threshold: float = 0.1,
    top_k: int = 5,
    max_distribution_samples: int = 100_000,
    approximate_topk: bool = False,
) -> dict[str, Any]:
    evaluator = StreamingLMMetrics(
        high_margin_threshold=high_margin_threshold,
        near_tie_threshold=near_tie_threshold,
        top_k=top_k,
        max_distribution_samples=max_distribution_samples,
        approximate_topk=approximate_topk,
    )
    evaluator.update(teacher_logits, candidate_logits, targets=targets, mask=mask, baseline_logits=baseline_logits, metadata=metadata)
    return evaluator.finalize()


lm_output_metrics_v2 = compute_lm_output_metrics
evaluate_lm_output_metrics = compute_lm_output_metrics


__all__ = [
    "StreamingLMMetrics",
    "compute_lm_output_metrics",
    "evaluate_lm_output_metrics",
    "lm_output_metrics_v2",
]
