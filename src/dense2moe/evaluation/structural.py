"""Streaming structural-fidelity metrics for dense FFN reconstruction.

The implementation keeps FIT-TRAIN and FIT-DEV separate.  ``StructuralMetrics
Accumulator`` can be updated with bounded batches and only retains a bounded
sample of token errors for percentile diagnostics; aggregate sums are exact.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .registry import (
    INSUFFICIENT_EVIDENCE,
    NOT_AVAILABLE,
    canonical_metric_id,
)


def _to_numpy(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover - numpy is a runtime dependency
        np = None  # type: ignore[assignment]
    if np is not None:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu")
            if hasattr(value, "float"):
                value = value.float()
            value = value.numpy()
        return np.asarray(value)
    return value


def _finite(value: Any) -> Any:
    array = _to_numpy(value)
    try:
        import numpy as np  # type: ignore

        return np.isfinite(array)
    except ImportError:  # pragma: no cover
        return [[math.isfinite(float(item)) for item in row] for row in array]


def _flatten_rows(value: Any) -> Any:
    array = _to_numpy(value)
    try:
        import numpy as np  # type: ignore

        if array.ndim == 0:
            return array.reshape(1, 1)
        return array.reshape(array.shape[0], -1)
    except AttributeError:  # pragma: no cover
        return array


def _mask_rows(mask: Any, count: int) -> Any:
    try:
        import numpy as np  # type: ignore

        if mask is None:
            return np.ones(count, dtype=bool)
        values = np.asarray(_to_numpy(mask), dtype=bool).reshape(-1)
        if values.size != count:
            raise ValueError(f"mask row count mismatch: expected {count}, got {values.size}")
        return values
    except ImportError:  # pragma: no cover
        if mask is None:
            return [True] * count
        values = list(mask)
        if len(values) != count:
            raise ValueError("mask row count mismatch")
        return [bool(value) for value in values]


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    try:
        import numpy as np  # type: ignore

        return float(np.quantile(np.asarray(values, dtype=np.float64), fraction, method="linear"))
    except TypeError:  # pragma: no cover - old numpy fallback
        import numpy as np  # type: ignore

        return float(np.quantile(np.asarray(values, dtype=np.float64), fraction, interpolation="linear"))


def _assignment_counts(assignments: Any, num_experts: int | None = None) -> list[float] | None:
    if assignments is None:
        return None
    values = _to_numpy(assignments)
    try:
        import numpy as np  # type: ignore

        values = np.asarray(values)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim > 1 and values.shape[-1] > 1 and np.issubdtype(values.dtype, np.floating):
            # Probabilities/weights are not hard utilization.  The caller can
            # still provide hard assignments explicitly; this path is for
            # integer top-k arrays and one-hot vectors.
            if np.all((values >= 0) & (values <= 1)) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-4):
                values = values.argmax(axis=-1)
            else:
                values = values.reshape(-1)
        values = values.reshape(-1).astype(int)
        if values.size == 0:
            return [0.0] * int(num_experts or 0)
        inferred = int(values.max()) + 1
        width = max(int(num_experts or 0), inferred)
        counts = np.bincount(values[values >= 0], minlength=width).astype(float)
        return counts.tolist()
    except ImportError:  # pragma: no cover
        values = [int(item) for item in values]
        width = max(int(num_experts or 0), (max(values) + 1 if values else 0))
        counts = [0.0] * width
        for value in values:
            if value >= 0:
                counts[value] += 1.0
        return counts


def _routing_summary(assignments: Any, *, num_experts: int | None = None, probabilities: Any = None) -> dict[str, Any]:
    counts = _assignment_counts(assignments, num_experts=num_experts)
    if counts is None and probabilities is None:
        return {
            "load_cv": None,
            "dead_experts": None,
            "dead_expert_rate": None,
            "expert_utilization": None,
            "expert_counts": None,
            "routing_entropy": None,
            "status": NOT_AVAILABLE,
        }
    if counts is None:
        probs = _to_numpy(probabilities)
        try:
            import numpy as np  # type: ignore

            probs = np.asarray(probs, dtype=np.float64)
            if probs.ndim == 1:
                probs = probs.reshape(1, -1)
            counts = probs.sum(axis=0).tolist()
        except ImportError:  # pragma: no cover
            counts = [float(sum(row[index] for row in probs)) for index in range(len(probs[0]))]
    try:
        import numpy as np  # type: ignore

        counts_array = np.asarray(counts, dtype=np.float64)
        mean = float(counts_array.mean()) if counts_array.size else 0.0
        std = float(counts_array.std()) if counts_array.size else 0.0
        load_cv = std / mean if mean > 0 else None
        dead = int((counts_array <= 0).sum()) if counts_array.size else 0
        total = int(counts_array.size)
        utilization = float((counts_array > 0).mean()) if total else None
    except ImportError:  # pragma: no cover
        total = len(counts)
        mean = sum(counts) / total if total else 0.0
        variance = sum((item - mean) ** 2 for item in counts) / total if total else 0.0
        load_cv = math.sqrt(variance) / mean if mean > 0 else None
        dead = sum(item <= 0 for item in counts)
        utilization = sum(item > 0 for item in counts) / total if total else None
    entropy = None
    if probabilities is not None:
        probs = _to_numpy(probabilities)
        try:
            import numpy as np  # type: ignore

            probs = np.asarray(probs, dtype=np.float64)
            probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
            entropy = float((-probs * np.log(np.maximum(probs, 1e-12))).sum(axis=-1).mean())
        except (ImportError, AttributeError):  # pragma: no cover
            entropy = None
    return {
        "load_cv": load_cv,
        "dead_experts": dead,
        "dead_expert_rate": (dead / total if total else None),
        "expert_utilization": utilization,
        "expert_counts": [float(item) for item in counts],
        "routing_entropy": entropy,
        "status": "COMPUTED",
    }


def _metadata_value(metadata: Sequence[Mapping[str, Any]] | None, key: str, index: int, default: Any = None) -> Any:
    if metadata is None or index >= len(metadata):
        return default
    return metadata[index].get(key, default)


def _aggregate_arrays(
    teacher: Any,
    candidate: Any,
    *,
    mask: Any = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    epsilon: float = 1e-8,
    max_distribution_samples: int = 100_000,
    learned_assignments: Any = None,
    oracle_assignments: Any = None,
    learned_probabilities: Any = None,
    oracle_probabilities: Any = None,
    include_slices: bool = True,
) -> dict[str, Any]:
    import numpy as np  # type: ignore

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    teacher_rows = np.asarray(_flatten_rows(teacher), dtype=np.float32)
    candidate_rows = np.asarray(_flatten_rows(candidate), dtype=np.float32)
    if teacher_rows.shape != candidate_rows.shape:
        raise ValueError(f"teacher/candidate shape mismatch: {teacher_rows.shape} != {candidate_rows.shape}")
    count = int(teacher_rows.shape[0])
    row_mask = _mask_rows(mask, count)
    finite = np.isfinite(teacher_rows).all(axis=1) & np.isfinite(candidate_rows).all(axis=1)
    non_finite_count = int((row_mask & ~finite).sum())
    masked_count = int((~row_mask).sum())
    scored = row_mask & finite
    if not scored.any():
        empty = {
            "status": INSUFFICIENT_EVIDENCE,
            "scored_token_count": 0,
            "masked_token_count": masked_count,
            "non_finite_token_count": non_finite_count,
            "invalid_token_count": non_finite_count,
            "independent_group_count": len({str(_metadata_value(metadata, "independent_group", i, _metadata_value(metadata, "group_identity", i, i))) for i in range(count) if row_mask[i]}),
        }
        return empty

    t = teacher_rows[scored]
    c = candidate_rows[scored]
    t_norm = np.linalg.norm(t, axis=1)
    c_norm = np.linalg.norm(c, axis=1)
    target_energy = np.maximum(t_norm * t_norm, epsilon)
    diff = c - t
    token_nmse = np.sum(diff * diff, axis=1) / target_energy
    norm_error = np.abs(c_norm - t_norm) / np.maximum(t_norm, epsilon)
    ratio = c_norm / np.maximum(t_norm, epsilon)
    cosine = np.sum(c * t, axis=1) / np.maximum(c_norm * t_norm, epsilon)
    cosine = np.clip(cosine, -1.0, 1.0)
    sample_limit = max(0, int(max_distribution_samples))
    cosine_errors = (1.0 - cosine)[:sample_limit].astype(float).tolist()
    nmse_sample = token_nmse[:sample_limit].astype(float).tolist()
    independent_groups = []
    valid_indices = np.flatnonzero(scored)
    for original_index in valid_indices:
        value = _metadata_value(metadata, "independent_group", int(original_index), None)
        if value is None:
            value = _metadata_value(metadata, "group_identity", int(original_index), int(original_index))
        independent_groups.append(str(value))

    def q(values: Sequence[float], fraction: float) -> float | None:
        return _percentile(values, fraction)

    result: dict[str, Any] = {
        "status": "COMPUTED",
        "cosine_similarity": float(cosine.mean()),
        "normalized_mse": float(token_nmse.mean()),
        "target_relative_norm_error": float(norm_error.mean()),
        "mean_prediction_to_target_norm_ratio": float(ratio.mean()),
        "p95_abs_relative_norm_error": q(norm_error.astype(float).tolist(), 0.95),
        "p50_token_cosine_error": q(cosine_errors, 0.50),
        "p90_token_cosine_error": q(cosine_errors, 0.90),
        "p95_token_cosine_error": q(cosine_errors, 0.95),
        "p99_token_cosine_error": q(cosine_errors, 0.99),
        "p50_token_normalized_error": q(nmse_sample, 0.50),
        "p90_token_normalized_error": q(nmse_sample, 0.90),
        "p95_token_normalized_error": q(nmse_sample, 0.95),
        "p99_token_normalized_error": q(nmse_sample, 0.99),
        "scored_token_count": int(scored.sum()),
        "masked_token_count": masked_count,
        "non_finite_token_count": non_finite_count,
        "invalid_token_count": non_finite_count,
        "dropped_token_count": 0,
        "independent_group_count": len(set(independent_groups)),
        "epsilon": float(epsilon),
        "denominator_policy": "max(target_norm_sq or target_norm, epsilon)",
        "distribution_sample_count": len(cosine_errors),
        "target_norm_buckets": {},
        "hard_token_buckets": {},
    }
    learned = _routing_summary(learned_assignments, probabilities=learned_probabilities)
    oracle = _routing_summary(oracle_assignments, probabilities=oracle_probabilities)
    result.update(
        {
            "learned_load_cv": learned["load_cv"],
            "dead_expert_count": learned["dead_experts"],
            "dead_expert_rate": learned["dead_expert_rate"],
            "expert_utilization": learned["expert_utilization"],
            "expert_counts": learned["expert_counts"],
            "routing_entropy": learned["routing_entropy"],
            "oracle_load_cv": oracle["load_cv"],
            "oracle_dead_expert_count": oracle["dead_experts"],
            "oracle_expert_counts": oracle["expert_counts"],
            "learned_router_status": learned["status"],
            "oracle_router_status": oracle["status"],
        }
    )
    if result["learned_load_cv"] is not None and result["expert_utilization"] is not None:
        result["routing_health"] = float(result["expert_utilization"] * max(0.0, 1.0 - min(1.0, result["learned_load_cv"])))
    else:
        result["routing_health"] = None

    # Use manifest metadata when available.  Quantile boundaries are computed
    # from the scored token norms but the identities themselves are preserved.
    quantile_edges = np.quantile(t_norm, [0.25, 0.50, 0.75]).tolist() if len(t_norm) >= 4 else []
    if quantile_edges:
        bucket_names = ("q1", "q2", "q3", "q4")
        bucket_masks = (
            t_norm <= quantile_edges[0],
            (t_norm > quantile_edges[0]) & (t_norm <= quantile_edges[1]),
            (t_norm > quantile_edges[1]) & (t_norm <= quantile_edges[2]),
            t_norm > quantile_edges[2],
        )
        for bucket_name, bucket_mask in zip(bucket_names, bucket_masks):
            result["target_norm_buckets"][bucket_name] = {
                "count": int(bucket_mask.sum()),
                "cosine_similarity": float(cosine[bucket_mask].mean()) if bucket_mask.any() else None,
                "normalized_mse": float(token_nmse[bucket_mask].mean()) if bucket_mask.any() else None,
                "target_relative_norm_error": float(norm_error[bucket_mask].mean()) if bucket_mask.any() else None,
            }
    hard_mask = np.asarray([bool(_metadata_value(metadata, "hard_token", int(i), False)) for i in valid_indices], dtype=bool)
    if not hard_mask.any() and len(norm_error) >= 4:
        hard_mask = norm_error >= np.quantile(norm_error, 0.95)
    if hard_mask.any():
        result["hard_token_buckets"]["hard"] = {
            "count": int(hard_mask.sum()),
            "cosine_similarity": float(cosine[hard_mask].mean()),
            "normalized_mse": float(token_nmse[hard_mask].mean()),
            "target_relative_norm_error": float(norm_error[hard_mask].mean()),
        }
    if include_slices and metadata:
        for key, output_key in (("source_family", "source_family_slices"), ("domain", "domain_slices"), ("residual_difficulty", "residual_difficulty_slices")):
            values = [str(_metadata_value(metadata, key, int(i), "unknown")) for i in valid_indices]
            slices: dict[str, Any] = {}
            for identity in sorted(set(values)):
                selector = np.asarray([value == identity for value in values], dtype=bool)
                if not selector.any():
                    continue
                slice_result = _aggregate_arrays(
                    t[selector],
                    c[selector],
                    epsilon=epsilon,
                    max_distribution_samples=max_distribution_samples,
                    include_slices=False,
                )
                slice_result["gate_eligible"] = bool(
                    int(slice_result.get("scored_token_count", 0)) >= 16
                    and int(slice_result.get("independent_group_count", 0)) >= 2
                )
                if not slice_result["gate_eligible"]:
                    slice_result["status"] = INSUFFICIENT_EVIDENCE
                    slice_result["reason"] = "slice below minimum sample/group count; diagnostic-only"
                slices[identity] = slice_result
            if slices:
                result[output_key] = slices
    # Canonical field aliases are generated from the registry mapping rather
    # than maintained as a second formula implementation.
    result["metric_ids"] = {
        "cosine_similarity": canonical_metric_id("cosine_similarity"),
        "normalized_mse": canonical_metric_id("normalized_mse"),
        "target_relative_norm_error": canonical_metric_id("target_relative_norm_error"),
        "learned_load_cv": canonical_metric_id("loadcv"),
        "oracle_load_cv": canonical_metric_id("oracle_loadcv"),
        "dead_expert_count": canonical_metric_id("dead_experts"),
    }
    return result


@dataclass
class StructuralMetricsAccumulator:
    """Streaming accumulator with exact scalar aggregates and bounded samples.

    Earlier V2 prototypes concatenated every hidden-state row until
    ``finalize``.  That made a nominally streaming API retain corpus-sized
    tensors.  This implementation keeps only scalar sums, bounded scalar
    diagnostic samples, metadata identities, and expert-count vectors.
    """

    epsilon: float = 1e-8
    max_distribution_samples: int = 100_000
    _count: int = field(default=0, init=False, repr=False)
    _masked_count: int = field(default=0, init=False, repr=False)
    _non_finite_count: int = field(default=0, init=False, repr=False)
    _sum_cosine: float = field(default=0.0, init=False, repr=False)
    _sum_nmse: float = field(default=0.0, init=False, repr=False)
    _sum_norm_error: float = field(default=0.0, init=False, repr=False)
    _sum_ratio: float = field(default=0.0, init=False, repr=False)
    _group_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _cosine_errors: list[float] = field(default_factory=list, init=False, repr=False)
    _nmse_samples: list[float] = field(default_factory=list, init=False, repr=False)
    _norm_error_samples: list[float] = field(default_factory=list, init=False, repr=False)
    _target_norm_samples: list[float] = field(default_factory=list, init=False, repr=False)
    _hard_count: int = field(default=0, init=False, repr=False)
    _hard_sum_cosine: float = field(default=0.0, init=False, repr=False)
    _hard_sum_nmse: float = field(default=0.0, init=False, repr=False)
    _hard_sum_norm_error: float = field(default=0.0, init=False, repr=False)
    _slice_stats: dict[str, dict[str, dict[str, Any]]] = field(default_factory=lambda: defaultdict(dict), init=False, repr=False)
    _learned_counts: list[float] | None = field(default=None, init=False, repr=False)
    _oracle_counts: list[float] | None = field(default=None, init=False, repr=False)
    _row_offset: int = field(default=0, init=False, repr=False)

    def _add_assignments(self, current: list[float] | None, assignments: Any) -> list[float] | None:
        counts = _assignment_counts(assignments)
        if counts is None:
            return current
        width = max(len(current or []), len(counts))
        merged = [0.0] * width
        for index, value in enumerate(current or []):
            merged[index] += float(value)
        for index, value in enumerate(counts):
            merged[index] += float(value)
        return merged

    def _add_slice(self, key: str, identity: str, cosine: float, nmse: float, norm_error: float, group: str) -> None:
        bucket = self._slice_stats[key].setdefault(
            identity,
            {"count": 0, "sum_cosine": 0.0, "sum_nmse": 0.0, "sum_norm_error": 0.0, "groups": set()},
        )
        bucket["count"] += 1
        bucket["sum_cosine"] += float(cosine)
        bucket["sum_nmse"] += float(nmse)
        bucket["sum_norm_error"] += float(norm_error)
        bucket["groups"].add(str(group))

    def update(
        self,
        teacher: Any,
        candidate: Any,
        *,
        mask: Any = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        learned_assignments: Any = None,
        oracle_assignments: Any = None,
    ) -> "StructuralMetricsAccumulator":
        import numpy as np  # type: ignore

        teacher_rows = np.asarray(_flatten_rows(_to_numpy(teacher)), dtype=np.float32)
        candidate_rows = np.asarray(_flatten_rows(_to_numpy(candidate)), dtype=np.float32)
        if teacher_rows.shape != candidate_rows.shape:
            raise ValueError(f"teacher/candidate shape mismatch: {teacher_rows.shape} != {candidate_rows.shape}")
        count = int(teacher_rows.shape[0])
        row_mask = _mask_rows(mask, count)
        finite = np.isfinite(teacher_rows).all(axis=1) & np.isfinite(candidate_rows).all(axis=1)
        self._masked_count += int((~row_mask).sum())
        self._non_finite_count += int((row_mask & ~finite).sum())
        scored = row_mask & finite
        original_indices = np.flatnonzero(scored)
        if original_indices.size:
            t = teacher_rows[scored]
            c = candidate_rows[scored]
            t_norm = np.linalg.norm(t, axis=1)
            c_norm = np.linalg.norm(c, axis=1)
            target_energy = np.maximum(t_norm * t_norm, self.epsilon)
            diff = c - t
            token_nmse = np.sum(diff * diff, axis=1) / target_energy
            norm_error = np.abs(c_norm - t_norm) / np.maximum(t_norm, self.epsilon)
            ratio = c_norm / np.maximum(t_norm, self.epsilon)
            cosine = np.clip(np.sum(c * t, axis=1) / np.maximum(c_norm * t_norm, self.epsilon), -1.0, 1.0)
            self._count += int(original_indices.size)
            self._sum_cosine += float(cosine.sum())
            self._sum_nmse += float(token_nmse.sum())
            self._sum_norm_error += float(norm_error.sum())
            self._sum_ratio += float(ratio.sum())
            remaining = max(0, int(self.max_distribution_samples) - len(self._cosine_errors))
            if remaining:
                take = min(remaining, len(cosine))
                self._cosine_errors.extend((1.0 - cosine[:take]).astype(float).tolist())
                self._nmse_samples.extend(token_nmse[:take].astype(float).tolist())
                self._norm_error_samples.extend(norm_error[:take].astype(float).tolist())
                self._target_norm_samples.extend(t_norm[:take].astype(float).tolist())
            rows_metadata = metadata if metadata is not None else ()
            for local_index, original_index in enumerate(original_indices.tolist()):
                item = rows_metadata[original_index] if original_index < len(rows_metadata) else {}
                item = item if isinstance(item, Mapping) else {}
                group = item.get("independent_group", item.get("group_identity", self._row_offset + original_index))
                group_text = str(group)
                self._group_ids.add(group_text)
                for metadata_key, output_key in (("source_family", "source_family"), ("domain", "domain"), ("residual_difficulty", "residual_difficulty")):
                    if metadata_key in item:
                        self._add_slice(output_key, str(item[metadata_key]), float(cosine[local_index]), float(token_nmse[local_index]), float(norm_error[local_index]), group_text)
                if bool(item.get("hard_token", False)):
                    self._hard_count += 1
                    self._hard_sum_cosine += float(cosine[local_index])
                    self._hard_sum_nmse += float(token_nmse[local_index])
                    self._hard_sum_norm_error += float(norm_error[local_index])
        # Keep routing evidence as counts only; assignments are never stored.
        self._learned_counts = self._add_assignments(self._learned_counts, learned_assignments)
        self._oracle_counts = self._add_assignments(self._oracle_counts, oracle_assignments)
        self._row_offset += count
        return self

    def finalize(self) -> dict[str, Any]:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self._count <= 0:
            return {
                "status": INSUFFICIENT_EVIDENCE,
                "scored_token_count": 0,
                "masked_token_count": self._masked_count,
                "non_finite_token_count": self._non_finite_count,
                "invalid_token_count": self._non_finite_count,
                "independent_group_count": len(self._group_ids),
            }
        result: dict[str, Any] = {
            "status": "COMPUTED",
            "cosine_similarity": self._sum_cosine / self._count,
            "normalized_mse": self._sum_nmse / self._count,
            "target_relative_norm_error": self._sum_norm_error / self._count,
            "mean_prediction_to_target_norm_ratio": self._sum_ratio / self._count,
            "p95_abs_relative_norm_error": _percentile(self._norm_error_samples, 0.95),
            "p50_token_cosine_error": _percentile(self._cosine_errors, 0.50),
            "p90_token_cosine_error": _percentile(self._cosine_errors, 0.90),
            "p95_token_cosine_error": _percentile(self._cosine_errors, 0.95),
            "p99_token_cosine_error": _percentile(self._cosine_errors, 0.99),
            "p50_token_normalized_error": _percentile(self._nmse_samples, 0.50),
            "p90_token_normalized_error": _percentile(self._nmse_samples, 0.90),
            "p95_token_normalized_error": _percentile(self._nmse_samples, 0.95),
            "p99_token_normalized_error": _percentile(self._nmse_samples, 0.99),
            "scored_token_count": self._count,
            "masked_token_count": self._masked_count,
            "non_finite_token_count": self._non_finite_count,
            "invalid_token_count": self._non_finite_count,
            "dropped_token_count": 0,
            "independent_group_count": len(self._group_ids),
            "epsilon": float(self.epsilon),
            "denominator_policy": "max(target_norm_sq or target_norm, epsilon)",
            "distribution_sample_count": len(self._cosine_errors),
            "target_norm_buckets": {},
            "hard_token_buckets": {},
        }
        # Quantile buckets are deliberately based on bounded samples.  The
        # scalar aggregate metrics above remain exact across all updates.
        if len(self._target_norm_samples) >= 4:
            import numpy as np  # type: ignore

            norms = np.asarray(self._target_norm_samples, dtype=np.float64)
            cosine_errors = np.asarray(self._cosine_errors, dtype=np.float64)
            nmse = np.asarray(self._nmse_samples, dtype=np.float64)
            edges = np.quantile(norms, [0.25, 0.50, 0.75]).tolist()
            masks = (norms <= edges[0], (norms > edges[0]) & (norms <= edges[1]), (norms > edges[1]) & (norms <= edges[2]), norms > edges[2])
            for name, selector in zip(("q1", "q2", "q3", "q4"), masks):
                result["target_norm_buckets"][name] = {
                    "count": int(selector.sum()),
                    "cosine_similarity": float((1.0 - cosine_errors[selector]).mean()) if selector.any() else None,
                    "normalized_mse": float(nmse[selector].mean()) if selector.any() else None,
                }
        if self._hard_count:
            result["hard_token_buckets"]["hard"] = {
                "count": self._hard_count,
                "cosine_similarity": self._hard_sum_cosine / self._hard_count,
                "normalized_mse": self._hard_sum_nmse / self._hard_count,
                "target_relative_norm_error": self._hard_sum_norm_error / self._hard_count,
            }

        def routing_from_counts(counts: list[float] | None) -> dict[str, Any]:
            if counts is None:
                return {"load_cv": None, "dead_experts": None, "dead_expert_rate": None, "expert_utilization": None, "expert_counts": None, "routing_entropy": None, "status": NOT_AVAILABLE}
            import numpy as np  # type: ignore

            values = np.asarray(counts, dtype=np.float64)
            mean = float(values.mean()) if values.size else 0.0
            dead = int((values <= 0).sum()) if values.size else 0
            return {"load_cv": float(values.std() / mean) if mean > 0 else None, "dead_experts": dead, "dead_expert_rate": dead / len(values) if len(values) else None, "expert_utilization": float((values > 0).mean()) if len(values) else None, "expert_counts": [float(item) for item in values.tolist()], "routing_entropy": None, "status": "COMPUTED"}

        learned = routing_from_counts(self._learned_counts)
        oracle = routing_from_counts(self._oracle_counts)
        result.update({"learned_load_cv": learned["load_cv"], "dead_expert_count": learned["dead_experts"], "dead_expert_rate": learned["dead_expert_rate"], "expert_utilization": learned["expert_utilization"], "expert_counts": learned["expert_counts"], "routing_entropy": learned["routing_entropy"], "oracle_load_cv": oracle["load_cv"], "oracle_dead_expert_count": oracle["dead_experts"], "oracle_expert_counts": oracle["expert_counts"], "learned_router_status": learned["status"], "oracle_router_status": oracle["status"]})
        if result["learned_load_cv"] is not None and result["expert_utilization"] is not None:
            result["routing_health"] = float(result["expert_utilization"] * max(0.0, 1.0 - min(1.0, result["learned_load_cv"])))
        else:
            result["routing_health"] = None
        for key, output_key in (("source_family", "source_family_slices"), ("domain", "domain_slices"), ("residual_difficulty", "residual_difficulty_slices")):
            slices: dict[str, Any] = {}
            for identity, values in self._slice_stats.get(key, {}).items():
                groups = values["groups"]
                gate_eligible = values["count"] >= 16 and len(groups) >= 2
                slices[identity] = {
                    "status": "COMPUTED" if gate_eligible else INSUFFICIENT_EVIDENCE,
                    "cosine_similarity": values["sum_cosine"] / values["count"],
                    "normalized_mse": values["sum_nmse"] / values["count"],
                    "target_relative_norm_error": values["sum_norm_error"] / values["count"],
                    "scored_token_count": values["count"],
                    "independent_group_count": len(groups),
                    "gate_eligible": gate_eligible,
                    **({} if gate_eligible else {"reason": "slice below minimum sample/group count; diagnostic-only"}),
                }
            if slices:
                result[output_key] = slices
        result["metric_ids"] = {"cosine_similarity": canonical_metric_id("cosine_similarity"), "normalized_mse": canonical_metric_id("normalized_mse"), "target_relative_norm_error": canonical_metric_id("target_relative_norm_error"), "learned_load_cv": canonical_metric_id("loadcv"), "oracle_load_cv": canonical_metric_id("oracle_loadcv"), "dead_expert_count": canonical_metric_id("dead_experts")}
        return result


def compute_structural_metrics(
    teacher: Any,
    candidate: Any,
    *,
    mask: Any = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    epsilon: float = 1e-8,
    max_distribution_samples: int = 100_000,
    learned_assignments: Any = None,
    oracle_assignments: Any = None,
    learned_probabilities: Any = None,
    oracle_probabilities: Any = None,
) -> dict[str, Any]:
    """Compute one split's structural V2 metrics without combining splits."""

    return _aggregate_arrays(
        teacher,
        candidate,
        mask=mask,
        metadata=metadata,
        epsilon=epsilon,
        max_distribution_samples=max_distribution_samples,
        learned_assignments=learned_assignments,
        oracle_assignments=oracle_assignments,
        learned_probabilities=learned_probabilities,
        oracle_probabilities=oracle_probabilities,
    )


structural_metrics_v2 = compute_structural_metrics
evaluate_structural_metrics = compute_structural_metrics


__all__ = [
    "StructuralMetricsAccumulator",
    "compute_structural_metrics",
    "evaluate_structural_metrics",
    "structural_metrics_v2",
]
