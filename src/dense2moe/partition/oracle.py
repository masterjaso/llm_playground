"""Frozen-slice routing diagnostics and exact small-k oracle baselines.

The functions in this module operate on a *frozen* partition of the dense
FFN.  They are useful for separating three questions that used to be mixed
together in the original ``oracle_topk`` diagnostic:

* can a normalized top-k convex combination of the unchanged slices fit the
  teacher residual (the simplex oracle)?
* does allowing non-negative output magnitude change the answer (the positive
  oracle)?
* do globally learned scalar output corrections help when the router remains
  normalized (the scaled-router oracle)?

These are structural diagnostics, not a hard ceiling on a trainable student.
The source slices are frozen, the routing oracle is target-assisted, and the
relaxed coefficients do not represent a jointly trained router or experts.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ffn import PartitionPlan


def _np():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError as exc:
        raise RuntimeError("numpy is required for oracle routing") from exc


def swiglu_contributions(inputs: Any, gate_proj: Any, up_proj: Any, down_proj: Any, plan: PartitionPlan) -> tuple[Any, Any]:
    """Return shared and routed per-expert outputs for each token."""

    np = _np()
    x = np.asarray(inputs)
    gate = np.asarray(gate_proj)
    up = np.asarray(up_proj)
    down = np.asarray(down_proj)
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("SwiGLU projection shapes do not match")
    plan.validate()

    def contribution(indices: Iterable[int]) -> Any:
        selected = list(indices)
        hidden = (x @ gate[selected].T) / (1.0 + np.exp(-(x @ gate[selected].T)))
        hidden *= x @ up[selected].T
        return hidden @ down[:, selected].T

    shared = contribution(plan.shared_indices)
    routed = np.stack([contribution(group) for group in plan.expert_indices], axis=1)
    return shared, routed


def _validate_oracle_arrays(shared: Any, routed: Any, target: Any, top_k: int) -> tuple[Any, Any, Any]:
    np = _np()
    shared_values = np.asarray(shared, dtype=np.float64)
    routed_values = np.asarray(routed, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if shared_values.ndim != 2 or routed_values.ndim != 3 or target_values.ndim != 2:
        raise ValueError("shared/target must be [tokens, output] and routed [tokens, experts, output]")
    if shared_values.shape != target_values.shape or routed_values.shape[0] != target_values.shape[0] or routed_values.shape[2] != target_values.shape[1]:
        raise ValueError("shared, routed, and target shapes do not match")
    if top_k <= 0 or top_k > routed_values.shape[1]:
        raise ValueError("invalid top_k")
    return shared_values, routed_values, target_values


def _candidate_error(matrix: Any, weights: Any, residual: Any) -> float:
    np = _np()
    prediction = np.asarray(matrix) @ np.asarray(weights)
    return float(np.mean((prediction - np.asarray(residual)) ** 2))


def _simplex_pair_weights(first: Any, second: Any, residual: Any) -> Any:
    """Solve the exact two-vector simplex problem in closed form.

    For ``a`` and ``b`` the objective is ``||alpha*a + (1-alpha)*b-r||²``.
    The unconstrained minimizer is projected onto ``[0, 1]``; this projection
    is exact because the objective is a one-dimensional convex quadratic.
    """

    np = _np()
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    direction = a - b
    denominator = float(direction @ direction)
    if denominator <= 1e-30:
        # Every alpha has the same prediction for coincident slices.  The
        # midpoint makes the result symmetric and deterministic.
        alpha = 0.5
    else:
        alpha = float(np.clip((direction @ (r - b)) / denominator, 0.0, 1.0))
    return np.asarray([alpha, 1.0 - alpha], dtype=np.float64)


def _simplex_weights_exact(matrix: Any, residual: Any) -> Any:
    """Solve a small equality-simplex least-squares problem exactly.

    The top-2 path is the closed form above.  For larger diagnostic ``top_k``
    values, enumerating active faces of the simplex gives the exact solution
    for the small values used by this project while retaining deterministic
    behavior for diagnostic profiles.
    """

    np = _np()
    values = np.asarray(matrix, dtype=np.float64)
    goal = np.asarray(residual, dtype=np.float64)
    columns = values.shape[1]
    if columns == 1:
        return np.ones(1, dtype=np.float64)
    if columns == 2:
        return _simplex_pair_weights(values[:, 0], values[:, 1], goal)

    best: Any | None = None
    best_error = float("inf")
    for width in range(1, columns + 1):
        for active in itertools.combinations(range(columns), width):
            if width == 1:
                candidate = np.ones(1, dtype=np.float64)
            else:
                active_matrix = values[:, list(active)]
                gram = active_matrix.T @ active_matrix
                rhs = active_matrix.T @ goal
                kkt = np.block([
                    [gram, np.ones((width, 1), dtype=np.float64)],
                    [np.ones((1, width), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
                ])
                rhs_kkt = np.concatenate([rhs, np.ones(1, dtype=np.float64)])
                try:
                    solution = np.linalg.solve(kkt, rhs_kkt)[:-1]
                except np.linalg.LinAlgError:
                    solution = np.linalg.lstsq(kkt, rhs_kkt, rcond=None)[0][:-1]
                candidate = np.asarray(solution, dtype=np.float64)
            if not np.isfinite(candidate).all() or np.min(candidate) < -1e-9:
                continue
            candidate = np.maximum(candidate, 0.0)
            candidate /= float(candidate.sum())
            full = np.zeros(columns, dtype=np.float64)
            full[list(active)] = candidate
            error = _candidate_error(values, full, goal)
            if error < best_error - 1e-15:
                best_error = error
                best = full
    if best is None:
        # A vertex is always feasible.  This branch only handles severe
        # floating-point pathologies in degenerate matrices.
        best = np.zeros(columns, dtype=np.float64)
        best[0] = 1.0
    return best


def _positive_pair_weights(first: Any, second: Any, residual: Any) -> Any:
    """Solve exact two-variable NNLS, including both boundary solutions."""

    np = _np()
    matrix = np.column_stack((np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)))
    goal = np.asarray(residual, dtype=np.float64)
    candidates: list[Any] = [np.zeros(2, dtype=np.float64)]

    # Interior unconstrained least-squares solution.
    gram = matrix.T @ matrix
    rhs = matrix.T @ goal
    try:
        unconstrained = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        unconstrained = np.linalg.lstsq(matrix, goal, rcond=None)[0]
    if np.isfinite(unconstrained).all() and np.min(unconstrained) >= -1e-9:
        candidates.append(np.maximum(unconstrained, 0.0))

    # Both one-dimensional faces, plus the origin above.
    first_norm = float(matrix[:, 0] @ matrix[:, 0])
    if first_norm > 1e-30:
        candidates.append(np.asarray([max(float(matrix[:, 0] @ goal) / first_norm, 0.0), 0.0], dtype=np.float64))
    second_norm = float(matrix[:, 1] @ matrix[:, 1])
    if second_norm > 1e-30:
        candidates.append(np.asarray([0.0, max(float(matrix[:, 1] @ goal) / second_norm, 0.0)], dtype=np.float64))

    errors = [_candidate_error(matrix, candidate, goal) for candidate in candidates]
    best_index = min(range(len(candidates)), key=lambda index: (errors[index], index))
    return candidates[best_index]


def _positive_weights_exact(matrix: Any, residual: Any) -> Any:
    """Solve small non-negative least squares by active-face enumeration."""

    np = _np()
    values = np.asarray(matrix, dtype=np.float64)
    goal = np.asarray(residual, dtype=np.float64)
    columns = values.shape[1]
    if columns == 1:
        denominator = float(values[:, 0] @ values[:, 0])
        if denominator <= 1e-30:
            return np.zeros(1, dtype=np.float64)
        return np.asarray([max(float(values[:, 0] @ goal) / denominator, 0.0)], dtype=np.float64)
    if columns == 2:
        return _positive_pair_weights(values[:, 0], values[:, 1], goal)

    best = np.zeros(columns, dtype=np.float64)
    best_error = _candidate_error(values, best, goal)
    for width in range(1, columns + 1):
        for active in itertools.combinations(range(columns), width):
            active_matrix = values[:, list(active)]
            try:
                candidate = np.linalg.lstsq(active_matrix, goal, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(candidate).all() or np.min(candidate) < -1e-9:
                continue
            full = np.zeros(columns, dtype=np.float64)
            full[list(active)] = np.maximum(candidate, 0.0)
            error = _candidate_error(values, full, goal)
            if error < best_error - 1e-15:
                best_error = error
                best = full
    return best


def _metrics(shared: Any, routed: Any, target: Any, ids: Any, weights: Any) -> dict[str, Any]:
    np = _np()
    shared_values = np.asarray(shared, dtype=np.float64)
    routed_values = np.asarray(routed, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    selected = np.zeros_like(shared_values)
    residual_by_token = np.zeros(target_values.shape[0], dtype=np.float64)
    for token in range(target_values.shape[0]):
        selected[token] = sum(
            float(weights[token, slot]) * routed_values[token, int(ids[token, slot])]
            for slot in range(ids.shape[1])
        )
        residual_by_token[token] = float(np.mean((selected[token] - (target_values[token] - shared_values[token])) ** 2))
    reconstruction = shared_values + selected
    mse = float(np.mean((reconstruction - target_values) ** 2))
    norm = float(np.mean(target_values**2)) + 1e-12
    dot = np.sum(reconstruction * target_values, axis=-1)
    cosine = float(np.mean(dot / ((np.linalg.norm(reconstruction, axis=-1) * np.linalg.norm(target_values, axis=-1)) + 1e-12)))
    return {
        "indices": np.asarray(ids, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float64),
        "reconstruction": reconstruction,
        "mse": mse,
        "normalized_mse": mse / norm,
        "cosine": cosine,
        "residual_by_token": residual_by_token,
    }


def _load_metrics(
    shared: Any,
    routed: Any,
    target: Any,
    ids: Any,
    weights: Any,
    *,
    hard_fraction: float = 0.25,
) -> dict[str, Any]:
    """Return reconstruction and dispatch-load metrics for an assignment.

    ``_metrics`` predates the load-aware diagnostic and intentionally keeps a
    small compatibility surface.  This companion adds the names used by the
    product gate (``global_nmse``, hard-quartile cosine, expert usage and dead
    experts) without changing the historical oracle result shape.
    """

    np = _np()
    shared_values = np.asarray(shared, dtype=np.float64)
    routed_values = np.asarray(routed, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    ids_values = np.asarray(ids, dtype=np.int64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if ids_values.ndim != 2 or weight_values.shape != ids_values.shape:
        raise ValueError("ids and weights must both be [tokens, top_k]")
    if ids_values.shape[0] != target_values.shape[0] or ids_values.shape[1] <= 0:
        raise ValueError("route count does not match target rows")
    if np.any(ids_values < 0) or np.any(ids_values >= routed_values.shape[1]):
        raise ValueError("route contains an expert outside routed")
    prediction = shared_values.copy()
    for slot in range(ids_values.shape[1]):
        prediction += weight_values[:, slot, None] * routed_values[np.arange(ids_values.shape[0]), ids_values[:, slot]]
    error = np.sum((prediction - target_values) ** 2, axis=1)
    target_norm = np.sum(target_values**2, axis=1)
    cosine = np.sum(prediction * target_values, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target_values, axis=1) + 1e-12
    )
    experts = int(routed_values.shape[1])
    usage = np.bincount(ids_values.reshape(-1), minlength=experts).astype(np.int64)
    usage_fraction = usage / max(ids_values.shape[0] * ids_values.shape[1], 1)
    load_cv = float(usage.std() / max(usage.mean(), 1e-12))
    hard_fraction = float(hard_fraction)
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")
    # Residual magnitude is known to dominate the selector's remaining error;
    # use it for a deterministic, target-assisted diagnostic quartile.
    hardness = np.linalg.norm(target_values - shared_values, axis=1)
    hard_count = max(1, int(np.ceil(hardness.shape[0] * hard_fraction)))
    hard_indices = np.argsort(-hardness, kind="stable")[:hard_count]
    return {
        "tokens": int(target_values.shape[0]),
        "mse": float(np.mean((prediction - target_values) ** 2)),
        "normalized_mse": float(np.mean((prediction - target_values) ** 2) / (np.mean(target_values**2) + 1e-12)),
        "global_nmse": float(np.sum(error) / (np.sum(target_norm) + 1e-12)),
        "mean_token_relative_mse": float(np.mean(error / np.maximum(target_norm, 1e-12))),
        "cosine": float(np.mean(cosine)),
        "hard_quartile_cosine": float(np.mean(cosine[hard_indices])),
        "expert_usage_counts": usage.tolist(),
        "expert_usage_fraction": usage_fraction.tolist(),
        "dead_experts": int(np.sum(usage == 0)),
        "load_cv": load_cv,
        "indices": ids_values,
        "weights": weight_values,
        "reconstruction": prediction,
        "residual_by_token": error / max(int(target_values.shape[1]), 1),
        "hard_indices": hard_indices,
    }


def _candidate_routes(
    routed: Any,
    residual: Any,
    top_k: int,
    *,
    simplex: bool,
    candidate_pool_size: int | None,
    max_combinations: int,
) -> list[tuple[float, tuple[int, ...], Any]]:
    """Build a bounded per-token route candidate set.

    p16/top4 has only 1,820 sets, so the default path is exhaustive.  For
    p32/top4/top5 the full combination count is much larger; the deterministic
    correlation pool keeps the diagnostic tractable while still exposing
    alternatives for price-based balancing.  The candidate metadata in the
    returned report makes this distinction explicit.
    """

    np = _np()
    routed_values = np.asarray(routed, dtype=np.float64)
    residual_value = np.asarray(residual, dtype=np.float64)
    experts = int(routed_values.shape[0])
    if top_k <= 0 or top_k > experts:
        raise ValueError("invalid top_k")
    if max_combinations <= 0:
        raise ValueError("max_combinations must be positive")
    correlations = routed_values @ residual_value
    norms = np.linalg.norm(routed_values, axis=1)
    pool = min(experts, int(candidate_pool_size or experts))
    # Exact enumeration is preferred whenever it is bounded.  Otherwise use
    # a larger pool than top-k and deterministic norm/correlation alternatives.
    combination_count = math.comb(pool, top_k) if pool >= top_k else 0
    if combination_count <= max_combinations:
        order = np.argsort(-correlations, kind="stable")[:pool]
        combinations = list(itertools.combinations((int(value) for value in order), top_k))
    else:
        pool = min(experts, max(top_k, int(candidate_pool_size or (2 * top_k + 4))))
        correlation_order = np.argsort(-correlations, kind="stable")[:pool]
        norm_order = np.argsort(-norms, kind="stable")[:pool]
        if math.comb(pool, top_k) <= max_combinations:
            # The fallback pool is small enough to retain every set.  This is
            # the preferred strongest-practical p32 screen (C(12,5)=792 by
            # default), while avoiding the full 201,376-set enumeration.
            combinations = list(itertools.combinations((int(value) for value in correlation_order), top_k))
            result: list[tuple[float, tuple[int, ...], Any]] = []
            for combo in combinations:
                matrix = routed_values[list(combo)].T
                weights = _simplex_weights_exact(matrix, residual_value) if simplex else _positive_weights_exact(matrix, residual_value)
                error = _candidate_error(matrix, weights, residual_value)
                result.append((float(error), tuple(int(value) for value in combo), weights))
            result.sort(key=lambda item: (item[0], item[1]))
            return result
        combinations_set: set[tuple[int, ...]] = set()
        combinations_set.add(tuple(sorted(int(value) for value in correlation_order[:top_k])))
        combinations_set.add(tuple(sorted(int(value) for value in norm_order[:top_k])))
        # Replacing one slot gives the Lagrangian loop useful alternatives
        # without pretending to be an exhaustive p32 oracle.
        base = min(combinations_set)
        for replacement in correlation_order:
            replacement_value = int(replacement)
            for slot in range(top_k):
                candidate = list(base)
                candidate[slot] = replacement_value
                if len(set(candidate)) == top_k:
                    combinations_set.add(tuple(sorted(candidate)))
        combinations = sorted(combinations_set)[:max_combinations]
    result: list[tuple[float, tuple[int, ...], Any]] = []
    for combo in combinations:
        matrix = routed_values[list(combo)].T
        weights = _simplex_weights_exact(matrix, residual_value) if simplex else _positive_weights_exact(matrix, residual_value)
        error = _candidate_error(matrix, weights, residual_value)
        result.append((float(error), tuple(int(value) for value in combo), weights))
    result.sort(key=lambda item: (item[0], item[1]))
    if not result:
        raise RuntimeError("route candidate generation produced no feasible set")
    return result


def _assign_priced_routes(
    candidates: Sequence[Sequence[tuple[float, tuple[int, ...], Any]]],
    prices: Any,
    *,
    penalty: float,
) -> tuple[Any, Any, Any]:
    np = _np()
    selected_ids: list[tuple[int, ...]] = []
    selected_weights: list[Any] = []
    errors = np.zeros(len(candidates), dtype=np.float64)
    provisional_usage = np.zeros(len(prices), dtype=np.int64)
    for token, options in enumerate(candidates):
        scored = [
            (
                item[0] + float(penalty) * sum(float(prices[index]) for index in item[1]),
                item[0],
                item,
            )
            for item in options
        ]
        minimum = min(value[0] for value in scored)
        # Independent Lagrangian choices can oscillate when several tokens
        # have exactly the same reconstruction cost (a common occurrence for
        # symmetric partitions).  Among numerically tied choices, assign the
        # least-used candidate first; this is a deterministic bounded repair,
        # not an unrecorded capacity constraint.
        tied = [value for value in scored if value[0] <= minimum + 1e-12]
        _, _, best = min(
            tied,
            key=lambda value: (
                sum(int(provisional_usage[index]) for index in value[2][1]),
                value[1],
                value[2][1],
            ),
        )
        errors[token] = best[0]
        selected_ids.append(best[1])
        selected_weights.append(best[2])
        for index in best[1]:
            provisional_usage[index] += 1
    return np.asarray(selected_ids, dtype=np.int64), np.asarray(selected_weights, dtype=np.float64), errors


def _assign_unconstrained_routes(candidates: Sequence[Sequence[tuple[float, tuple[int, ...], Any]]]) -> tuple[Any, Any]:
    """Select each token's best reconstruction route without load tie-breaks."""

    np = _np()
    selected = [min(options, key=lambda item: (item[0], item[1])) for options in candidates]
    return (
        np.asarray([item[1] for item in selected], dtype=np.int64),
        np.asarray([item[2] for item in selected], dtype=np.float64),
    )


def _pareto_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep points not dominated in cosine, NMSE, and load CV."""

    frontier: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        cosine = float(point["cosine"])
        nmse = float(point["global_nmse"])
        load_cv = float(point["load_cv"])
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            other_cosine = float(other["cosine"])
            other_nmse = float(other["global_nmse"])
            other_load_cv = float(other["load_cv"])
            no_worse = (
                other_cosine >= cosine - 1e-12
                and other_nmse <= nmse + 1e-12
                and other_load_cv <= load_cv + 1e-12
            )
            strictly_better = (
                other_cosine > cosine + 1e-12
                or other_nmse < nmse - 1e-12
                or other_load_cv < load_cv - 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(dict(point))
    return sorted(frontier, key=lambda point: (float(point["load_cv"]), float(point["global_nmse"]), -float(point["cosine"])))


def _stream_validate_oracle_arrays(shared: Any, routed: Any, target: Any, top_k: int) -> tuple[Any, Any, Any]:
    """Validate without copying full real-scale contribution tensors."""

    np = _np()
    shared_values = np.asarray(shared)
    routed_values = np.asarray(routed)
    target_values = np.asarray(target)
    if shared_values.ndim != 2 or routed_values.ndim != 3 or target_values.ndim != 2:
        raise ValueError("shared/target must be [tokens, output] and routed [tokens, experts, output]")
    if shared_values.shape != target_values.shape or routed_values.shape[0] != target_values.shape[0] or routed_values.shape[2] != target_values.shape[1]:
        raise ValueError("shared, routed, and target shapes do not match")
    if top_k <= 0 or top_k > routed_values.shape[1]:
        raise ValueError("invalid top_k")
    return shared_values, routed_values, target_values


def _candidate_templates(
    experts: int,
    top_k: int,
    *,
    candidate_pool_size: int | None,
    max_combinations: int,
) -> tuple[Any, int, int, str]:
    """Return compact global or local candidate identities."""

    np = _np()
    if max_combinations <= 0:
        raise ValueError("max_combinations must be positive")
    configured_pool = experts if candidate_pool_size is None else min(experts, int(candidate_pool_size))
    configured_pool = max(configured_pool, top_k)
    full_count = math.comb(configured_pool, top_k)
    if configured_pool == experts and full_count <= max_combinations:
        return (
            np.asarray(list(itertools.combinations(range(experts), top_k)), dtype=np.int16 if experts <= 32767 else np.int32),
            experts,
            full_count,
            "exact_candidate_sets",
        )
    # p32 uses a deterministic correlation-ranked pool.  The template stores
    # local positions in that pool; each token maps them to actual experts.
    effective_pool = configured_pool
    if candidate_pool_size is None:
        effective_pool = min(experts, max(top_k, 2 * top_k + 4))
    combination_count = math.comb(effective_pool, top_k)
    if combination_count <= max_combinations:
        templates = list(itertools.combinations(range(effective_pool), top_k))
    else:
        combinations_set: set[tuple[int, ...]] = {tuple(range(top_k))}
        for replacement in range(effective_pool):
            for slot in range(top_k):
                candidate = list(range(top_k))
                candidate[slot] = replacement
                if len(set(candidate)) == top_k:
                    combinations_set.add(tuple(sorted(candidate)))
        templates = sorted(combinations_set)[:max_combinations]
    return (
        np.asarray(templates, dtype=np.int16),
        effective_pool,
        len(templates),
        "bounded_correlation_candidate_pool",
    )


def _batched_projected_fit(
    gram: Any,
    rhs: Any,
    residual_norm: Any,
    hidden: int,
    *,
    simplex: bool,
) -> tuple[Any, Any]:
    """Project batched candidate normal equations onto the route constraint."""

    np = _np()
    gram = np.asarray(gram, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    residual_norm = np.asarray(residual_norm, dtype=np.float32)
    batch, candidates, width, _ = gram.shape
    work_dtype = np.dtype(np.float32)
    ridge = np.asarray(1e-6, dtype=work_dtype)
    gram += ridge * np.eye(width, dtype=work_dtype)
    if simplex:
        kkt = np.zeros((batch, candidates, width + 1, width + 1), dtype=work_dtype)
        kkt[..., :width, :width] = gram
        kkt[..., :width, -1] = 1.0
        kkt[..., -1, :width] = 1.0
        kkt_rhs = np.zeros((batch, candidates, width + 1), dtype=work_dtype)
        kkt_rhs[..., :width] = rhs
        kkt_rhs[..., -1] = 1.0
        try:
            coefficients = np.linalg.solve(kkt, kkt_rhs[..., None])[..., :width, 0]
        except np.linalg.LinAlgError:
            coefficients = np.zeros((batch, candidates, width), dtype=work_dtype)
    else:
        try:
            coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = np.zeros((batch, candidates, width), dtype=work_dtype)
    coefficients = np.where(np.isfinite(coefficients), coefficients, 0.0)
    coefficients = np.maximum(coefficients, 0.0)
    if simplex:
        coefficient_sum = np.sum(coefficients, axis=-1, keepdims=True)
        normalized = coefficients / np.maximum(coefficient_sum, 1e-12)
        invalid = coefficient_sum[..., 0] <= 1e-12
        if np.any(invalid):
            vertex = np.argmax(rhs, axis=-1)
            fallback = np.eye(width, dtype=work_dtype)[vertex]
            normalized = np.where(invalid[..., None], fallback, normalized)
        coefficients = normalized
    else:
        # If every unconstrained coefficient is negative, compare the zero
        # vector with the best positive vertex instead of retaining a useless
        # all-zero route for pricing.
        invalid = np.sum(coefficients, axis=-1) <= 1e-12
        if np.any(invalid):
            vertex = np.argmax(rhs, axis=-1)
            diagonal = np.diagonal(gram, axis1=-2, axis2=-1)
            vertex_denominator = np.maximum(np.take_along_axis(diagonal, vertex[..., None], axis=-1)[..., 0], ridge)
            vertex_value = np.maximum(np.take_along_axis(rhs, vertex[..., None], axis=-1)[..., 0] / vertex_denominator, 0.0)
            fallback = np.eye(width, dtype=work_dtype)[vertex] * vertex_value[..., None]
            zero_error = residual_norm[:, None] / max(hidden, 1)
            vertex_error = (
                residual_norm[:, None]
                - 2.0 * vertex_value * np.take_along_axis(rhs, vertex[..., None], axis=-1)[..., 0]
                + vertex_value * vertex_value * vertex_denominator
            ) / max(hidden, 1)
            use_vertex = invalid & (vertex_error < zero_error)
            coefficients = np.where(use_vertex[..., None], fallback, coefficients)
    error = (
        residual_norm[:, None]
        - 2.0 * np.sum(coefficients * rhs, axis=-1)
        + np.einsum("bck,bckl,bcl->bc", coefficients, gram, coefficients, dtype=work_dtype)
    ) / max(hidden, 1)
    # A singular/collinear KKT system can return a minimum-norm solution that
    # projects poorly (for example [1, -1] for a positive two-vector fit).
    # Compare each one-vector face in a bounded width-sized loop; this keeps
    # the work vectorized over tokens/candidates and recovers the important
    # boundary cases without enumerating all 2^k faces.
    for slot in range(width):
        if simplex:
            vertex_coefficient = np.ones((batch, candidates), dtype=work_dtype)
        else:
            denominator = np.maximum(gram[..., slot, slot], ridge)
            vertex_coefficient = np.maximum(rhs[..., slot] / denominator, 0.0)
        vertex_error = (
            residual_norm[:, None]
            - 2.0 * vertex_coefficient * rhs[..., slot]
            + vertex_coefficient * vertex_coefficient * np.maximum(gram[..., slot, slot], ridge)
        ) / max(hidden, 1)
        update = vertex_error < error
        vertex_weights = np.zeros_like(coefficients)
        vertex_weights[..., slot] = vertex_coefficient
        coefficients = np.where(update[..., None], vertex_weights, coefficients)
        error = np.where(update, vertex_error, error)
    return error, coefficients


def _batched_candidate_fit(vectors: Any, residual: Any, *, simplex: bool) -> tuple[Any, Any]:
    """Fit bounded float32 coefficients for [batch, candidates, k, hidden].

    A full active-face NNLS enumeration is exact for tiny diagnostic calls but
    becomes the dominant cost at 16k tokens × 1,820 p16 candidates.  The
    real-scale path uses one ridge-stabilized batched KKT solve, followed by a
    deterministic projection onto the non-negative orthant (and simplex when
    requested).  Selected routes are solved again in bounded batches before
    final metrics, so candidate pricing never requires Python route objects or
    an unbounded per-token optimizer.
    """

    np = _np()
    values = np.asarray(vectors, dtype=np.float32)
    goal = np.asarray(residual, dtype=np.float32)
    _batch, _candidates, _width, hidden = values.shape
    gram = np.einsum("bckh,bclh->bckl", values, values, dtype=np.float32)
    rhs = np.einsum("bckh,bh->bck", values, goal, dtype=np.float32)
    residual_norm = np.sum(goal * goal, axis=1, dtype=np.float32)
    return _batched_projected_fit(gram, rhs, residual_norm, hidden, simplex=simplex)


def _batched_exact_positive_fit(vectors: Any, residual: Any) -> tuple[Any, Any]:
    """Exact non-negative active-face refit for a bounded selected-route batch.

    Candidate pricing at real scale intentionally uses the projected float32
    scorer above.  Once a route is selected, however, ``top_k`` is small and
    the final coefficients can be solved exactly by enumerating its active
    faces.  This vectorized implementation keeps the refit bounded over
    ``[batch, candidate, top_k]`` route systems and avoids the old per-token
    Python route-object loop.
    """

    np = _np()
    values = np.asarray(vectors, dtype=np.float64)
    goal = np.asarray(residual, dtype=np.float64)
    batch, candidates, top_k, hidden = values.shape
    if candidates != 1:
        raise ValueError("exact selected-route refit expects one candidate per token")
    gram = np.einsum("bckh,bclh->bckl", values, values, dtype=np.float64)
    rhs = np.einsum("bckh,bh->bck", values, goal, dtype=np.float64)
    residual_norm = np.sum(goal * goal, axis=1, dtype=np.float64)[:, None]
    best_error = residual_norm.copy()
    best_weights = np.zeros((batch, candidates, top_k), dtype=np.float64)
    for mask in range(1, 1 << top_k):
        active = tuple(index for index in range(top_k) if mask & (1 << index))
        width = len(active)
        raw_g = np.take(np.take(gram, active, axis=2), active, axis=3)
        raw_b = np.take(rhs, active, axis=2)
        flat_g = raw_g.reshape(batch * candidates, width, width)
        flat_b = raw_b.reshape(batch * candidates, width)
        try:
            flat_solution = np.linalg.solve(flat_g, flat_b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # ``numpy.linalg.lstsq`` is not batch-aware on all supported
            # NumPy versions.  Singular/collinear faces are uncommon for
            # learned routes but are expected in deterministic fixtures, so
            # keep this bounded fallback explicitly per selected system.
            flat_solution = np.stack(
                [np.linalg.lstsq(matrix, vector, rcond=None)[0] for matrix, vector in zip(flat_g, flat_b)],
                axis=0,
            )
        solution = flat_solution.reshape(batch, candidates, width)
        valid = np.isfinite(solution).all(axis=-1) & (solution.min(axis=-1) >= -1e-9)
        solution = np.maximum(solution, 0.0)
        quadratic = np.einsum("bcw,bcwv,bcv->bc", solution, raw_g, solution, dtype=np.float64)
        candidate_error = residual_norm - 2.0 * np.sum(solution * raw_b, axis=-1) + quadratic
        candidate_error = np.where(valid, candidate_error, np.inf)
        update = candidate_error < best_error
        if np.any(update):
            full = np.zeros_like(best_weights)
            full[:, :, active] = solution
            best_weights = np.where(update[..., None], full, best_weights)
            best_error = np.where(update, candidate_error, best_error)
    return best_error / max(hidden, 1), best_weights.astype(np.float32)


def _batched_candidate_vectors(routed: Any, candidate_ids: Any) -> Any:
    np = _np()
    routed_values = np.asarray(routed)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.ndim == 2:
        # Selected routes are [batch, top_k]; add the singleton candidate
        # axis.  Broadcasting a leading batch axis here would accidentally
        # create [batch, batch, top_k] and make ``fitted[:, 0]`` use the first
        # token's route for every token.
        ids = ids[:, None, :]
    return np.take_along_axis(routed_values[:, None, :, :], ids[..., None], axis=2)


def _score_candidate_batches(
    shared: Any,
    routed: Any,
    target: Any,
    templates: Any,
    *,
    exact_global: bool,
    effective_pool: int,
    top_k: int,
    simplex: bool,
    batch_size: int,
    max_in_memory_bytes: int,
    storage_dir: Path | None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Stream candidate scoring into compact arrays/memmaps."""

    np = _np()
    tokens, experts, hidden = routed.shape
    candidates = int(templates.shape[0])
    itemsize = np.dtype(np.float32).itemsize
    requested_batch = max(1, min(int(batch_size), tokens))
    # A caller may have a large historical batch-size default even when the
    # contribution store is a read-only memmap.  Bound the decompressed input
    # window itself before considering candidate score storage; otherwise one
    # ``routed[start:stop]`` slice can still recreate the multi-gigabyte NPZ
    # failure that this streaming path is meant to avoid.
    input_bytes_per_token = ((experts + 2) * hidden + experts * experts + experts) * itemsize
    max_batch_by_input = int(max_in_memory_bytes // max(input_bytes_per_token, 1))
    if max_batch_by_input < 1:
        raise ValueError(
            "max_in_memory_bytes is smaller than one decompressed contribution row; "
            f"need at least {int(input_bytes_per_token)} bytes"
        )
    effective_batch = max(1, min(requested_batch, max_batch_by_input))
    input_batch_bytes = int(effective_batch * input_bytes_per_token)
    estimated_scores = tokens * candidates * np.dtype(np.float32).itemsize
    estimated_ids = tokens * candidates * top_k * np.dtype(np.int16).itemsize if not exact_global else 0
    available_storage = max(1, int(max_in_memory_bytes - input_batch_bytes))
    use_memmap = estimated_scores > available_storage
    temporary_path: Path | None = None
    ids_path: Path | None = None
    if use_memmap:
        if storage_dir is None:
            import tempfile

            storage_dir = Path(tempfile.mkdtemp(prefix="d2m-oracle-"))
        storage_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = storage_dir / "candidate-errors.float32.mmap"
        errors = np.memmap(temporary_path, mode="w+", dtype=np.float32, shape=(tokens, candidates))
    else:
        errors = np.empty((tokens, candidates), dtype=np.float32)
    ids_store: Any | None = None
    if not exact_global:
        remaining_storage = available_storage if use_memmap else max(1, available_storage - estimated_scores)
        if estimated_ids > remaining_storage:
            if storage_dir is None:
                import tempfile

                storage_dir = Path(tempfile.mkdtemp(prefix="d2m-oracle-"))
            storage_dir.mkdir(parents=True, exist_ok=True)
            ids_path = storage_dir / "candidate-ids.int16.mmap"
            ids_store = np.memmap(ids_path, mode="w+", dtype=np.int16, shape=(tokens, candidates, top_k))
        else:
            ids_store = np.empty((tokens, candidates, top_k), dtype=np.int16)
    # Candidate scoring uses the per-batch expert Gram matrix and correlation
    # vector, so the candidate block is [batch, candidates, top_k, top_k], not
    # [batch, candidates, top_k, hidden].
    resident_storage = 0 if use_memmap else estimated_scores
    if not exact_global and not isinstance(ids_store, np.memmap):
        resident_storage += estimated_ids
    working_budget = max(1, int(max_in_memory_bytes - input_batch_bytes - resident_storage))
    # ``selected_rows`` is [batch, candidates, top_k, experts], so account for
    # the expert dimension as well as the compact Gram/KKT blocks.  This keeps
    # candidate chunks bounded even for p32 where experts > top_k.
    bytes_per_candidate = max(
        int(effective_batch * (top_k * experts + (top_k + 1) ** 2 * 3 + top_k) * itemsize),
        1,
    )
    candidate_chunk = max(1, min(candidates, working_budget // bytes_per_candidate))
    candidate_chunk = int(max(1, candidate_chunk))
    for start in range(0, tokens, effective_batch):
        stop = min(tokens, start + effective_batch)
        shared_batch = np.asarray(shared[start:stop], dtype=np.float32)
        routed_batch = np.asarray(routed[start:stop], dtype=np.float32)
        target_batch = np.asarray(target[start:stop], dtype=np.float32)
        residual_batch = target_batch - shared_batch
        residual_norm = np.sum(residual_batch * residual_batch, axis=1, dtype=np.float32)
        gram_all = np.einsum("beh,bfh->bef", routed_batch, routed_batch, dtype=np.float32)
        correlations_all = np.einsum("beh,bh->be", routed_batch, residual_batch, dtype=np.float32)
        if exact_global:
            ids_batch = templates
        else:
            order = np.argsort(-correlations_all, axis=1, kind="stable")[:, :effective_pool]
            ids_batch = np.take_along_axis(
                np.broadcast_to(order[:, None, :], (stop - start, candidates, effective_pool)),
                np.broadcast_to(templates[None, :, :], (stop - start, candidates, top_k)),
                axis=2,
            )
            ids_store[start:stop] = ids_batch
        for candidate_start in range(0, candidates, candidate_chunk):
            candidate_stop = min(candidates, candidate_start + candidate_chunk)
            block_ids = (
                ids_batch[candidate_start:candidate_stop]
                if exact_global
                else ids_batch[:, candidate_start:candidate_stop, :]
            )
            if block_ids.ndim == 2:
                block_ids = np.broadcast_to(block_ids[None, ...], (stop - start, *block_ids.shape))
            block_rhs = np.take_along_axis(correlations_all[:, None, :], block_ids, axis=2)
            selected_rows = np.take_along_axis(
                gram_all[:, None, :, :],
                block_ids[..., None],
                axis=2,
            )
            block_gram = np.take_along_axis(
                selected_rows,
                block_ids[:, :, None, :],
                axis=3,
            )
            block_errors, _ = _batched_projected_fit(
                block_gram,
                block_rhs,
                residual_norm,
                hidden,
                simplex=simplex,
            )
            errors[start:stop, candidate_start:candidate_stop] = np.asarray(block_errors, dtype=np.float32)
    if isinstance(errors, np.memmap):
        errors.flush()
    if isinstance(ids_store, np.memmap):
        ids_store.flush()
    metadata = {
        "candidate_error_storage": "memmap" if use_memmap else "ram",
        "candidate_error_path": str(temporary_path) if temporary_path is not None else None,
        "candidate_id_storage": "memmap" if isinstance(ids_store, np.memmap) else "ram" if ids_store is not None else "global",
        "candidate_id_path": str(ids_path) if ids_path is not None else None,
        "candidate_count": candidates,
        "candidate_batch_size": effective_batch,
        "requested_batch_size": requested_batch,
        "input_bytes_per_token": int(input_bytes_per_token),
        "input_batch_bytes": input_batch_bytes,
        "candidate_chunk_size": candidate_chunk,
    }
    return errors, ids_store, metadata


def _candidate_ids_for_rows(ids_store: Any | None, templates: Any, start: int, stop: int) -> Any:
    if ids_store is None:
        return templates
    return ids_store[start:stop]


def _select_stream_assignment(
    errors: Any,
    ids_store: Any | None,
    templates: Any,
    *,
    prices: Any,
    penalty: float,
    experts: int,
    top_k: int,
    batch_size: int,
    token_offset: int = 0,
) -> tuple[Any, Any]:
    """Select candidates from compact scores and return IDs plus usage."""

    np = _np()
    tokens, candidates = errors.shape
    selected = np.empty(tokens, dtype=np.int32)
    usage = np.zeros(experts, dtype=np.int64)
    for start in range(0, tokens, max(1, int(batch_size))):
        stop = min(tokens, start + max(1, int(batch_size)))
        ids = _candidate_ids_for_rows(ids_store, templates, start, stop)
        price_cost = np.sum(prices[ids], axis=-1)
        scores = np.asarray(errors[start:stop], dtype=np.float64) + float(penalty) * price_cost
        choices = np.argmin(scores, axis=1).astype(np.int32)
        for row in range(stop - start):
            row_scores = scores[row]
            minimum = float(row_scores[choices[row]])
            tied = np.flatnonzero(row_scores <= minimum + 1e-10)
            if tied.size > 1:
                if tied.size == candidates:
                    choices[row] = int((token_offset + start + row) % candidates)
                else:
                    choices[row] = min(
                        (int(candidate) for candidate in tied),
                        key=lambda candidate: (
                            int(np.sum(usage[ids[candidate] if ids.ndim == 2 else ids[row, candidate]])),
                            candidate,
                        ),
                    )
            selected_route = ids[choices[row]] if ids.ndim == 2 else ids[row, choices[row]]
            usage += np.bincount(np.atleast_1d(selected_route), minlength=experts)
        selected[start:stop] = choices
    return selected, usage


def _selected_ids(ids_store: Any | None, templates: Any, selected: Any) -> Any:
    np = _np()
    if ids_store is None:
        return np.asarray(templates, dtype=np.int64)[np.asarray(selected, dtype=np.int64)]
    rows = np.arange(len(selected), dtype=np.int64)
    return np.asarray(ids_store[rows, np.asarray(selected, dtype=np.int64)], dtype=np.int64)


def _stream_selected_weights(
    routed: Any,
    target: Any,
    shared: Any,
    ids: Any,
    *,
    simplex: bool,
    batch_size: int,
    exact_positive: bool = False,
) -> Any:
    np = _np()
    tokens = int(ids.shape[0])
    weights = np.zeros((tokens, ids.shape[1]), dtype=np.float32)
    for start in range(0, tokens, max(1, int(batch_size))):
        stop = min(tokens, start + max(1, int(batch_size)))
        residual = np.asarray(target[start:stop], dtype=np.float32) - np.asarray(shared[start:stop], dtype=np.float32)
        vectors = _batched_candidate_vectors(np.asarray(routed[start:stop], dtype=np.float32), ids[start:stop])
        if exact_positive and not simplex:
            _, fitted = _batched_exact_positive_fit(vectors, residual)
        else:
            _, fitted = _batched_candidate_fit(vectors, residual, simplex=simplex)
        weights[start:stop] = fitted[:, 0]
    return weights


def _stream_metrics(
    shared: Any,
    routed: Any,
    target: Any,
    ids: Any,
    weights: Any,
    *,
    batch_size: int,
    hard_fraction: float,
    materialize_outputs: bool,
) -> dict[str, Any]:
    """Calculate quality/load metrics without a full float64 prediction cube."""

    np = _np()
    tokens, _experts, hidden = routed.shape
    if not 0.0 < float(hard_fraction) <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")
    work_dtype = np.dtype(np.float32)
    reconstruction = np.empty((tokens, hidden), dtype=work_dtype) if materialize_outputs else None
    cosine_by_token = np.empty(tokens, dtype=np.float64)
    hardness = np.empty(tokens, dtype=np.float64)
    usage = np.bincount(np.asarray(ids).reshape(-1), minlength=routed.shape[1]).astype(np.int64)
    error_sum = 0.0
    target_norm_sum = 0.0
    cosine_sum = 0.0
    relative_sum = 0.0
    for start in range(0, tokens, max(1, int(batch_size))):
        stop = min(tokens, start + max(1, int(batch_size)))
        shared_batch = np.asarray(shared[start:stop], dtype=work_dtype)
        routed_batch = np.asarray(routed[start:stop], dtype=work_dtype)
        target_batch = np.asarray(target[start:stop], dtype=work_dtype)
        selected_ids = np.asarray(ids[start:stop], dtype=np.int64)
        selected = np.take_along_axis(routed_batch, selected_ids[:, :, None], axis=1)
        prediction = shared_batch + np.sum(selected * np.asarray(weights[start:stop], dtype=work_dtype)[:, :, None], axis=1)
        delta = prediction - target_batch
        error = np.sum(delta * delta, axis=1, dtype=np.float64)
        target_norm = np.sum(target_batch * target_batch, axis=1, dtype=np.float64)
        prediction_norm = np.linalg.norm(prediction, axis=1)
        target_length = np.linalg.norm(target_batch, axis=1)
        cosine = np.sum(prediction * target_batch, axis=1, dtype=np.float64) / (prediction_norm * target_length + 1e-12)
        relative = error / np.maximum(target_norm, 1e-12)
        cosine_by_token[start:stop] = cosine
        hardness[start:stop] = np.linalg.norm(target_batch - shared_batch, axis=1)
        error_sum += float(np.sum(error))
        target_norm_sum += float(np.sum(target_norm))
        cosine_sum += float(np.sum(cosine))
        relative_sum += float(np.sum(relative))
        if reconstruction is not None:
            reconstruction[start:stop] = prediction
    hard_count = max(1, int(np.ceil(tokens * float(hard_fraction))))
    hard_indices = np.argsort(-hardness, kind="stable")[:hard_count]
    load_cv = float(usage.std() / max(usage.mean(), 1e-12))
    result: dict[str, Any] = {
        "tokens": int(tokens),
        "mse": float(error_sum / max(tokens * hidden, 1)),
        "normalized_mse": float(error_sum / max(target_norm_sum, 1e-12)),
        "global_nmse": float(error_sum / max(target_norm_sum, 1e-12)),
        "mean_token_relative_mse": float(relative_sum / max(tokens, 1)),
        "cosine": float(cosine_sum / max(tokens, 1)),
        "hard_quartile_cosine": float(np.mean(cosine_by_token[hard_indices])),
        "expert_usage_counts": usage.tolist(),
        "expert_usage_fraction": (usage / max(tokens * ids.shape[1], 1)).tolist(),
        "dead_experts": int(np.sum(usage == 0)),
        "load_cv": load_cv,
        "indices": np.asarray(ids, dtype=np.int64),
        "weights": np.asarray(weights),
        "hard_indices": hard_indices,
    }
    if reconstruction is not None:
        result["reconstruction"] = reconstruction
    return result


def frozen_slice_load_aware_oracle(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    top_k: int = 2,
    target_load_cv: float = 0.50,
    simplex: bool = False,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
    iterations: int = 32,
    price_step: float = 0.5,
    price_decay: float = 0.95,
    penalty_grid: Sequence[float] | None = None,
    hard_fraction: float = 0.25,
    batch_size: int = 64,
    max_in_memory_bytes: int = 256 * 1024 * 1024,
    storage_dir: str | Path | None = None,
    materialize_outputs: bool | None = None,
) -> dict[str, Any]:
    """Stream a bounded load-aware frozen-slice oracle.

    p16/top4 uses one global ``C(16,4)=1820`` identity table and streams the
    candidate-error matrix in token/candidate chunks.  p32/top4/top5 use a
    deterministic correlation-ranked local pool and are explicitly bounded.
    Candidate coefficients are recomputed only for selected routes, so the
    implementation does not retain Python route objects or a full float64
    contribution cube.
    """

    np = _np()
    shared_values, routed_values, target_values = _stream_validate_oracle_arrays(shared, routed, target, top_k)
    if not 0.0 <= float(target_load_cv):
        raise ValueError("target_load_cv must be non-negative")
    if candidate_pool_size is not None and candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive when provided")
    if iterations <= 0 or price_step <= 0 or not 0.0 < price_decay <= 1.0:
        raise ValueError("iterations/price_step/price_decay are invalid")
    if batch_size <= 0 or max_in_memory_bytes <= 0:
        raise ValueError("batch_size and max_in_memory_bytes must be positive")
    tokens, experts, hidden = routed_values.shape
    if penalty_grid is None:
        penalty_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    penalties = tuple(float(value) for value in penalty_grid)
    if not penalties or any(value < 0.0 for value in penalties):
        raise ValueError("penalty_grid must contain non-negative values")
    templates, effective_pool, candidate_count, assurance = _candidate_templates(
        experts,
        top_k,
        candidate_pool_size=candidate_pool_size,
        max_combinations=max_combinations,
    )
    exact_global = assurance == "exact_candidate_sets"
    configured_storage = Path(storage_dir) if storage_dir is not None else None
    owned_storage: Any | None = None
    estimated_storage = tokens * candidate_count * (4 + (0 if exact_global else top_k * 2))
    if configured_storage is None and estimated_storage > max_in_memory_bytes:
        import tempfile

        owned_storage = tempfile.TemporaryDirectory(prefix="d2m-load-aware-")
        configured_storage = Path(owned_storage.name)
    if materialize_outputs is None:
        materialize_outputs = tokens * hidden * np.dtype(np.float32).itemsize <= max_in_memory_bytes
    errors, ids_store, score_metadata = _score_candidate_batches(
        shared_values,
        routed_values,
        target_values,
        templates,
        exact_global=exact_global,
        effective_pool=effective_pool,
        top_k=top_k,
        simplex=simplex,
        batch_size=batch_size,
        max_in_memory_bytes=max_in_memory_bytes,
        storage_dir=configured_storage,
    )
    try:
        zero_prices = np.zeros(experts, dtype=np.float64)
        unconstrained_choice, _ = _select_stream_assignment(
            errors,
            ids_store,
            templates,
            prices=zero_prices,
            penalty=0.0,
            experts=experts,
            top_k=top_k,
            batch_size=batch_size,
        )
        unconstrained_ids = _selected_ids(ids_store, templates, unconstrained_choice)
        unconstrained_weights = _stream_selected_weights(
            routed_values,
            target_values,
            shared_values,
            unconstrained_ids,
            simplex=simplex,
            batch_size=batch_size,
            exact_positive=True,
        )
        unconstrained_metric = _stream_metrics(
            shared_values,
            routed_values,
            target_values,
            unconstrained_ids,
            unconstrained_weights,
            batch_size=batch_size,
            hard_fraction=hard_fraction,
            materialize_outputs=bool(materialize_outputs),
        )
        points: list[dict[str, Any]] = []
        selected_by_point: list[tuple[Any, Any]] = []
        target_usage = tokens * top_k / max(experts, 1)
        for penalty in penalties:
            prices = np.zeros(experts, dtype=np.float64)
            best_assignment: Any | None = None
            best_metric: dict[str, Any] | None = None
            best_key: tuple[float, ...] | None = None
            for iteration in range(int(iterations)):
                choice, usage = _select_stream_assignment(
                    errors,
                    ids_store,
                    templates,
                    prices=prices,
                    penalty=penalty,
                    experts=experts,
                    top_k=top_k,
                    batch_size=batch_size,
                )
                ids = _selected_ids(ids_store, templates, choice)
                weights = _stream_selected_weights(
                    routed_values,
                    target_values,
                    shared_values,
                    ids,
                    simplex=simplex,
                    batch_size=batch_size,
                    exact_positive=True,
                )
                metric = _stream_metrics(
                    shared_values,
                    routed_values,
                    target_values,
                    ids,
                    weights,
                    batch_size=batch_size,
                    hard_fraction=hard_fraction,
                    materialize_outputs=False,
                )
                hard_feasible = bool(
                    float(metric["global_nmse"]) <= 0.05
                    and int(metric["dead_experts"]) == 0
                    and float(metric["load_cv"]) <= float(target_load_cv)
                )
                if hard_feasible:
                    key = (0.0, -float(metric["cosine"]), float(metric["global_nmse"]), float(metric["load_cv"]))
                else:
                    key = (1.0, float(metric["load_cv"]), -float(metric["cosine"]), float(metric["global_nmse"]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_assignment = (ids.copy(), weights.copy())
                    best_metric = metric
                imbalance = usage.astype(np.float64) / max(target_usage, 1e-12) - 1.0
                prices += float(price_step) * (float(price_decay) ** iteration) * imbalance
                prices -= prices.mean()
            if best_assignment is None or best_metric is None:  # pragma: no cover
                raise RuntimeError("load-aware assignment produced no route")
            ids, weights = best_assignment
            point = {
                "penalty": float(penalty),
                "global_nmse": best_metric["global_nmse"],
                "mean_token_relative_mse": best_metric["mean_token_relative_mse"],
                "cosine": best_metric["cosine"],
                "hard_quartile_cosine": best_metric["hard_quartile_cosine"],
                "load_cv": best_metric["load_cv"],
                "dead_experts": best_metric["dead_experts"],
                "expert_usage_counts": best_metric["expert_usage_counts"],
                "feasible_load_target": bool(float(best_metric["load_cv"]) <= float(target_load_cv)),
                "hard_feasible": bool(
                    float(best_metric["global_nmse"]) <= 0.05
                    and int(best_metric["dead_experts"]) == 0
                    and float(best_metric["load_cv"]) <= float(target_load_cv)
                ),
                "green_gate": bool(
                    float(best_metric["global_nmse"]) <= 0.05
                    and float(best_metric["cosine"]) >= 0.98
                    and float(best_metric["load_cv"]) <= float(target_load_cv)
                    and int(best_metric["dead_experts"]) == 0
                ),
                "iterations": int(iterations),
            }
            points.append(point)
            selected_by_point.append((ids, weights))

        hard_feasible_points = [index for index, point in enumerate(points) if point["hard_feasible"]]
        load_feasible_points = [index for index, point in enumerate(points) if point["feasible_load_target"]]
        if hard_feasible_points:
            selected_index = min(
                hard_feasible_points,
                key=lambda index: (-float(points[index]["cosine"]), float(points[index]["global_nmse"]), float(points[index]["load_cv"])),
            )
        elif load_feasible_points:
            selected_index = min(
                load_feasible_points,
                key=lambda index: (-float(points[index]["cosine"]), float(points[index]["global_nmse"]), float(points[index]["load_cv"])),
            )
        else:
            selected_index = min(
                range(len(points)),
                key=lambda index: (float(points[index]["load_cv"]), -float(points[index]["cosine"]), float(points[index]["global_nmse"])),
            )
        ids, weights = selected_by_point[selected_index]
        metric = _stream_metrics(
            shared_values,
            routed_values,
            target_values,
            ids,
            weights,
            batch_size=batch_size,
            hard_fraction=hard_fraction,
            materialize_outputs=bool(materialize_outputs),
        )
        pareto = _pareto_points(points)
        result: dict[str, Any] = {
            "method": "frozen_slice_load_aware_oracle",
            "routing_constraint": "nonnegative weights" if not simplex else "nonnegative weights summing to one",
            "assignment_method": "priced_lagrangian_iterative_load_balancing_streaming",
            "assurance": assurance,
            "candidate_pool_size": int(candidate_pool_size or experts),
            "effective_candidate_pool_size": int(effective_pool),
            "combinations_considered_per_token": int(candidate_count),
            "max_combinations": int(max_combinations),
            "target_load_cv": float(target_load_cv),
            "feasible_load_target": bool(float(metric["load_cv"]) <= float(target_load_cv)),
            "hard_feasible": bool(
                float(metric["global_nmse"]) <= 0.05
                and int(metric["dead_experts"]) == 0
                and float(metric["load_cv"]) <= float(target_load_cv)
            ),
            "green_gate": bool(
                float(metric["global_nmse"]) <= 0.05
                and float(metric["cosine"]) >= 0.98
                and float(metric["load_cv"]) <= float(target_load_cv)
                and int(metric["dead_experts"]) == 0
            ),
            "selected_penalty": float(penalties[selected_index]),
            "price_iterations": int(iterations),
            "batch_size": int(batch_size),
            "max_in_memory_bytes": int(max_in_memory_bytes),
            "materialize_outputs": bool(materialize_outputs),
            "candidate_storage_ephemeral": owned_storage is not None,
            "coefficient_solver": "batched_projected_float32_candidate_scoring_with_exact_active_face_selected_refit",
            "candidate_fit_exact": False,
            "selected_fit_exact": bool(not simplex),
            **score_metadata,
            "unconstrained": {
                "global_nmse": unconstrained_metric["global_nmse"],
                "mean_token_relative_mse": unconstrained_metric["mean_token_relative_mse"],
                "cosine": unconstrained_metric["cosine"],
                "hard_quartile_cosine": unconstrained_metric["hard_quartile_cosine"],
                "load_cv": unconstrained_metric["load_cv"],
                "expert_usage_counts": unconstrained_metric["expert_usage_counts"],
                "dead_experts": unconstrained_metric["dead_experts"],
                "indices": unconstrained_metric["indices"],
                "weights": unconstrained_metric["weights"],
            },
            "indices": metric["indices"],
            "weights": metric["weights"],
            "mse": metric["mse"],
            "normalized_mse": metric["normalized_mse"],
            "global_nmse": metric["global_nmse"],
            "mean_token_relative_mse": metric["mean_token_relative_mse"],
            "cosine": metric["cosine"],
            "hard_quartile_cosine": metric["hard_quartile_cosine"],
            "expert_usage_counts": metric["expert_usage_counts"],
            "expert_usage_fraction": metric["expert_usage_fraction"],
            "dead_experts": metric["dead_experts"],
            "load_cv": metric["load_cv"],
            "pareto": pareto,
            "pareto_all_points": points,
        }
        if "reconstruction" in metric:
            result["reconstruction"] = metric["reconstruction"]
        return result
    finally:
        if owned_storage is not None:
            try:
                if isinstance(errors, np.memmap):
                    errors.flush()
                if isinstance(ids_store, np.memmap):
                    ids_store.flush()
                del errors, ids_store
            except UnboundLocalError:
                pass
            owned_storage.cleanup()


# Names used by reports and external experiments.  Keep both spellings so a
# diagnostic can be adopted without coupling callers to an implementation
# detail of the frozen-slice module.
load_aware_oracle = frozen_slice_load_aware_oracle
load_constrained_oracle = frozen_slice_load_aware_oracle


def frozen_slice_simplex_oracle(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    top_k: int = 2,
) -> dict[str, Any]:
    """Find the exact frozen-slice normalized top-k oracle.

    For p8/top-2 this enumerates all 28 expert pairs and solves each pair's
    constrained problem in closed form by clipping the scalar simplex optimum
    to ``[0, 1]``.  The returned ``weights`` therefore sum to one for every
    token.  This is a target-assisted frozen-slice diagnostic, not a trained
    router result or a hard ceiling on trainable experts.
    """

    np = _np()
    shared_values, routed_values, target_values = _validate_oracle_arrays(shared, routed, target, top_k)
    tokens, experts, _ = routed_values.shape
    combinations = list(itertools.combinations(range(experts), top_k))
    ids = np.zeros((tokens, top_k), dtype=np.int64)
    weights = np.zeros((tokens, top_k), dtype=np.float64)
    residual = np.zeros(tokens, dtype=np.float64)
    for token in range(tokens):
        goal = target_values[token] - shared_values[token]
        best_error = float("inf")
        best_combo = combinations[0]
        best_weights = np.ones(top_k, dtype=np.float64) / top_k
        for combo in combinations:
            matrix = routed_values[token, list(combo)].T
            candidate_weights = _simplex_weights_exact(matrix, goal)
            error = _candidate_error(matrix, candidate_weights, goal)
            if error < best_error - 1e-15 or (abs(error - best_error) <= 1e-15 and combo < best_combo):
                best_error = error
                best_combo = combo
                best_weights = candidate_weights
        ids[token] = best_combo
        weights[token] = best_weights
        residual[token] = best_error
    result = _metrics(shared_values, routed_values, target_values, ids, weights)
    result["method"] = "frozen_slice_simplex_oracle"
    result["routing_constraint"] = "nonnegative weights summing to one"
    return result


def frozen_slice_positive_oracle(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    top_k: int = 2,
) -> dict[str, Any]:
    """Find the exact frozen-slice non-negative top-k oracle.

    For top-2, every pair is solved as a two-variable NNLS problem.  The
    unconstrained interior solution and both one-variable boundaries (plus
    the origin) are evaluated explicitly, so coefficients are non-negative
    but need not sum to one.  This isolates magnitude/scaling effects from
    expert-selection effects.
    """

    np = _np()
    shared_values, routed_values, target_values = _validate_oracle_arrays(shared, routed, target, top_k)
    tokens, experts, _ = routed_values.shape
    combinations = list(itertools.combinations(range(experts), top_k))
    ids = np.zeros((tokens, top_k), dtype=np.int64)
    weights = np.zeros((tokens, top_k), dtype=np.float64)
    residual = np.zeros(tokens, dtype=np.float64)
    for token in range(tokens):
        goal = target_values[token] - shared_values[token]
        best_error = float("inf")
        best_combo = combinations[0]
        best_weights = np.zeros(top_k, dtype=np.float64)
        for combo in combinations:
            matrix = routed_values[token, list(combo)].T
            candidate_weights = _positive_weights_exact(matrix, goal)
            error = _candidate_error(matrix, candidate_weights, goal)
            if error < best_error - 1e-15 or (abs(error - best_error) <= 1e-15 and combo < best_combo):
                best_error = error
                best_combo = combo
                best_weights = candidate_weights
        ids[token] = best_combo
        weights[token] = best_weights
        residual[token] = best_error
    result = _metrics(shared_values, routed_values, target_values, ids, weights)
    result["method"] = "frozen_slice_positive_oracle"
    result["routing_constraint"] = "nonnegative weights with no sum-to-one constraint"
    return result


def _split_arrays(
    shared: Any | None,
    routed: Any | None,
    target: Any | None,
    positional_holdout: Sequence[Any],
    *,
    train_shared: Any | None,
    train_routed: Any | None,
    train_target: Any | None,
    holdout_shared: Any | None,
    holdout_routed: Any | None,
    holdout_target: Any | None,
    train_indices: Iterable[int] | None,
    holdout_indices: Iterable[int] | None,
    train_mask: Any | None,
    holdout_mask: Any | None,
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    np = _np()
    if positional_holdout:
        if len(positional_holdout) != 3:
            raise TypeError("scaled oracle accepts exactly three positional holdout arrays")
        if any(value is not None for value in (train_shared, train_routed, train_target, holdout_shared, holdout_routed, holdout_target)):
            raise TypeError("do not mix positional and keyword split arrays")
        train_shared, train_routed, train_target = shared, routed, target
        holdout_shared, holdout_routed, holdout_target = positional_holdout
    explicit = (train_shared, train_routed, train_target, holdout_shared, holdout_routed, holdout_target)
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("all train_* and holdout_* arrays are required together")
        return (train_shared, train_routed, train_target), (holdout_shared, holdout_routed, holdout_target)  # type: ignore[return-value]
    if shared is None or routed is None or target is None:
        raise ValueError("combined shared/routed/target arrays are required")
    shared_values = np.asarray(shared)
    if train_indices is not None or holdout_indices is not None:
        if train_indices is None or holdout_indices is None:
            raise ValueError("train_indices and holdout_indices must be supplied together")
        train_ids = np.asarray(list(train_indices), dtype=np.int64)
        holdout_ids = np.asarray(list(holdout_indices), dtype=np.int64)
    elif train_mask is not None or holdout_mask is not None:
        if train_mask is None or holdout_mask is None:
            raise ValueError("train_mask and holdout_mask must be supplied together")
        train_ids = np.flatnonzero(np.asarray(train_mask, dtype=bool))
        holdout_ids = np.flatnonzero(np.asarray(holdout_mask, dtype=bool))
    else:
        raise ValueError("scaled oracle requires explicit train/holdout arrays or split indices/masks")
    if len(train_ids) == 0 or len(holdout_ids) == 0:
        raise ValueError("train and holdout splits must both be non-empty")
    if np.intersect1d(train_ids, holdout_ids).size:
        raise ValueError("train and holdout splits must be disjoint")
    return (
        shared_values[train_ids], np.asarray(routed)[train_ids], np.asarray(target)[train_ids],
    ), (
        shared_values[holdout_ids], np.asarray(routed)[holdout_ids], np.asarray(target)[holdout_ids],
    )


def _scaled_features(routed: Any, ids: Any, weights: Any) -> Any:
    np = _np()
    routed_values = np.asarray(routed, dtype=np.float64)
    token_count, expert_count, output_size = routed_values.shape
    features = np.zeros((token_count, expert_count, output_size), dtype=np.float64)
    for token in range(token_count):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            features[token, expert] += float(weights[token, slot]) * routed_values[token, expert]
    return features


def _scaled_metrics(shared: Any, target: Any, features: Any, scales: Any) -> dict[str, Any]:
    np = _np()
    shared_values = np.asarray(shared, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    routed_part = np.sum(np.asarray(features, dtype=np.float64) * np.asarray(scales, dtype=np.float64)[None, :, None], axis=1)
    reconstruction = shared_values + routed_part
    mse = float(np.mean((reconstruction - target_values) ** 2))
    normalized_mse = mse / (float(np.mean(target_values**2)) + 1e-12)
    dot = np.sum(reconstruction * target_values, axis=-1)
    cosine = float(np.mean(dot / ((np.linalg.norm(reconstruction, axis=-1) * np.linalg.norm(target_values, axis=-1)) + 1e-12)))
    return {"mse": mse, "normalized_mse": normalized_mse, "cosine": cosine, "reconstruction": reconstruction}


def frozen_slice_scaled_router_oracle(
    shared: Any | None = None,
    routed: Any | None = None,
    target: Any | None = None,
    *positional_holdout: Any,
    top_k: int = 2,
    train_shared: Any | None = None,
    train_routed: Any | None = None,
    train_target: Any | None = None,
    holdout_shared: Any | None = None,
    holdout_routed: Any | None = None,
    holdout_target: Any | None = None,
    train_indices: Iterable[int] | None = None,
    holdout_indices: Iterable[int] | None = None,
    train_mask: Any | None = None,
    holdout_mask: Any | None = None,
) -> dict[str, Any]:
    """Fit global expert output scales on train and evaluate on holdout.

    Routing weights are obtained independently for each split from the exact
    frozen-slice simplex oracle, then held fixed while scalar ``scale_i``
    values are fitted by least squares on TRAIN only.  The optional six-array
    form is convenient for callers with physically separate shards; the
    combined-array form requires explicit disjoint indices or masks and never
    silently creates a sequential split.

    Because target-assisted oracle routing is used for both splits, this is a
    learned-global-scale *diagnostic*, not an unbiased trained-router
    evaluation.  The holdout target is used only to score the fitted scales.
    """

    np = _np()
    train_data, holdout_data = _split_arrays(
        shared,
        routed,
        target,
        positional_holdout,
        train_shared=train_shared,
        train_routed=train_routed,
        train_target=train_target,
        holdout_shared=holdout_shared,
        holdout_routed=holdout_routed,
        holdout_target=holdout_target,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        train_mask=train_mask,
        holdout_mask=holdout_mask,
    )
    train_shared_values, train_routed_values, train_target_values = _validate_oracle_arrays(*train_data, top_k)
    holdout_shared_values, holdout_routed_values, holdout_target_values = _validate_oracle_arrays(*holdout_data, top_k)
    train_routing = frozen_slice_simplex_oracle(train_shared_values, train_routed_values, train_target_values, top_k=top_k)
    holdout_routing = frozen_slice_simplex_oracle(holdout_shared_values, holdout_routed_values, holdout_target_values, top_k=top_k)
    train_features = _scaled_features(train_routed_values, train_routing["indices"], train_routing["weights"])
    holdout_features = _scaled_features(holdout_routed_values, holdout_routing["indices"], holdout_routing["weights"])
    train_residual = train_target_values - train_shared_values
    design = np.asarray(train_features, dtype=np.float64).transpose(0, 2, 1).reshape(-1, train_features.shape[1])
    response = np.asarray(train_residual, dtype=np.float64).reshape(-1)
    scales = np.linalg.lstsq(design, response, rcond=None)[0]
    train_metrics = _scaled_metrics(train_shared_values, train_target_values, train_features, scales)
    holdout_metrics = _scaled_metrics(holdout_shared_values, holdout_target_values, holdout_features, scales)
    return {
        "method": "frozen_slice_scaled_router_oracle",
        "routing_constraint": "normalized simplex weights from frozen-slice oracle",
        "fit_scope": "train_only",
        "holdout_routing_target_assisted": True,
        "scales": np.asarray(scales, dtype=np.float64),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "train_normalized_mse": train_metrics["normalized_mse"],
        "holdout_normalized_mse": holdout_metrics["normalized_mse"],
        "holdout_cosine": holdout_metrics["cosine"],
        "train_routing": train_routing,
        "holdout_routing": holdout_routing,
    }


def trainable_student_proxy(
    frozen_result: Mapping[str, Any] | None = None,
    *,
    student_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the distinction between frozen diagnostics and a trainable MoE.

    This helper intentionally does not fabricate a trained result.  It gives
    reports a machine-readable interpretation: frozen-slice oracle error is a
    reference for the unchanged initialization, while a trainable student is
    constrained by optimization/data/compute, router capacity and balancing,
    expert parameterization, and its fixed evaluation split.  Consequently
    ``frozen_slice_is_hard_ceiling`` is always ``False``.
    """

    return {
        "method": "trainable_student_proxy",
        "frozen_slice_is_hard_ceiling": False,
        "frozen_slice_reference": dict(frozen_result) if frozen_result is not None else None,
        "student_metrics": dict(student_metrics) if student_metrics is not None else None,
        "genuine_constraints": [
            "optimization and distillation data",
            "trainable expert parameterization and shared capacity",
            "router expressivity, load balance, and top-k selection",
            "compute, memory, and convergence budget",
            "fixed train/holdout protocol and end-to-end quality gates",
        ],
        "interpretation": "frozen slices diagnose initialization; trained experts and router may change the attainable error",
    }


def oracle_topk(shared: Any, routed: Any, target: Any, *, top_k: int) -> dict[str, Any]:
    """Compatibility alias for the exact frozen-slice simplex oracle.

    Historical callers keep working, but new reports should name the method
    ``frozen_slice_simplex_oracle`` explicitly.
    """

    return frozen_slice_simplex_oracle(shared, routed, target, top_k=top_k)


def sparse_baseline(shared: Any, routed: Any, *, top_k: int, mode: str = "normalized_topk") -> dict[str, Any]:
    """Evaluate deterministic sparse baselines.

    ``capacity_scaled`` assigns each selected contribution weight ``E/k``.
    Thus a uniformly sampled subset estimates the full sum without bias:
    ``E[selected_sum] * (E/k) = full_sum``.  The deterministic norm-ranked
    selection used here is not itself a uniform sampler, so the unbiasedness
    statement applies to the coefficient semantics, not to this ranking.
    """

    np = _np()
    shared_values = np.asarray(shared)
    routed_values = np.asarray(routed)
    if routed_values.ndim != 3 or shared_values.ndim != 2 or shared_values.shape != (routed_values.shape[0], routed_values.shape[2]):
        raise ValueError("shared must be [tokens, output] and routed [tokens, experts, output]")
    tokens, experts, _ = routed_values.shape
    if top_k <= 0 or top_k > experts:
        raise ValueError("invalid top_k")
    if mode not in {"normalized_topk", "capacity_scaled", "first_topk", "random_balanced"}:
        raise ValueError(f"unknown sparse baseline: {mode}")
    if mode == "first_topk":
        ids = np.tile(np.arange(top_k), (tokens, 1))
    elif mode == "random_balanced":
        ids = np.asarray([np.arange(token * top_k, token * top_k + top_k) % experts for token in range(tokens)])
    else:
        scores = np.linalg.norm(routed_values, axis=-1)
        ids = np.argsort(-scores, axis=-1, kind="stable")[:, :top_k]
    weights = np.ones((tokens, top_k), dtype=np.float64) / top_k
    if mode == "capacity_scaled":
        # Start from unit coefficients, not normalized probabilities: each
        # selected contribution must receive E/k exactly.
        weights = np.full((tokens, top_k), experts / top_k, dtype=np.float64)
    reconstruction = np.asarray(shared_values, dtype=np.float64).copy()
    for token in range(tokens):
        for slot in range(top_k):
            reconstruction[token] += weights[token, slot] * routed_values[token, ids[token, slot]]
    return {"mode": mode, "indices": ids, "weights": weights, "reconstruction": reconstruction}


__all__ = [
    "frozen_slice_load_aware_oracle",
    "frozen_slice_positive_oracle",
    "frozen_slice_scaled_router_oracle",
    "frozen_slice_simplex_oracle",
    "load_aware_oracle",
    "load_constrained_oracle",
    "oracle_topk",
    "sparse_baseline",
    "swiglu_contributions",
    "trainable_student_proxy",
]
