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
from collections.abc import Iterable, Mapping, Sequence
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
    "frozen_slice_positive_oracle",
    "frozen_slice_scaled_router_oracle",
    "frozen_slice_simplex_oracle",
    "oracle_topk",
    "sparse_baseline",
    "swiglu_contributions",
    "trainable_student_proxy",
]
