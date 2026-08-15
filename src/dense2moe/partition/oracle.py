"""Contribution-aware sparse routing baselines and exact p8 oracle search."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
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


def _normalized_positive_weights(values: Any, target: Any) -> Any:
    np = _np()
    matrix = np.asarray(values, dtype=np.float64)
    target_vec = np.asarray(target, dtype=np.float64)
    gram = matrix.T @ matrix
    rhs = matrix.T @ target_vec
    try:
        weights = np.linalg.solve(gram + np.eye(gram.shape[0]) * 1e-9, rhs)
    except np.linalg.LinAlgError:
        weights = np.ones(matrix.shape[1], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
        weights = np.ones(matrix.shape[1], dtype=np.float64)
    return (weights / weights.sum()).astype(np.float32)


def oracle_topk(
    shared: Any,
    routed: Any,
    target: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Select experts by exact residual minimization for each token.

    For p8/top-2 this enumerates all 28 pairs and is a correctness reference.
    Larger profiles remain deterministic but may be slower; callers can use a
    greedy approximation outside this bounded oracle function.
    """

    np = _np()
    shared_values = np.asarray(shared)
    routed_values = np.asarray(routed)
    target_values = np.asarray(target)
    if routed_values.ndim != 3 or target_values.ndim != 2:
        raise ValueError("routed must be [tokens, experts, output] and target [tokens, output]")
    tokens, experts, _ = routed_values.shape
    if top_k <= 0 or top_k > experts:
        raise ValueError("invalid top_k")
    combinations = list(itertools.combinations(range(experts), top_k))
    ids = np.zeros((tokens, top_k), dtype=np.int64)
    weights = np.zeros((tokens, top_k), dtype=np.float32)
    residual = np.zeros(tokens, dtype=np.float64)
    for token in range(tokens):
        base = shared_values[token]
        goal = target_values[token] - base
        best_error = float("inf")
        best_combo = combinations[0]
        best_weights = np.ones(top_k, dtype=np.float32) / top_k
        for combo in combinations:
            matrix = routed_values[token, list(combo)].T
            candidate_weights = _normalized_positive_weights(matrix, goal)
            candidate = matrix @ candidate_weights
            error = float(np.mean((candidate - goal) ** 2))
            if error < best_error - 1e-15 or (abs(error - best_error) <= 1e-15 and combo < best_combo):
                best_error = error
                best_combo = combo
                best_weights = candidate_weights
        ids[token] = best_combo
        weights[token] = best_weights
        residual[token] = best_error
    reconstruction = np.asarray(shared_values).copy()
    for token in range(tokens):
        reconstruction[token] += sum(weights[token, slot] * routed_values[token, ids[token, slot]] for slot in range(top_k))
    mse = float(np.mean((reconstruction - target_values) ** 2))
    norm = float(np.mean(target_values**2)) + 1e-12
    dot = np.sum(reconstruction * target_values, axis=-1)
    cosine = float(np.mean(dot / ((np.linalg.norm(reconstruction, axis=-1) * np.linalg.norm(target_values, axis=-1)) + 1e-12)))
    return {
        "indices": ids,
        "weights": weights,
        "reconstruction": reconstruction,
        "mse": mse,
        "normalized_mse": mse / norm,
        "cosine": cosine,
        "residual_by_token": residual,
    }


def sparse_baseline(shared: Any, routed: Any, *, top_k: int, mode: str = "normalized_topk") -> dict[str, Any]:
    """Evaluate deterministic negative/control sparse baselines."""

    np = _np()
    shared_values = np.asarray(shared)
    routed_values = np.asarray(routed)
    tokens, experts, _ = routed_values.shape
    if mode not in {"normalized_topk", "capacity_scaled", "first_topk", "random_balanced"}:
        raise ValueError(f"unknown sparse baseline: {mode}")
    if mode == "first_topk":
        ids = np.tile(np.arange(top_k), (tokens, 1))
    elif mode == "random_balanced":
        ids = np.asarray([np.arange(token * top_k, token * top_k + top_k) % experts for token in range(tokens)])
    else:
        scores = np.linalg.norm(routed_values, axis=-1)
        ids = np.argsort(-scores, axis=-1, kind="stable")[:, :top_k]
    weights = np.ones((tokens, top_k), dtype=np.float32) / top_k
    if mode == "capacity_scaled":
        weights *= experts / top_k
    reconstruction = shared_values.copy()
    for token in range(tokens):
        for slot in range(top_k):
            reconstruction[token] += weights[token, slot] * routed_values[token, ids[token, slot]]
    return {"mode": mode, "indices": ids, "weights": weights, "reconstruction": reconstruction}


__all__ = ["oracle_topk", "sparse_baseline", "swiglu_contributions"]
