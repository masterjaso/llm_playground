"""Framework-neutral top-k routing primitives."""

from __future__ import annotations

import math
from typing import Any


def _softmax_row(row: list[float]) -> list[float]:
    maximum = max(row) if row else 0.0
    exp = [math.exp(value - maximum) for value in row]
    total = sum(exp) or 1.0
    return [value / total for value in exp]


def normalize_topk_weights(weights: Any, *, axis: int = -1) -> Any:
    """Normalize selected router weights so each token sums to one."""

    if getattr(weights.__class__, "__module__", "").startswith("torch"):
        denominator = weights.sum(dim=axis, keepdim=True)
        return weights / denominator.clamp_min(1e-12)
    try:
        import numpy as np  # type: ignore

        values = np.asarray(weights)
        denominator = np.sum(values, axis=axis, keepdims=True)
        return values / np.where(denominator == 0, 1.0, denominator)
    except ImportError:
        if isinstance(weights, list) and weights and isinstance(weights[0], list):
            return [[value / (sum(row) or 1.0) for value in row] for row in weights]
        total = sum(weights) or 1.0
        return [value / total for value in weights]


def topk_router(logits: Any, top_k: int) -> tuple[Any, Any]:
    """Return deterministic selected expert indices and normalized weights."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if getattr(logits.__class__, "__module__", "").startswith("torch"):
        import torch  # type: ignore

        values = logits if logits.ndim > 1 else logits.unsqueeze(0)
        if top_k > values.shape[-1]:
            raise ValueError("top_k cannot exceed expert count")
        if not torch.isfinite(values).all():
            raise ValueError("router logits must be finite")
        selected_logits, torch_indices = torch.topk(values, top_k, dim=-1)
        torch_weights = torch.softmax(selected_logits, dim=-1)
        return torch_indices, torch_weights
    try:
        import numpy as np  # type: ignore

        numpy_values = np.asarray(logits)
        if numpy_values.ndim == 1:
            numpy_values = numpy_values[None, :]
        if top_k > numpy_values.shape[-1]:
            raise ValueError("top_k cannot exceed expert count")
        if not np.isfinite(numpy_values).all():
            raise ValueError("router logits must be finite")
        # Stable descending sort makes ties reproducible across platforms.
        numpy_order = np.argsort(-numpy_values, axis=-1, kind="stable")[:, :top_k]
        numpy_selected_logits = np.take_along_axis(numpy_values, numpy_order, axis=-1)
        selected = np.exp(numpy_selected_logits - np.max(numpy_selected_logits, axis=-1, keepdims=True))
        weights = normalize_topk_weights(selected)
        return numpy_order, weights
    except ImportError:
        rows = logits if isinstance(logits, list) and logits and isinstance(logits[0], list) else [logits]
        fallback_indices: list[list[int]] = []
        fallback_weights: list[list[float]] = []
        for row in rows:
            if top_k > len(row):
                raise ValueError("top_k cannot exceed expert count")
            if not all(math.isfinite(float(value)) for value in row):
                raise ValueError("router logits must be finite")
            chosen = sorted(range(len(row)), key=lambda index: (-row[index], index))[:top_k]
            selected = _softmax_row([row[index] for index in chosen])
            fallback_indices.append(chosen)
            fallback_weights.append(selected)
        return (fallback_indices[0], fallback_weights[0]) if not (isinstance(logits, list) and logits and isinstance(logits[0], list)) else (fallback_indices, fallback_weights)


def shared_gate_initialization(size: int, pass_through: float = 0.99) -> Any:
    """Return logits whose sigmoid is a near-pass-through shared gate."""

    if not 0.0 < pass_through < 1.0:
        raise ValueError("pass_through must be between zero and one")
    logit = math.log(pass_through / (1.0 - pass_through))
    try:
        import numpy as np  # type: ignore

        return np.full(size, logit, dtype=np.float32)
    except ImportError:
        return [logit for _ in range(size)]


def weighted_expert_sum(expert_outputs: Any, indices: Any, weights: Any) -> Any:
    """Combine selected expert outputs for a token batch."""

    try:
        import numpy as np  # type: ignore

        outputs = np.asarray(expert_outputs)
        selected = np.asarray(indices)
        selected_weights = np.asarray(weights)
        result: Any = np.zeros((selected.shape[0], outputs.shape[-1]), dtype=outputs.dtype)
        for token in range(selected.shape[0]):
            for slot in range(selected.shape[1]):
                result[token] += selected_weights[token, slot] * outputs[selected[token, slot], token]
        return result
    except ImportError:
        result = []
        for token, token_indices in enumerate(indices):
            width = len(expert_outputs[token_indices[0]][token])
            row = [0.0] * width
            for slot, expert in enumerate(token_indices):
                for col in range(width):
                    row[col] += weights[token][slot] * expert_outputs[expert][token][col]
            result.append(row)
        return result
