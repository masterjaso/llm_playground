"""Tiny numpy models for end-to-end structural smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .router import topk_router


def _numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError as exc:
        raise RuntimeError("numpy is required for tiny model tests") from exc


@dataclass
class TinyDenseFFN:
    input_size: int
    hidden_size: int
    output_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        np = _numpy()
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0, 0.1, size=(self.input_size, self.hidden_size)).astype("float32")
        self.w2 = rng.normal(0, 0.1, size=(self.hidden_size, self.output_size)).astype("float32")

    def __call__(self, x: Any) -> Any:
        np = _numpy()
        hidden = np.maximum(0, np.asarray(x) @ self.w1)
        return hidden @ self.w2


class TinyMoE:
    def __init__(self, dense: TinyDenseFFN, experts: int = 2, top_k: int = 1):
        np = _numpy()
        self.dense = dense
        self.experts = experts
        self.top_k = top_k
        # An exact all-expert initialization is useful for testing the routing
        # contract before any sparse optimization.
        self.w1 = np.stack([dense.w1.copy() for _ in range(experts)], axis=0)
        self.w2 = np.stack([dense.w2.copy() for _ in range(experts)], axis=0)
        self.router = np.zeros((dense.input_size, experts), dtype="float32")

    def __call__(self, x: Any, *, all_experts: bool = False) -> Any:
        np = _numpy()
        values = np.asarray(x)
        hidden = np.maximum(0, np.einsum("ti,eih->teh", values, self.w1))
        outputs = np.einsum("teh,eho->teo", hidden, self.w2)
        if all_experts:
            return outputs.mean(axis=1)
        logits = values @ self.router
        indices, weights = topk_router(logits, self.top_k)
        result = np.zeros((values.shape[0], self.dense.output_size), dtype="float32")
        for token in range(values.shape[0]):
            for slot, expert in enumerate(indices[token]):
                result[token] += weights[token, slot] * outputs[token, expert]
        return result
