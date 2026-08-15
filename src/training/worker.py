"""Small, testable training primitives used by independent layer workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OOMBackoff:
    microbatch: int
    max_retries: int = 3
    retries: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def next(self, error: BaseException) -> int:
        message = str(error).lower()
        if "out of memory" not in message and "cuda" not in message:
            raise error
        if self.retries >= self.max_retries or self.microbatch <= 1:
            raise error
        old = self.microbatch
        self.retries += 1
        self.microbatch = max(1, self.microbatch // 2)
        self.history.append({"retry": self.retries, "old_microbatch": old, "new_microbatch": self.microbatch, "error": str(error)})
        return self.microbatch

    def run(self, operation: Callable[[int], Any]) -> Any:
        while True:
            try:
                return operation(self.microbatch)
            except RuntimeError as exc:
                self.next(exc)


def train_tiny_layer(inputs: Any, targets: Any, *, epochs: int = 3, learning_rate: float = 1e-2) -> dict[str, Any]:
    """Fit a linear output map using numpy; deterministic and CPU-friendly."""

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for tiny training") from exc
    x = np.asarray(inputs, dtype="float64")
    y = np.asarray(targets, dtype="float64")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("inputs and targets must be rank-2 arrays with matching rows")
    weights = np.zeros((x.shape[1], y.shape[1]), dtype="float64")
    losses: list[float] = []
    for _ in range(max(1, epochs)):
        prediction = x @ weights
        error = prediction - y
        losses.append(float(np.mean(error * error)))
        gradient = (2.0 / x.shape[0]) * (x.T @ error)
        weights -= learning_rate * gradient
    return {"weights": weights.astype("float32"), "losses": losses, "final_loss": losses[-1]}

