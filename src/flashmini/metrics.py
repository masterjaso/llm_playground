"""Metrics engine — continuous machine-readable metric emission (JSONL + summary).

No benchmark result may exist only in console output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch


def _jsonable(value: Any) -> Any:
    """Convert tensors/numpy scalars to JSON-serializable Python values."""
    if isinstance(value, torch.Tensor):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class MetricsLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def log(self, **kwargs: Any) -> None:
        record = {"ts": time.time(), **_jsonable(kwargs)}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_summary(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))