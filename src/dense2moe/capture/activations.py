"""Small activation-capture backend that remains resumable and deterministic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def activation_partition_deterministic(num_items: int, partitions: int, seed: int = 0) -> list[list[int]]:
    if num_items < 0 or partitions <= 0:
        raise ValueError("num_items must be non-negative and partitions positive")
    # Hash-based ordering avoids dependence on process hash randomization.
    order = sorted(range(num_items), key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest())
    result: list[list[int]] = [[] for _ in range(partitions)]
    for offset, item in enumerate(order):
        result[offset % partitions].append(item)
    return result


def capture_activations(activations: Iterable[Any], destination: str | Path, *, layer: int, dtype: str = "float32") -> dict[str, Any]:
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    values = list(activations)
    output = dst / f"layer-{layer:04d}.json"
    payload = {"layer": layer, "dtype": dtype, "count": len(values), "values": values}
    output.write_text(json.dumps(payload, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"path": str(output), "layer": layer, "count": len(values), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
