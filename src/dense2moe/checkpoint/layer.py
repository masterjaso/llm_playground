"""JSON layer-checkpoint round trips for lightweight worker artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..state import atomic_write_json


@dataclass(frozen=True)
class LayerCheckpoint:
    layer: int
    profile: str
    tensors: dict[str, Any]
    metrics: dict[str, Any]
    schema_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "layer": self.layer, "profile": self.profile, "tensors": self.tensors, "metrics": self.metrics}


def save_layer_checkpoint(checkpoint: LayerCheckpoint, path: str | Path) -> Path:
    target = Path(path)
    atomic_write_json(target, checkpoint.as_dict())
    return target


def load_layer_checkpoint(path: str | Path) -> LayerCheckpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LayerCheckpoint(int(payload["layer"]), str(payload["profile"]), dict(payload.get("tensors", {})), dict(payload.get("metrics", {})), int(payload.get("schema_version", 1)))

