"""Versioned layer checkpoint metadata and real ``safetensors`` artifacts.

JSON is descriptive metadata only.  Assembly accepts a layer only when the
metadata points at a hashed, finite safetensors file and its quality gate is
green (or an explicit research-candidate override is present).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..state import atomic_write_json

LAYER_SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_fingerprint(profile: Mapping[str, Any] | str) -> str:
    payload = profile if isinstance(profile, Mapping) else {"name": profile}
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LayerCheckpoint:
    # The first four fields retain the original lightweight fixture API.
    layer: int
    profile: str
    tensors: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LAYER_SCHEMA_VERSION
    status: str = "PENDING"
    profile_hash: str = ""
    source_revision: str = "unknown"
    source_config_hash: str = ""
    source_index_hash: str = ""
    dataset_hash: str = ""
    partition_strategy: str = "contiguous"
    partition_hash: str = ""
    router_architecture: str = "topk-normalized"
    training_seed: int = 0
    training_config: dict[str, Any] = field(default_factory=dict)
    tensor_file: str | None = None
    tensor_sha256: str | None = None
    tensor_inventory: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    router_metrics: dict[str, Any] = field(default_factory=dict)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    code_commit: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        now = _now()
        created = self.created_at or now
        updated = self.updated_at or now
        return {
            "schema_version": self.schema_version,
            "layer": self.layer,
            "status": self.status,
            "profile": self.profile,
            "profile_hash": self.profile_hash,
            "source_revision": self.source_revision,
            "source_config_hash": self.source_config_hash,
            "source_index_hash": self.source_index_hash,
            "dataset_hash": self.dataset_hash,
            "partition_strategy": self.partition_strategy,
            "partition_hash": self.partition_hash,
            "router_architecture": self.router_architecture,
            "training_seed": self.training_seed,
            "training_config": self.training_config,
            "tensor_file": self.tensor_file,
            "tensor_sha256": self.tensor_sha256,
            "tensor_inventory": self.tensor_inventory,
            "train_metrics": self.train_metrics,
            "holdout_metrics": self.holdout_metrics,
            "router_metrics": self.router_metrics,
            "quality_gate": self.quality_gate,
            "created_at": created,
            "updated_at": updated,
            "code_commit": self.code_commit,
            # Compatibility fields are retained but never satisfy assembly.
            "tensors": self.tensors,
            "metrics": self.metrics,
        }


def _metadata_to_checkpoint(payload: Mapping[str, Any]) -> LayerCheckpoint:
    return LayerCheckpoint(
        layer=int(payload["layer"]),
        profile=str(payload.get("profile", "unknown")),
        tensors=dict(payload.get("tensors", {})),
        metrics=dict(payload.get("metrics", {})),
        schema_version=int(payload.get("schema_version", 1)),
        status=str(payload.get("status", "PENDING")),
        profile_hash=str(payload.get("profile_hash", "")),
        source_revision=str(payload.get("source_revision", "unknown")),
        source_config_hash=str(payload.get("source_config_hash", "")),
        source_index_hash=str(payload.get("source_index_hash", "")),
        dataset_hash=str(payload.get("dataset_hash", "")),
        partition_strategy=str(payload.get("partition_strategy", "contiguous")),
        partition_hash=str(payload.get("partition_hash", "")),
        router_architecture=str(payload.get("router_architecture", "topk-normalized")),
        training_seed=int(payload.get("training_seed", 0)),
        training_config=dict(payload.get("training_config", {})),
        tensor_file=payload.get("tensor_file"),
        tensor_sha256=payload.get("tensor_sha256"),
        tensor_inventory=dict(payload.get("tensor_inventory", {})),
        train_metrics=dict(payload.get("train_metrics", {})),
        holdout_metrics=dict(payload.get("holdout_metrics", {})),
        router_metrics=dict(payload.get("router_metrics", {})),
        quality_gate=dict(payload.get("quality_gate", {})),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        code_commit=str(payload.get("code_commit", "unknown")),
    )


def save_layer_checkpoint(checkpoint: LayerCheckpoint, path: str | Path) -> Path:
    """Write metadata only; real workers must publish a tensor artifact too."""

    target = Path(path)
    atomic_write_json(target, checkpoint.as_dict())
    return target


def load_layer_checkpoint(path: str | Path) -> LayerCheckpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("layer checkpoint metadata must be a JSON object")
    return _metadata_to_checkpoint(payload)


def _finite(value: Any) -> bool:
    try:
        import numpy as np  # type: ignore

        return bool(np.isfinite(np.asarray(value)).all())
    except ImportError:
        if isinstance(value, (list, tuple)):
            return all(_finite(item) for item in value)
        return math.isfinite(float(value))


def publish_tensor_artifact(tensors: Mapping[str, Any], path: str | Path) -> tuple[Path, dict[str, Any], str]:
    """Write a finite tensor mapping to safetensors and return inventory/hash."""

    if not tensors:
        raise ValueError("tensor artifact cannot be empty")
    if not all(str(name) for name in tensors):
        raise ValueError("tensor names must be non-empty")
    if not all(_finite(value) for value in tensors.values()):
        raise ValueError("tensor artifact contains NaN or Inf")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np  # type: ignore
        from safetensors.numpy import save_file  # type: ignore

        converted = {str(name): np.ascontiguousarray(np.asarray(value)) for name, value in tensors.items()}
        save_file(converted, str(target))
        inventory = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in converted.items()
        }
    except ImportError as exc:
        raise RuntimeError("safetensors and numpy are required for real layer artifacts") from exc
    return target, inventory, sha256_file(target)


def validate_layer_checkpoint(
    metadata_path: str | Path,
    *,
    expected_profile: str | None = None,
    expected_profile_hash: str | None = None,
    expected_source_revision: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_layer: int | None = None,
    require_quality: bool = True,
) -> tuple[bool, list[str], LayerCheckpoint | None]:
    """Validate metadata, safetensors existence/hash/inventory, and gates."""

    path = Path(metadata_path)
    errors: list[str] = []
    try:
        checkpoint = load_layer_checkpoint(path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, [f"invalid metadata: {exc}"], None
    if checkpoint.schema_version < 2:
        errors.append("legacy schema")
    if checkpoint.status != "TRAINED_VALIDATED":
        errors.append(f"status is {checkpoint.status!r}")
    if expected_layer is not None and checkpoint.layer != expected_layer:
        errors.append("layer mismatch")
    if expected_profile and checkpoint.profile != expected_profile:
        errors.append("profile mismatch")
    if expected_profile_hash and checkpoint.profile_hash != expected_profile_hash:
        errors.append("profile hash mismatch")
    if expected_source_revision and checkpoint.source_revision != expected_source_revision:
        errors.append("source revision mismatch")
    if expected_dataset_hash and checkpoint.dataset_hash != expected_dataset_hash:
        errors.append("dataset hash mismatch")
    if not checkpoint.tensor_file:
        errors.append("missing tensor_file")
    else:
        tensor_path = Path(checkpoint.tensor_file)
        if not tensor_path.is_absolute():
            tensor_path = path.parent / tensor_path
        if not tensor_path.exists():
            errors.append("missing tensor artifact")
        else:
            if checkpoint.tensor_sha256 != sha256_file(tensor_path):
                errors.append("tensor hash mismatch")
            try:
                from safetensors import safe_open  # type: ignore
                with safe_open(str(tensor_path), framework="numpy") as handle:
                    actual_names = sorted(handle.keys())
                    if sorted(checkpoint.tensor_inventory) != actual_names:
                        errors.append("tensor inventory mismatch")
                    for name in actual_names:
                        value = handle.get_tensor(name)
                        if not _finite(value):
                            errors.append(f"non-finite tensor: {name}")
                        expected = checkpoint.tensor_inventory.get(name, {})
                        if list(getattr(value, "shape", ())) != list(expected.get("shape", [])):
                            errors.append(f"shape mismatch: {name}")
                        if expected.get("dtype") and str(getattr(value, "dtype", "")) != str(expected.get("dtype")):
                            errors.append(f"dtype mismatch: {name}")
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"cannot read tensor artifact: {exc}")
    if require_quality and checkpoint.quality_gate.get("overall") not in {"green", "research-candidate"}:
        errors.append("quality gate is not green")
    return not errors, errors, checkpoint
