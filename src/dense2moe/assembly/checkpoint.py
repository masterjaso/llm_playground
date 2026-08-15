"""Strict assembly validation for real layer checkpoints.

The assembler intentionally produces a manifest even on rejection so a
resumable run has durable diagnostics.  ``complete`` is true only when every
expected layer is a validated safetensors-backed checkpoint with matching
fingerprints.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..checkpoint.layer import load_layer_checkpoint, validate_layer_checkpoint
from ..provenance import current_git_commit
from ..state import atomic_write_json


def target_state_dict_inventory(
    *,
    num_layers: int,
    hidden_size: int,
    routed_experts: int,
    expert_intermediate_size: int,
    shared_intermediate_size: int,
    attention_keys: Iterable[str] | None = None,
) -> list[str]:
    """Generate the exact versioned text-target inventory.

    ``attention_keys`` can be supplied from a native Qwen 3.5 config.  The
    default inventory focuses on the target-owned tensors and uses explicit
    SwiGLU names (never generic ReLU ``w1/w2`` placeholders).
    """

    if min(num_layers, hidden_size, routed_experts, expert_intermediate_size, shared_intermediate_size) <= 0:
        raise ValueError("target dimensions must be positive")
    keys: list[str] = ["model.embed_tokens.weight"]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        keys.extend([f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"])
        if attention_keys:
            keys.extend(f"{prefix}.{suffix}" for suffix in attention_keys)
        mlp = f"{prefix}.mlp"
        keys.extend(
            [
                f"{mlp}.router.weight",
                f"{mlp}.shared_expert.gate_proj.weight",
                f"{mlp}.shared_expert.up_proj.weight",
                f"{mlp}.shared_expert.down_proj.weight",
            ]
        )
        for expert in range(routed_experts):
            base = f"{mlp}.experts.{expert}"
            keys.extend([f"{base}.gate_proj.weight", f"{base}.up_proj.weight", f"{base}.down_proj.weight"])
    keys.extend(["model.norm.weight", "lm_head.weight"])
    return keys


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_checkpoint(
    layer_paths: Iterable[str | Path],
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and publish an assembly manifest.

    Invalid/placeholder checkpoints are recorded in ``errors`` and can never
    make ``complete`` true.  No existing layer artifact is overwritten.
    """

    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    manifest_metadata = dict(metadata or {})
    manifest_commit = current_git_commit()
    manifest_metadata["code_commit"] = manifest_commit
    expected_layers_raw = manifest_metadata.get("expected_layers")
    try:
        expected_layers = int(expected_layers_raw) if expected_layers_raw is not None else None
    except (TypeError, ValueError):
        expected_layers = None
    expected_profile = manifest_metadata.get("profile")
    expected_profile_hash = manifest_metadata.get("profile_hash")
    expected_revision = manifest_metadata.get("source_revision")
    expected_dataset = manifest_metadata.get("dataset_hash")
    paths = [Path(item) for item in layer_paths]
    errors: list[str] = []
    layers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(paths, key=lambda item: item.name):
        if not path.exists():
            errors.append(f"missing metadata: {path}")
            continue
        try:
            checkpoint = load_layer_checkpoint(path)
            layer = checkpoint.layer
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid metadata {path}: {exc}")
            continue
        if layer in seen:
            errors.append(f"duplicate layer: {layer}")
            continue
        seen.add(layer)
        valid, layer_errors, _ = validate_layer_checkpoint(
            path,
            expected_profile=str(expected_profile) if expected_profile else None,
            expected_profile_hash=str(expected_profile_hash) if expected_profile_hash else None,
            expected_source_revision=str(expected_revision) if expected_revision else None,
            expected_dataset_hash=str(expected_dataset) if expected_dataset else None,
            expected_layer=layer,
        )
        if not valid:
            errors.extend(f"layer {layer}: {item}" for item in layer_errors)
        layers.append(
            {
                "layer": layer,
                "metadata": str(path),
                "metadata_sha256": _digest(path),
                "status": checkpoint.status,
                "profile": checkpoint.profile,
                "profile_hash": checkpoint.profile_hash,
                "source_revision": checkpoint.source_revision,
                "dataset_hash": checkpoint.dataset_hash,
                "code_commit": checkpoint.code_commit,
                "tensor_file": checkpoint.tensor_file,
                "tensor_sha256": checkpoint.tensor_sha256,
                "tensor_inventory": checkpoint.tensor_inventory,
                "quality_gate": checkpoint.quality_gate,
                "valid": valid,
            }
        )
    if expected_layers is not None:
        if expected_layers <= 0:
            errors.append("expected_layers must be positive")
        missing = sorted(set(range(max(0, expected_layers))) - seen)
        unexpected = sorted(layer for layer in seen if layer < 0 or layer >= expected_layers)
        if missing:
            errors.append(f"missing layers: {missing}")
        if unexpected:
            errors.append(f"unexpected layers: {unexpected}")
    # A model assembled from independently generated corpora, source pins, or
    # code versions is scientifically incomparable.  Enforce consistency even
    # when the caller did not provide explicit expected fingerprints.
    if layers:
        for field in ("profile_hash", "source_revision", "dataset_hash", "code_commit"):
            values = {str(item.get(field, "")) for item in layers}
            if "" in values:
                errors.append(f"missing {field} fingerprint")
            elif len(values) > 1:
                errors.append(f"mismatched {field} fingerprints")
    complete = bool(layers) and not errors and all(item["valid"] for item in layers)
    if expected_layers is not None:
        complete = complete and len(layers) == expected_layers
    manifest = {
        "format": "dense2moe-manifest-v2",
        "complete": complete,
        "errors": errors,
        "layers": layers,
        "metadata": manifest_metadata,
        "code_commit": manifest_commit,
        "strict": True,
    }
    atomic_write_json(dst / "manifest.json", manifest)
    return manifest


__all__ = ["assemble_checkpoint", "target_state_dict_inventory"]
