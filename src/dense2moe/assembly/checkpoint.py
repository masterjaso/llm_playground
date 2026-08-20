"""Strict assembly validation for real layer checkpoints.

The assembler intentionally produces a manifest even on rejection so a
resumable run has durable diagnostics.  ``complete`` is true only when every
expected layer is a validated safetensors-backed checkpoint with matching
fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..checkpoint.layer import load_layer_checkpoint, validate_layer_checkpoint
from ..provenance import current_git_commit
from ..state import atomic_write_json

ASSEMBLY_MANIFEST_FORMAT = "dense2moe-manifest-v3"
ASSEMBLY_RECEIPT_FORMAT = "dense2moe-assembly-receipt-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_inventory(value: Any, *, field_name: str) -> tuple[dict[str, Any], str | None]:
    """Return a name-keyed inventory while retaining optional shape metadata."""

    if isinstance(value, Mapping):
        return {str(name): item for name, item in value.items()}, None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return {str(name): {} for name in value}, None
    return {}, f"{field_name} must be a mapping or iterable of tensor names"


def _resolve_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return base / candidate


def _valid_sha256(value: Any) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value)))



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
    if expected_layers is None:
        errors.append("expected_layers is required for strict assembly")
    elif expected_layers <= 0:
        errors.append("expected_layers must be positive")
    if not paths:
        errors.append("no layer checkpoints supplied")

    for path in sorted(paths, key=lambda item: (item.name, item.as_posix())):
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
        try:
            metadata_digest = _digest(path)
        except OSError as exc:
            errors.append(f"layer {layer}: cannot hash metadata: {exc}")
            metadata_digest = ""
        tensor_inventory = dict(checkpoint.tensor_inventory) if isinstance(checkpoint.tensor_inventory, Mapping) else {}
        layers.append(
            {
                "layer": layer,
                "metadata": str(path),
                "metadata_sha256": metadata_digest,
                "status": checkpoint.status,
                "profile": checkpoint.profile,
                "profile_hash": checkpoint.profile_hash,
                "source_revision": checkpoint.source_revision,
                "source_config_hash": checkpoint.source_config_hash,
                "source_index_hash": checkpoint.source_index_hash,
                "dataset_hash": checkpoint.dataset_hash,
                "partition_hash": checkpoint.partition_hash,
                "routing_mode": checkpoint.routing_mode,
                "code_commit": checkpoint.code_commit,
                "tensor_file": checkpoint.tensor_file,
                "tensor_sha256": checkpoint.tensor_sha256,
                "tensor_inventory": tensor_inventory,
                "tensor_inventory_sha256": _canonical_digest(tensor_inventory),
                "quality_gate": checkpoint.quality_gate,
                "valid": valid,
            }
        )
    if expected_layers is not None:
        missing = sorted(set(range(max(0, expected_layers))) - seen)
        unexpected = sorted(layer for layer in seen if layer < 0 or layer >= expected_layers)
        if missing:
            errors.append(f"missing layers: {missing}")
        if unexpected:
            errors.append(f"unexpected layers: {unexpected}")

    # Preserve the exact tensor inventory from every validated layer.  The
    # aggregate is deliberately name-keyed so a duplicate tensor silently
    # shadowing another layer can never be promoted.
    tensor_inventory: dict[str, Any] = {}
    tensor_artifacts: list[dict[str, Any]] = []
    for item in layers:
        for name, details in sorted(item["tensor_inventory"].items()):
            if name in tensor_inventory:
                errors.append(f"duplicate tensor inventory entry: {name}")
            else:
                tensor_inventory[name] = details
        tensor_artifacts.append(
            {
                "layer": item["layer"],
                "metadata": item["metadata"],
                "tensor_file": item["tensor_file"],
                "tensor_sha256": item["tensor_sha256"],
                "tensor_inventory_sha256": item["tensor_inventory_sha256"],
            }
        )
    if not tensor_inventory:
        errors.append("missing aggregate tensor inventory")

    preserved_inventory: dict[str, Any] = {}
    preserved_raw = manifest_metadata.get("preserved_tensor_inventory")
    if preserved_raw is not None:
        preserved_inventory, inventory_error = _normalise_inventory(preserved_raw, field_name="preserved_tensor_inventory")
        if inventory_error:
            errors.append(inventory_error)
        for name, details in sorted(preserved_inventory.items()):
            if name in tensor_inventory:
                errors.append(f"preserved tensor overlaps layer inventory: {name}")
            else:
                tensor_inventory[name] = details
    preserved_hashes: dict[str, str] = {}
    preserved_hashes_raw = manifest_metadata.get("preserved_tensor_hashes")
    if preserved_hashes_raw is not None:
        if not isinstance(preserved_hashes_raw, Mapping):
            errors.append("preserved_tensor_hashes must be a mapping")
        else:
            preserved_hashes = {str(name): str(value) for name, value in preserved_hashes_raw.items()}
            for name, value in preserved_hashes.items():
                if not _valid_sha256(value):
                    errors.append(f"invalid preserved tensor hash: {name}")
            for name in preserved_inventory:
                if name not in preserved_hashes:
                    errors.append(f"missing preserved tensor hash: {name}")
    if preserved_inventory and not manifest_metadata.get("preserved_tensor_file") and not manifest_metadata.get("preserved_tensor_files"):
        errors.append("preserved tensor artifact path is required for strict assembly")

    # A preservation claim is only useful when its source artifact is present
    # and content-addressed.  The per-tensor hashes remain in the receipt;
    # this file hash guards the artifact boundary itself.
    preserved_artifact_specs: list[tuple[str, str | None]] = []
    preserved_file = manifest_metadata.get("preserved_tensor_file")
    if preserved_file:
        preserved_artifact_specs.append(
            (str(preserved_file), str(manifest_metadata.get("preserved_tensor_file_sha256")) if manifest_metadata.get("preserved_tensor_file_sha256") else None)
        )
    preserved_files = manifest_metadata.get("preserved_tensor_files")
    if preserved_files is not None:
        if not isinstance(preserved_files, Iterable) or isinstance(preserved_files, (str, bytes, bytearray)):
            errors.append("preserved_tensor_files must be an iterable")
        else:
            for item in preserved_files:
                if isinstance(item, Mapping):
                    path_value = item.get("path")
                    hash_value = item.get("sha256")
                else:
                    path_value = item
                    hash_value = None
                if not path_value:
                    errors.append("preserved tensor artifact path is missing")
                else:
                    preserved_artifact_specs.append((str(path_value), str(hash_value) if hash_value else None))
    preserved_artifacts: list[dict[str, Any]] = []
    for path_value, expected_hash in preserved_artifact_specs:
        artifact_path = _resolve_path(dst, path_value)
        if not artifact_path.is_file():
            errors.append(f"missing preserved tensor artifact: {artifact_path}")
            continue
        if not expected_hash or not _valid_sha256(expected_hash):
            errors.append(f"missing or invalid preserved tensor artifact hash: {artifact_path}")
            continue
        actual_hash = _digest(artifact_path)
        if actual_hash != expected_hash:
            errors.append(f"preserved tensor artifact hash mismatch: {artifact_path}")
        preserved_artifacts.append({"path": str(artifact_path), "sha256": actual_hash})

    expected_inventory: dict[str, Any] | None = None
    expected_inventory_raw = manifest_metadata.get("expected_tensor_inventory")
    if expected_inventory_raw is not None:
        expected_inventory, inventory_error = _normalise_inventory(expected_inventory_raw, field_name="expected_tensor_inventory")
        if inventory_error:
            errors.append(inventory_error)
        else:
            missing_tensors = sorted(set(expected_inventory) - set(tensor_inventory))
            unexpected_tensors = sorted(set(tensor_inventory) - set(expected_inventory))
            if missing_tensors:
                errors.append(f"missing tensors: {missing_tensors}")
            if unexpected_tensors:
                errors.append(f"unexpected tensors: {unexpected_tensors}")
            for name, expected_details in expected_inventory.items():
                actual_details = tensor_inventory.get(name)
                if isinstance(expected_details, Mapping) and isinstance(actual_details, Mapping):
                    for detail_name in ("shape", "dtype"):
                        if detail_name in expected_details and expected_details[detail_name] != actual_details.get(detail_name):
                            errors.append(f"tensor {name} {detail_name} mismatch")

    # Re-check all identity fields at the assembly boundary.  The layer
    # validator checks presence and file integrity; this check prevents a
    # complete manifest from mixing independently produced layer artifacts.
    if layers:
        for field in ("profile_hash", "source_revision", "dataset_hash", "code_commit"):
            values = {str(item.get(field, "")) for item in layers}
            if "" in values:
                errors.append(f"missing {field} fingerprint")
            elif len(values) > 1:
                errors.append(f"mismatched {field} fingerprints")
        for item in layers:
            if str(item.get("code_commit", "")) != manifest_commit:
                errors.append(f"layer {item['layer']}: code_commit does not match assembly code_commit")

    aggregate_inventory_hash = _canonical_digest(tensor_inventory)
    expected_inventory_hash = manifest_metadata.get("expected_tensor_inventory_sha256")
    if expected_inventory_hash is not None:
        if not _valid_sha256(expected_inventory_hash):
            errors.append("invalid expected_tensor_inventory_sha256")
        elif str(expected_inventory_hash) != aggregate_inventory_hash:
            errors.append("tensor inventory hash mismatch")

    artifact_scope = str(manifest_metadata.get("artifact_scope", "validated-layer-tensors"))
    complete = bool(layers) and not errors and all(item["valid"] for item in layers)
    if expected_layers is not None:
        complete = complete and len(layers) == expected_layers
    manifest = {
        "format": ASSEMBLY_MANIFEST_FORMAT,
        "complete": complete,
        "errors": errors,
        "layers": layers,
        "layer_inventory": {
            "expected": expected_layers,
            "observed": sorted(seen),
            "count": len(seen),
        },
        "tensor_inventory": tensor_inventory,
        "tensor_inventory_sha256": aggregate_inventory_hash,
        "tensor_artifacts": tensor_artifacts,
        "preserved_tensor_inventory": preserved_inventory,
        "preserved_tensor_hashes": preserved_hashes,
        "preserved_tensor_artifacts": preserved_artifacts,
        "artifact_scope": artifact_scope,
        "metadata": manifest_metadata,
        "code_commit": manifest_commit,
        "strict": True,
        "receipt": {
            "format": ASSEMBLY_RECEIPT_FORMAT,
            "path": "assembly-receipt.json",
        },
    }
    manifest_path = dst / "manifest.json"
    receipt_path = dst / "assembly-receipt.json"
    atomic_write_json(manifest_path, manifest)
    manifest_digest = _digest(manifest_path)
    receipt = {
        "format": ASSEMBLY_RECEIPT_FORMAT,
        "status": "ASSEMBLY_COMPLETE" if complete else "ASSEMBLY_BLOCKED",
        "complete": complete,
        "manifest": "manifest.json",
        "manifest_sha256": manifest_digest,
        "assembly_fingerprint": _canonical_digest(
            {
                "code_commit": manifest_commit,
                "layers": [
                    {
                        "layer": item["layer"],
                        "metadata_sha256": item["metadata_sha256"],
                        "tensor_sha256": item["tensor_sha256"],
                        "tensor_inventory_sha256": item["tensor_inventory_sha256"],
                    }
                    for item in layers
                ],
                "tensor_inventory_sha256": aggregate_inventory_hash,
            }
        ),
        "code_commit": manifest_commit,
        "profile": manifest_metadata.get("profile"),
        "profile_hash": manifest_metadata.get("profile_hash"),
        "source_revision": manifest_metadata.get("source_revision"),
        "dataset_hash": manifest_metadata.get("dataset_hash"),
        "artifact_scope": artifact_scope,
        "layer_inventory": manifest["layer_inventory"],
        "tensor_inventory": tensor_inventory,
        "tensor_inventory_sha256": aggregate_inventory_hash,
        "tensor_artifacts": tensor_artifacts,
        "preserved_tensor_inventory": preserved_inventory,
        "preserved_tensor_hashes": preserved_hashes,
        "preserved_tensor_artifacts": preserved_artifacts,
        "errors": errors,
    }
    atomic_write_json(receipt_path, receipt)
    return manifest


def validate_assembly_receipt(path: str | Path) -> dict[str, Any]:
    """Validate the durable assembly manifest/receipt pair without mutation."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    if candidate.name == "assembly-receipt.json":
        receipt_path = candidate
        manifest_path = candidate.with_name("manifest.json")
    else:
        manifest_path = candidate
        receipt_path = manifest_path.with_name("assembly-receipt.json")
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return {"valid": False, "errors": [f"cannot read assembly manifest: {exc}"]}
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return {"valid": False, "errors": [f"cannot read assembly receipt: {exc}"], "manifest": manifest}
    if not isinstance(manifest, Mapping) or not isinstance(receipt, Mapping):
        errors.append("assembly manifest and receipt must be JSON objects")
    else:
        manifest_receipt = manifest.get("receipt")
        if not isinstance(manifest_receipt, Mapping) or manifest_receipt.get("format") != ASSEMBLY_RECEIPT_FORMAT:
            errors.append("assembly manifest receipt format mismatch")
        if receipt.get("format") != ASSEMBLY_RECEIPT_FORMAT:
            errors.append("assembly receipt format mismatch")
        if receipt.get("manifest") != manifest_path.name:
            errors.append("assembly receipt manifest path mismatch")
        try:
            manifest_digest = _digest(manifest_path)
        except OSError as exc:
            errors.append(f"cannot hash assembly manifest: {exc}")
        else:
            if receipt.get("manifest_sha256") != manifest_digest:
                errors.append("assembly manifest hash mismatch")
        if receipt.get("complete") != manifest.get("complete"):
            errors.append("assembly receipt completion mismatch")
        expected_status = "ASSEMBLY_COMPLETE" if manifest.get("complete") else "ASSEMBLY_BLOCKED"
        if receipt.get("status") != expected_status:
            errors.append("assembly receipt status mismatch")
        if receipt.get("tensor_inventory_sha256") != manifest.get("tensor_inventory_sha256"):
            errors.append("assembly tensor inventory hash mismatch")
        if receipt.get("code_commit") != manifest.get("code_commit"):
            errors.append("assembly code provenance mismatch")
        if manifest.get("complete") and (manifest.get("errors") or receipt.get("errors")):
            errors.append("complete assembly contains errors")
    return {"valid": not errors, "errors": errors, "manifest": manifest, "receipt": receipt}


__all__ = ["assemble_checkpoint", "target_state_dict_inventory", "validate_assembly_receipt"]
