"""Strict GGUF artifacts and provenance receipts.

``write_tiny_gguf`` remains available for the historical structural smoke
test. Product exports use :func:`export_gguf`, which requires a complete,
receipt-backed assembly and writes the validated layer tensors into a real
tensor-bearing GGUF container.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..assembly.checkpoint import validate_assembly_receipt
from ..provenance import current_git_commit
from ..state import atomic_write_json

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_EXPORT_RECEIPT_FORMAT = "dense2moe-gguf-export-receipt-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# GGML scalar type IDs used by GGUF. Quantized types are deliberately not
# accepted here: quantization is a separate, gated phase.
_DTYPE_INFO: dict[str, tuple[int, int]] = {
    "float32": (0, 4),
    "float16": (1, 2),
    "float64": (28, 8),
    "int8": (24, 1),
    "uint8": (24, 1),
    "int16": (25, 2),
    "uint16": (25, 2),
    "int32": (26, 4),
    "uint32": (26, 4),
    "int64": (27, 8),
    "uint64": (27, 8),
    "bool": (24, 1),
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pack_string(value: str) -> bytes:
    encoded = str(value).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_string_kv(key: str, value: Any) -> bytes:
    # GGUF type 8 is a UTF-8 string. JSON keeps nested provenance values
    # deterministic while avoiding a dependency on a particular converter.
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        if isinstance(value, (Mapping, list, tuple))
        else str(value)
    )
    return _pack_string(key) + struct.pack("<I", 8) + _pack_string(encoded)


def _align(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _resolve_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return base / candidate


def _dtype_info(dtype: Any) -> tuple[int, int]:
    name = str(dtype).replace("<", "").replace(">", "").replace("=", "").lower()
    try:
        return _DTYPE_INFO[name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype for GGUF export: {dtype}") from exc


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"tensor {name} inventory shape is missing")  # noqa: TRY004
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"tensor {name} inventory shape is invalid")
    return result


def _encode_tensor_info(name: str, shape: tuple[int, ...], tensor_type: int, offset: int) -> bytes:
    encoded = bytearray()
    encoded.extend(_pack_string(name))
    encoded.extend(struct.pack("<I", len(shape)))
    if shape:
        encoded.extend(struct.pack(f"<{len(shape)}Q", *shape))
    encoded.extend(struct.pack("<IQ", tensor_type, offset))
    return bytes(encoded)


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read assembly manifest: {candidate}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("assembly manifest must be a JSON object")  # noqa: TRY004
    return candidate, dict(payload)


def _collect_tensor_records(manifest_path: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("assembly manifest has no layer inventory")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for layer in sorted(
        layers,
        key=lambda item: int(item.get("layer", -1)) if isinstance(item, Mapping) else -1,
    ):
        if not isinstance(layer, Mapping):
            raise ValueError("assembly layer inventory entry must be an object")  # noqa: TRY004
        if not layer.get("valid"):
            raise ValueError(f"layer {layer.get('layer')} is not valid")
        metadata_value = layer.get("metadata")
        tensor_value = layer.get("tensor_file")
        if not metadata_value or not tensor_value:
            raise ValueError(f"layer {layer.get('layer')} is missing tensor provenance")
        metadata_path = _resolve_path(manifest_path.parent, str(metadata_value))
        tensor_path = _resolve_path(metadata_path.parent, str(tensor_value))
        if not tensor_path.is_file():
            raise ValueError(f"missing layer tensor artifact: {tensor_path}")
        expected_hash = str(layer.get("tensor_sha256", ""))
        if not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"layer {layer.get('layer')} has invalid tensor hash")
        actual_hash = _digest(tensor_path)
        if actual_hash != expected_hash:
            raise ValueError(f"layer {layer.get('layer')} tensor hash mismatch")
        inventory = layer.get("tensor_inventory")
        if not isinstance(inventory, Mapping) or not inventory:
            raise ValueError(f"layer {layer.get('layer')} has no tensor inventory")
        for name, details in sorted(inventory.items(), key=lambda item: str(item[0])):
            tensor_name = str(name)
            if tensor_name in names:
                raise ValueError(f"duplicate tensor in assembly export: {tensor_name}")
            if not isinstance(details, Mapping):
                raise ValueError(  # noqa: TRY004
                    f"tensor {tensor_name} inventory must be an object"
                )
            shape = _shape(details.get("shape"), name=tensor_name)
            tensor_type, item_size = _dtype_info(details.get("dtype"))
            records.append(
                {
                    "name": tensor_name,
                    "shape": shape,
                    "tensor_type": tensor_type,
                    "item_size": item_size,
                    "path": tensor_path,
                    "layer": int(layer.get("layer", -1)),
                    "dtype": str(details.get("dtype")),
                }
            )
            names.add(tensor_name)
    manifest_inventory = manifest.get("tensor_inventory")
    if not isinstance(manifest_inventory, Mapping):
        raise ValueError("assembly manifest is missing aggregate tensor inventory")  # noqa: TRY004
    if {str(name) for name in manifest_inventory} != names:
        raise ValueError("assembly tensor inventory does not match layer inventories")
    inventory_hash = manifest.get("tensor_inventory_sha256")
    if inventory_hash and str(inventory_hash) != _canonical_digest(manifest_inventory):
        raise ValueError("assembly tensor inventory hash mismatch")
    return sorted(records, key=lambda item: item["name"])


def _tensor_payload(record: Mapping[str, Any]) -> bytes:
    try:
        import numpy as np  # type: ignore
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy and safetensors are required for strict GGUF export") from exc
    path = Path(record["path"])
    try:
        with safe_open(str(path), framework="numpy") as handle:
            available_names = handle.keys()
            if record["name"] not in available_names:
                raise ValueError(f"tensor missing from safetensors artifact: {record['name']}")
            value = handle.get_tensor(record["name"])
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot read tensor {record['name']}: {exc}") from exc
    actual_shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if actual_shape != tuple(record["shape"]):
        raise ValueError(f"tensor {record['name']} shape mismatch")
    actual_type, actual_size = _dtype_info(getattr(value, "dtype", ""))
    if actual_type != record["tensor_type"] or actual_size != record["item_size"]:
        raise ValueError(f"tensor {record['name']} dtype mismatch")
    array = np.ascontiguousarray(value)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and np.little_endian is False):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    payload = array.tobytes(order="C")
    expected_size = math.prod(record["shape"], start=1) * int(record["item_size"])
    if len(payload) != expected_size:
        raise ValueError(f"tensor {record['name']} byte size mismatch")
    return payload


def _write_tensor_gguf(target: Path, records: list[dict[str, Any]], kv: Mapping[str, Any]) -> None:
    kv_bytes = b"".join(_pack_string_kv(key, value) for key, value in sorted(kv.items()))
    prefix_size = 24 + len(kv_bytes)
    info_size = sum(
        len(_encode_tensor_info(record["name"], record["shape"], record["tensor_type"], 0))
        for record in records
    )
    data_start = _align(prefix_size + info_size)
    data_offset = 0
    tensor_infos: list[bytes] = []
    for record in records:
        record["offset"] = data_offset
        tensor_infos.append(
            _encode_tensor_info(record["name"], record["shape"], record["tensor_type"], data_offset)
        )
        data_size = math.prod(record["shape"], start=1) * int(record["item_size"])
        data_offset = _align(data_offset + data_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite GGUF artifact: {target}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(GGUF_MAGIC)
            handle.write(struct.pack("<IQQ", GGUF_VERSION, len(records), len(kv)))
            handle.write(kv_bytes)
            for info in tensor_infos:
                handle.write(info)
            handle.write(b"\0" * (data_start - handle.tell()))
            for record in records:
                payload = _tensor_payload(record)
                handle.write(payload)
                handle.write(b"\0" * (_align(handle.tell()) - handle.tell()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_tiny_gguf(path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write the legacy zero-tensor structural smoke artifact.

    This artifact intentionally carries a ``STRUCTURAL_SMOKE_ONLY`` receipt
    and is rejected by ``validate_gguf(..., require_receipt=True)``.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(GGUF_MAGIC)
        handle.write(struct.pack("<IQQ", GGUF_VERSION, 0, 0))
    digest = _digest(target)
    sidecar = target.with_suffix(target.suffix + ".json")
    atomic_write_json(
        sidecar,
        {
            "format": "GGUF",
            "version": GGUF_VERSION,
            "sha256": digest,
            "artifact_sha256": digest,
            "receipt_type": "dense2moe-structural-smoke-receipt-v1",
            "status": "STRUCTURAL_SMOKE_ONLY",
            "provenance_complete": False,
            "metadata": dict(metadata or {}),
        },
    )
    return target


def export_gguf(
    assembly_manifest: str | Path,
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export validated layer tensors into a durable, receipt-bearing GGUF.

    The assembly manifest and its sidecar receipt are treated as immutable
    inputs. Missing/changed provenance or tensor bytes fails closed before a
    product artifact is published.
    """

    manifest_path, manifest = _load_manifest(assembly_manifest)
    receipt_check = validate_assembly_receipt(manifest_path)
    if not receipt_check.get("valid"):
        raise ValueError(f"assembly receipt validation failed: {receipt_check.get('errors', [])}")
    if manifest.get("format") != "dense2moe-manifest-v3" or not manifest.get("strict"):
        raise ValueError("GGUF export requires a strict assembly manifest")
    if not manifest.get("complete"):
        raise ValueError("GGUF export requires a complete assembly")
    receipt = receipt_check.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("assembly receipt is missing")  # noqa: TRY004
    assembly_commit = str(receipt.get("code_commit", ""))
    if not _GIT_COMMIT_PATTERN.fullmatch(assembly_commit):
        raise ValueError("assembly receipt is missing code provenance")
    records = _collect_tensor_records(manifest_path, manifest)
    if not records:
        raise ValueError("strict GGUF export requires at least one tensor")

    target = Path(destination)
    sidecar = target.with_suffix(target.suffix + ".json")
    assembly_receipt_path = manifest_path.with_name("assembly-receipt.json")
    export_metadata = dict(metadata or {})
    reserved = {
        "assembly_manifest_sha256": receipt.get("manifest_sha256"),
        "assembly_receipt_sha256": _digest(assembly_receipt_path),
        "assembly_fingerprint": receipt.get("assembly_fingerprint"),
        "assembly_code_commit": assembly_commit,
        "source_revision": receipt.get("source_revision"),
        "profile": receipt.get("profile"),
        "profile_hash": receipt.get("profile_hash"),
        "dataset_hash": receipt.get("dataset_hash"),
        "tensor_inventory_sha256": manifest.get("tensor_inventory_sha256"),
        "tensor_count": len(records),
    }
    kv = {
        "dense2moe.artifact_scope": manifest.get("artifact_scope", receipt.get("artifact_scope")),
        "dense2moe.assembly_manifest_sha256": reserved["assembly_manifest_sha256"],
        "dense2moe.assembly_fingerprint": reserved["assembly_fingerprint"],
        "dense2moe.assembly_code_commit": assembly_commit,
        "dense2moe.tensor_inventory_sha256": reserved["tensor_inventory_sha256"],
        "dense2moe.tensor_count": len(records),
    }
    _write_tensor_gguf(target, records, kv)
    artifact_digest = _digest(target)
    export_receipt = {
        "receipt_type": GGUF_EXPORT_RECEIPT_FORMAT,
        "format": "GGUF",
        "version": GGUF_VERSION,
        "status": "GGUF_EXPORT_COMPLETE",
        "artifact": target.name,
        "artifact_sha256": artifact_digest,
        "artifact_size": target.stat().st_size,
        "tensor_count": len(records),
        "kv_count": len(kv),
        "assembly_manifest": str(manifest_path),
        "assembly_manifest_sha256": reserved["assembly_manifest_sha256"],
        "assembly_receipt": str(assembly_receipt_path),
        "assembly_receipt_sha256": reserved["assembly_receipt_sha256"],
        "assembly_fingerprint": reserved["assembly_fingerprint"],
        "assembly_code_commit": assembly_commit,
        "export_code_commit": current_git_commit(),
        "source_revision": reserved["source_revision"],
        "profile": reserved["profile"],
        "profile_hash": reserved["profile_hash"],
        "dataset_hash": reserved["dataset_hash"],
        "tensor_inventory_sha256": reserved["tensor_inventory_sha256"],
        "metadata": export_metadata,
    }
    atomic_write_json(sidecar, export_receipt)
    validation = validate_gguf(target, require_receipt=True)
    return {"path": str(target), "receipt": str(sidecar), "validation": validation, "provenance": reserved}


def write_gguf(
    path: str | Path,
    *,
    assembly_manifest: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Path-returning convenience wrapper for :func:`export_gguf`."""

    export_gguf(assembly_manifest, path, metadata=metadata)
    return Path(path)


def validate_gguf(path: str | Path, *, require_receipt: bool = False) -> dict[str, Any]:
    """Validate GGUF framing and, optionally, its product provenance receipt."""

    target = Path(path)
    with target.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:4] != GGUF_MAGIC:
        raise ValueError("invalid GGUF header")
    version, tensor_count, kv_count = struct.unpack("<IQQ", header[4:24])
    if version not in {2, GGUF_VERSION}:
        raise ValueError(f"unsupported GGUF version: {version}")
    sidecar = target.with_suffix(target.suffix + ".json")
    receipt: dict[str, Any] | None = None
    if sidecar.exists():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid GGUF receipt: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("GGUF receipt must be a JSON object")
        receipt = dict(loaded)
        digest = _digest(target)
        for field_name in ("sha256", "artifact_sha256"):
            if field_name in receipt and receipt[field_name] != digest:
                raise ValueError(f"GGUF receipt {field_name} mismatch")
    if require_receipt:
        if receipt is None:
            raise ValueError("strict GGUF export requires a provenance receipt")
        required = {
            "receipt_type": GGUF_EXPORT_RECEIPT_FORMAT,
            "status": "GGUF_EXPORT_COMPLETE",
        }
        for field_name, expected in required.items():
            if receipt.get(field_name) != expected:
                raise ValueError(f"strict GGUF export requires a product provenance receipt ({field_name})")
        if tensor_count <= 0:
            raise ValueError("strict GGUF export cannot contain zero tensors")
        if receipt.get("tensor_count") != tensor_count:
            raise ValueError("GGUF tensor count does not match receipt")
        if not _GIT_COMMIT_PATTERN.fullmatch(str(receipt.get("assembly_code_commit", ""))):
            raise ValueError("GGUF receipt is missing assembly code provenance")
        if not _GIT_COMMIT_PATTERN.fullmatch(str(receipt.get("export_code_commit", ""))):
            raise ValueError("GGUF receipt is missing export code provenance")
        for field_name in ("assembly_manifest_sha256", "assembly_receipt_sha256", "tensor_inventory_sha256"):
            if not _SHA256_PATTERN.fullmatch(str(receipt.get(field_name, ""))):
                raise ValueError(f"GGUF receipt is missing {field_name}")
    return {
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "size": target.stat().st_size,
        "receipt_path": str(sidecar) if sidecar.exists() else None,
        "receipt": receipt,
    }


__all__ = ["export_gguf", "validate_gguf", "write_gguf", "write_tiny_gguf"]
