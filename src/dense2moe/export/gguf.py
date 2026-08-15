"""Minimal GGUF writer used for converter smoke tests and evidence manifests."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"


def write_tiny_gguf(path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # GGUF v3 header with zero tensors and zero key/value pairs is valid for a
    # structural converter smoke test. Real conversion adds typed KV entries
    # and tensor data through the optional llama.cpp integration.
    with target.open("wb") as handle:
        handle.write(GGUF_MAGIC)
        handle.write(struct.pack("<IQQ", 3, 0, 0))
    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar.write_text(json.dumps({"format": "GGUF", "version": 3, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "metadata": dict(metadata or {})}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def validate_gguf(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:4] != GGUF_MAGIC:
        raise ValueError("invalid GGUF header")
    version, tensor_count, kv_count = struct.unpack("<IQQ", header[4:24])
    return {"magic": "GGUF", "version": version, "tensor_count": tensor_count, "kv_count": kv_count, "size": target.stat().st_size}

