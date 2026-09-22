"""Deterministic train/validation assignment by stable hash (v4)."""

from __future__ import annotations

import hashlib

SPLIT_SALT_DEFAULT = "flashmini-v4-split-v1"
VAL_FRACTION_DEFAULT = 0.005
SPLIT_ALGORITHM = "sha256_document_hash_threshold_v1"


def _salt_bytes(salt: str | bytes) -> bytes:
    if isinstance(salt, bytes):
        if not salt:
            raise ValueError("split salt must be non-empty")
        return salt
    if not salt:
        raise ValueError("split salt must be non-empty")
    return salt.encode("utf-8")


def assign_split(document_id_hex: str, *, salt: str | bytes = SPLIT_SALT_DEFAULT,
                 val_fraction: float = VAL_FRACTION_DEFAULT) -> str:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    try:
        digest = bytes.fromhex(document_id_hex)
    except ValueError as exc:
        raise ValueError("document_id must be hex") from exc
    salted = hashlib.sha256(_salt_bytes(salt) + b"\0" + digest).digest()
    threshold = int(val_fraction * (1 << 64))
    return "val" if int.from_bytes(salted[:8], "big") < threshold else "train"
