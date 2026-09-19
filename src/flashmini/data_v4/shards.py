"""Parquet+ZSTD canonical shard writer + manifest (v4)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

SHARD_FORMAT = "parquet_zstd_v1"
SHARD_VERSION = "flashmini-v4-shard-v1"


def write_shard(documents: list[dict], out_path: Path, *,
                shard_id: str, recipe_name: str = "",
                recipe_hash: str = "") -> dict:
    """Write canonical documents to Parquet/ZSTD and return shard manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([
        {
            "document_id": d["document_id"],
            "text": d["text"],
            "source_id": d.get("source_id", ""),
            "domain": d.get("domain", ""),
            "language": d.get("language", "en"),
            "license": d.get("license", ""),
            "redistribution_class": d.get("redistribution_class", "review_required"),
            "split": d.get("split", "train"),
            "content_hash": d.get("content_hash", ""),
            "char_count": len(d.get("text", "")),
            "token_estimate": max(1, len(d.get("text", "")) // 4),
        }
        for d in documents
    ])
    pq.write_table(table, out_path, compression="zstd")
    raw = out_path.read_bytes()
    chars = sum(len(d.get("text", "")) for d in documents)
    return {
        "shard_id": shard_id,
        "path": out_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "document_count": len(documents),
        "character_count": chars,
        "estimated_token_count": max(1, chars // 4),
        "source_distribution": dict(Counter(d.get("source_id", "") for d in documents)),
        "domain_distribution": dict(Counter(d.get("domain", "") for d in documents)),
        "language_distribution": dict(Counter(d.get("language", "en") for d in documents)),
        "license_distribution": dict(Counter(d.get("license", "") for d in documents)),
        "build_version": SHARD_VERSION,
        "format": SHARD_FORMAT,
        "recipe_name": recipe_name,
        "recipe_hash": recipe_hash,
        "created_at": int(time.time()),
    }


def corpus_fingerprint(*, registry_hash: str, recipe_hash: str,
                       filter_version: str, dedupe_version: str,
                       split_salt: str, shard_hashes: list[str],
                       tokenizer_identity: str = "") -> str:
    payload = json.dumps({
        "registry_hash": registry_hash,
        "recipe_hash": recipe_hash,
        "filter_version": filter_version,
        "dedupe_version": dedupe_version,
        "split_salt": split_salt,
        "shard_hashes": sorted(shard_hashes),
        "tokenizer_identity": tokenizer_identity,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
