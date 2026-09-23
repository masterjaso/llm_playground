"""Parquet+ZSTD canonical shard writer + manifest (v4)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

SHARD_FORMAT = "parquet_zstd_v1"
SHARD_VERSION = "flashmini-v4-shard-v1"


def write_shard(documents: list[dict], out_path: Path, *,
                shard_id: str, recipe_name: str = "",
                recipe_hash: str = "", exact_token_counts: list[int] | None = None,
                tokenizer: dict | None = None, remote_path: str = "",
                hf_revision: str = "") -> dict:
    """Write canonical documents to Parquet/ZSTD and return shard manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exact = list(exact_token_counts or [])
    if exact and len(exact) != len(documents):
        raise ValueError("exact_token_counts must match document count")
    table = pa.Table.from_pylist([
        {
            "document_id": d["document_id"],
            "text": d["text"],
            "source_id": d.get("source_id", ""),
            "source_revision": d.get("source_revision", ""),
            "record_id": d.get("record_id", ""),
            "source_cursor": json.dumps(d.get("source_cursor", {}), sort_keys=True),
            "domain": d.get("domain", ""),
            "language": d.get("language", "en"),
            "license": d.get("license", ""),
            "redistribution_class": d.get("redistribution_class", "review_required"),
            "split": d.get("split", "train"),
            "content_hash": d.get("content_hash", ""),
            "char_count": len(d.get("text", "")),
            "token_estimate": max(1, len(d.get("text", "")) // 4),
            "exact_token_count": (int(exact[i]) if exact else None),
        }
        for i, d in enumerate(documents)
    ])
    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".part",
                                    dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    raw = out_path.read_bytes()
    chars = sum(len(d.get("text", "")) for d in documents)
    train_docs = sum(d.get("split", "train") == "train" for d in documents)
    val_docs = sum(d.get("split", "train") == "val" for d in documents)
    exact_total = sum(exact) if exact else None
    exact_by_domain: dict[str, int] = {}
    exact_by_source: dict[str, int] = {}
    exact_by_split: dict[str, int] = {}
    if exact:
        for d, count in zip(documents, exact):
            exact_by_domain[d.get("domain", "")] = exact_by_domain.get(d.get("domain", ""), 0) + int(count)
            exact_by_source[d.get("source_id", "")] = exact_by_source.get(d.get("source_id", ""), 0) + int(count)
            split = d.get("split", "train")
            exact_by_split[split] = exact_by_split.get(split, 0) + int(count)
    return {
        "shard_id": shard_id,
        "path": out_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "document_count": len(documents),
        "character_count": chars,
        "estimated_token_count": max(1, chars // 4),
        "exact_token_count": exact_total,
        "exact_tokens_by_domain": exact_by_domain,
        "exact_tokens_by_source": exact_by_source,
        "exact_tokens_by_split": exact_by_split,
        "source_distribution": dict(Counter(d.get("source_id", "") for d in documents)),
        "domain_distribution": dict(Counter(d.get("domain", "") for d in documents)),
        "split_distribution": dict(Counter(d.get("split", "train") for d in documents)),
        "language_distribution": dict(Counter(d.get("language", "en") for d in documents)),
        "license_distribution": dict(Counter(d.get("license", "") for d in documents)),
        "license_redistribution_summary": dict(Counter(
            d.get("redistribution_class", "review_required") for d in documents)),
        "train_document_count": train_docs,
        "validation_document_count": val_docs,
        "build_version": SHARD_VERSION,
        "format": SHARD_FORMAT,
        "recipe_name": recipe_name,
        "recipe_hash": recipe_hash,
        "created_at": int(time.time()),
        "remote_path": remote_path,
        "hf_revision": hf_revision,
        "tokenizer": tokenizer or {},
    }


def should_rollover(documents: list[dict], *, target_bytes: int = 512 * 1024 * 1024,
                    target_tokens: int | None = None, max_documents: int | None = None,
                    exact_token_counts: list[int] | None = None) -> bool:
    """Conservative pre-write rollover check for bounded physical shards."""
    if not documents:
        return False
    if max_documents is not None and len(documents) >= int(max_documents):
        return True
    # Parquet+ZSTD compression varies by source.  A 1.25x character safety
    # bound prevents pathological HTML/code rows from creating multi-GiB shards;
    # the post-write manifest remains authoritative.
    projected = sum(len(str(d.get("text", ""))) for d in documents) * 1.25
    if projected >= int(target_bytes):
        return True
    return (target_tokens is not None and exact_token_counts is not None
            and sum(int(x) for x in exact_token_counts) >= int(target_tokens))


def split_documents(documents: list[dict], *, target_bytes: int = 512 * 1024 * 1024,
                    target_tokens: int | None = None, max_documents: int | None = None,
                    exact_token_counts: list[int] | None = None) -> list[list[dict]]:
    """Split documents into bounded shard candidates without tiny fixed counts."""
    if exact_token_counts is not None and len(exact_token_counts) != len(documents):
        raise ValueError("exact_token_counts must match documents")
    groups: list[list[dict]] = []
    group: list[dict] = []
    group_tokens: list[int] = []
    for i, doc in enumerate(documents):
        group.append(doc)
        if exact_token_counts is not None:
            group_tokens.append(int(exact_token_counts[i]))
        if should_rollover(group, target_bytes=target_bytes, target_tokens=target_tokens,
                           max_documents=max_documents, exact_token_counts=group_tokens):
            groups.append(group)
            group, group_tokens = [], []
    if group:
        groups.append(group)
    return groups


def corpus_fingerprint(*, registry_hash: str, recipe_hash: str,
                       filter_version: str, dedupe_version: str,
                       split_salt: str, shard_hashes: list[str],
                       tokenizer_identity: str = "", release_id: str = "",
                       token_format_version: str = "",
                       benchmark_exclusion_version: str = "",
                       validation_tokens: int = 0,
                       validation_salt: str = "") -> str:
    payload = json.dumps({
        "registry_hash": registry_hash,
        "recipe_hash": recipe_hash,
        "filter_version": filter_version,
        "dedupe_version": dedupe_version,
        "split_salt": split_salt,
        "shard_hashes": sorted(shard_hashes),
        "tokenizer_identity": tokenizer_identity,
        "release_id": release_id,
        "token_format_version": token_format_version,
        "benchmark_exclusion_version": benchmark_exclusion_version,
        "validation_tokens": int(validation_tokens),
        "validation_salt": validation_salt,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
