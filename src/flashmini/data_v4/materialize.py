"""Tokenizer-specific token-store materialization for a frozen release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import packing
from . import tokenizer as tokenizer_mod


def materialize_canonical_shard(canonical_path: str | Path, output_prefix: str | Path,
                                *, tokenizer, tokenizer_spec: tokenizer_mod.TokenizerSpec,
                                sequence_length: int | None = None,
                                packing_policy: str = "document_mix") -> dict:
    import pyarrow.parquet as pq
    canonical_path = Path(canonical_path)
    table = pq.read_table(canonical_path,
                          columns=["text", "document_id", "domain", "source_id", "split"])
    texts = table.column("text").to_pylist()
    ids = table.column("document_id").to_pylist()
    tokens, offsets, lengths = packing.tokenize_documents(texts, ids, tokenizer=tokenizer)
    prefix = Path(output_prefix)
    token_meta = packing.write_token_store(
        tokens, offsets, prefix,
        tokenizer_id=tokenizer_spec.tokenizer_id,
        tokenizer_revision=tokenizer_spec.revision,
        tokenizer_fingerprint=tokenizer_spec.fingerprint,
        document_ids=ids)
    packed_meta = None
    if sequence_length is not None:
        packed, boundaries = packing.pack_token_stream(
            tokens, offsets, seq_len=int(sequence_length),
            eos_token_id=int(tokenizer_spec.eos_token_id),
            pad_token_id=(int(getattr(tokenizer, "pad_token_id", None))
                          if getattr(tokenizer, "pad_token_id", None) is not None
                          else int(tokenizer_spec.eos_token_id)),
            policy=packing_policy)
        packed_path = prefix.with_suffix(f".s{int(sequence_length)}.npy")
        packed_tmp = packed_path.with_name(packed_path.name + ".part")
        with open(packed_tmp, "wb") as handle:
            import numpy as np
            np.save(handle, packed, allow_pickle=False)
        packed_tmp.replace(packed_path)
        packed_meta = {
            "path": str(packed_path), "sequence_count": int(packed.shape[0]),
            "sequence_length": int(sequence_length), "boundaries": boundaries,
            "sha256": hashlib.sha256(packed_path.read_bytes()).hexdigest(),
        }
    domains = table.column("domain").to_pylist()
    sources = table.column("source_id").to_pylist()
    splits = table.column("split").to_pylist()
    exact_by_domain: dict[str, int] = {}
    exact_by_source: dict[str, int] = {}
    exact_by_split: dict[str, int] = {}
    for length, domain, source, split in zip(lengths, domains, sources, splits):
        exact_by_domain[domain] = exact_by_domain.get(domain, 0) + int(length)
        exact_by_source[source] = exact_by_source.get(source, 0) + int(length)
        exact_by_split[split] = exact_by_split.get(split, 0) + int(length)
    return {
        "canonical_path": str(canonical_path),
        "token_store": token_meta,
        "token_count": int(tokens.size),
        "exact_tokens_by_domain": exact_by_domain,
        "exact_tokens_by_source": exact_by_source,
        "exact_tokens_by_split": exact_by_split,
        "packed": packed_meta,
        "tokenizer": tokenizer_spec.as_dict(),
        "packing_policy": packing_policy,
    }


def materialize_manifest(manifest_path: str | Path, output_dir: str | Path, *,
                         tokenizer, tokenizer_spec: tokenizer_mod.TokenizerSpec,
                         local_base: str | Path | None = None,
                         sequence_length: int | None = None,
                         packing_policy: str = "document_mix") -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    output_dir = Path(output_dir)
    rows = []
    base = Path(local_base) if local_base else manifest_path.parent / "shards"
    for shard in manifest.get("shards", []):
        source = base / shard["path"]
        if not source.exists():
            rows.append({"shard_id": shard.get("shard_id"), "status": "unavailable"})
            continue
        prefix = output_dir / Path(shard["path"]).stem
        result = materialize_canonical_shard(
            source, prefix, tokenizer=tokenizer, tokenizer_spec=tokenizer_spec,
            sequence_length=sequence_length, packing_policy=packing_policy)
        rows.append({"shard_id": shard.get("shard_id"), "status": "materialized", **result})
    return {
        "format": packing.TOKEN_FORMAT_VERSION,
        "tokenizer": tokenizer_spec.as_dict(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "shards": rows,
        "materialized": all(row["status"] == "materialized" for row in rows),
    }
