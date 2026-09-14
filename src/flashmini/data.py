"""Deterministic local dataset preparation.

Streams a corpus once, tokenizes once, packs into deterministic token sequences,
and stores memory-mapped/sharded local representation so reruns do not
re-download/tokenize. Records manifest hashes for reproducibility.

All model variants consume the exact same token sequence for matched comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


V3_DATA_FORMAT_VERSION = 3
V3_SPLIT_ALGORITHM = "sha256_document_hash_threshold_v1"
V3_DEDUPE_ALGORITHM = "sqlite_sha256_token_bytes_v1"


def _normalise_split_salt(seed: int, split_salt: str | bytes | None) -> bytes:
    """Return the stable byte salt used by the v3 document partitioner."""
    if split_salt is None:
        split_salt = f"flashmini-v3-seed-{seed}"
    if isinstance(split_salt, bytes):
        return split_salt
    if not isinstance(split_salt, str) or not split_salt:
        raise ValueError("split_salt must be a non-empty string or bytes")
    return split_salt.encode("utf-8")


def _document_digest(tokens: Sequence[int]) -> bytes:
    """Hash one complete tokenized document, including its EOS marker."""
    # A document is bounded by the tokenizer output.  Keeping this conversion
    # local means the streaming preparation path never accumulates a corpus.
    return hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).digest()


def _split_document(digest: bytes, split_salt: bytes, val_fraction: float) -> str:
    salted = hashlib.sha256(split_salt + b"\0" + digest).digest()
    threshold = int(val_fraction * (1 << 64))
    return "val" if int.from_bytes(salted[:8], "big") < threshold else "train"


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pack_raw_split(
    raw_path: Path,
    out_path: Path,
    *,
    num_tokens: int,
    seq_len: int,
    eos_id: int,
    block_tokens: int = 1 << 20,
) -> dict[str, Any]:
    """Pack an int32 raw stream to compact, numpy-readable input/label arrays.

    This is deliberately a second streaming pass over the staging file.  The
    final arrays use int32 on disk (the vocabulary and ignore index fit), while
    ``MemmapDataset.get_batch`` exposes int64 tensors to the existing model
    code.  At most ``block_tokens`` values are resident during this operation.
    """
    if num_tokens < 2:
        raise ValueError("each dataset split needs at least two tokens")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")

    num_sequences = (num_tokens - 1 + seq_len - 1) // seq_len
    input_mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.int32, shape=(num_sequences, seq_len)
    )
    raw = np.memmap(raw_path, mode="r", dtype=np.int32, shape=(num_tokens,))
    flat_input = input_mm.reshape(-1)

    rows_per_block = max(1, block_tokens // seq_len)
    for row_start in range(0, num_sequences, rows_per_block):
        row_end = min(num_sequences, row_start + rows_per_block)
        start = row_start * seq_len
        end = min(row_end * seq_len, num_tokens)
        count = end - start
        current = np.asarray(raw[start:end], dtype=np.int32)
        target = flat_input[start : start + count]
        target[:] = current
        if count < (row_end - row_start) * seq_len:
            flat_input[start + count : row_end * seq_len] = eos_id
        labels = np.full((row_end - row_start) * seq_len, -100, dtype=np.int32)
        if start < num_tokens - 1:
            label_count = min(end, num_tokens - 1) - start
            labels[:label_count] = raw[start + 1 : start + 1 + label_count]
        label_path = out_path.with_name(out_path.stem.replace("_input", "_labels") + ".npy")
        # Labels are created lazily on the first block so no full-corpus label
        # allocation is needed.
        if row_start == 0:
            labels_mm = np.lib.format.open_memmap(
                label_path, mode="w+", dtype=np.int32, shape=(num_sequences, seq_len)
            )
        labels_mm[row_start:row_end] = labels.reshape(row_end - row_start, seq_len)

    input_mm.flush()
    labels_mm.flush()
    del raw, input_mm, labels_mm
    _fsync_file(out_path)
    _fsync_file(label_path)
    return {
        "num_sequences": int(num_sequences),
        "tokens": int(num_sequences * seq_len),
        "raw_tokens": int(num_tokens),
        "scored_tokens": int(max(0, num_tokens - 1)),
        "input_dtype": "int32",
        "labels_dtype": "int32",
        "input_sha256": sha256_file(out_path),
        "labels_sha256": sha256_file(label_path),
    }


def prepare_streaming_documents(
    documents: Iterable[Sequence[int]],
    out_dir: Path,
    *,
    seq_len: int,
    eos_id: int,
    target_train_tokens: int | None = None,
    max_source_tokens: int | None = None,
    max_docs: int | None = None,
    seed: int = 0,
    split_salt: str | bytes | None = None,
    val_fraction: float = 0.02,
    provenance: dict[str, Any] | None = None,
    block_tokens: int = 1 << 20,
) -> dict[str, Any]:
    """Prepare a reproducible v3 dataset from a stream of tokenized documents.

    Documents are consumed once and never retained in memory.  Exact duplicate
    tokenized documents are recorded in a SQLite ``WITHOUT ROWID`` table before
    a salted SHA-256 partition assigns each unique document to train or val.
    Raw split streams are then packed in a bounded second pass.  Stopping occurs
    only at complete document boundaries, so no document can straddle splits.

    ``target_train_tokens`` is measured in scored tokens.  The final token of a
    split has no successor and is therefore ignored by the language-model loss.
    Set it to ``None`` only when the caller intentionally wants to consume the
    complete source stream (or uses ``max_source_tokens`` as the limit).
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"dataset output directory already contains artifacts: {out_dir}"
        )
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if target_train_tokens is not None and target_train_tokens <= 0:
        raise ValueError("target_train_tokens must be positive")
    if max_source_tokens is not None and max_source_tokens <= 0:
        raise ValueError("max_source_tokens must be positive")
    if max_docs is not None and max_docs <= 0:
        raise ValueError("max_docs must be positive when provided")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v3-staging-", dir=out_dir))
    db_path = out_dir / "document_dedupe.sqlite3"
    split_salt_bytes = _normalise_split_salt(seed, split_salt)
    raw_paths = {name: staging / f"{name}.tokens.i32" for name in ("train", "val")}
    raw_handles = {name: path.open("wb") for name, path in raw_paths.items()}
    counts = {name: {"raw_tokens": 0, "documents": 0} for name in raw_paths}
    source_doc_count = 0
    source_empty_doc_count = 0
    source_body_tokens = 0
    duplicate_docs = 0
    unique_doc_count = 0

    db = sqlite3.connect(db_path)
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE seen_docs (digest BLOB PRIMARY KEY, split TEXT NOT NULL, "
            "token_count INTEGER NOT NULL) WITHOUT ROWID"
        )
        db.commit()
        db.execute("BEGIN")
        for document in documents:
            if max_docs is not None and source_doc_count >= max_docs:
                break
            # Materialise one document only.  The caller contract says document
            # tokens exclude EOS; accepting an existing terminal EOS would make
            # boundaries ambiguous, so reject it loudly.
            body = [int(token) for token in document]
            source_doc_count += 1
            if max_source_tokens is not None and source_body_tokens + len(body) > max_source_tokens:
                break
            if not body:
                source_empty_doc_count += 1
                continue
            if eos_id in body:
                raise ValueError("v3 document streams must omit the EOS token")
            source_body_tokens += len(body)
            complete = body + [int(eos_id)]
            digest = _document_digest(complete)
            split = _split_document(digest, split_salt_bytes, val_fraction)
            inserted = db.execute(
                "INSERT OR IGNORE INTO seen_docs(digest, split, token_count) VALUES (?, ?, ?)",
                (digest, split, len(complete)),
            ).rowcount
            if not inserted:
                duplicate_docs += 1
                continue
            unique_doc_count += 1
            payload = np.asarray(complete, dtype=np.int32)
            payload.tofile(raw_handles[split])
            counts[split]["raw_tokens"] += len(complete)
            counts[split]["documents"] += 1
            if unique_doc_count % 10000 == 0:
                db.commit()
                db.execute("BEGIN")
            train_scored = max(0, counts["train"]["raw_tokens"] - 1)
            # A val document is useful for evaluation, but do not consume an
            # unbounded source tail just to find one under a tiny fraction.
            if (
                target_train_tokens is not None
                and train_scored >= target_train_tokens
                and counts["val"]["raw_tokens"] >= 2
            ):
                break
        db.commit()
    finally:
        for handle in raw_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        db.close()

    train_scored = max(0, counts["train"]["raw_tokens"] - 1)
    if target_train_tokens is not None and train_scored < target_train_tokens:
        raise ValueError(
            f"source ended before target_train_tokens: scored={train_scored}, "
            f"target={target_train_tokens}, source_docs={source_doc_count}"
        )
    if counts["val"]["raw_tokens"] < 2:
        raise ValueError("document hash split produced fewer than two validation tokens")

    split_metadata: dict[str, Any] = {}
    for name in ("train", "val"):
        input_path = out_dir / f"{name}_input.npy"
        metadata = _pack_raw_split(
            raw_paths[name],
            input_path,
            num_tokens=counts[name]["raw_tokens"],
            seq_len=seq_len,
            eos_id=eos_id,
            block_tokens=block_tokens,
        )
        metadata["documents"] = counts[name]["documents"]
        split_metadata[name] = metadata

    # The SQLite table is the durable disk-backed dedupe evidence and lets an
    # independent checker verify that a digest has one and only one split.
    _fsync_file(db_path)
    manifest: dict[str, Any] = {
        "format_version": V3_DATA_FORMAT_VERSION,
        "split_algorithm": V3_SPLIT_ALGORITHM,
        "split_method": V3_SPLIT_ALGORITHM,
        "dedupe_algorithm": V3_DEDUPE_ALGORITHM,
        "seq_len": int(seq_len),
        "eos_id": int(eos_id),
        "seed": int(seed),
        "split_salt": split_salt_bytes.decode("utf-8", errors="backslashreplace"),
        "val_fraction": float(val_fraction),
        "target_train_scored_tokens": target_train_tokens,
        "max_source_tokens": max_source_tokens,
        "source_doc_count": int(source_doc_count),
        "source_empty_doc_count": int(source_empty_doc_count),
        "source_body_tokens": int(source_body_tokens),
        "unique_doc_count": int(unique_doc_count),
        "duplicate_doc_count": int(duplicate_docs),
        "unique_tokens_including_eos": int(
            counts["train"]["raw_tokens"] + counts["val"]["raw_tokens"]
        ),
        "document_dedupe_db": {
            "path": db_path.name,
            "sha256": sha256_file(db_path),
            "bytes": db_path.stat().st_size,
        },
        "splits": split_metadata,
        "provenance": dict(provenance or {}),
    }
    # Keep immutable provenance easy for training/comparison guardrails to
    # consume without knowing the nested provenance schema.
    if provenance:
        manifest["dataset"] = provenance.get("dataset_id")
        manifest["dataset_revision"] = provenance.get("dataset_revision")
        manifest["tokenizer"] = provenance.get("tokenizer_id")
        manifest["tokenizer_revision"] = provenance.get("tokenizer_revision")
    manifest_path = out_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _fsync_file(manifest_path)
    # Staging contains only raw int32 streams and is safe to remove after the
    # final memmaps are hashed.  Its lifetime is bounded even for multi-billion
    # token corpora and it is never part of the training data contract.
    for path in raw_paths.values():
        path.unlink(missing_ok=True)
    staging.rmdir()
    return manifest


def verify_dataset_integrity(data_dir: Path) -> dict[str, Any]:
    """Verify v3 shard hashes, shapes, counts, and dedupe evidence.

    This read-only check is intended for the official-run preflight.  It reads
    shard bytes sequentially for hashing but never loads a full shard into
    memory.  v2 directories remain readable and receive the same shard checks;
    v3 additionally requires its durable dedupe database.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("dataset manifest must contain an object")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise TypeError("dataset manifest has no split metadata")
    checked: dict[str, Any] = {}
    for split in ("train", "val"):
        metadata = splits.get(split)
        if not isinstance(metadata, dict):
            raise TypeError(f"dataset manifest has no {split} metadata")
        input_path = data_dir / f"{split}_input.npy"
        labels_path = data_dir / f"{split}_labels.npy"
        if sha256_file(input_path) != metadata.get("input_sha256"):
            raise ValueError(f"{split} input shard hash mismatch")
        if sha256_file(labels_path) != metadata.get("labels_sha256"):
            raise ValueError(f"{split} labels shard hash mismatch")
        input_mm = np.load(input_path, mmap_mode="r")
        labels_mm = np.load(labels_path, mmap_mode="r")
        if input_mm.shape != labels_mm.shape or input_mm.ndim != 2:
            raise ValueError(f"{split} shard shape mismatch")
        if int(input_mm.shape[1]) != int(manifest.get("seq_len", input_mm.shape[1])):
            raise ValueError(f"{split} shard sequence length mismatch")
        if int(metadata.get("num_sequences", input_mm.shape[0])) != int(input_mm.shape[0]):
            raise ValueError(f"{split} manifest sequence count mismatch")
        if int(manifest.get("format_version", 1)) >= 3:
            for start in range(0, len(labels_mm), 4096):
                if not np.all(np.any(labels_mm[start:start + 4096] != -100, axis=1)):
                    raise ValueError(f"{split} contains an entirely unscored sequence")
        flat_labels = labels_mm.reshape(-1)
        counted = sum(int((flat_labels[start:start + (1 << 20)] != -100).sum())
                      for start in range(0, flat_labels.size, 1 << 20))
        if counted != int(metadata.get("scored_tokens", counted)):
            raise ValueError(f"{split} manifest scored-token count mismatch")
        checked[split] = {
            "num_sequences": int(input_mm.shape[0]),
            "seq_len": int(input_mm.shape[1]),
            "scored_tokens": counted,
        }

    if int(manifest.get("format_version", 1)) >= V3_DATA_FORMAT_VERSION:
        dedupe = manifest.get("document_dedupe_db")
        if not isinstance(dedupe, dict):
            raise ValueError("v3 manifest has no dedupe database metadata")
        db_path = data_dir / str(dedupe.get("path", "document_dedupe.sqlite3"))
        if db_path.resolve().parent != data_dir.resolve():
            raise ValueError("dedupe database must be directly inside the dataset directory")
        if sha256_file(db_path) != dedupe.get("sha256"):
            raise ValueError("v3 dedupe database hash mismatch")
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            total_docs = int(db.execute("SELECT COUNT(*) FROM seen_docs").fetchone()[0])
            if total_docs != int(manifest.get("unique_doc_count", total_docs)):
                raise ValueError("v3 unique document count mismatch")
            for split in ("train", "val"):
                docs = int(
                    db.execute("SELECT COUNT(*) FROM seen_docs WHERE split = ?", (split,)).fetchone()[0]
                )
                if docs != int(splits[split].get("documents", docs)):
                    raise ValueError(f"v3 {split} document count mismatch")
        checked["dedupe"] = {"path": db_path.name, "unique_documents": total_docs}
    return {"valid": True, "format_version": int(manifest.get("format_version", 1)), "splits": checked}


def pack_tokens(
    token_ids: Iterable[int],
    seq_len: int,
    eos_id: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a stream of token IDs into fixed-length sequences with document boundaries.

    Documents are separated by EOS. Sequences are packed contiguously; a document
    that crosses a sequence boundary is continued in the next sequence (no padding
    waste). Returns (input_ids, labels) arrays of shape (N, seq_len).
    """
    tokens = list(token_ids)
    n = len(tokens)
    if n < 2 or seq_len <= 0:
        raise ValueError("Packing needs at least two tokens and a positive sequence length")
    n_seq = (n + seq_len - 1) // seq_len
    # Pad to multiple of seq_len with EOS
    pad = n_seq * seq_len - n
    tokens = tokens + [eos_id] * pad
    arr = np.array(tokens, dtype=np.int64)
    input_ids = arr.reshape(n_seq, seq_len)
    # Preserve the real next token across packed sequence boundaries. Padding
    # and the last token with no known successor must not improve measured loss.
    flat_labels = np.full(n_seq * seq_len, -100, dtype=np.int64)
    flat_labels[:max(0, n - 1)] = tokens[1:n]
    labels = flat_labels.reshape(n_seq, seq_len)
    scored = (labels != -100).any(axis=1)
    input_ids, labels = input_ids[scored], labels[scored]
    return input_ids, labels


def prepare_dataset(
    token_ids: Iterable[int],
    out_dir: Path,
    seq_len: int,
    eos_id: int,
    split: str = "train",
    seed: int = 0,
    val_fraction: float = 0.01,
    document_split: bool = False,
) -> dict:
    """Tokenize-once, pack, shard to memmap, and write a manifest.

    Splits the token stream deterministically into train/val with no leakage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if (out_dir / "data_manifest.json").exists():
        raise FileExistsError("Frozen dataset already exists; use a new output directory")
    tokens = list(token_ids)
    rng = np.random.default_rng(seed)
    duplicate_docs = 0
    if document_split:
        # Remove exact duplicate complete documents before splitting. A truncated
        # final document is excluded so it cannot straddle train and validation.
        unique_tokens, boundaries, seen, start = [], [], set(), 0
        for end, token in enumerate(tokens):
            if token != eos_id:
                continue
            doc = tokens[start:end + 1]
            start = end + 1
            digest = hashlib.sha256(np.asarray(doc, dtype=np.int64).tobytes()).digest()
            if digest in seen:
                duplicate_docs += 1
                continue
            seen.add(digest)
            unique_tokens.extend(doc)
            boundaries.append(len(unique_tokens))
        tokens = unique_tokens
        if len(boundaries) < 2:
            raise ValueError("Document split requires at least two distinct EOS-terminated documents")
    n = len(tokens)
    n_val = int(n * val_fraction)
    split_at = n - n_val
    if document_split:
        split_at = min(boundaries[:-1], key=lambda boundary: abs(boundary - split_at))
    val_tokens = tokens[split_at:]
    train_tokens = tokens[:split_at]
    if len(train_tokens) < 2 or len(val_tokens) < 2:
        raise ValueError("Each split needs at least two tokens")

    splits = {"train": train_tokens, "val": val_tokens}
    manifest: dict = {
        "split": split,
        "seq_len": seq_len,
        "eos_id": eos_id,
        "seed": seed,
        "val_fraction": val_fraction,
        "total_tokens": n,
        "format_version": 2,
        "split_method": "deduplicated_document_tail" if document_split else "token_tail",
        "exact_duplicate_documents_removed": duplicate_docs,
        "labels_ignore_index": -100,
        "splits": {},
        "shards": {},
    }

    for name, toks in splits.items():
        input_ids, labels = pack_tokens(toks, seq_len, eos_id, rng)
        in_path = out_dir / f"{name}_input.npy"
        lab_path = out_dir / f"{name}_labels.npy"
        np.save(in_path, input_ids)
        np.save(lab_path, labels)
        manifest["splits"][name] = {
            "num_sequences": int(input_ids.shape[0]),
            "tokens": int(input_ids.shape[0] * seq_len),
            "scored_tokens": int((labels != -100).sum()),
            "input_sha256": sha256_file(in_path),
            "labels_sha256": sha256_file(lab_path),
        }

    manifest_path = out_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


class MemmapDataset:
    """Memory-mapped dataset over packed token shards."""

    def __init__(self, data_dir: Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.input = np.load(self.data_dir / f"{split}_input.npy", mmap_mode="r")
        self.labels = np.load(self.data_dir / f"{split}_labels.npy", mmap_mode="r")
        if self.input.ndim != 2 or self.labels.ndim != 2:
            raise ValueError("dataset shards must be rank-2 arrays")
        if self.input.shape != self.labels.shape:
            raise ValueError(
                f"input/label shape mismatch for {split}: "
                f"input={self.input.shape}, labels={self.labels.shape}"
            )
        self.num_sequences = self.input.shape[0]
        self.seq_len = self.input.shape[1]
        self.manifest: dict[str, Any] | None = None
        manifest_path = self.data_dir / "data_manifest.json"
        if manifest_path.is_file():
            try:
                raw_manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dataset manifest: {manifest_path}") from exc
            if not isinstance(raw_manifest, dict):
                raise ValueError("dataset manifest must contain an object")
            self.manifest = raw_manifest
            declared_seq_len = raw_manifest.get("seq_len")
            if declared_seq_len is not None and int(declared_seq_len) != self.seq_len:
                raise ValueError(
                    f"dataset manifest sequence length {declared_seq_len} does not match "
                    f"shard {self.seq_len}"
                )
            split_meta = raw_manifest.get("splits", {}).get(split, {})
            if (
                isinstance(split_meta, dict)
                and split_meta.get("num_sequences") is not None
                and int(split_meta["num_sequences"]) != self.num_sequences
            ):
                raise ValueError(
                    f"dataset manifest sequence count for {split} does not match shard"
                )
            self.format_version = int(raw_manifest.get("format_version", 1))
            self.scored_tokens = int(
                split_meta.get("scored_tokens", max(0, self.num_sequences * self.seq_len - 1))
            )
            self.raw_tokens = int(
                split_meta.get("raw_tokens", self.num_sequences * self.seq_len)
            )
        else:
            self.format_version = 1
            self.raw_tokens = self.num_sequences * self.seq_len
            self.scored_tokens = int((self.labels != -100).sum())

    def __len__(self) -> int:
        return self.num_sequences

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # v3 stores token arrays as int32 to halve disk use.  The model and
        # cross-entropy path historically receive int64, so cast only the
        # requested batch and never materialise the corpus.
        return (
            np.asarray(self.input[indices], dtype=np.int64),
            np.asarray(self.labels[indices], dtype=np.int64),
        )

    def iter_epoch_batches(
        self,
        *,
        seed: int,
        epoch: int = 0,
        batch_size: int,
        drop_last: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield each sequence once in a deterministic per-epoch permutation.

        The permutation is the only O(number-of-sequences) allocation.  It is
        bounded by the sequence index width (8 bytes each), rather than token
        storage, and avoids replacement sampling that silently reuses a small
        corpus.  ``drop_last`` is explicit because dropping examples changes
        token accounting.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        permutation = np.arange(self.num_sequences, dtype=np.int64)
        np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)])).shuffle(permutation)
        limit = self.num_sequences - (self.num_sequences % batch_size if drop_last else 0)
        for start in range(0, limit, batch_size):
            yield permutation[start : min(start + batch_size, limit)]

    def verify_integrity(self) -> dict[str, Any]:
        """Run the full read-only manifest/shard integrity check."""
        return verify_dataset_integrity(self.data_dir)
