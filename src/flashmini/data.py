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
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        self.num_sequences = self.input.shape[0]
        self.seq_len = self.input.shape[1]

    def __len__(self) -> int:
        return self.num_sequences

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.input[indices], self.labels[indices]
