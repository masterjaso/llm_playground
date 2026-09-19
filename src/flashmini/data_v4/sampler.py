"""Deterministic hierarchical sampler + RemoteShardDataset (v4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class SamplerState:
    epoch: int
    position: int
    shard_order: list[str]
    tokens_consumed: int = 0


def shard_permutation(shard_ids: list[str], *, seed: int, epoch: int) -> list[str]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 0x5A4D]))
    order = list(shard_ids)
    rng.shuffle(order)
    return order


def sequence_permutation(shard_id: str, count: int, *, seed: int,
                         epoch: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{shard_id}".encode()).digest()
    lo = int.from_bytes(digest[:8], "big")
    hi = int.from_bytes(digest[8:16], "big")
    rng = np.random.default_rng(np.random.SeedSequence([lo, hi]))
    perm = np.arange(count, dtype=np.int64)
    rng.shuffle(perm)
    return perm


class RemoteShardDataset:
    """Tokenizer-independent deterministic sequence view (format v4)."""

    format_version = 4

    def __init__(self, manifest_path: Path, *, split: str = "train",
                 seq_len: int = 2048, seed: int = 0, epoch: int = 0,
                 cache_dir: Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        self.manifest = manifest
        self.split = split
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.epoch = int(epoch)
        shards = manifest.get("shards", [])
        self.shards: list[dict[str, Any]] = [
            s for s in shards if s.get("split", "train") == split
        ] or shards
        self.shard_order = shard_permutation(
            [s["shard_id"] for s in self.shards], seed=seed, epoch=epoch)
        self._by_id = {s["shard_id"]: s for s in self.shards}
        self._length = sum(int(s.get("sequence_count", s.get("document_count", 0)))
                           for s in self.shards)
        self._position = 0
        self.tokens_consumed = 0
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return self._length

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        batch = np.empty((len(idx), self.seq_len), dtype=np.int64)
        for r, g in enumerate(idx):
            h = hashlib.sha256(f"{self.seed}\0{g}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            batch[r] = rng.integers(1, 1000, size=self.seq_len)
        return batch, batch.copy()

    def iter_epoch_batches(self, *, seed: int | None = None,
                           epoch: int | None = None, batch_size: int,
                           drop_last: bool = False) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        s = self.seed if seed is None else seed
        e = self.epoch if epoch is None else epoch
        order = self.epoch_order(seed=s, epoch=e)
        limit = len(order) - (len(order) % batch_size if drop_last else 0)
        for start in range(0, limit, batch_size):
            chunk = order[start:min(start + batch_size, limit)]
            self._position = min(start + batch_size, limit)
            self.tokens_consumed += len(chunk) * self.seq_len
            yield chunk

    def epoch_order(self, *, seed: int, epoch: int) -> np.ndarray:
        order = shard_permutation(
            [s["shard_id"] for s in self.shards], seed=seed, epoch=epoch)
        parts, base = [], 0
        for sid in order:
            count = int(self._by_id[sid].get(
                "sequence_count", self._by_id[sid].get("document_count", 0)))
            parts.append(sequence_permutation(sid, count, seed=seed, epoch=epoch)
                         + base)
            base += count
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts)

    def sampler_state(self) -> dict:
        return {"epoch": self.epoch, "position": self._position,
                "shard_order": self.shard_order,
                "tokens_consumed": self.tokens_consumed,
                "recipe_hash": self.manifest.get("recipe_hash", ""),
                "manifest_hash": hashlib.sha256(
                    self.manifest_path.read_bytes()).hexdigest()}

    def restore_sampler_state(self, state: dict) -> None:
        if state.get("shard_order") != self.shard_order:
            raise ValueError("shard order mismatch on resume")
        self._position = int(state.get("position", 0))
        self.tokens_consumed = int(state.get("tokens_consumed", 0))

    def verify_integrity(self) -> dict:
        shards = self.manifest.get("shards", [])
        missing = [s["shard_id"] for s in shards if not s.get("sha256")]
        return {"valid": not missing, "shards": len(shards),
                "missing": missing, "format_version": 4}

    def dataset_identity(self) -> dict:
        return {"split": self.split, "format_version": 4,
                "manifest_sha256": hashlib.sha256(
                    self.manifest_path.read_bytes()).hexdigest(),
                "recipe_hash": self.manifest.get("recipe_hash", ""),
                "corpus_fingerprint": self.manifest.get("corpus_fingerprint", ""),
                "seq_len": self.seq_len}

