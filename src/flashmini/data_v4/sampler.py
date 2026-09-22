"""Deterministic hierarchical sampler + RemoteShardDataset (v4).

Sampling is hierarchical so a batch never sprays reads across every remote
shard: ``epoch/seed -> shard permutation -> document permutation per shard``.
Global sequence indices are assigned in *manifest order* (stable, independent
of the epoch permutation), so resume is exact and independent of iteration
order.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import cache as cache_mod
from . import hf_store, packing


@dataclass
class SamplerState:
    epoch: int
    position: int
    shard_order: list[str]
    tokens_consumed: int = 0
    tokenizer_identity: str = ""
    packing_version: str = packing.PACKING_VERSION
    manifest_hash: str = ""
    recipe_hash: str = ""


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
    """Reads canonical v4 shards from Hugging Face through a bounded cache.

    Each training sequence is one canonical document packed to ``seq_len``
    (see :mod:`flashmini.data_v4.packing`).  Shards are downloaded on demand,
    sha256-verified against the frozen manifest, tokenized once into a packed
    ``.npy`` cache entry, and evicted once the cache bound is reached.
    """

    format_version = 4

    def __init__(self, manifest_path: Path, *, split: str = "train",
                 seq_len: int = 2048, seed: int = 0, epoch: int = 0,
                 cache_dir: Path | None = None, cache_gb: float | None = None,
                 hf_repo: str | None = None, revision: str | None = None,
                 tokenizer_id: str = "gpt2",
                 tokenizer_revision: str | None = None,
                 local_base: Path | None = None,
                 min_avail_gb: float = 10.0,
                 max_open_shards: int = 4,
                 auto_evict: bool = True,
                 packing_policy: str = "legacy_document",
                 view_id: str = "",
                 stage: str = "",
                 world_size: int = 1,
                 rank: int = 0,
                 production: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        raw = self.manifest_path.read_bytes()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        self.manifest = manifest
        self.split = split
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.hf_repo = hf_repo or manifest.get("hf_repo", "")
        self.revision = revision or manifest.get("hf_revision", "") or None
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = tokenizer_revision
        if production:
            from .tokenizer import _IMMUTABLE_REVISION
            if tokenizer_revision is None or not _IMMUTABLE_REVISION.fullmatch(tokenizer_revision):
                raise ValueError("production dataset requires an immutable tokenizer revision")
        declared_tok = (manifest.get("tokenizer") or {}).get("identity")
        if production and declared_tok and declared_tok != packing.tokenizer_identity(
                tokenizer_id, tokenizer_revision):
            raise ValueError("manifest tokenizer identity does not match loader")
        self.tokenizer_identity = packing.tokenizer_identity(
            tokenizer_id, tokenizer_revision)
        self.local_base = Path(local_base) if local_base else None
        self.cache_root = cache_mod.cache_root(str(cache_dir) if cache_dir else None)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = cache_mod.cache_max_bytes(
            int(cache_gb * 1024 ** 3) if cache_gb else None)
        self.min_avail_bytes = int(min_avail_gb * 1024 ** 3)
        self.auto_evict = auto_evict
        if packing_policy not in {"legacy_document", "document_mix", "coherent"}:
            raise ValueError("unknown packing_policy")
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError("rank must be within world_size")
        self.packing_policy = packing_policy
        self.view_id = str(view_id or manifest.get("view_id", ""))
        self.stage = str(stage or manifest.get("stage", ""))
        self.world_size = int(world_size)
        self.rank = int(rank)

        all_shards = manifest.get("shards", [])
        # Document-level split membership (``split_distribution``) is
        # authoritative: shards normally hold both train and val documents, so
        # split isolation must never be implemented by dropping whole shards.
        document_level_splits = any(
            isinstance(s.get("split_distribution"), dict) and s["split_distribution"]
            for s in all_shards)
        self.document_level_splits = document_level_splits
        if document_level_splits:
            self.shards: list[dict[str, Any]] = list(all_shards)
        else:
            self.shards = [s for s in all_shards
                           if s.get("split", "train") == split] or list(all_shards)
        self._by_id = {s["shard_id"]: s for s in self.shards}
        self.shard_order = shard_permutation(
            [s["shard_id"] for s in self.shards], seed=seed, epoch=epoch)

        self._seq_counts: dict[str, int] = {}
        self._offsets: dict[str, int] = {}
        self.exact_counts = True
        total = 0
        for s in self.shards:
            sid = s["shard_id"]
            dist = s.get("split_distribution")
            if isinstance(dist, dict) and dist:
                count = int(dist.get(split, 0))
            else:
                # Legacy manifest (pre split_distribution): document-level split
                # membership is only knowable after unpacking, so the declared
                # sequence_count is used as the sampling budget instead.
                count = int(s.get("sequence_count", s.get("document_count", 0)))
                self.exact_counts = False
            if self.packing_policy != "legacy_document":
                count = int(s.get("tokenized_sequence_count", s.get("sequence_count", count)))
            self._seq_counts[sid] = count
            self._offsets[sid] = total
            total += count
        self._length = total
        self._position = 0
        self.tokens_consumed = 0
        self._tokenizer = None
        self._packed_cache: dict[str, np.ndarray] = {}
        self._loaded_order: list[str] = []
        self.max_open_shards = max(1, int(max_open_shards))
        self._local_verified: set[str] = set()
        self.downloads = 0
        self.evictions = 0
        self.notes: list[str] = []
        self.metrics = {
            "cache_hits": 0, "downloads": 0, "decode_tokens": 0,
            "data_wait_seconds": 0.0, "prefetch_queue_depth": 0,
            "shard_transition_seconds": 0.0,
        }
        self.telemetry = cache_mod.CacheTelemetry()
        if not self.exact_counts:
            self.notes.append(
                "legacy manifest without split_distribution: per-split counts "
                "are the declared sequence_count budget")

    def __len__(self) -> int:
        if self.packing_policy != "legacy_document":
            unknown = [s for s in self.shards if "tokenized_sequence_count" not in s]
            for shard in unknown:
                self._packed_for(shard)
        return self._length

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from .tokenizer import load_tokenizer
            self._tokenizer = load_tokenizer(self.tokenizer_id,
                                             self.tokenizer_revision)
        return self._tokenizer

    # ------------------------------------------------------------ shard access
    def remote_path(self, shard: dict[str, Any]) -> str:
        return shard.get("remote_path") or f"shards/{shard['path']}"

    def local_shard_path(self, shard: dict[str, Any]) -> Path:
        return self.cache_root / "shards" / f"{shard['sha256'][:40]}.parquet"

    def _fetch_shard(self, shard: dict[str, Any]) -> Path:
        """Ensure a sha256-verified local copy of one canonical shard."""
        dest = self.local_shard_path(shard)
        sha = shard["sha256"]
        if dest.is_file():
            if sha in self._local_verified:
                self.telemetry.cache_hits += 1
                return dest
            try:
                cache_mod.verify_sha256(dest, sha)
                self._local_verified.add(sha)
                self.telemetry.cache_hits += 1
                return dest
            except ValueError:
                dest.unlink(missing_ok=True)  # truncated/corrupt: re-fetch
        if self.local_base is not None:
            import shutil
            src = Path(self.local_base) / shard["path"]
            if not src.is_file():
                raise FileNotFoundError(f"local_base missing {src}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
        else:
            if not self.hf_repo:
                raise ValueError("no hf_repo in manifest and none supplied")
            started = time.monotonic()
            hf_store.download_file(
                self.hf_repo, self.remote_path(shard), dest,
                revision=self.revision, cache_dir=self.cache_root / "hf",
                min_avail_bytes=self.min_avail_bytes)
            self.downloads += 1
            self.telemetry.cache_misses += 1
            self.telemetry.record_download(dest.stat().st_size, time.monotonic() - started)
        cache_mod.verify_sha256(dest, sha)
        self._local_verified.add(sha)
        cache_mod.mark_in_use(dest)
        return dest

    def _texts_of(self, shard: dict[str, Any]) -> tuple[list[str], list[str]]:
        import pyarrow.parquet as pq
        table = pq.read_table(self._fetch_shard(shard),
                              columns=["text", "document_id", "split"])
        texts = table.column("text").to_pylist()
        doc_ids = table.column("document_id").to_pylist()
        splits = table.column("split").to_pylist()
        keep = [i for i, s in enumerate(splits) if s == self.split]
        return [texts[i] for i in keep], [doc_ids[i] for i in keep]

    def _packed_for(self, shard: dict[str, Any]) -> np.ndarray:
        sid = shard["shard_id"]
        cached = self._packed_cache.get(sid)
        if cached is not None:
            return cached
        texts, doc_ids = self._texts_of(shard)
        expected = self._seq_counts[sid]
        if self.exact_counts and len(texts) != expected:
            raise ValueError(
                f"{sid}: manifest split_distribution={expected} but shard "
                f"contains {len(texts)} documents for split {self.split!r}")
        if not self.exact_counts and len(texts) != expected:
            # Legacy manifests only: adopt the real count so sampling stays
            # within bounds instead of fabricating missing documents.
            self.notes.append(
                f"{sid}: legacy count {expected} -> actual {len(texts)}")
            self._seq_counts[sid] = len(texts)
        if self.packing_policy == "legacy_document":
            packed = packing.load_or_build_packed(
                cache_root=self.cache_root, shard_sha256=shard["sha256"],
                texts=texts, document_ids=doc_ids, tokenizer=self.tokenizer,
                tokenizer_id=self.tokenizer_id,
                tokenizer_revision=self.tokenizer_revision,
                seq_len=self.seq_len, epoch=self.epoch)
        else:
            key = packing.tokenizer_key(self.tokenizer_id, self.tokenizer_revision)
            target = packing.packed_path(
                self.cache_root, shard["sha256"], key,
                seq_len=self.seq_len, epoch=self.epoch).with_name(
                    packing.packed_path(self.cache_root, shard["sha256"], key,
                                       seq_len=self.seq_len, epoch=self.epoch).stem
                    + "-contiguous.npy")
            try:
                packed = np.load(target, mmap_mode="r")
                if packed.shape[1] != self.seq_len:
                    raise ValueError("packed sequence length mismatch")
                self.metrics["cache_hits"] += 1
            except (OSError, ValueError, FileNotFoundError):
                packed = packing.pack_documents(
                    texts, doc_ids, tokenizer=self.tokenizer,
                    seq_len=self.seq_len, policy=self.packing_policy)
                tmp = target.with_name(target.name + ".part")
                with open(tmp, "wb") as handle:
                    np.save(handle, packed, allow_pickle=False)
                tmp.replace(target)
            self._seq_counts[sid] = int(packed.shape[0])
            self._recompute_offsets()
            self.metrics["decode_tokens"] += int(packed.size)
        loaded = self._open_packed(sid, shard, packed)
        if self.auto_evict:
            # The packed array just loaded is "active" (memory-mapped) and is
            # therefore protected; the source parquet and stale shards are not.
            self.enforce_cache_bound()
        return loaded

    def _recompute_offsets(self) -> None:
        total = 0
        for s in self.shards:
            sid = s["shard_id"]
            self._offsets[sid] = total
            total += self._seq_counts[sid]
        self._length = total

    def _shard_for_index(self, index: int) -> str:
        for s in self.shards:
            sid = s["shard_id"]
            start = self._offsets[sid]
            if start <= index < start + self._seq_counts[sid]:
                return sid
        raise IndexError(f"sequence index {index} outside corpus ({self._length})")

    # -------------------------------------------------------------- batching
    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return real token ids and next-token targets for global indices."""
        idx = [int(g) for g in np.asarray(indices).reshape(-1)]
        out = np.empty((len(idx), self.seq_len), dtype=np.int64)
        for row, g in enumerate(idx):
            sid = self._shard_for_index(g)
            packed = self._packed_for(self._by_id[sid])
            local = g - self._offsets[sid]
            if local >= packed.shape[0]:
                raise IndexError(f"sequence {g} missing in {sid}")
            out[row] = packed[local]
        labels = np.empty_like(out)
        labels[:, :-1] = out[:, 1:]
        labels[:, -1] = out[:, -1]
        return out, labels

    def epoch_order(self, *, seed: int, epoch: int) -> np.ndarray:
        """Global sequence indices in deterministic per-epoch order."""
        order = shard_permutation(
            [s["shard_id"] for s in self.shards], seed=seed, epoch=epoch)
        parts, base = [], 0
        for sid in order:
            count = self._seq_counts[sid]
            parts.append(sequence_permutation(sid, count, seed=seed, epoch=epoch)
                         + base)
            base += count
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts)

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
            self.prefetch_for(chunk)
            yield chunk

    # ---------------------------------------------------------- cache control
    def active_cache_files(self) -> set[str]:
        """Packed shards currently memory-mapped: never evict these."""
        key = packing.tokenizer_key(self.tokenizer_id, self.tokenizer_revision)
        active: set[str] = set()
        for sid in self._packed_cache:
            shard = self._by_id.get(sid)
            if shard is None:
                continue
            path = packing.packed_path(
                self.cache_root, shard["sha256"], key,
                seq_len=self.seq_len, epoch=self.epoch)
            if self.packing_policy != "legacy_document":
                path = path.with_name(path.stem + "-contiguous.npy")
            active.add(path.name)
        return active

    def _open_packed(self, sid: str, shard: dict[str, Any],
                     arr) -> np.ndarray:
        """Register a loaded packed array, closing the least-recently-used one."""
        self._packed_cache[sid] = arr
        if sid in self._loaded_order:
            self._loaded_order.remove(sid)
        self._loaded_order.append(sid)
        while len(self._loaded_order) > self.max_open_shards:
            old = self._loaded_order.pop(0)
            self._packed_cache.pop(old, None)
        return arr

    def prefetch_for(self, indices: np.ndarray) -> str:
        """Warm the cache for the shard owning the next indices.

        Prefetch is best-effort: a fetch failure is recorded and sampling
        continues, because the real read in ``get_batch`` will raise properly.
        """
        try:
            sid = self._shard_for_index(int(indices[0]))
        except (IndexError, ValueError, TypeError):
            return ""
        shard = self._by_id.get(sid)
        if shard is None:
            return ""
        try:
            self._packed_for(shard)
        except Exception as exc:  # noqa: BLE001 - prefetch must not break training
            self.notes.append(f"prefetch {sid} failed: {type(exc).__name__}")
            return ""
        return sid

    def prefetch_shards(self, shard_ids: list[str], *, workers: int = 2) -> list[str]:
        """Warm several upcoming shards using bounded background workers."""
        ids = [sid for sid in shard_ids if sid in self._by_id]
        self.telemetry.prefetch_queue_depth = len(ids)
        if not ids:
            return []

        def warm(sid: str) -> str:
            self._packed_for(self._by_id[sid])
            return sid

        completed: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(warm, sid) for sid in ids]
            for future in futures:
                try:
                    completed.append(future.result())
                except Exception as exc:  # noqa: BLE001 - foreground read remains authoritative
                    self.notes.append(f"prefetch failed: {type(exc).__name__}")
        self.telemetry.prefetch_queue_depth = 0
        return completed

    def enforce_cache_bound(self) -> list[str]:
        evicted = cache_mod.enforce_bound(
            self.cache_root, self.max_bytes, active=self.active_cache_files())
        self.evictions += len(evicted)
        self.telemetry.evictions += len(evicted)
        return evicted

    def telemetry_snapshot(self) -> dict:
        return {**self.telemetry.as_dict(), "downloads": self.downloads,
                "evictions": self.evictions, "notes": list(self.notes)}

    # ------------------------------------------------------- identity / state
    def sampler_state(self) -> dict:
        return {
            "epoch": self.epoch,
            "position": self._position,
            "shard_order": self.shard_order,
            "tokens_consumed": self.tokens_consumed,
            "recipe_hash": self.manifest.get("recipe_hash", ""),
            "manifest_hash": self.manifest_hash,
            "tokenizer_identity": self.tokenizer_identity,
            "packing_version": packing.PACKING_VERSION,
            "split": self.split,
            "seed": self.seed,
            "view_id": self.view_id,
            "stage": self.stage,
            "release_fingerprint": self.manifest.get("training_view_fingerprint", self.manifest.get("corpus_fingerprint", "")),
            "world_size": self.world_size,
            "rank": self.rank,
            "packing_policy": self.packing_policy,
        }

    def restore_sampler_state(self, state: dict) -> None:
        if state.get("shard_order") != self.shard_order:
            raise ValueError("shard order mismatch on resume")
        if state.get("manifest_hash") and state["manifest_hash"] != self.manifest_hash:
            raise ValueError("manifest changed since checkpoint")
        if state.get("tokenizer_identity") and \
                state["tokenizer_identity"] != self.tokenizer_identity:
            raise ValueError("tokenizer identity mismatch on resume")
        if state.get("view_id", self.view_id) != self.view_id or state.get("stage", self.stage) != self.stage:
            raise ValueError("training view/stage mismatch on resume")
        if int(state.get("world_size", self.world_size)) != self.world_size or \
                int(state.get("rank", self.rank)) != self.rank:
            raise ValueError("distributed worker topology mismatch on resume")
        self._position = int(state.get("position", 0))
        self.tokens_consumed = int(state.get("tokens_consumed", 0))

    def verify_integrity(self, *, check_local: bool = True) -> dict:
        """Validate manifest completeness plus any locally cached shards."""
        missing = [s["shard_id"] for s in self.shards if not s.get("sha256")]
        bad = []
        if check_local:
            for shard in self.shards:
                path = self.local_shard_path(shard)
                if not path.is_file():
                    continue
                try:
                    cache_mod.verify_sha256(path, shard["sha256"])
                except ValueError:
                    bad.append(shard["shard_id"])
                    path.unlink(missing_ok=True)
        return {"valid": not missing and not bad, "format_version": 4,
                "split": self.split, "shards": len(self.shards),
                "sequences": self._length, "missing": missing,
                "corrupt_local": bad, "exact_counts": self.exact_counts,
                "notes": list(self.notes)}

    def dataset_identity(self) -> dict:
        return {
            "split": self.split,
            "format_version": 4,
            "manifest_sha256": self.manifest_hash,
            "manifest_path": str(self.manifest_path),
            "recipe_hash": self.manifest.get("recipe_hash", ""),
            "corpus_fingerprint": self.manifest.get("corpus_fingerprint", ""),
            "hf_repo": self.hf_repo,
            "hf_revision": self.revision or "",
            "tokenizer_identity": self.tokenizer_identity,
            "packing_version": packing.PACKING_VERSION,
            "seq_len": self.seq_len,
            "view_id": self.view_id,
            "stage": self.stage,
            "packing_policy": self.packing_policy,
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def sequence_text(self, index: int, tokenizer=None) -> str:
        """Decode one sequence back to text (provenance/debug convenience)."""
        tok = tokenizer or self.tokenizer
        _x, _y = self.get_batch(np.asarray([index]))
        return tok.decode([int(t) for t in _x[0]], skip_special_tokens=False)

    def sampler_for(self, seed: int, consumed: int = 0, *,
                    world_size: int | None = None, rank: int | None = None) -> HierarchicalSampler:
        """Explicit training-integration hook (see ``HierarchicalSampler``)."""
        return HierarchicalSampler(
            self, seed=seed, consumed=consumed,
            world_size=self.world_size if world_size is None else world_size,
            rank=self.rank if rank is None else rank)


class HierarchicalSampler:
    """Index source compatible with :class:`flashmini.experiment.EpochSampler`.

    Same ``take(count)`` contract, but the underlying permutation is the
    hierarchical v4 order (shard permutation, then per-shard sequence
    permutation) so consecutive batches stay inside one remote shard.
    """

    def __init__(self, dataset: RemoteShardDataset, *, seed: int,
                 consumed: int = 0, world_size: int = 1, rank: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.consumed = int(consumed)
        self.global_rows = len(dataset)
        self.epoch = -1
        self.order: np.ndarray | None = None
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError("rank must be within world_size")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.rows = (self.global_rows + self.world_size - 1 - self.rank) // self.world_size

    def take(self, count: int) -> np.ndarray:
        if self.rows <= 0:
            return np.zeros((0,), dtype=np.int64)
        epoch, offset = divmod(self.consumed, self.rows)
        if epoch != self.epoch:
            global_order = self.dataset.epoch_order(seed=self.seed, epoch=epoch)
            self.order = global_order[self.rank::self.world_size]
            self.epoch = epoch
        count = min(count, self.rows - offset)
        result = self.order[offset:offset + count]
        self.consumed += count
        self.dataset.prefetch_for(result)
        return result

    def state(self) -> dict:
        return {"seed": self.seed, "consumed": self.consumed,
                "rows": self.rows, "world_size": self.world_size,
                "rank": self.rank, "dataset": self.dataset.dataset_identity()}


class DistributedHierarchicalSampler(HierarchicalSampler):
    """Explicit name for the distributed sampler contract."""
