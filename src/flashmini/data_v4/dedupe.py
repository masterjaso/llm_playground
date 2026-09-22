"""Restartable exact and near-duplicate indexes.

Exact: streaming set/sqlite of content hashes (order-independent).
Near: 128-dim MinHash over word-5-shingles, LSH bands (8 bands x 16 rows),
partitioned by band-prefix so only one partition is resident at a time.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Self

DEDUPE_VERSION = "flashmini-v4-dedupe-v3"
NEAR_ALGORITHM = "minhash_lsh_128_8x16_word5_bounded_stride_v1"
NEAR_THRESHOLD = 0.8
MAX_SHINGLES = 4096


def _shingles(text: str, k: int = 5, *, max_shingles: int = MAX_SHINGLES) -> set[str]:
    """Return a deterministic, bounded word-shingle sample.

    Building every shingle is impractical for long books and code files:
    MinHash revisits the collection once per permutation. Evenly spaced
    windows preserve coverage across the document while bounding both the
    retained set and subsequent hash work.
    """
    if max_shingles <= 0:
        raise ValueError("max_shingles must be positive")
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    window_count = len(words) - k + 1
    if window_count <= max_shingles:
        return {" ".join(words[i:i + k]) for i in range(window_count)}
    # Integer arithmetic keeps the sample deterministic across platforms and
    # guarantees at most ``max_shingles`` windows.
    return {
        " ".join(words[(i * window_count) // max_shingles:
                        (i * window_count) // max_shingles + k])
        for i in range(max_shingles)
    }


def minhash_signature(text: str, num_perm: int = 128) -> list[int]:
    sh = _shingles(text)
    if not sh:
        return [0] * num_perm
    sig: list[int] = []
    for seed in range(num_perm):
        best = None
        for s in sh:
            h = int.from_bytes(
                hashlib.sha256(f"{seed}\0{s}".encode()).digest()[:8], "big")
            if best is None or h < best:
                best = h
        sig.append(int(best or 0))
    return sig


def lsh_buckets(sig: list[int], bands: int = 8) -> list[str]:
    rows = len(sig) // bands
    out = []
    for b in range(bands):
        chunk = sig[b * rows:(b + 1) * rows]
        out.append(f"{b:02d}-" + hashlib.sha256(json.dumps(chunk).encode()).hexdigest()[:8])
    return out


class ExactDedupe:
    """Order-independent exact filter with an optional SQLite backing store.

    Production builds pass a path, which keeps only the current hash lookup in
    SQLite and makes every accepted/rejected decision durable.  The no-path
    mode remains a tiny in-memory adapter for callers and hermetic unit tests;
    it is never used by the production build command.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._seen: set[str] = set()
        self.duplicates = 0
        self.path = Path(path) if path is not None else None
        self._db: sqlite3.Connection | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), timeout=30)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA busy_timeout=30000")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS content_hashes ("
                "hash TEXT PRIMARY KEY, first_document_id TEXT NOT NULL DEFAULT '', "
                "accepted_at INTEGER NOT NULL)"
            )
            self._db.commit()

    def check(self, content_hash_hex: str, document_id: str = "") -> bool:
        """Return True if kept (first occurrence), False if duplicate."""
        if len(content_hash_hex) != 64:
            raise ValueError("content hash must be a 64-character SHA256 hex string")
        if self._db is not None:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO content_hashes(hash, first_document_id, accepted_at) "
                "VALUES (?, ?, ?)", (content_hash_hex, document_id, int(time.time()))
            )
            self._db.commit()
            if cur.rowcount == 0:
                self.duplicates += 1
                return False
            return True
        if content_hash_hex in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(content_hash_hex)
        return True

    def import_hashes(self, hashes: list[str] | set[str]) -> int:
        """Migrate a legacy state list into the durable index once."""
        if self._db is None:
            before = len(self._seen)
            self._seen.update(hashes)
            return len(self._seen) - before
        rows = [(str(h), "", int(time.time())) for h in hashes]
        self._db.executemany(
            "INSERT OR IGNORE INTO content_hashes(hash, first_document_id, accepted_at) "
            "VALUES (?, ?, ?)", rows)
        self._db.commit()
        return int(self._db.execute("SELECT changes()").fetchone()[0])

    def __contains__(self, content_hash_hex: str) -> bool:
        if self._db is not None:
            return self._db.execute(
                "SELECT 1 FROM content_hashes WHERE hash = ?", (content_hash_hex,)
            ).fetchone() is not None
        return content_hash_hex in self._seen

    def __len__(self) -> int:
        if self._db is not None:
            return int(self._db.execute("SELECT COUNT(*) FROM content_hashes").fetchone()[0])
        return len(self._seen)

    def close(self) -> None:
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _pack_signature(signature: list[int]) -> bytes:
    return struct.pack(f"<{len(signature)}Q", *(int(x) for x in signature))


def _unpack_signature(payload: bytes, count: int = 128) -> list[int]:
    if len(payload) != count * 8:
        raise ValueError("invalid MinHash signature payload")
    return list(struct.unpack(f"<{count}Q", payload))


class NearDedupeIndex:
    """Persistent MinHash/LSH filter with bounded candidate lookup.

    Only rows sharing an LSH bucket are compared, so ingestion does not do a
    global pairwise scan.  SQLite transactions make a crash between documents
    restartable; the index is independent of the canonical shard commit.
    """

    def __init__(self, path: str | Path, *, threshold: float = NEAR_THRESHOLD,
                 num_perm: int = 128, bands: int = 8,
                 max_candidates: int = 4096) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("near-dedupe threshold must be in (0,1]")
        if num_perm <= 0 or bands <= 0 or num_perm % bands:
            raise ValueError("num_perm must be positive and divisible by bands")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.num_perm = int(num_perm)
        self.bands = int(bands)
        self.max_candidates = int(max_candidates)
        self.duplicates = 0
        self._db = sqlite3.connect(str(self.path), timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS near_signatures ("
            "document_id TEXT PRIMARY KEY, signature BLOB NOT NULL, "
            "buckets TEXT NOT NULL, accepted_at INTEGER NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS near_buckets ("
            "bucket TEXT NOT NULL, document_id TEXT NOT NULL, "
            "PRIMARY KEY(bucket, document_id))"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_near_bucket ON near_buckets(bucket)")
        self._db.commit()

    def check(self, text: str, document_id: str) -> bool:
        """Return ``True`` for a new document and persist its signature."""
        signature = minhash_signature(text, num_perm=self.num_perm)
        buckets = lsh_buckets(signature, bands=self.bands)
        candidates: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for bucket in buckets:
            rows = self._db.execute(
                "SELECT s.document_id, s.signature FROM near_buckets b "
                "JOIN near_signatures s ON s.document_id=b.document_id "
                "WHERE b.bucket=? LIMIT ?", (bucket, self.max_candidates)
            ).fetchall()
            for candidate_id, payload in rows:
                if candidate_id not in seen:
                    seen.add(candidate_id)
                    candidates.append((candidate_id, payload))
                    if len(candidates) >= self.max_candidates:
                        break
            if len(candidates) >= self.max_candidates:
                break
        for _candidate_id, payload in candidates:
            other = _unpack_signature(payload, self.num_perm)
            agree = sum(a == b for a, b in zip(signature, other))
            if agree / self.num_perm >= self.threshold:
                self.duplicates += 1
                return False
        try:
            self._db.execute(
                "INSERT INTO near_signatures(document_id, signature, buckets, accepted_at) "
                "VALUES (?, ?, ?, ?)",
                (document_id, _pack_signature(signature), json.dumps(buckets), int(time.time())),
            )
            self._db.executemany(
                "INSERT OR IGNORE INTO near_buckets(bucket, document_id) VALUES (?, ?)",
                [(bucket, document_id) for bucket in buckets],
            )
            self._db.commit()
        except sqlite3.IntegrityError:
            # Another process accepted the same stable document ID first.
            self._db.rollback()
            self.duplicates += 1
            return False
        return True

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM near_signatures").fetchone()[0])

    def close(self) -> None:
        self._db.commit()
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def partition_signatures(records: list[dict], out_dir: Path,
                         prefix_chars: int = 2) -> list[Path]:
    """Persist signature partitions keyed by deterministic bucket prefix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[dict]] = {}
    for rec in records:
        for bucket in rec["buckets"]:
            key = bucket[:prefix_chars]
            parts.setdefault(key, []).append(
                {"document_id": rec["document_id"], "bucket": bucket,
                 "signature": rec["signature"]})
    paths = []
    for key, rows in sorted(parts.items()):
        p = out_dir / f"part-{key}.json"
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(rows, sort_keys=True))
        tmp.replace(p)
        paths.append(p)
    return paths


def resolve_partition(part_path: Path, threshold: float = NEAR_THRESHOLD) -> dict:
    """Resolve near-duplicate groups within one partition (deterministic)."""
    rows = json.loads(part_path.read_text())
    by_bucket: dict[str, list[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)
    removed: set[str] = set()
    for _bucket, members in sorted(by_bucket.items()):
        members.sort(key=lambda r: r["document_id"])
        keeper = None
        keeper_sig: list[int] | None = None
        for m in members:
            if m["document_id"] in removed:
                continue
            if keeper is None:
                keeper, keeper_sig = m["document_id"], m["signature"]
                continue
            assert keeper_sig is not None
            agree = sum(1 for a, b in zip(keeper_sig, m["signature"]) if a == b)
            if agree / len(keeper_sig) >= threshold:
                removed.add(m["document_id"])
    return {"partition": part_path.name, "members": len(rows),
            "removed": sorted(removed), "algorithm": NEAR_ALGORITHM,
            "threshold": threshold, "version": DEDUPE_VERSION}
