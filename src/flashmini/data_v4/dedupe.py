"""Exact + MinHash near-duplicate handling (v4).

Exact: streaming set/sqlite of content hashes (order-independent).
Near: 128-dim MinHash over word-5-shingles, LSH bands (8 bands x 16 rows),
partitioned by band-prefix so only one partition is resident at a time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEDUPE_VERSION = "flashmini-v4-dedupe-v1"
NEAR_ALGORITHM = "minhash_lsh_128_8x16_word5_v1"
NEAR_THRESHOLD = 0.8


def _shingles(text: str, k: int = 5) -> set[str]:
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(text: str, num_perm: int = 128) -> list[int]:
    sh = _shingles(text)
    if not sh:
        return [0] * num_perm
    sig: list[int] = []
    for seed in range(num_perm):
        best = None
        for s in sh:
            h = int.from_bytes(
                hashlib.sha256(f"{seed}\0{s}".encode("utf-8")).digest()[:8], "big")
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
    """Order-independent exact filter keyed on content hash."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.duplicates = 0

    def check(self, content_hash_hex: str) -> bool:
        """Return True if kept (first occurrence), False if duplicate."""
        if content_hash_hex in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(content_hash_hex)
        return True


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
        p.write_text(json.dumps(rows))
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
