"""Bounded LRU local cache with watermarks and atomic writes (v4)."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheStats:
    root: str
    max_bytes: int
    used_bytes: int
    files: int
    avail_bytes: int


@dataclass
class CacheTelemetry:
    """Small in-process telemetry surface for training data observability."""

    cache_hits: int = 0
    cache_misses: int = 0
    downloads: int = 0
    downloaded_bytes: int = 0
    download_seconds: float = 0.0
    evictions: int = 0
    data_wait_seconds: float = 0.0
    prefetch_queue_depth: int = 0

    def record_download(self, size: int, elapsed: float) -> None:
        self.downloads += 1
        self.downloaded_bytes += max(0, int(size))
        self.download_seconds += max(0.0, float(elapsed))

    @property
    def download_mb_s(self) -> float:
        return (self.downloaded_bytes / 1024**2 / self.download_seconds
                if self.download_seconds > 0 else 0.0)

    def as_dict(self) -> dict:
        return {
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "downloads": self.downloads,
            "downloaded_bytes": self.downloaded_bytes,
            "download_mb_s": self.download_mb_s,
            "data_wait_seconds": self.data_wait_seconds,
            "prefetch_queue_depth": self.prefetch_queue_depth,
            "evictions": self.evictions,
        }


def cache_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    import os as _os
    return Path(_os.environ.get("FLASHMINI_DATA_CACHE_DIR", ".cache/flashmini-data-v4"))


def cache_max_bytes(explicit: int | None = None) -> int:
    import os as _os
    if explicit is not None:
        return explicit
    try:
        return int(float(_os.environ.get("FLASHMINI_DATA_CACHE_GB", "30")) * (1024 ** 3))
    except ValueError:
        return 30 * (1024 ** 3)


def disk_avail_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path if path.exists() else path.parent).free
    except OSError:
        return 0


def check_watermark(path: Path, min_avail_bytes: int = 10 * 1024 ** 3) -> None:
    avail = disk_avail_bytes(path)
    if avail < min_avail_bytes:
        raise RuntimeError(
            f"disk watermark: only {avail / 1024**3:.2f} GiB available "
            f"(minimum {min_avail_bytes / 1024**3:.0f} GiB); refusing ingestion")


def _dir_size(root: Path) -> tuple[int, int]:
    total, count = 0, 0
    if not root.exists():
        return 0, 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix != ".tmp":
            try:
                total += p.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def enforce_bound(root: Path, max_bytes: int, active: set[str] | None = None) -> list[str]:
    """Evict oldest files (by mtime) until under bound. Never evict active."""
    active = active or set()
    evicted: list[str] = []
    used, _ = _dir_size(root)
    if used <= max_bytes:
        return evicted
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in active
         and not p.name.endswith(".tmp")),
        key=lambda p: p.stat().st_mtime)
    for p in files:
        if used <= max_bytes:
            break
        if str(p) in active or p.name in active:
            continue
        try:
            used -= p.stat().st_size
            p.unlink()
            evicted.append(str(p))
        except OSError:
            continue
    if used > max_bytes:
        raise RuntimeError(f"cache bound exceeded and nothing further evictable ({used} bytes)")
    return evicted


def mark_in_use(path: Path) -> None:
    """Touch a managed cache file so LRU eviction keeps recently used data."""
    try:
        now = time.time()
        os.utime(path, (now, now))
    except OSError:
        pass


def atomic_write_bytes(path: Path, data: bytes,
                       *, min_avail_bytes: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if min_avail_bytes is not None and path.parent.exists():
        check_watermark(path.parent, min_avail_bytes)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {path.name}")


def stats(root: Path, max_bytes: int) -> CacheStats:
    used, count = _dir_size(root)
    try:
        avail = shutil.disk_usage(root if root.exists() else Path(".")).free
    except OSError:
        avail = -1
    return CacheStats(str(root), max_bytes, used, count, avail)
