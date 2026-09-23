"""Streaming source adapter (v4): bounded windows, fail-local semantics."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class SourceCursor:
    config: str | None = None
    split: str | None = None
    offset: int = 0
    source_file: str | None = None
    row_group: int | None = None
    row_index: int | None = None
    revision: str | None = None

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "split": self.split,
            "offset": int(self.offset),
            "source_file": self.source_file,
            "row_group": self.row_group,
            "row_index": self.row_index,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: dict | None) -> SourceCursor:
        value = value or {}
        return cls(
            config=value.get("config"), split=value.get("split"),
            offset=int(value.get("offset", 0) or 0),
            source_file=value.get("source_file"),
            row_group=(int(value["row_group"]) if value.get("row_group") is not None else None),
            row_index=(int(value["row_index"]) if value.get("row_index") is not None else None),
            revision=value.get("revision"),
        )


@dataclass
class SourceResult:
    ok: bool
    status: str = "OK"  # OK | SOURCE_BLOCKED | SOURCE_ERROR
    reason: str = ""
    records: list[dict] = field(default_factory=list)
    cursor: SourceCursor = field(default_factory=SourceCursor)
    decode_seconds: float = 0.0


def open_source_stream(source: dict) -> SourceResult | tuple[SourceResult, Iterator]:
    """Open a streaming iterator once per source (avoids repeated skip rescans)."""
    from datasets import load_dataset

    # ``hf_store`` also resolves the repository's .env fallback.  Passing the
    # token explicitly prevents source streaming from silently falling back to
    # unauthenticated Hub requests even when publishing authentication works.
    from . import hf_store
    native = SourceCursor.from_dict(source.get("_cursor"))
    cur = SourceCursor(
        config=source.get("config"), split=source.get("split", "train"),
        offset=native.offset, source_file=native.source_file,
        row_group=native.row_group, row_index=native.row_index,
        revision=source.get("revision"),
    )
    try:
        ds = load_dataset(
            source["dataset_id"],
            name=source.get("config"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
            token=hf_store.load_token(),
            cache_dir=source.get("_cache_dir"),
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 - classify blocked/error source locally
        return _classify_open_error(exc, cur)
    text_field = source.get("text_field", "text")
    features = getattr(ds, "features", None)
    if features is not None and text_field not in features:
        return SourceResult(
            False, "SOURCE_BLOCKED", f"text_field_missing:{text_field}", [], cur)
    offset = int(source.get("_offset", cur.offset) or 0)
    # The streaming API does not expose a universal seek primitive.  Keep the
    # legacy offset fallback for generic HF datasets, but carry the native
    # cursor fields through every checkpoint so file/row-group adapters can
    # resume without a linear skip when they provide one.
    seek = getattr(ds, "seek", None)
    if callable(seek) and cur.source_file is not None:
        iterator = iter(seek(cur.source_file, cur.row_group or 0, cur.row_index or 0))
    else:
        iterator = iter(ds.skip(offset) if offset else iter(ds))
    return SourceResult(True, "OK", "", [], cur), iterator


def _classify_open_error(exc: Exception, cur: SourceCursor) -> SourceResult:
    from datasets.exceptions import DatasetNotFoundError
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in ("gated", "401", "403", "access denied",
                              "terms", "login", "private")):
        return SourceResult(False, "SOURCE_BLOCKED", "gated_access_not_accepted", [], cur)
    if "not found" in low or isinstance(exc, DatasetNotFoundError):
        return SourceResult(False, "SOURCE_BLOCKED", "dataset_not_found", [], cur)
    return SourceResult(False, "SOURCE_ERROR", f"{type(exc).__name__}: {msg[:200]}", [], cur)


def stream_records(iterator, source: dict, *, limit: int, start_offset: int = 0) -> SourceResult:
    """Pull one bounded window from an already-open iterator."""
    native = SourceCursor.from_dict(source.get("_cursor"))
    cur = SourceCursor(config=source.get("config"), split=source.get("split", "train"),
                       offset=start_offset, source_file=native.source_file,
                       row_group=native.row_group, row_index=native.row_index,
                       revision=source.get("revision"))
    records: list[dict] = []
    decode_seconds = 0.0
    try:
        for i, row in enumerate(iterator):
            if i >= limit:
                break
            ordinal = start_offset + i
            decode_started = time.perf_counter()
            records.append(_to_record(source, row, ordinal))
            decode_seconds += time.perf_counter() - decode_started
        cur.offset += len(records)
        if cur.row_index is not None:
            cur.row_index += len(records)
        else:
            cur.row_index = cur.offset
    except Exception as exc:  # noqa: BLE001 - classify per-window source failure
        return SourceResult(False, "SOURCE_ERROR",
                            f"{type(exc).__name__}: {str(exc)[:200]}", records, cur,
                            decode_seconds)
    return SourceResult(True, "OK", "", records, cur, decode_seconds)


def stream_source_window(source: dict, *, limit: int = 50,
                         cursor: SourceCursor | None = None,
                         max_retries: int = 5) -> SourceResult:
    """Stream one bounded window. Never raises for gated/missing sources.
    Retries transient network errors with exponential backoff."""
    cur = cursor or SourceCursor(
        config=source.get("config"), split=source.get("split", "train"))
    src = dict(source)
    src["_cursor"] = cur.as_dict()
    src["_offset"] = cur.offset
    for attempt in range(max_retries + 1):
        opened = open_source_stream(src)
        if isinstance(opened, SourceResult):
            if attempt == max_retries or not _is_retryable(opened):
                return opened
            time.sleep(min(2 ** attempt, 30) + _jitter())
            continue
        _result, iterator = opened
        try:
            res = stream_records(iterator, src, limit=limit, start_offset=cur.offset)
        finally:
            closer = getattr(iterator, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001, S110 - iterator cleanup is best effort
                    pass
        # On transient network errors during streaming, retry from scratch
        if attempt < max_retries and _is_retryable(res):
            time.sleep(min(2 ** attempt, 30) + _jitter())
            continue
        return res
    return res


def _is_retryable(result: SourceResult) -> bool:
    """Only retry transient network/IO errors, not auth/blocked/missing."""
    if result.status == "OK":
        return False
    msg = (result.reason or "").lower()
    retryable_keywords = (
        "bad file descriptor", "broken pipe", "connection reset",
        "timeout", "temporary failure", "network", "http error",
        "too many requests", "rate limit", "server error",
        "remote end closed", "errno 9", "io error", "socket",
    )
    return any(k in msg for k in retryable_keywords)


def _jitter() -> float:
    """Small random jitter to avoid thundering herd."""
    import random as _random
    return _random.uniform(0, 0.5)



def _to_record(source: dict, row: dict, ordinal: int) -> dict:
    text_field = source.get("text_field", "text")
    text = row.get(text_field, "") if isinstance(row, dict) else ""
    if isinstance(text, list):
        # Chat-style rows (e.g. conversations): join assistant/user values.
        text = "\n\n".join(
            str(t.get("value", "")) for t in text if isinstance(t, dict))
    if not isinstance(text, str):
        text = ""
    rid = str(row.get("id", row.get("__index_level_0__", ordinal)))
    return {
        "source_id": source.get("source_id", source.get("dataset_id", "")),
        "dataset_id": source.get("dataset_id", ""),
        "revision": source.get("revision", ""),
        "config": source.get("config"),
        "split": source.get("split"),
        "record_id": rid,
        "text": text if isinstance(text, str) else "",
        "extra": {k: v for k, v in (row.items() if isinstance(row, dict) else [])
                  if k not in ("text", text_field) and isinstance(v, (str, int, float, bool))},
        "source_cursor": {
            "source_file": row.get("__source_file") if isinstance(row, dict) else None,
            "row_group": row.get("__row_group") if isinstance(row, dict) else None,
            "row_index": ordinal,
        },
    }
