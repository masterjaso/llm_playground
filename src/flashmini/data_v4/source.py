"""Streaming source adapter (v4): bounded windows, fail-local semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class SourceCursor:
    config: str | None = None
    split: str | None = None
    offset: int = 0


@dataclass
class SourceResult:
    ok: bool
    status: str = "OK"  # OK | SOURCE_BLOCKED | SOURCE_ERROR
    reason: str = ""
    records: list[dict] = field(default_factory=list)
    cursor: SourceCursor = field(default_factory=SourceCursor)


def open_source_stream(source: dict) -> SourceResult:
    """Open a streaming iterator once per source (avoids repeated skip rescans)."""
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    cur = SourceCursor(config=source.get("config"), split=source.get("split", "train"))
    try:
        ds = load_dataset(
            source["dataset_id"],
            name=source.get("config"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
            streaming=True,
        )
    except Exception as exc:
        return _classify_open_error(exc, cur)
    offset = int(source.get("_offset", 0) or 0)
    iterator = iter(ds.skip(offset) if offset else iter(ds))
    return SourceResult(True, "OK", "", [], cur), iterator  # type: ignore[return-value]


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
    cur = SourceCursor(config=source.get("config"), split=source.get("split", "train"),
                       offset=start_offset)
    records: list[dict] = []
    try:
        for i, row in enumerate(iterator):
            if i >= limit:
                break
            records.append(_to_record(source, row, start_offset + i))
        cur.offset += len(records)
    except Exception as exc:
        return SourceResult(False, "SOURCE_ERROR",
                            f"{type(exc).__name__}: {str(exc)[:200]}", records, cur)
    return SourceResult(True, "OK", "", records, cur)


def stream_source_window(source: dict, *, limit: int = 50,
                         cursor: SourceCursor | None = None,
                         max_retries: int = 5) -> SourceResult:
    """Stream one bounded window. Never raises for gated/missing sources.
    Retries transient network errors with exponential backoff."""
    cur = cursor or SourceCursor(
        config=source.get("config"), split=source.get("split", "train"))
    src = dict(source)
    src["_offset"] = cur.offset
    for attempt in range(max_retries + 1):
        opened = open_source_stream(src)
        if isinstance(opened, SourceResult):
            if attempt == max_retries or not _is_retryable(opened):
                return opened
            time.sleep(min(2 ** attempt, 30) + _jitter())
            continue
        _result, iterator = opened
        res = stream_records(iterator, source, limit=limit, start_offset=cur.offset)
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
    }
