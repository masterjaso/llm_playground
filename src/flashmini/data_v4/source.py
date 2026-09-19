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


def stream_source_window(source: dict, *, limit: int = 50,
                         cursor: SourceCursor | None = None) -> SourceResult:
    """Stream one bounded window. Never raises for gated/missing sources."""
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    cur = cursor or SourceCursor(
        config=source.get("config"), split=source.get("split", "train"))
    try:
        ds = load_dataset(
            source["dataset_id"],
            name=source.get("config"),
            split=source.get("split", "train"),
            revision=source.get("revision"),
            streaming=True,
        )
    except Exception as exc:  # fail-local: record and continue other sources
        msg = str(exc)
        low = msg.lower()
        if any(k in low for k in ("gated", "401", "403", "access denied",
                                  "terms", "login", "private")):
            return SourceResult(False, "SOURCE_BLOCKED",
                                "gated_access_not_accepted", [], cur)
        if "not found" in low or isinstance(exc, DatasetNotFoundError):
            return SourceResult(False, "SOURCE_BLOCKED", "dataset_not_found", [], cur)
        return SourceResult(False, "SOURCE_ERROR", f"{type(exc).__name__}: {msg[:200]}",
                            [], cur)
    records: list[dict] = []
    try:
        it = iter(ds.skip(cur.offset) if cur.offset else iter(ds))
        for i, row in enumerate(it):
            if i >= limit:
                break
            records.append(_to_record(source, row, cur.offset + i))
        cur.offset += len(records)
    except Exception as exc:
        return SourceResult(False, "SOURCE_ERROR",
                            f"{type(exc).__name__}: {str(exc)[:200]}", records, cur)
    return SourceResult(True, "OK", "", records, cur)


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
