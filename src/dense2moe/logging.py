"""Small JSONL event logger used by every command."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one JSON record, creating the parent directory first."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

