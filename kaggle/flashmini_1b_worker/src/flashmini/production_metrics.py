"""Append-only production metrics and 25M milestone ledger."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .metrics import _jsonable
from .observability import atomic_write_json


class MetricsLedger:
    """Keep training history forever while checkpoint retention stays bounded."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.milestones_path = self.run_dir / "metrics" / "checkpoints.jsonl"
        self.snapshot_path = self.run_dir / "metrics" / "metrics_checkpoint.json"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.milestones_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **values: Any) -> dict[str, Any]:
        record = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "event": event, **_jsonable(values)}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def _read_milestones(self) -> list[dict[str, Any]]:
        if not self.milestones_path.is_file():
            return []
        rows = []
        for line in self.milestones_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def append_milestone(self, **values: Any) -> dict[str, Any]:
        required = {"checkpoint_threshold_tokens", "actual_tokens_seen", "global_step",
                    "checkpoint_sha256"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"milestone missing required fields: {missing}")
        row = {"event": "checkpoint_milestone", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **_jsonable(values)}
        previous = self._read_milestones()
        threshold = int(row["checkpoint_threshold_tokens"])
        if previous and threshold <= int(previous[-1]["checkpoint_threshold_tokens"]):
            if previous[-1].get("checkpoint_sha256") == row.get("checkpoint_sha256"):
                return previous[-1]
            raise ValueError("checkpoint milestone thresholds must increase monotonically")
        with self.milestones_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_write_json(self.snapshot_path, {"schema_version": 1, "milestones": previous + [row]})
        self.append("checkpoint_milestone", **{key: value for key, value in row.items() if key != "event"})
        return row

    def tail(self, count: int = 25) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("tail count cannot be negative")
        if not self.metrics_path.is_file() or count == 0:
            return []
        rows = [json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()]
        return rows[-count:]

    def milestones(self) -> list[dict[str, Any]]:
        return self._read_milestones()


def reconcile_metrics_history(path: Path | str, *, max_step: int, max_tokens: int) -> dict[str, int]:
    """Retain append-only rows and identify orphaned post-checkpoint records."""
    path = Path(path)
    if not path.is_file():
        return {"retained": 0, "orphaned": 0}
    retained, orphaned = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("global_step", row.get("step", 0)) or 0)
        tokens = int(row.get("global_exact_tokens", row.get("tokens_seen", 0)) or 0)
        if step <= max_step and tokens <= max_tokens:
            retained.append(row)
        else:
            orphaned.append(row)
    # Never erase forensic rows: rewrite the coherent stream and place the
    # displaced rows beside it for inspection.
    if orphaned:
        orphan = path.with_name(path.name + ".orphaned.jsonl")
        with orphan.open("a", encoding="utf-8") as handle:
            for row in orphaned:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary = path.with_name(path.name + ".reconciled.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"retained": len(retained), "orphaned": len(orphaned)}


__all__ = ["MetricsLedger", "reconcile_metrics_history"]
