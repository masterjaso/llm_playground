"""SQLite WAL job leasing with duplicate-lease protection."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    id: int
    layer: int
    status: str
    worker: str | None
    attempts: int
    payload: dict[str, Any]


class JobQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, layer INTEGER UNIQUE NOT NULL, status TEXT NOT NULL, worker TEXT, attempts INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL DEFAULT '{}', leased_at REAL, completed_at REAL, error TEXT)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, id)")

    def enqueue(self, layer: int, payload: dict[str, Any] | None = None) -> int:
        cursor = self.connection.execute("INSERT OR IGNORE INTO jobs(layer,status,payload) VALUES(?,?,?)", (layer, "pending", _json(payload or {})))
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self.connection.execute("SELECT id FROM jobs WHERE layer=?", (layer,)).fetchone()
        assert row is not None
        return int(row[0])

    def enqueue_layers(self, layers: list[int]) -> None:
        for layer in layers:
            self.enqueue(layer)

    def lease(self, worker: str, *, now: float | None = None) -> Job | None:
        timestamp = now if now is not None else time.time()
        token = worker or str(uuid.uuid4())
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            updated = self.connection.execute("UPDATE jobs SET status='leased',worker=?,attempts=attempts+1,leased_at=? WHERE id=? AND status='pending'", (token, timestamp, row["id"]))
            if updated.rowcount != 1:
                self.connection.execute("ROLLBACK")
                return None
            self.connection.execute("COMMIT")
            return self._row(row["id"])
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def complete(self, job_id: int, worker: str) -> None:
        updated = self.connection.execute("UPDATE jobs SET status='complete',completed_at=? WHERE id=? AND status='leased' AND worker=?", (time.time(), job_id, worker))
        if updated.rowcount != 1:
            raise ValueError("job is not leased by this worker")

    def fail(self, job_id: int, worker: str, error: str, *, retry: bool = True) -> None:
        status = "pending" if retry else "failed"
        updated = self.connection.execute("UPDATE jobs SET status=?,error=?,worker=NULL WHERE id=? AND status='leased' AND worker=?", (status, error, job_id, worker))
        if updated.rowcount != 1:
            raise ValueError("job is not leased by this worker")

    def reclaim_expired(self, timeout_seconds: float) -> int:
        cutoff = time.time() - timeout_seconds
        cursor = self.connection.execute("UPDATE jobs SET status='pending',worker=NULL WHERE status='leased' AND leased_at < ?", (cutoff,))
        return int(cursor.rowcount)

    def get(self, job_id: int) -> Job:
        return self._row(job_id)

    def summary(self) -> dict[str, int]:
        rows = self.connection.execute("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _row(self, job_id: int) -> Job:
        row = self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return Job(int(row["id"]), int(row["layer"]), str(row["status"]), row["worker"], int(row["attempts"]), _unjson(row["payload"]))

    def close(self) -> None:
        self.connection.close()


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _unjson(value: str) -> dict[str, Any]:
    import json

    return json.loads(value)

