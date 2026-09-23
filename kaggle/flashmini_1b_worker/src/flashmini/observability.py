"""Crash-safe status, heartbeat, and freeze detection for long TPU runs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_PREFIX = "FLASHMINI_STATUS "


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    """Write parseable JSON with fsync + atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some mounted filesystems do not permit directory fsync; the
            # atomic replace remains safe and the limitation is not fatal.
            pass
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StatusPaths:
    run_dir: Path

    @property
    def status_dir(self) -> Path:
        return self.run_dir / "status"

    @property
    def heartbeat(self) -> Path:
        return self.status_dir / "heartbeat.json"

    @property
    def progress(self) -> Path:
        return self.status_dir / "progress.json"

    @property
    def run_status(self) -> Path:
        return self.run_dir / "run_status.json"

    @property
    def events(self) -> Path:
        return self.status_dir / "events.jsonl"


_DEFAULT_FIELDS: dict[str, Any] = {
    "phase": "starting", "global_step": 0, "global_exact_tokens": 0,
    "foundation_tokens": 0, "target_tokens": 100_000_000_000,
    "percent_to_10b": 0.0, "percent_to_100b": 0.0,
    "current_lr": None, "recent_loss": None, "ema_loss": None,
    "recent_total_loss": None, "grad_norm": None, "router_aux_loss": None,
    "router_entropy": None, "expert_load_ratio": None, "ple_scale": None,
    "ple_norm_ratio": None, "kvc_active": True, "tokens_per_sec_recent": None,
    "tokens_per_sec_session": None, "tokens_per_sec_lifetime": None,
    "elapsed_session_seconds": 0.0, "elapsed_lifetime_seconds": 0.0,
    "estimated_data_wait_seconds": 0.0, "checkpoint_state": "none",
    "seconds_since_last_optimizer_progress": 0.0, "parent_checkpoint_sha256": None,
    "session_id": None, "parent_session_id": None, "latest_checkpoint": None,
    "latest_checkpoint_sha256": None, "compile_elapsed_seconds": None,
    "compile_count": 0, "restart_count": 0, "data_state": "ready",
}


class StatusLogger:
    """Human + machine status surface shared by local and XLA workers."""

    def __init__(self, run_dir: Path | str, *, run_id: str, session_id: str | None = None,
                 parent_session_id: str | None = None, target_tokens: int = 100_000_000_000,
                 preview_pause_tokens: int = 10_000_000_000, heartbeat_interval: float = 60.0,
                 stream=None) -> None:
        self.paths = StatusPaths(Path(run_dir))
        self.run_id = str(run_id)
        self.session_id = session_id or uuid.uuid4().hex
        self.parent_session_id = parent_session_id
        self.target_tokens = int(target_tokens)
        self.preview_pause_tokens = int(preview_pause_tokens)
        self.heartbeat_interval = max(0.1, float(heartbeat_interval))
        self.stream = stream or sys.stdout
        self._started = time.monotonic()
        self._lifetime_started = self._started
        self._last_progress = self._started
        self._state: dict[str, Any] = dict(_DEFAULT_FIELDS)
        self._state.update({"run_id": self.run_id, "session_id": self.session_id,
                            "parent_session_id": self.parent_session_id,
                            "target_tokens": self.target_tokens})
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _with_derived(self, values: Mapping[str, Any]) -> dict[str, Any]:
        state = dict(self._state)
        state.update(values)
        tokens = int(state.get("global_exact_tokens") or 0)
        state["percent_to_10b"] = 100.0 * min(tokens, self.preview_pause_tokens) / max(self.preview_pause_tokens, 1)
        state["percent_to_100b"] = 100.0 * min(tokens, self.target_tokens) / max(self.target_tokens, 1)
        state["last_heartbeat_utc"] = utc_now()
        state["last_progress_utc"] = state.get("last_progress_utc")
        state["elapsed_session_seconds"] = max(0.0, time.monotonic() - self._started)
        state["elapsed_lifetime_seconds"] = state["elapsed_session_seconds"]
        state["seconds_since_last_optimizer_progress"] = max(0.0, time.monotonic() - self._last_progress)
        return state

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        with self._lock:
            record = self._with_derived({"event": event, **values, "timestamp_utc": utc_now()})
            self._state.update(record)
            self.paths.events.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()
            print(STATUS_PREFIX + json.dumps(record, sort_keys=True, default=str), file=self.stream, flush=True)
            return record

    def update(self, *, progress: bool = False, event: str = "train_status", **values: Any) -> dict[str, Any]:
        with self._lock:
            if progress:
                self._last_progress = time.monotonic()
                values.setdefault("last_progress_utc", utc_now())
            record = self._with_derived(values)
            self._state.update(record)
            atomic_write_json(self.paths.heartbeat, {**record, "event": "heartbeat"})
            atomic_write_json(self.paths.progress, {**record, "event": event})
            status = {
                "run_id": self.run_id, "status": self._status_for_phase(str(record.get("phase", "starting"))),
                "session_id": self.session_id, "parent_session_id": self.parent_session_id,
                "target_tokens": self.target_tokens, "preview_pause_tokens": self.preview_pause_tokens,
                "tokens_seen": int(record.get("global_exact_tokens") or 0),
                "step": int(record.get("global_step") or 0), "stage": record.get("stage", "foundation"),
                "recent_loss": record.get("recent_loss"), "ema_loss": record.get("ema_loss"),
                "recent_tokens_per_sec": record.get("tokens_per_sec_recent"),
                "lifetime_tokens_per_sec": record.get("tokens_per_sec_lifetime"),
                "latest_checkpoint": record.get("latest_checkpoint"),
                "latest_checkpoint_sha256": record.get("latest_checkpoint_sha256"),
                "last_progress_utc": record.get("last_progress_utc"),
                "last_heartbeat_utc": record.get("last_heartbeat_utc"),
            }
            atomic_write_json(self.paths.run_status, status)
            if event:
                self.emit(event, **values)
            return record

    @staticmethod
    def _status_for_phase(phase: str) -> str:
        if phase in {"xla_compile", "compile"}:
            return "compiling"
        if phase in {"checkpoint", "checkpointing"}:
            return "checkpointing"
        if phase in {"paused", "paused_quota"}:
            return "paused"
        if phase in {"failed", "stall"}:
            return "failed"
        if phase == "complete":
            return "complete"
        return "training"

    def heartbeat(self) -> dict[str, Any]:
        return self.update(event="heartbeat")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="flashmini-heartbeat", daemon=True)
        self._thread.start()
        self.heartbeat()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                self.heartbeat()
            except Exception as exc:  # noqa: BLE001 - telemetry must not kill the worker
                print(STATUS_PREFIX + json.dumps({"event": "heartbeat_error", "error": str(exc)}),
                      file=self.stream, flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.heartbeat_interval * 2))


@dataclass
class Watchdog:
    """Phase-aware stall detector; compilation is never mistaken for a hang."""

    compile_timeout_seconds: float = 1800.0
    stall_timeout_seconds: float = 300.0
    hard_stall_timeout_seconds: float = 900.0
    now: Callable[[], float] = time.monotonic
    phase: str = "starting"
    phase_started: float = field(default_factory=time.monotonic)
    last_step: int = 0
    last_tokens: int = 0
    last_progress: float = field(default_factory=time.monotonic)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _suspected: bool = False

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)
        self.phase_started = self.now()
        self.last_progress = self.phase_started
        self._suspected = False

    def observe(self, *, step: int, tokens: int, **diagnostics: Any) -> None:
        self.diagnostics.update(diagnostics)
        if int(step) > self.last_step or int(tokens) > self.last_tokens:
            self.last_step, self.last_tokens = int(step), int(tokens)
            self.last_progress = self.now()
            self._suspected = False

    def poll(self) -> dict[str, Any] | None:
        current = self.now()
        elapsed_phase = current - self.phase_started
        stalled = current - self.last_progress
        if self.phase in {"xla_compile", "compile"}:
            if elapsed_phase > self.compile_timeout_seconds:
                return self._event("XLA_COMPILE_TIMEOUT", elapsed_phase, stalled)
            return None
        if self.phase not in {"training", "active"}:
            return None
        if stalled >= self.hard_stall_timeout_seconds:
            return self._event("STALL_HARD", elapsed_phase, stalled)
        if stalled >= self.stall_timeout_seconds and not self._suspected:
            self._suspected = True
            return self._event("STALL_SUSPECTED", elapsed_phase, stalled)
        return None

    def _event(self, event: str, phase_elapsed: float, stalled: float) -> dict[str, Any]:
        cause = "training"
        if self.diagnostics.get("checkpoint_state") in {"saving", "uploading"}:
            cause = "checkpoint_upload"
        elif self.diagnostics.get("network_error"):
            cause = "network_stall"
        elif float(self.diagnostics.get("estimated_data_wait_seconds", 0.0) or 0.0) > stalled / 2:
            cause = "data_starvation"
        return {
            "event": event, "phase": self.phase, "phase_elapsed_seconds": phase_elapsed,
            "stall_elapsed_seconds": stalled, "last_step": self.last_step,
            "last_tokens": self.last_tokens, "cause": cause,
            "diagnostics": dict(self.diagnostics),
        }


__all__ = ["STATUS_PREFIX", "StatusLogger", "StatusPaths", "Watchdog", "atomic_write_json", "utc_now"]
