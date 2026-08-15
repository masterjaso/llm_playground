"""Durable run state, atomic artifact publication, and handoff generation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging import append_jsonl
from .provenance import current_git_commit

SCHEMA_VERSION = 2

# Facts are monotonic: an observation with lower assurance never replaces a
# pinned/verified value.  Keeping the ordering in one place makes doctor and
# source inspection agree on merge semantics.
FACT_PRECEDENCE = {"unknown": 0, "inferred": 1, "measured": 2, "verified": 3}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RunState:
    schema_version: int
    run_id: str
    current_phase: str = "discovery"
    phase_status: str = "pending"
    source_revision: str = "unknown"
    selected_profile: str | None = None
    last_successful_command: str | None = None
    next_exact_command: str | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, str] = field(default_factory=lambda: {"created": utc_now(), "updated": utc_now()})
    terminal_state: str | None = None
    active_blocker: str | None = None
    prediction_contract: dict[str, Any] = field(default_factory=dict)
    parent_run_id: str | None = None
    source_config_hash: str | None = None
    source_index_hash: str | None = None
    code_commit: str | None = None

    @classmethod
    def new(cls, run_id: str) -> RunState:
        return cls(schema_version=SCHEMA_VERSION, run_id=run_id, code_commit=current_git_commit())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunState:
        data = dict(payload)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("run_id", "unknown")
        data.setdefault("timestamps", {"created": utc_now(), "updated": utc_now()})
        # Ignore forward-compatible fields from a newer writer, while filling
        # defaults for fields introduced by this schema.
        return cls(**{field_name: data[field_name] for field_name in cls.__dataclass_fields__ if field_name in data})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStore:
    """A run directory with atomic state and append-only command/event logs."""

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("evidence", "predictions", "logs", "metrics", "reports", "artifacts", "partitions", "layer-checkpoints", "capture"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.commands_path = self.run_dir / "commands.jsonl"
        self.events_path = self.run_dir / "events.jsonl"

    @property
    def run_id(self) -> str:
        if self.state_path.exists():
            return self.load().run_id
        return self.run_dir.name

    def load(self) -> RunState:
        if not self.state_path.exists():
            state = RunState.new(self.run_dir.name)
            self.save(state)
            return state
        return RunState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))

    def save(self, state: RunState) -> None:
        # Every newly written state receipt identifies the code that wrote it.
        # This intentionally updates active continuation state only when it is
        # saved; historical run files are never rewritten by this module.
        state.code_commit = current_git_commit()
        state.timestamps["updated"] = utc_now()
        payload = json.dumps(state.as_dict(), indent=2, sort_keys=True, default=str) + "\n"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="state-", suffix=".json", dir=self.run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def transition(self, **changes: Any) -> RunState:
        state = self.load()
        for key, value in changes.items():
            if not hasattr(state, key):
                raise AttributeError(f"unknown run-state field: {key}")
            setattr(state, key, value)
        # BLOCKED is a resumable condition.  A successful transition that
        # clears the blocker must also clear the legacy BLOCKED marker; final
        # outcomes remain immutable unless explicitly changed by the caller.
        if changes.get("active_blocker") is None and "active_blocker" in changes:
            if state.terminal_state == "BLOCKED":
                state.terminal_state = None
            if state.phase_status == "blocked":
                state.phase_status = "pending"
        if state.terminal_state == "BLOCKED" and state.active_blocker is None:
            state.terminal_state = None
        self.save(state)
        append_jsonl(self.events_path, {"event": "state_transition", "changes": changes})
        return state

    def record_command(self, command: str, *, argv: list[str] | None = None, ok: bool = True, result: Any = None) -> None:
        append_jsonl(
            self.commands_path,
            {"command": command, "argv": argv or [], "ok": ok, "result": result, "code_commit": current_git_commit()},
        )

    def record_event(self, event: str, **details: Any) -> None:
        append_jsonl(self.events_path, {"event": event, **details, "code_commit": current_git_commit()})

    def write_prediction(self, phase: str, contract: Mapping[str, Any], result: str | None = None) -> Path:
        payload = {"phase": phase, "contract": dict(contract), "result": result, "timestamp": utc_now(), "code_commit": current_git_commit()}
        path = self.run_dir / "predictions" / f"{phase}.json"
        atomic_write_json(path, payload)
        state = self.load()
        state.prediction_contract = dict(contract)
        self.save(state)
        return path

    def write_handoff(self, *, next_command: str | None = None, expected_output: str = "", blocker: str | None = None) -> Path:
        state = self.load()
        next_command = next_command or state.next_exact_command or "d2m status --run-dir <run-dir> --json"
        state.next_exact_command = next_command
        state.active_blocker = blocker
        if blocker is None:
            if state.terminal_state == "BLOCKED":
                state.terminal_state = None
            if state.phase_status == "blocked":
                state.phase_status = "pending"
        elif state.terminal_state not in {"SUCCEEDED", "RESEARCH_CANDIDATE", "FAILED_SAFELY"}:
            # Retain the marker for old reports, but it is intentionally not a
            # terminal state for resume logic.
            state.terminal_state = "BLOCKED"
        self.save(state)
        handoff = self.run_dir / "HANDOFF.md"
        lines = [
            f"# Handoff for `{state.run_id}`",
            "",
            f"- Current status: `{state.terminal_state or state.phase_status}`",
            f"- Current phase: `{state.current_phase}`",
            f"- Last completed gate: `{state.last_successful_command or 'none'}`",
            f"- Active blocker: `{blocker or 'none'}`",
            f"- Exact next command: `{next_command}`",
            f"- Expected output: {expected_output or 'see state.json and the latest event log'}",
            f"- Code commit: `{current_git_commit()}`",
            f"- Relevant log: `{self.run_dir / 'events.jsonl'}`",
            f"- Resume command: `d2m run --run-dir {self.run_dir} --resume`",
            "",
        ]
        handoff.write_text("\n".join(lines), encoding="utf-8")
        return handoff


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_fact_ledgers(existing: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """Merge environment/source facts without downgrading assurance.

    A fact is represented as ``{"status": ..., "value": ...}``.  Unknown or
    malformed observations are retained only when no stronger fact exists.
    The function is pure so it can be tested independently of the CLI.
    """

    result: dict[str, Any] = {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in existing.items()}
    for key, candidate_raw in observed.items():
        candidate = dict(candidate_raw) if isinstance(candidate_raw, Mapping) else {"status": "unknown", "value": candidate_raw}
        candidate_status = str(candidate.get("status", "unknown")).lower()
        if candidate_status not in FACT_PRECEDENCE:
            candidate_status = "unknown"
        candidate["status"] = candidate_status
        previous_raw = result.get(str(key))
        previous = dict(previous_raw) if isinstance(previous_raw, Mapping) else {"status": "unknown", "value": previous_raw}
        previous_status = str(previous.get("status", "unknown")).lower()
        if previous_status not in FACT_PRECEDENCE:
            previous_status = "unknown"
        # Only an equal/stronger observation may replace a value.  For equal
        # assurance, prefer a non-null value and merge auxiliary evidence.
        if FACT_PRECEDENCE[candidate_status] >= FACT_PRECEDENCE[previous_status] and (
            candidate.get("value") is not None or previous.get("value") is None
        ):
            merged = dict(previous)
            merged.update(candidate)
            result[str(key)] = merged
        elif str(key) not in result:
            result[str(key)] = candidate
    return result


def atomic_artifact_publish(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> Path:
    """Atomically publish a validated file without overwriting an existing one."""

    src, dst = Path(source), Path(destination)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite completed artifact: {dst}")
    staged = dst.with_name(dst.name + ".publishing")
    if staged.exists():
        staged.unlink()
    shutil.copy2(src, staged)
    os.replace(staged, dst)
    return dst


def bootstrap_run(
    run_dir: str | os.PathLike[str],
    run_id: str | None = None,
    *,
    parent_run_id: str | None = None,
) -> StateStore:
    store = StateStore(run_dir)
    if not store.state_path.exists():
        state = RunState.new(run_id or Path(run_dir).name)
        state.parent_run_id = parent_run_id
        store.save(state)
        store.write_handoff()
    return store
