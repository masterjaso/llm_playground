"""Bounded, non-interactive command execution.

The research workflow launches a mixture of quick repository diagnostics and
deliberately long CUDA jobs.  ``subprocess.run`` by itself is a poor control
plane primitive for that workflow: a stalled child provides no durable signal
that it is still alive.  This module provides one small cross-platform
wrapper with explicit start/terminal markers and optional heartbeat receipts.

The wrapper is intentionally independent of the CLI state store so scripts,
PowerShell entry points, and tests can use the same contract.  It never
invokes a shell by default and disables the common Git/GitHub prompt and
pager environment variables.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TIMEOUTS: dict[str, float | None] = {
    "FAST": 60.0,
    "MEDIUM": 300.0,
    # LONG_RUNNING still accepts an explicit timeout.  ``None`` means the
    # caller has deliberately opted into heartbeat-only supervision.
    "LONG_RUNNING": None,
}


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _command_argv(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        # ``posix=False`` preserves Windows paths and quoting when this helper
        # is called from PowerShell, while still being useful on POSIX hosts.
        return shlex.split(command, posix=(os.name != "nt"))
    values = [str(value) for value in command]
    if not values:
        raise ValueError("command must not be empty")
    return values


def _safe_name(value: str | None, argv: Sequence[str]) -> str:
    if value:
        return str(value)
    return Path(argv[0]).name or "command"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class CommandResult:
    """Machine-readable terminal state for a guarded process."""

    name: str
    argv: tuple[str, ...]
    status: str
    returncode: int | None
    elapsed_seconds: float
    started_at: str
    finished_at: str
    stdout: str
    stderr: str
    timeout_seconds: float | None
    heartbeat_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "DONE" and self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok}


class GuardedCommandError(RuntimeError):
    """Raised when ``run_guarded(..., check=True)`` does not succeed."""

    def __init__(self, result: CommandResult):
        self.result = result
        super().__init__(
            f"guarded command {result.name!r} ended with {result.status} "
            f"(returncode={result.returncode}, elapsed={result.elapsed_seconds:.3f}s)"
        )


def _default_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    environment = dict(os.environ)
    # These variables make unattended Git/GitHub calls fail fast rather than
    # waiting for credentials, an editor, or a pager on stdin.
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "GH_PAGER": "cat", "PAGER": "cat"})
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _terminate_process_tree(process: subprocess.Popen[str], *, grace_seconds: float = 2.0) -> None:
    """Terminate a child and descendants without relying on a shell."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        # ``taskkill`` is available on supported Windows installations and is
        # the only reliable way to stop a Python/CUDA descendant tree.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, grace_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass


def _emit_line(emit: Callable[[str], Any], line: str) -> None:
    try:
        emit(line)
    except (BrokenPipeError, OSError, RuntimeError, TypeError, ValueError):
        # Logging must never keep the child alive or mask its terminal state.
        pass


def run_guarded(
    command: str | Sequence[str],
    *,
    name: str | None = None,
    category: str = "FAST",
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    long_running: bool = False,
    heartbeat_interval: float = 30.0,
    heartbeat_path: str | Path | None = None,
    emit: Callable[[str], Any] | None = None,
    check: bool = False,
    terminate_grace_seconds: float = 2.0,
) -> CommandResult:
    """Run a non-interactive command with explicit terminal markers.

    ``category`` supplies the normal timeout contract (FAST=60 seconds,
    MEDIUM=5 minutes).  LONG_RUNNING commands may omit a timeout only when
    ``long_running=True``; those commands emit a heartbeat and update the
    optional JSON receipt at least every ``heartbeat_interval`` seconds.
    ``emit`` receives markers and child output, and defaults to stdout.
    """

    category = str(category).upper()
    if category not in TIMEOUTS:
        raise ValueError(f"unknown command category: {category!r}")
    # A caller that opts into heartbeat supervision without spelling out a
    # category is almost always launching a deliberate long job.  Promote the
    # implicit FAST default so the convenience API cannot accidentally kill a
    # training/capture process after one minute.  Explicit ``timeout`` still
    # wins, so bounded long-running diagnostics remain possible.
    if long_running and category == "FAST" and timeout is None:
        category = "LONG_RUNNING"
    argv = _command_argv(command)
    command_name = _safe_name(name, argv)
    if timeout is None:
        timeout = TIMEOUTS[category]
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("timeout must be positive or None")
    if category == "LONG_RUNNING" and not long_running:
        raise ValueError("LONG_RUNNING commands require long_running=True for heartbeat supervision")
    if long_running and heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    if emit is None:
        emit = print
    heartbeat_file = Path(heartbeat_path) if heartbeat_path is not None else None
    if long_running and heartbeat_file is None:
        raise ValueError("long-running commands require heartbeat_path")
    started_wall = _utc_now()
    started = time.perf_counter()
    marker = f"__CMD_START__ name={command_name} timestamp={started_wall}"
    _emit_line(emit, marker)
    if heartbeat_file is not None:
        _write_json(
            heartbeat_file,
            {
                "status": "RUNNING",
                "task": command_name,
                "argv": argv,
                "started": started_wall,
                "last_heartbeat": started_wall,
                "completed": 0,
                "total": None,
                "pid": None,
            },
        )

    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd is not None else None,
        "env": _default_environment(env),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP is enough for cooperative children; the
        # timeout path additionally uses taskkill /T for non-cooperative trees.
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(argv, **popen_kwargs)
    except OSError as exc:
        elapsed = time.perf_counter() - started
        result = CommandResult(command_name, tuple(argv), "FAILED", getattr(exc, "errno", 1), elapsed, started_wall, _utc_now(), "", str(exc), timeout, str(heartbeat_file) if heartbeat_file else None)
        _emit_line(emit, f"__CMD_FAILED__ name={command_name} rc={result.returncode} elapsed={elapsed:.3f}")
        if heartbeat_file is not None:
            _write_json(
                heartbeat_file,
                {
                    "status": "FAILED",
                    "terminal_status": "FAILED",
                    "task": command_name,
                    "started": started_wall,
                    "last_heartbeat": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                },
            )
        if check:
            raise GuardedCommandError(result)
        return result

    if heartbeat_file is not None:
        _write_json(heartbeat_file, {"status": "RUNNING", "task": command_name, "argv": argv, "started": started_wall, "last_heartbeat": _utc_now(), "completed": 0, "total": None, "pid": process.pid})

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def _drain(stream: Any, label: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((label, line))
        finally:
            output_queue.put((label, None))

    threads = [threading.Thread(target=_drain, args=(process.stdout, "stdout"), daemon=True), threading.Thread(target=_drain, args=(process.stderr, "stderr"), daemon=True)]
    for thread in threads:
        thread.start()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    closed_streams = 0
    last_heartbeat = started
    timed_out = False
    while True:
        now = time.perf_counter()
        if long_running and now - last_heartbeat >= heartbeat_interval:
            elapsed = now - started
            heartbeat_timestamp = _utc_now()
            _emit_line(emit, f"__HEARTBEAT__ task={command_name} elapsed={elapsed:.3f} progress=unknown")
            if heartbeat_file is not None:
                _write_json(heartbeat_file, {"status": "RUNNING", "task": command_name, "started": started_wall, "last_heartbeat": heartbeat_timestamp, "elapsed_seconds": elapsed, "completed": 0, "total": None, "pid": process.pid})
            last_heartbeat = now
        try:
            label, line = output_queue.get(timeout=0.1)
            if line is None:
                closed_streams += 1
            else:
                if label == "stdout":
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)
                _emit_line(emit, line.rstrip("\r\n"))
        except queue.Empty:
            pass
        if timeout is not None and now - started >= float(timeout) and process.poll() is None:
            timed_out = True
            _terminate_process_tree(process, grace_seconds=terminate_grace_seconds)
        if process.poll() is not None and closed_streams >= 2:
            break
        # A stream reader can be delayed very slightly after process exit; do
        # not spin forever if it failed to enqueue its sentinel.
        if process.poll() is not None and now - started > (float(timeout) + 5.0 if timeout is not None else now - started + 5.0):
            break
    for thread in threads:
        thread.join(timeout=0.2)
    elapsed = time.perf_counter() - started
    returncode = process.returncode
    status = "TIMEOUT" if timed_out else "DONE" if returncode == 0 else "FAILED"
    result = CommandResult(command_name, tuple(argv), status, returncode, elapsed, started_wall, _utc_now(), "".join(stdout_lines), "".join(stderr_lines), timeout, str(heartbeat_file) if heartbeat_file else None)
    if status == "DONE":
        _emit_line(emit, f"__CMD_DONE__ name={command_name} rc={returncode} elapsed={elapsed:.3f}")
    elif status == "TIMEOUT":
        _emit_line(emit, f"__CMD_TIMEOUT__ name={command_name} elapsed={elapsed:.3f}")
    else:
        _emit_line(emit, f"__CMD_FAILED__ name={command_name} rc={returncode} elapsed={elapsed:.3f}")
    if heartbeat_file is not None:
        _write_json(
            heartbeat_file,
            {
                # The command stream uses DONE/FAILED/TIMEOUT markers.  The
                # durable progress receipt follows the takeover contract's
                # SUCCESS/FAILED/TIMEOUT vocabulary while retaining the exact
                # terminal marker state for machine consumers.
                "status": "SUCCESS" if status == "DONE" else status,
                "terminal_status": status,
                "task": command_name,
                "started": started_wall,
                "last_heartbeat": _utc_now(),
                "elapsed_seconds": elapsed,
                "returncode": returncode,
                "pid": process.pid,
            },
        )
    if check and not result.ok:
        raise GuardedCommandError(result)
    return result


# Friendly aliases for callers that prefer a noun or a shorter verb.
guarded_command = run_guarded
run_command = run_guarded


__all__ = [
    "TIMEOUTS",
    "CommandResult",
    "GuardedCommandError",
    "guarded_command",
    "run_command",
    "run_guarded",
]
