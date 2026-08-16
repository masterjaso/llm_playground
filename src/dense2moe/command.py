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
from collections import deque
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
    stdout_tail: str = ""
    stderr_tail: str = ""
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    output_lines_dropped: int = 0
    last_child_output_at: str | None = None
    child_output_stale: bool = False

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


class _TailBuffer:
    """Bounded UTF-8 tail retained for a potentially very verbose child."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("tail byte limit must be positive")
        self.max_bytes = int(max_bytes)
        self._chunks: deque[bytes] = deque()
        self._size = 0

    def append(self, value: str) -> None:
        chunk = value.encode("utf-8", errors="replace")
        if len(chunk) >= self.max_bytes:
            self._chunks.clear()
            self._chunks.append(chunk[-self.max_bytes :])
            self._size = self.max_bytes
            return
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self._size > self.max_bytes and self._chunks:
            self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


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
    stdout_log_path: str | Path | None = None,
    stderr_log_path: str | Path | None = None,
    tail_bytes: int = 4_000_000,
    child_output_stale_after: float | None = None,
    emit: Callable[[str], Any] | None = None,
    check: bool = False,
    terminate_grace_seconds: float = 2.0,
) -> CommandResult:
    """Run a non-interactive command with explicit terminal markers.

    ``category`` supplies the normal timeout contract (FAST=60 seconds,
    MEDIUM=5 minutes).  LONG_RUNNING commands may omit a timeout only when
    ``long_running=True``; those commands emit a heartbeat and update the
    optional JSON receipt at least every ``heartbeat_interval`` seconds.
    ``emit`` receives markers and child output, and defaults to stdout.  For
    long-running commands the complete streams are spooled to log files and
    only a bounded UTF-8 tail is retained in the result/receipt.
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
    if tail_bytes <= 0:
        raise ValueError("tail_bytes must be positive")
    if child_output_stale_after is not None and child_output_stale_after <= 0:
        raise ValueError("child_output_stale_after must be positive or None")
    if emit is None:
        emit = print
    heartbeat_file = Path(heartbeat_path) if heartbeat_path is not None else None
    if long_running and heartbeat_file is None:
        raise ValueError("long-running commands require heartbeat_path")
    stale_after = float(child_output_stale_after if child_output_stale_after is not None else max(60.0, heartbeat_interval * 2.0))
    spool_logs = bool(long_running or stdout_log_path is not None or stderr_log_path is not None)
    stdout_log = Path(stdout_log_path) if stdout_log_path is not None else None
    stderr_log = Path(stderr_log_path) if stderr_log_path is not None else None
    if spool_logs and long_running and heartbeat_file is not None:
        stdout_log = stdout_log or heartbeat_file.with_name(heartbeat_file.stem + ".stdout.log")
        stderr_log = stderr_log or heartbeat_file.with_name(heartbeat_file.stem + ".stderr.log")
    started_wall = _utc_now()
    started = time.perf_counter()
    marker = f"__CMD_START__ name={command_name} timestamp={started_wall}"
    _emit_line(emit, marker)

    child_state: dict[str, Any] = {
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "output_lines_dropped": 0,
        "last_monotonic": None,
        "last_wall": None,
    }
    state_lock = threading.Lock()

    def _heartbeat_payload(status: str, *, pid: int | None, elapsed: float) -> dict[str, Any]:
        with state_lock:
            last_monotonic = child_state["last_monotonic"]
            last_wall = child_state["last_wall"]
            stdout_bytes = int(child_state["stdout_bytes"])
            stderr_bytes = int(child_state["stderr_bytes"])
            output_lines_dropped = int(child_state["output_lines_dropped"])
        age = elapsed if last_monotonic is None else max(0.0, time.perf_counter() - float(last_monotonic))
        stale = bool(age >= stale_after)
        return {
            "status": status,
            "terminal_status": status,
            "task": command_name,
            "argv": argv,
            "started": started_wall,
            "last_heartbeat": _utc_now(),
            "last_child_output_at": last_wall,
            "last_child_output_age_seconds": age,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "output_lines_dropped": output_lines_dropped,
            "child_output_stale": stale,
            "completed": 0,
            "total": None,
            "pid": pid,
            "elapsed_seconds": elapsed,
            "returncode": None,
            "stdout_log_path": str(stdout_log) if stdout_log is not None else None,
            "stderr_log_path": str(stderr_log) if stderr_log is not None else None,
        }

    if heartbeat_file is not None:
        _write_json(heartbeat_file, _heartbeat_payload("RUNNING", pid=None, elapsed=0.0))

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

    log_handles: dict[str, Any] = {}
    try:
        for label, path in (("stdout", stdout_log), ("stderr", stderr_log)):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                log_handles[label] = path.open("w", encoding="utf-8")
        process = subprocess.Popen(argv, **popen_kwargs)
    except OSError as exc:
        for handle in log_handles.values():
            handle.close()
        elapsed = time.perf_counter() - started
        result = CommandResult(
            name=command_name,
            argv=tuple(argv),
            status="FAILED",
            returncode=getattr(exc, "errno", 1),
            elapsed_seconds=elapsed,
            started_at=started_wall,
            finished_at=_utc_now(),
            stdout="",
            stderr=str(exc),
            timeout_seconds=timeout,
            heartbeat_path=str(heartbeat_file) if heartbeat_file else None,
            stdout_log_path=str(stdout_log) if stdout_log else None,
            stderr_log_path=str(stderr_log) if stderr_log else None,
        )
        _emit_line(emit, f"__CMD_FAILED__ name={command_name} rc={result.returncode} elapsed={elapsed:.3f}")
        if heartbeat_file is not None:
            _write_json(heartbeat_file, _heartbeat_payload("FAILED", pid=None, elapsed=elapsed) | {"returncode": result.returncode})
        if check:
            raise GuardedCommandError(result)
        return result

    if heartbeat_file is not None:
        _write_json(heartbeat_file, _heartbeat_payload("RUNNING", pid=process.pid, elapsed=0.0))

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=256)
    stop_event = threading.Event()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_tail = _TailBuffer(tail_bytes)
    stderr_tail = _TailBuffer(tail_bytes)

    def _enqueue(event: tuple[str, str | None]) -> None:
        # Full spool files and bounded tails are authoritative for long jobs.
        # Do not apply pipe backpressure merely because the optional live
        # output consumer is slower than a verbose child; dropping a live line
        # is preferable to stalling the child.  Short commands remain lossless.
        if spool_logs:
            try:
                output_queue.put_nowait(event)
            except queue.Full:
                with state_lock:
                    child_state["output_lines_dropped"] += 1
            return
        while not stop_event.is_set():
            try:
                output_queue.put(event, timeout=0.1)
                return
            except queue.Full:
                continue

    def _drain(stream: Any, label: str) -> None:
        handle = log_handles.get(label)
        try:
            for line in iter(stream.readline, ""):
                if handle is not None:
                    handle.write(line)
                    handle.flush()
                encoded_size = len(line.encode("utf-8", errors="replace"))
                with state_lock:
                    child_state[f"{label}_bytes"] += encoded_size
                    child_state["last_monotonic"] = time.perf_counter()
                    child_state["last_wall"] = _utc_now()
                if spool_logs:
                    (stdout_tail if label == "stdout" else stderr_tail).append(line)
                _enqueue((label, line))
        finally:
            _enqueue((label, None))

    threads = [
        threading.Thread(target=_drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    closed_streams = 0
    last_heartbeat = started
    timed_out = False
    timeout_seen_at: float | None = None
    process_exit_seen_at: float | None = None
    while True:
        now = time.perf_counter()
        if long_running and now - last_heartbeat >= heartbeat_interval:
            elapsed = now - started
            payload = _heartbeat_payload("RUNNING", pid=process.pid, elapsed=elapsed)
            age = payload["last_child_output_age_seconds"]
            _emit_line(
                emit,
                "__HEARTBEAT__ "
                f"task={command_name} elapsed={elapsed:.3f} "
                f"progress=unknown last_child_output_age_seconds={age} "
                f"stdout_bytes={payload['stdout_bytes']} stderr_bytes={payload['stderr_bytes']} "
                f"child_output_stale={payload['child_output_stale']}",
            )
            if heartbeat_file is not None:
                _write_json(heartbeat_file, payload)
            last_heartbeat = now
        try:
            label, line = output_queue.get(timeout=0.1)
            if line is None:
                closed_streams += 1
            else:
                if label == "stdout":
                    if not spool_logs:
                        stdout_lines.append(line)
                else:
                    if not spool_logs:
                        stderr_lines.append(line)
                _emit_line(emit, line.rstrip("\r\n"))
        except queue.Empty:
            pass
        process_returncode = process.poll()
        if process_returncode is not None and process_exit_seen_at is None:
            process_exit_seen_at = time.perf_counter()
        if timeout is not None and now - started >= float(timeout) and process_returncode is None:
            timed_out = True
            timeout_seen_at = timeout_seen_at or time.perf_counter()
            _terminate_process_tree(process, grace_seconds=terminate_grace_seconds)
            process_returncode = process.poll()
            if process_returncode is not None and process_exit_seen_at is None:
                process_exit_seen_at = time.perf_counter()
        if process_returncode is not None and closed_streams >= 2:
            break
        # A child can exit while a descendant or a faulty reader keeps one
        # pipe open.  Track exit time explicitly so both bounded and
        # heartbeat-only commands have a finite terminal-collection window.
        if process_exit_seen_at is not None and time.perf_counter() - process_exit_seen_at >= 5.0:
            break
        if timeout_seen_at is not None and time.perf_counter() - timeout_seen_at >= 5.0:
            # A platform-specific kill failure must still produce a terminal
            # timeout receipt rather than waiting forever on a broken child.
            break

    stop_event.set()
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    for thread in threads:
        thread.join(timeout=1.0)
    # A platform-specific tree termination failure must not leave a live
    # process behind after the supervisor has emitted its terminal receipt.
    # The normal child-exit path has already populated ``returncode``; this
    # bounded fallback is only for a stubborn timeout path.
    if process.poll() is None:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process, grace_seconds=terminate_grace_seconds)
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    for handle in log_handles.values():
        handle.close()
    elapsed = time.perf_counter() - started
    returncode = process.poll()
    status = "TIMEOUT" if timed_out else "DONE" if returncode == 0 else "FAILED"
    with state_lock:
        stdout_bytes = int(child_state["stdout_bytes"])
        stderr_bytes = int(child_state["stderr_bytes"])
        output_lines_dropped = int(child_state["output_lines_dropped"])
        last_child_output_at = child_state["last_wall"]
        last_monotonic = child_state["last_monotonic"]
    child_output_age = elapsed if last_monotonic is None else max(0.0, time.perf_counter() - float(last_monotonic))
    child_output_stale = bool(child_output_age >= stale_after)
    stdout_value = stdout_tail.text() if spool_logs else "".join(stdout_lines)
    stderr_value = stderr_tail.text() if spool_logs else "".join(stderr_lines)
    result = CommandResult(
        name=command_name,
        argv=tuple(argv),
        status=status,
        returncode=returncode,
        elapsed_seconds=elapsed,
        started_at=started_wall,
        finished_at=_utc_now(),
        stdout=stdout_value,
        stderr=stderr_value,
        timeout_seconds=timeout,
        heartbeat_path=str(heartbeat_file) if heartbeat_file else None,
        stdout_tail=stdout_value,
        stderr_tail=stderr_value,
        stdout_log_path=str(stdout_log) if stdout_log else None,
        stderr_log_path=str(stderr_log) if stderr_log else None,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        output_lines_dropped=output_lines_dropped,
        last_child_output_at=last_child_output_at,
        child_output_stale=child_output_stale,
    )
    if status == "DONE":
        _emit_line(emit, f"__CMD_DONE__ name={command_name} rc={returncode} elapsed={elapsed:.3f}")
    elif status == "TIMEOUT":
        _emit_line(emit, f"__CMD_TIMEOUT__ name={command_name} elapsed={elapsed:.3f}")
    else:
        _emit_line(emit, f"__CMD_FAILED__ name={command_name} rc={returncode} elapsed={elapsed:.3f}")
    if heartbeat_file is not None:
        _write_json(
            heartbeat_file,
            _heartbeat_payload("SUCCESS" if status == "DONE" else status, pid=process.pid, elapsed=elapsed)
            | {
                "terminal_status": status,
                "returncode": returncode,
                "last_child_output_at": last_child_output_at,
                "last_child_output_age_seconds": child_output_age,
                "child_output_stale": child_output_stale,
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
