from __future__ import annotations

import json
import sys
import time

from dense2moe.command import run_guarded


def test_guarded_command_emits_terminal_contract() -> None:
    markers: list[str] = []
    result = run_guarded(
        [sys.executable, "-c", "print('ready')"],
        name="unit-fast",
        category="FAST",
        emit=markers.append,
    )

    assert result.ok
    assert result.status == "DONE"
    assert any(line.startswith("__CMD_START__ name=unit-fast") for line in markers)
    assert any(line.startswith("__CMD_DONE__ name=unit-fast rc=0") for line in markers)
    assert "ready" in result.stdout


def test_guarded_command_times_out_and_writes_receipt(tmp_path) -> None:
    receipt = tmp_path / "progress.json"
    markers: list[str] = []
    result = run_guarded(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        name="unit-timeout",
        timeout=0.05,
        heartbeat_path=receipt,
        emit=markers.append,
    )

    assert result.status == "TIMEOUT"
    assert any(line.startswith("__CMD_TIMEOUT__ name=unit-timeout") for line in markers)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "TIMEOUT"


def test_long_running_default_category_uses_heartbeat_only_receipt(tmp_path) -> None:
    receipt = tmp_path / "success.json"
    result = run_guarded(
        [sys.executable, "-c", "print('long-job-complete')"],
        name="unit-long-success",
        long_running=True,
        heartbeat_path=receipt,
        heartbeat_interval=0.01,
        emit=lambda _line: None,
    )

    assert result.ok
    assert result.timeout_seconds is None
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["terminal_status"] == "DONE"


def test_long_running_spools_full_output_and_retains_bounded_tail(tmp_path) -> None:
    receipt = tmp_path / "verbose.json"
    stdout_log = tmp_path / "stdout.log"
    result = run_guarded(
        [sys.executable, "-c", "print(''.join(f'{i:04d}\\n' for i in range(5000)), end=''); import time; time.sleep(.08)"],
        name="unit-verbose-long",
        long_running=True,
        heartbeat_path=receipt,
        stdout_log_path=stdout_log,
        tail_bytes=128,
        heartbeat_interval=0.02,
        child_output_stale_after=0.05,
        emit=lambda _line: None,
    )

    assert result.ok
    assert result.stdout_log_path == str(stdout_log)
    assert stdout_log.read_text(encoding="utf-8").count("\n") == 5000
    assert len(result.stdout_tail.encode("utf-8")) <= 128
    assert result.stdout_bytes > len(result.stdout_tail.encode("utf-8"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["stdout_bytes"] == result.stdout_bytes
    assert payload["last_child_output_at"] is not None
    assert payload["stderr_bytes"] == 0


def test_long_running_heartbeat_marks_silent_child_stale(tmp_path) -> None:
    receipt = tmp_path / "silent.json"
    result = run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.16)"],
        name="unit-silent-long",
        long_running=True,
        heartbeat_path=receipt,
        heartbeat_interval=0.03,
        child_output_stale_after=0.05,
        emit=lambda _line: None,
    )

    assert result.ok
    assert result.child_output_stale
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["child_output_stale"] is True
    assert payload["last_child_output_age_seconds"] >= 0.05


def test_exit_with_inherited_pipe_cannot_wait_forever(tmp_path) -> None:
    # The grandchild inherits stdout and keeps one drain thread blocked after
    # the direct child has exited.  The supervisor must use process_exit_seen_at
    # and return after its bounded terminal-collection window.
    child_code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5.5)']); "
        "sys.exit(0)"
    )
    started = time.perf_counter()
    result = run_guarded(
        [sys.executable, "-c", child_code],
        name="unit-inherited-pipe",
        category="FAST",
        timeout=10.0,
        emit=lambda _line: None,
    )
    elapsed = time.perf_counter() - started

    assert result.ok
    assert elapsed < 7.0
    # The direct child is gone at this point, but the deliberately leaked
    # grandchild still owns the pipe for a fraction of a second.  Let it exit
    # before pytest tears down its capture file.
    time.sleep(0.75)
