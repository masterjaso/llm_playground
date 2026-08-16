from __future__ import annotations

import json
import sys

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
