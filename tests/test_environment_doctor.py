from __future__ import annotations

import json
import subprocess
import sys

from dense2moe.hardware import load_recovery_pin, probe_torch, run_environment_doctor


def test_torch_probe_is_structured_when_torch_is_unavailable() -> None:
    result = probe_torch()

    assert result["installed"] is False
    assert result["cuda_available"] is False
    assert result["devices"] == []
    assert result["reason"]


def test_doctor_fails_closed_with_named_no_cuda_blockers(tmp_path) -> None:
    environment = {
        "platform": {"system": "Linux", "python": sys.version},
        "windows": False,
        "wsl": True,
        "torch": {
            "installed": True,
            "import_ok": True,
            "version": "2.13.0+cpu",
            "compiled_cuda": None,
            "cuda_available": False,
            "devices": [],
            "reason": "torch.cuda.is_available() is false",
        },
    }

    result = run_environment_doctor(environment=environment, repo_root=tmp_path, recovery_pin_path=tmp_path / "missing.json")

    assert result["status"] == "BLOCKED"
    assert result["ok"] is False
    assert any(item.startswith("cuda_available:") for item in result["blockers"])
    assert any(item.startswith("p16_forward:") for item in result["blockers"])
    assert any(item.startswith("dense_teacher_ffn_forward:") for item in result["blockers"])


def test_recovery_pin_reads_historical_receipt() -> None:
    pin = load_recovery_pin()

    assert pin["status"] == "available"
    assert pin["torch_version"] == "2.13.0+cu130"
    assert pin["cuda_runtime"] == "13.0"
    assert pin["expected_gpu_count"] == 2


def test_cli_doctor_returns_blocked_json_without_cuda(tmp_path) -> None:
    run_dir = tmp_path / "doctor-run"
    completed = subprocess.run(
        [sys.executable, "-m", "dense2moe.cli", "doctor", "--run-dir", str(run_dir), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["environment_doctor"]["status"] == "BLOCKED"
    assert (run_dir / "environment.json").exists()
