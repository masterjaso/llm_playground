from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dense2moe import hardware
from dense2moe.hardware import (
    load_recovery_pin,
    probe_torch,
    run_environment_doctor,
    write_runtime_lock,
)


@pytest.mark.skipif(probe_torch().get("installed") is True, reason="requires an interpreter without torch")
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
    assert any("PLATFORM_POLICY_VIOLATION" in item for item in result["blockers"])
    assert any(item.startswith("p16_forward:") for item in result["blockers"])
    assert any(item.startswith("dense_teacher_ffn_forward:") for item in result["blockers"])


def test_recovery_pin_reads_historical_receipt() -> None:
    pin = load_recovery_pin()

    assert pin["status"] == "available"
    assert pin["torch_version"] == "2.13.0+cu130"
    assert pin["cuda_runtime"] == "13.0"
    assert pin["expected_gpu_count"] == 2


@pytest.mark.skipif(
    os.name == "nt" and bool(probe_torch().get("cuda_available")),
    reason="native CUDA runtime is expected to pass discovery",
)
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
    assert "wsl" not in payload["next_exact_command"].lower()
    assert "powershell.exe" in payload["next_exact_command"]
    assert (run_dir / "environment.json").exists()


def _capable_windows_environment(root: Path) -> dict[str, object]:
    interpreter = root / ".venv" / "Scripts" / "python.exe"
    return {
        "platform": {
            "system": "Windows",
            "release": "11",
            "version": "10.0.26200",
            "python": "3.12.6",
            "python_executable": str(interpreter),
        },
        "windows": True,
        "wsl": False,
        "python_executable": str(interpreter),
        "python_version": "3.12.6",
        "selected_training_gpus": [0, 1],
        "torch": {
            "installed": True,
            "import_ok": True,
            "version": "2.13.0+cu130",
            "compiled_cuda": "13.0",
            "cuda_available": True,
            "bf16_tensor": True,
            "small_cuda_gemm": True,
            "devices": [
                {"index": 0, "name": "GPU A", "bf16_supported": True, "small_cuda_gemm": True},
                {"index": 1, "name": "GPU B", "bf16_supported": True, "small_cuda_gemm": True},
            ],
        },
        "packages": {"safetensors": {"version": "0.5.0"}},
    }


def test_historical_recovery_mismatch_is_advisory_and_lock_drift_is_blocking(tmp_path, monkeypatch) -> None:
    environment = _capable_windows_environment(tmp_path)
    monkeypatch.setattr(hardware, "_probe_safetensors", lambda _path: {"status": "PASS", "ok": True})
    monkeypatch.setattr(hardware, "_probe_d2m_checkpoint", lambda _path: {"status": "PASS", "ok": True})
    monkeypatch.setattr(hardware, "_probe_torch_models", lambda _torch, *, device: {
        "p16_forward": {"status": "PASS", "ok": True, "device": device},
        "dense_teacher_ffn_forward": {"status": "PASS", "ok": True, "device": device},
    })
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    recovery = tmp_path / "historical.json"
    recovery.write_text(json.dumps({"platform": "historical", "python": "3.12.5", "torch": "2.12.0", "torch_cuda": "12.8", "devices": []}), encoding="utf-8")
    lock_path = tmp_path / "runs" / "windows-runtime-lock.json"

    doctor = run_environment_doctor(environment=environment, repo_root=tmp_path, recovery_pin_path=recovery, runtime_lock_path=lock_path)

    assert doctor["ok"] is True
    assert doctor["runtime_lock"]["status"] == "MISSING"
    assert doctor["recovery_profile_advisory"] is True
    lock = write_runtime_lock(environment, doctor, repo_root=tmp_path, path=lock_path, code_commit="abc123")
    assert lock["status"] == "APPROVED"
    assert hardware.load_runtime_lock(lock_path)["status"] == "LOCKED"

    environment["torch"]["version"] = "2.13.1+cu130"  # type: ignore[index]
    drifted = run_environment_doctor(environment=environment, repo_root=tmp_path, recovery_pin_path=recovery, runtime_lock_path=lock_path)

    assert drifted["ok"] is False
    assert drifted["runtime_lock"]["status"] == hardware.WINDOWS_RUNTIME_DRIFT
    assert any(item.startswith("runtime_lock: WINDOWS_RUNTIME_DRIFT") for item in drifted["blockers"])

    lock_path.write_text("{\"schema_version\": 1, \"lock_sha256\": \"forged\"}", encoding="utf-8")
    invalid = run_environment_doctor(environment=environment, repo_root=tmp_path, recovery_pin_path=recovery, runtime_lock_path=lock_path)
    assert invalid["runtime_lock"]["status"] == hardware.WINDOWS_RUNTIME_DRIFT
    assert any(item.startswith("runtime_lock: WINDOWS_RUNTIME_DRIFT") for item in invalid["blockers"])

    malformed = {"schema_version": 1, "runtime_fingerprint": ["not", "a", "mapping"]}
    malformed["lock_sha256"] = hardware._canonical_sha256(malformed)
    lock_path.write_text(json.dumps(malformed), encoding="utf-8")
    malformed_result = hardware.check_runtime_lock(environment, lock_path)
    assert malformed_result["status"] == hardware.WINDOWS_RUNTIME_DRIFT
    assert malformed_result["drift"] == ["runtime_fingerprint"]


def test_setup_and_test_share_current_lock_defaults() -> None:
    setup = Path("scripts/Setup-Windows.ps1").read_text(encoding="utf-8")
    test = Path("scripts/Test-Windows-Cuda.ps1").read_text(encoding="utf-8")
    assert 'runs\\windows-runtime-lock.json' in setup
    assert 'runs\\windows-runtime-lock.json' in test
    assert 'ForceRecovery' in setup
    assert 'WINDOWS_RUNTIME_LOCK_BACKED_UP' in setup
    assert 'WINDOWS_RUNTIME_LOCK_REAPPROVED' in setup
    assert 'windows-cuda-v3.json' not in test
    assert 'WINDOWS_RUNTIME_DRIFT' in test
    assert 'WINDOWS_RUNTIME_UNRESOLVED' in test
