param(
    [Alias("Receipt")][string]$EnvironmentReceipt = "",
    [string]$RuntimeLock = "",
    [switch]$Lightweight
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Missing .venv\Scripts\python.exe. Run scripts\Setup-Windows.ps1 first."
}
if (-not $EnvironmentReceipt) {
    $EnvironmentReceipt = Join-Path $projectRoot "runs\windows-environment-receipt.json"
}
if (-not $RuntimeLock) {
    $RuntimeLock = Join-Path $projectRoot "runs\windows-runtime-lock.json"
}

Push-Location $projectRoot
try {
    $script = @'
import json
import sys
from pathlib import Path

from dense2moe.hardware import (
    collect_environment,
    load_runtime_lock,
    run_environment_doctor,
    write_runtime_lock,
)

receipt_path = Path(r'''__ENVIRONMENT_RECEIPT__''')
runtime_lock_path = Path(r'''__RUNTIME_LOCK__''')
environment = collect_environment(repo_root=Path.cwd())
environment["environment_receipt_path"] = str(receipt_path)
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text(json.dumps(environment, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
doctor = run_environment_doctor(
    environment=environment,
    repo_root=Path.cwd(),
    runtime_lock_path=runtime_lock_path,
)
if not doctor["ok"]:
    status = "WINDOWS_RUNTIME_DRIFT" if any("WINDOWS_RUNTIME_DRIFT" in item for item in doctor.get("blockers", [])) else "WINDOWS_RUNTIME_UNRESOLVED"
    blocked = {
        "status": status,
        "environment": environment,
        "doctor": doctor,
        "blockers": doctor.get("blockers", []),
        "runtime_lock": load_runtime_lock(runtime_lock_path),
    }
    print(json.dumps(blocked, indent=2, sort_keys=True, default=str))
    raise SystemExit("WINDOWS_CUDA_NOT_READY: environment capability gate is blocked")

lock = load_runtime_lock(runtime_lock_path)
if lock.get("status") == "MISSING":
    lock = write_runtime_lock(
        environment,
        doctor,
        repo_root=Path.cwd(),
        path=runtime_lock_path,
        environment_receipts=[receipt_path],
    )
payload = {
    "status": "WINDOWS_RUNTIME_LOCKED",
    "environment": environment,
    "doctor": doctor,
    "runtime_lock": lock,
    "lightweight": bool(__LIGHTWEIGHT__),
}
print(json.dumps(payload, indent=2, sort_keys=True, default=str))
'@
    $script = $script.Replace("__ENVIRONMENT_RECEIPT__", $EnvironmentReceipt.Replace("'", "''"))
    $script = $script.Replace("__RUNTIME_LOCK__", $RuntimeLock.Replace("'", "''"))
    $lightweightValue = if ($Lightweight) { "True" } else { "False" }
    $script = $script.Replace("__LIGHTWEIGHT__", $lightweightValue)
    $temporaryScript = Join-Path $env:TEMP ("d2m-cuda-gate-" + [guid]::NewGuid().ToString("N") + ".py")
    Set-Content -LiteralPath $temporaryScript -Value $script -Encoding UTF8
    try {
        $wrapper = Join-Path $projectRoot "scripts\run_guarded_command.py"
        $guarded = @($wrapper, "--name", "windows-runtime-lock", "--category", "MEDIUM", "--timeout", "300", "--", $venvPython, $temporaryScript)
        & $venvPython @guarded
        if ($LASTEXITCODE -ne 0) { throw "Windows runtime capability gate failed" }
    }
    finally {
        Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
    }
    Write-Host "WINDOWS_RUNTIME_LOCKED"
    Write-Host "Interpreter: $venvPython"
    Write-Host "Environment receipt: $EnvironmentReceipt"
    Write-Host "Runtime lock: $RuntimeLock"
}
finally {
    Pop-Location
}
