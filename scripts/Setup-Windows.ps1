param(
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu130",
    [string]$TorchVersion = "2.13.0",
    [string]$PythonVersion = "3.12",
    [string]$EnvironmentReceipt = "",
    [switch]$SkipTorchInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not $EnvironmentReceipt) {
    $EnvironmentReceipt = Join-Path $projectRoot "runs\windows-environment-pin.json"
}

function Invoke-GuardedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Interpreter,
        [Parameter(Mandatory = $true)][string]$Name,
        [ValidateSet("FAST", "MEDIUM", "LONG_RUNNING")][string]$Category = "MEDIUM",
        [double]$Timeout = 300,
        [Parameter(Mandatory = $true)][string[]]$ChildArgs
    )
    $wrapper = Join-Path $projectRoot "scripts\run_guarded_command.py"
    $guarded = @($wrapper, "--name", $Name, "--category", $Category, "--timeout", $Timeout, "--")
    $guarded += @($Interpreter) + $ChildArgs
    & $Interpreter @guarded
    if ($LASTEXITCODE -ne 0) {
        throw "Guarded Windows Python command failed: $Name ($($ChildArgs -join ' '))"
    }
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    Invoke-GuardedPython -Interpreter $venvPython -Name $Name -ChildArgs $Args
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        $knownPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        if (Test-Path -LiteralPath $knownPython -PathType Leaf) {
            $launcher = @{ Source = $knownPython }
        }
        else {
            $launcher = Get-Command py -ErrorAction SilentlyContinue
            if (-not $launcher) {
                $launcher = Get-Command python -ErrorAction SilentlyContinue
            }
        }
        if (-not $launcher) {
            throw "A native Windows Python 3.10+ launcher is required to create .venv. Install Python from python.org, then rerun this script."
        }
        $launcherName = [System.IO.Path]::GetFileName($launcher.Source).ToLowerInvariant()
        $launcherArgs = if ($launcherName -eq "py.exe") { @("-$PythonVersion") } else { @() }
        Invoke-GuardedPython -Interpreter $launcher.Source -Name "create-venv" -ChildArgs ($launcherArgs + @("-m", "venv", ".venv"))
    }

    Invoke-Python -Name "pip-bootstrap" -Args @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    # Install project/runtime/dev dependencies without allowing the generic
    # torch requirement to replace the CUDA wheel below.
    Invoke-Python -Name "pip-project-runtime-dev" -Args @("-m", "pip", "install", "-e", ".[runtime,dev]")
    if (-not $SkipTorchInstall) {
        # Keep the previously successful CUDA wheel pinned.  A floating torch
        # requirement can silently replace it with a CPU or newer wheel.
        Invoke-Python -Name "pip-install-cuda-torch" -Args @("-m", "pip", "install", "--index-url", $CudaIndexUrl, "torch==$TorchVersion")
    }
    # Install ML packages after torch so dependency resolution cannot silently
    # downgrade it to a CPU wheel.
    Invoke-Python -Name "pip-project-ml" -Args @("-m", "pip", "install", "-e", ".[ml]")
    Invoke-Python -Name "pip-install-transformers" -Args @("-m", "pip", "install", "transformers", "accelerate", "safetensors")
    Invoke-Python -Name "pip-show-environment" -Args @("-m", "pip", "show", "dense2moe", "torch", "transformers", "accelerate", "safetensors", "numpy", "pytest", "ruff", "mypy")
    $pinCode = @'
import json
import sys
from pathlib import Path
from dense2moe.hardware import collect_environment

target = Path(sys.argv[1])
payload = collect_environment()
payload["setup_request"] = {
    "python_version": "__PYTHON_VERSION__",
    "torch_version": "__TORCH_VERSION__",
    "cuda_index_url": "__CUDA_INDEX_URL__",
    "torch_requirement": "torch==__TORCH_VERSION__",
}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
print(f"WINDOWS_ENVIRONMENT_PIN_WRITTEN: {target}")
'@
    $pinCode = $pinCode.Replace("__PYTHON_VERSION__", $PythonVersion.Replace("'", "''"))
    $pinCode = $pinCode.Replace("__TORCH_VERSION__", $TorchVersion.Replace("'", "''"))
    $pinCode = $pinCode.Replace("__CUDA_INDEX_URL__", $CudaIndexUrl.Replace("'", "''"))
    Invoke-Python -Name "write-environment-pin" -Args @("-c", $pinCode, $EnvironmentReceipt)
    Write-Host "WINDOWS_PYTHON_READY"
    Write-Host "Interpreter: $venvPython"
    Write-Host "Environment receipt: $EnvironmentReceipt"
}
finally {
    Pop-Location
}
