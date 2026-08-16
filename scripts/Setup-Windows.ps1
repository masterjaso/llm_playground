param(
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$SkipTorchInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

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
        $launcherArgs = if ($launcherName -eq "py.exe") { @("-3") } else { @() }
        Invoke-GuardedPython -Interpreter $launcher.Source -Name "create-venv" -ChildArgs ($launcherArgs + @("-m", "venv", ".venv"))
    }

    Invoke-Python -Name "pip-bootstrap" -Args @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    # Install project/runtime/dev dependencies without allowing the generic
    # torch requirement to replace the CUDA wheel below.
    Invoke-Python -Name "pip-project-runtime-dev" -Args @("-m", "pip", "install", "-e", ".[runtime,dev]")
    if (-not $SkipTorchInstall) {
        Invoke-Python -Name "pip-install-cuda-torch" -Args @("-m", "pip", "install", "--index-url", $CudaIndexUrl, "torch>=2.3")
    }
    # Install ML packages after torch so dependency resolution cannot silently
    # downgrade it to a CPU wheel.
    Invoke-Python -Name "pip-project-ml" -Args @("-m", "pip", "install", "-e", ".[ml]")
    Invoke-Python -Name "pip-install-transformers" -Args @("-m", "pip", "install", "transformers", "accelerate", "safetensors")
    Invoke-Python -Name "pip-show-environment" -Args @("-m", "pip", "show", "dense2moe", "torch", "transformers", "accelerate", "safetensors", "numpy", "pytest", "ruff", "mypy")
    Write-Host "WINDOWS_PYTHON_READY"
    Write-Host "Interpreter: $venvPython"
}
finally {
    Pop-Location
}
