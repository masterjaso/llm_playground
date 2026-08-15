param(
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu130",
    [switch]$SkipTorchInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    $process = Start-Process -FilePath $venvPython -ArgumentList $Args -Wait -PassThru -NoNewWindow
    if ($process.ExitCode -ne 0) {
        throw "Windows Python command failed: $($Args -join ' ')"
    }
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
        if ($launcherName -eq "py.exe") {
            $process = Start-Process -FilePath $launcher.Source -ArgumentList @("-3", "-m", "venv", ".venv") -Wait -PassThru -NoNewWindow
        }
        else {
            $process = Start-Process -FilePath $launcher.Source -ArgumentList @("-m", "venv", ".venv") -Wait -PassThru -NoNewWindow
        }
        if ($process.ExitCode -ne 0) { throw "Unable to create .venv" }
    }

    Invoke-Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    # Install project/runtime/dev dependencies without allowing the generic
    # torch requirement to replace the CUDA wheel below.
    Invoke-Python @("-m", "pip", "install", "-e", ".[runtime,dev]")
    if (-not $SkipTorchInstall) {
        Invoke-Python @("-m", "pip", "install", "--index-url", $CudaIndexUrl, "torch>=2.3")
    }
    # Install ML packages after torch so dependency resolution cannot silently
    # downgrade it to a CPU wheel.
    Invoke-Python @("-m", "pip", "install", "-e", ".[ml]")
    Invoke-Python @("-m", "pip", "install", "transformers", "accelerate", "safetensors")
    Invoke-Python @("-m", "pip", "show", "dense2moe", "torch", "transformers", "accelerate", "safetensors", "numpy", "pytest", "ruff", "mypy")
    Write-Host "WINDOWS_PYTHON_READY"
    Write-Host "Interpreter: $venvPython"
}
finally {
    Pop-Location
}
