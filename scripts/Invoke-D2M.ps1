param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = $null
if ($env:D2M_PYTHON) {
    $python = $env:D2M_PYTHON
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "D2M_PYTHON points to a missing interpreter: $python"
    }
}
if (-not $python) {
    $candidate = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $python = $candidate
    }
}
if (-not $python) {
    throw "No project-local Windows Python was found. Run scripts\Setup-Windows.ps1 or set D2M_PYTHON to an explicit interpreter. Scientific execution refuses arbitrary global Python."
}
Push-Location $projectRoot
try {
    $process = Start-Process -FilePath $python -ArgumentList (@("-m", "dense2moe.cli", $Command) + $Arguments) -Wait -PassThru -NoNewWindow
    exit $process.ExitCode
}
finally {
    Pop-Location
}
