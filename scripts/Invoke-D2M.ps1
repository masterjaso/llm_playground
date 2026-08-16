param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command,
    [ValidateSet("FAST", "MEDIUM", "LONG_RUNNING")]
    [string]$Category = "FAST",
    [double]$Timeout = 0,
    [switch]$LongRunning,
    [double]$HeartbeatInterval = 30,
    [string]$HeartbeatPath = "",
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
    $guarded = @(
        (Join-Path $projectRoot "scripts\run_guarded_command.py"),
        "--name", "d2m-$Command",
        "--category", $Category,
        "--heartbeat-interval", $HeartbeatInterval
    )
    if ($Timeout -gt 0) { $guarded += @("--timeout", $Timeout) }
    if ($LongRunning) { $guarded += "--long-running" }
    if ($HeartbeatPath) { $guarded += @("--heartbeat-path", $HeartbeatPath) }
    $guarded += "--"
    $guarded += @($python, "-m", "dense2moe.cli", $Command) + $Arguments
    # The reusable Python wrapper owns markers, timeout enforcement, and
    # process-tree termination; unsupervised native waits are not used.
    & $python @guarded
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
