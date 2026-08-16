param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name,
    [ValidateSet("FAST", "MEDIUM", "LONG_RUNNING")]
    [string]$Category = "FAST",
    [double]$Timeout = 0,
    [switch]$LongRunning,
    [double]$HeartbeatInterval = 30,
    [string]$HeartbeatPath = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"
if (-not $Command -or $Command.Count -eq 0) {
    throw "A child command is required"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:D2M_PYTHON) { $env:D2M_PYTHON } else { Join-Path $projectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "No project-local Python found; run scripts\Setup-Windows.ps1 first"
}
$arguments = @((Join-Path $projectRoot "scripts\run_guarded_command.py"), "--name", $Name, "--category", $Category, "--heartbeat-interval", $HeartbeatInterval)
if ($Timeout -gt 0) { $arguments += @("--timeout", $Timeout) }
if ($LongRunning) { $arguments += "--long-running" }
if ($HeartbeatPath) { $arguments += @("--heartbeat-path", $HeartbeatPath) }
$arguments += "--"
$arguments += $Command
Push-Location $projectRoot
try {
    # The Python wrapper owns the timeout, process-tree termination, markers,
    # and receipt contract; keep native waits behind this supervision layer.
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
