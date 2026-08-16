<#
.SYNOPSIS
    Run the required native-Windows smoke matrix for Invoke-GuardedCommand.ps1.

This script deliberately exercises the PowerShell entry point rather than
calling the Python supervisor directly.  It writes a concise JSON receipt and
exits non-zero if any expected marker, timeout, heartbeat, or process-tree
check fails.
#>

param(
    [string]$Output = "runs\windows-guarded-command-smoke.json",
    [int]$HeartbeatSeconds = 60
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "Native Windows smoke tests must run under Windows PowerShell; this host is not Windows."
}
if ($HeartbeatSeconds -lt 60) {
    throw "HeartbeatSeconds must be at least 60 for the long-running smoke case."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$guarded = Join-Path $PSScriptRoot "Invoke-GuardedCommand.ps1"
$python = if ($env:D2M_PYTHON) { $env:D2M_PYTHON } else { Join-Path $projectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "No project-local Windows Python found: $python"
}
$outputPath = if ([IO.Path]::IsPathRooted($Output)) { [IO.Path]::GetFullPath($Output) } else { Join-Path $projectRoot $Output }
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$scratch = Join-Path $outputDir "guarded-command-smoke"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

function Invoke-SmokeCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string[]]$Child,
        [int]$ExpectedExit = 0,
        [double]$Timeout = 0,
        [switch]$LongRunning,
        [double]$HeartbeatInterval = 30,
        [string]$HeartbeatPath = "",
        [int]$MinHeartbeatCount = 0,
        [string[]]$RequiredMarkers = @()
    )

    $arguments = @("-Name", $Name, "-Category", $Category)
    if ($Timeout -gt 0) { $arguments += @("-Timeout", [string]$Timeout) }
    if ($LongRunning) { $arguments += "-LongRunning" }
    if ($HeartbeatInterval -gt 0) { $arguments += @("-HeartbeatInterval", [string]$HeartbeatInterval) }
    if ($HeartbeatPath) { $arguments += @("-HeartbeatPath", $HeartbeatPath) }
    $arguments += "--"
    $arguments += $Child
    $lines = @(& $guarded @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $text = ($lines | ForEach-Object { $_.ToString() }) -join "`n"
    foreach ($marker in $RequiredMarkers) {
        if ($text -notmatch [regex]::Escape($marker)) {
            throw "$Name did not emit required marker: $marker`n$text"
        }
    }
    $heartbeatCount = @($lines | Where-Object { $_.ToString() -match "__HEARTBEAT__" }).Count
    if ($heartbeatCount -lt $MinHeartbeatCount) {
        throw "$Name emitted $heartbeatCount heartbeats; expected at least $MinHeartbeatCount`n$text"
    }
    if ($exitCode -ne $ExpectedExit) {
        throw "$Name returned $exitCode; expected $ExpectedExit`n$text"
    }
    return [ordered]@{
        name = $Name
        status = "PASS"
        exit_code = $exitCode
        required_markers = @($RequiredMarkers)
        heartbeat_count = $heartbeatCount
        output_tail = if ($text.Length -gt 2048) { $text.Substring($text.Length - 2048) } else { $text }
    }
}

$cases = [ordered]@{}
$pidFile = Join-Path $scratch "timeout-descendant.pid"
$heartbeatPath = Join-Path $scratch "heartbeat.json"
$env:D2M_SMOKE_PID_FILE = $pidFile
try {
    # A. Successful child.
    $cases.success = Invoke-SmokeCase -Name "windows-smoke-success" -Category "FAST" -Timeout 30 `
        -Child @($python, "-c", "print('smoke-success', flush=True)") -RequiredMarkers @("__CMD_START__", "__CMD_DONE__", "rc=0")

    # B. Non-zero child.
    $cases.failure = Invoke-SmokeCase -Name "windows-smoke-failure" -Category "FAST" -Timeout 30 `
        -Child @($python, "-c", "import sys; print('smoke-failure', flush=True); sys.exit(7)") `
        -ExpectedExit 1 -RequiredMarkers @("__CMD_START__", "__CMD_FAILED__")

    # C. Timeout with a descendant.  The child writes the descendant PID so
    # the receipt can verify that taskkill /T removed the complete tree.
    $timeoutCode = "import os,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); open(os.environ['D2M_SMOKE_PID_FILE'],'w').write(str(p.pid)); time.sleep(300)"
    $cases.timeout = Invoke-SmokeCase -Name "windows-smoke-timeout-tree" -Category "FAST" -Timeout 2 `
        -Child @($python, "-c", $timeoutCode) -ExpectedExit 1 -RequiredMarkers @("__CMD_START__", "__CMD_TIMEOUT__")
    if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
        throw "Timeout smoke child did not record its descendant PID"
    }
    $descendantPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    $descendantAlive = $false
    try { $descendantAlive = $null -ne (Get-Process -Id $descendantPid -ErrorAction Stop) } catch { $descendantAlive = $false }
    if ($descendantAlive) { throw "Timeout smoke descendant $descendantPid is still alive" }
    $cases["timeout"]["descendant_pid"] = $descendantPid
    $cases["timeout"]["descendant_terminated"] = $true

    # D. Heartbeat-only long-running child.  The child is intentionally quiet
    # between two structured lines; the heartbeat receipt must still advance.
    $heartbeatCode = "import time; print('child-progress-start', flush=True); time.sleep($HeartbeatSeconds); print('child-progress-done', flush=True)"
    $cases.heartbeat = Invoke-SmokeCase -Name "windows-smoke-heartbeat" -Category "LONG_RUNNING" `
        -Timeout ($HeartbeatSeconds + 30) -LongRunning -HeartbeatInterval 10 -HeartbeatPath $heartbeatPath `
        -Child @($python, "-c", $heartbeatCode) -MinHeartbeatCount 2 `
        -RequiredMarkers @("__CMD_START__", "__HEARTBEAT__", "__CMD_DONE__")
    $heartbeatPayload = Get-Content -LiteralPath $heartbeatPath -Raw | ConvertFrom-Json
    if ($heartbeatPayload.status -ne "SUCCESS" -or $heartbeatPayload.terminal_status -ne "DONE") {
        throw "Heartbeat smoke receipt did not finish successfully"
    }
    if ($heartbeatPayload.last_child_output_at -eq $null -or $heartbeatPayload.stdout_bytes -le 0) {
        throw "Heartbeat smoke receipt is missing child progress fields"
    }
    $cases["heartbeat"]["heartbeat_status"] = $heartbeatPayload.status
    $cases["heartbeat"]["stdout_bytes"] = $heartbeatPayload.stdout_bytes
    $cases["heartbeat"]["child_output_stale"] = $heartbeatPayload.child_output_stale

    # E. Git must complete without a pager or credential prompt.
    $cases.git = Invoke-SmokeCase -Name "windows-smoke-git" -Category "FAST" -Timeout 30 `
        -Child @("git", "--no-pager", "log", "-1", "--format=%H") -RequiredMarkers @("__CMD_START__", "__CMD_DONE__")
    $gitHeadMatch = [regex]::Match($cases["git"]["output_tail"], "\b[0-9a-f]{40}\b")
    if (-not $gitHeadMatch.Success) { throw "Guarded git smoke output did not contain a commit hash" }

    $receipt = [ordered]@{
        schema_version = 1
        status = "WINDOWS_GUARDED_COMMAND_SMOKE_COMPLETE"
        generated_at = [DateTime]::UtcNow.ToString("o")
        host_os = $env:OS
        powershell = $PSVersionTable.PSVersion.ToString()
        python = $python
        git_head = $gitHeadMatch.Value
        cases = $cases
    }
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputPath -Encoding UTF8
    Write-Output (ConvertTo-Json $receipt -Depth 8)
}
catch {
    $failure = [ordered]@{
        schema_version = 1
        status = "WINDOWS_GUARDED_COMMAND_SMOKE_FAILED"
        generated_at = [DateTime]::UtcNow.ToString("o")
        host_os = $env:OS
        powershell = $PSVersionTable.PSVersion.ToString()
        error = $_.Exception.Message
        cases = $cases
    }
    $failure | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputPath -Encoding UTF8
    throw
}
finally {
    Remove-Item Env:D2M_SMOKE_PID_FILE -ErrorAction SilentlyContinue
}
