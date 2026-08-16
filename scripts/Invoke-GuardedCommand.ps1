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

function ConvertTo-WindowsProcessArgument {
    param([AllowEmptyString()][string]$Value)

    if ($null -eq $Value -or $Value.Length -eq 0) {
        return '""'
    }
    # ProcessStartInfo.ArgumentList is available on modern PowerShell/.NET,
    # but Windows PowerShell 5.1 only exposes one command-line string.  Keep
    # this fallback compatible with CommandLineToArgvW so paths and child
    # snippets containing spaces/quotes survive the native launch unchanged.
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            if ($backslashes -gt 0) {
                [void]$builder.Append((('\' * ($backslashes * 2 + 1)) -join ''))
            } else {
                [void]$builder.Append('\')
            }
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append((('\' * $backslashes) -join ''))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append((('\' * ($backslashes * 2)) -join ''))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-GuardedPython {
    param([Parameter(Mandatory = $true)][string[]]$PythonArguments)

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $python
    $startInfo.WorkingDirectory = $projectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    if ($startInfo.PSObject.Properties.Name -contains "ArgumentList") {
        foreach ($value in $PythonArguments) {
            [void]$startInfo.ArgumentList.Add([string]$value)
        }
    } else {
        $startInfo.Arguments = (($PythonArguments | ForEach-Object { ConvertTo-WindowsProcessArgument $_ }) -join ' ')
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Unable to start guarded Python process: $python"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    foreach ($line in ($stdout -split "`r?`n")) {
        if ($line.Length -gt 0) { Write-Output $line }
    }
    foreach ($line in ($stderr -split "`r?`n")) {
        if ($line.Length -gt 0) { [Console]::Error.WriteLine($line) }
    }
    $script:GuardedPythonExitCode = $process.ExitCode
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
    Invoke-GuardedPython -PythonArguments $arguments
    exit $script:GuardedPythonExitCode
}
finally {
    Pop-Location
}
