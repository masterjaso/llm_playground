<!-- nsp:meta
id: docs.windows
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/windows.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Native Windows / PowerShell workflow

This project is operated from `C:\workplace\llm_playground` with Windows
Python and PowerShell.  WSL is not required and the pipeline never installs a
Linux NVIDIA driver.

```powershell
Set-Location C:\workplace\llm_playground
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Setup-Windows.ps1
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Invoke-GuardedCommand-Smoke.ps1 -Output runs\windows-guarded-command-smoke-current.json -HeartbeatSeconds 60
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Test-Windows-Cuda.ps1
& .\.venv\Scripts\python.exe -m dense2moe.cli doctor --run-dir runs\<run-id> --runtime-lock runs\windows-runtime-lock.json --json
& .\.venv\Scripts\python.exe -m dense2moe.cli status --run-dir runs\<run-id> --json
```

Only `C:\workplace\llm_playground\.venv\Scripts\python.exe` is
authoritative for D2M receipts. The guarded smoke and CUDA doctor must pass
before a new `runs\windows-runtime-lock.json` is used for method-proof
capture. Runtime-lock drift is a hard stop (`WINDOWS_RUNTIME_DRIFT`), not a
reason to fall back to WSL or the global Python installation.

If the current lock is explicitly reviewed as stale, rerun setup with
`-ForceRecovery`. The script preserves the old lock as a timestamped
`.drift-*.json` backup, rebuilds the project environment, and asks the
capability gate to issue a new lock; it never silently overwrites drift.

For a durable long-running command, use `scripts\Invoke-GuardedCommand.ps1`
with `-LongRunning` and a heartbeat receipt; the wrapper owns process-tree
supervision and can still be pointed at a run's `logs` directory.  `HANDOFF.md`
contains the exact resume command.  Model snapshots are written under the run
directory only after a pinned commit revision has been recorded.

## Guarded command contract

Repository diagnostics and report scripts should use
`dense2moe.command.run_guarded` (or `scripts/run_guarded_command.py`).  Every
invocation emits a start marker and exactly one terminal marker:

```text
__CMD_START__ name=<name> timestamp=<utc>
__CMD_DONE__ name=<name> rc=0 elapsed=<seconds>
```

Failures and bounded timeouts use `__CMD_FAILED__` and `__CMD_TIMEOUT__`.
FAST commands default to 60 seconds, MEDIUM commands to five minutes, and
only explicitly classified LONG_RUNNING work may omit a timeout.  Long jobs
must set `--long-running` and a heartbeat receipt; the wrapper emits
`__HEARTBEAT__` records and terminates the complete child process tree on
timeout.  Child stdin is closed and Git/GitHub pagers and credential prompts
are disabled.

PowerShell callers can use the equivalent
`scripts\Invoke-GuardedCommand.ps1 -Name <name> -- <program> <args>` entry
point; it delegates to the same Python implementation instead of performing
an unsupervised native wait.

Long-running commands spool complete stdout/stderr to sibling `.stdout.log`
and `.stderr.log` files next to the heartbeat receipt while retaining only a
bounded tail in the JSON result.  Heartbeats include child output age and byte
counters, plus `child_output_stale`; this marks a quiet child without killing
it automatically.  Before a real Windows experiment, run
`scripts\Invoke-GuardedCommand-Smoke.ps1`; its native result is written to
`runs\windows-guarded-command-smoke.json`.  The checked-in
`docs\windows-guarded-command-smoke-receipt.json` records the most recent
successful native run and should be refreshed whenever the guard or host
environment changes.
