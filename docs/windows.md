# Native Windows / PowerShell workflow

This project is operated from `C:\workplace\llm_playground` with Windows
Python and PowerShell.  WSL is not required and the pipeline never installs a
Linux NVIDIA driver.

```powershell
Set-Location C:\workplace\llm_playground
python -m pip install -e ".[runtime,dev,ml]"
python -m dense2moe.cli doctor --run-dir runs\<run-id> --json
python -m dense2moe.cli status --run-dir runs\<run-id> --json
```

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
`docs\windows-guarded-command-smoke-receipt.json` is explicitly marked
`NOT_RUN_LINUX_ENVIRONMENT` until that native run occurs.
