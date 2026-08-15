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

For a durable long-running command, start it with `Start-Process` and redirect
stdout/stderr to that run's `logs` directory.  `HANDOFF.md` contains the exact
resume command.  Model snapshots are written under the run directory only
after a pinned commit revision has been recorded.

