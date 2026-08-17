"""CLI entry point for the repository's bounded command contract.

Examples::

    & .\\.venv\\Scripts\\python.exe scripts\\run_guarded_command.py --name git-log --category FAST -- git log -1
    & .\\.venv\\Scripts\\python.exe scripts\\run_guarded_command.py --name capture --long-running `
        --heartbeat-path runs\\example\\capture-progress.json -- .\\.venv\\Scripts\\python.exe scripts\\train.py

The child receives no stdin and Git/GitHub pagers/prompts are disabled by the
shared :mod:`dense2moe.command` implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep the documented checkout-local invocation usable before an editable
# install exists (notably during native Windows bootstrap).  Installed users
# simply resolve the same package from their environment.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.command import run_guarded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None)
    parser.add_argument("--category", choices=("FAST", "MEDIUM", "LONG_RUNNING"), default="FAST")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--heartbeat-path", type=Path, default=None)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--stdout-log-path", type=Path, default=None)
    parser.add_argument("--stderr-log-path", type=Path, default=None)
    parser.add_argument("--tail-bytes", type=int, default=4_000_000)
    parser.add_argument("--child-output-stale-after", type=float, default=None)
    parser.add_argument("--long-running", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    result = run_guarded(
        command,
        name=args.name,
        category=args.category,
        timeout=args.timeout,
        cwd=args.cwd,
        heartbeat_path=args.heartbeat_path,
        heartbeat_interval=args.heartbeat_interval,
        stdout_log_path=args.stdout_log_path,
        stderr_log_path=args.stderr_log_path,
        tail_bytes=args.tail_bytes,
        child_output_stale_after=args.child_output_stale_after,
        long_running=args.long_running,
        check=args.check,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
