"""Runtime provenance for newly generated pipeline artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def current_git_commit() -> str:
    """Return the full commit that produced the current source tree.

    Provenance is deliberately fail-closed.  A receipt or validated artifact
    must never be labelled with a placeholder when the repository identity
    cannot be established.
    """

    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to determine current git commit") from exc
    commit = completed.stdout.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError(f"git returned an invalid commit: {commit!r}")
    return commit


__all__ = ["current_git_commit"]
