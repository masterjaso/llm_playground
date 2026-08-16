"""Runtime provenance for newly generated pipeline artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from .command import run_guarded

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def current_git_commit() -> str:
    """Return the full commit that produced the current source tree.

    Provenance is deliberately fail-closed.  A receipt or validated artifact
    must never be labelled with a placeholder when the repository identity
    cannot be established.
    """

    repository = Path(__file__).resolve().parents[2]
    try:
        completed = run_guarded(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            name="git-rev-parse-head",
            category="FAST",
            timeout=60.0,
            emit=lambda _line: None,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("unable to determine current git commit") from exc
    if not completed.ok:
        raise RuntimeError("unable to determine current git commit")
    commit = completed.stdout.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError(f"git returned an invalid commit: {commit!r}")
    return commit


__all__ = ["current_git_commit"]
