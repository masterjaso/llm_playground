"""Immutable milestone checkpoint preservation for v3 gate decisions.

After a gate checkpoint has been fully validated, an immutable milestone copy
is preserved under the run so that later keep-latest training cannot erase the
only weights used for an important decision. The copy is made atomically, read
back, hashed, and its SHA-256 recorded in the gate report. Training continues
from the current official ``checkpoints/`` checkpoint, not from a milestone.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preserve_milestone(
    checkpoint_path: Path,
    run_dir: Path,
    milestone_name: str,
) -> dict[str, Any]:
    """Atomically copy a validated checkpoint to an immutable milestone.

    The milestone is written to a temporary file, fsynced, read back, and
    atomically installed at ``run_dir/milestones/<milestone_name>``. The
    returned record contains the milestone path and its SHA-256. A partial or
    unverified file is never left in place.
    """
    checkpoint_path = Path(checkpoint_path)
    run_dir = Path(run_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    milestones_dir = run_dir / "milestones"
    milestones_dir.mkdir(parents=True, exist_ok=True)
    destination = milestones_dir / milestone_name

    # Refuse to overwrite an existing milestone with a different hash; an
    # existing identical milestone is idempotent.
    if destination.exists():
        existing = _sha256_file(destination)
        source = _sha256_file(checkpoint_path)
        if existing != source:
            raise ValueError(
                f"milestone {destination} already exists with a different hash; "
                "refusing to overwrite an immutable milestone"
            )
        return {"milestone": str(destination), "sha256": existing, "created": False}

    tmp = Path(tempfile.mkstemp(dir=str(milestones_dir), suffix=".tmp")[1])
    try:
        shutil.copyfile(checkpoint_path, tmp)
        with open(tmp, "rb") as saved:
            os.fsync(saved.fileno())
        # Read back and verify the copy matches the source before installing.
        if _sha256_file(tmp) != _sha256_file(checkpoint_path):
            raise ValueError("milestone copy failed read-back verification")
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    directory_fd = os.open(str(milestones_dir), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"milestone": str(destination), "sha256": _sha256_file(destination), "created": True}


def verify_milestone(milestone_path: Path, expected_sha256: str) -> bool:
    """Return True if the milestone file exists and matches the recorded hash."""
    milestone_path = Path(milestone_path)
    if not milestone_path.is_file():
        return False
    return _sha256_file(milestone_path) == expected_sha256
