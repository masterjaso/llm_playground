"""Tests for immutable milestone checkpoint preservation (1E)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from flashmini.milestones import preserve_milestone, verify_milestone


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_preserve_milestone_copies_and_hashes(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.bin"
    _write(ckpt, b"checkpoint-bytes")
    run_dir = tmp_path / "run"
    result = preserve_milestone(ckpt, run_dir, "gate_2p1m")
    milestone = Path(result["milestone"])
    assert milestone.is_file()
    assert milestone.read_bytes() == b"checkpoint-bytes"
    assert result["sha256"] == _sha256(b"checkpoint-bytes")
    assert result["created"] is True
    assert verify_milestone(milestone, result["sha256"]) is True


def test_preserve_milestone_is_idempotent_for_identical_hash(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.bin"
    _write(ckpt, b"checkpoint-bytes")
    run_dir = tmp_path / "run"
    first = preserve_milestone(ckpt, run_dir, "gate_2p1m")
    second = preserve_milestone(ckpt, run_dir, "gate_2p1m")
    assert second["created"] is False
    assert second["sha256"] == first["sha256"]


def test_preserve_milestone_refuses_overwrite_with_different_hash(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.bin"
    _write(ckpt, b"checkpoint-bytes")
    run_dir = tmp_path / "run"
    preserve_milestone(ckpt, run_dir, "gate_2p1m")
    # A different checkpoint must not overwrite the immutable milestone.
    other = tmp_path / "other.bin"
    _write(other, b"different-bytes")
    with pytest.raises(ValueError):
        preserve_milestone(other, run_dir, "gate_2p1m")
    # The original milestone is untouched.
    milestone = run_dir / "milestones" / "gate_2p1m"
    assert milestone.read_bytes() == b"checkpoint-bytes"


def test_verify_milestone_detects_corruption(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.bin"
    _write(ckpt, b"checkpoint-bytes")
    run_dir = tmp_path / "run"
    result = preserve_milestone(ckpt, run_dir, "gate_2p1m")
    milestone = Path(result["milestone"])
    assert verify_milestone(milestone, result["sha256"]) is True
    assert verify_milestone(milestone, "0" * 64) is False
    # A missing milestone fails verification.
    assert verify_milestone(tmp_path / "missing.bin", result["sha256"]) is False


def test_preserve_milestone_missing_source_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preserve_milestone(tmp_path / "nope.bin", tmp_path / "run", "gate_2p1m")
