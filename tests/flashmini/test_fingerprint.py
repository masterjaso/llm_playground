"""Tests for the machine-readable execution fingerprint (1B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from flashmini.fingerprint import (
    collect_fingerprint,
    enforce_clean_tree,
    enforce_fingerprint_match,
    fingerprint_sha,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_collect_fingerprint_is_stable() -> None:
    fp1 = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    fp2 = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    assert fp1["fingerprint_sha256"] == fp2["fingerprint_sha256"]
    assert len(fp1["fingerprint_sha256"]) == 64


def test_fingerprint_sha_excludes_own_key() -> None:
    fp = {"git_commit": "abc", "source_sha256": "xyz"}
    fp["fingerprint_sha256"] = fingerprint_sha(fp)
    # Recomputing over the same payload (minus the hash key) must match.
    assert fingerprint_sha(fp) == fp["fingerprint_sha256"]


def test_matching_fingerprint_passes() -> None:
    fp = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    # A clean tree is required; if the repo is dirty, allow_dirty for the test.
    enforce_fingerprint_match(fp, fp, allow_dirty=True)


def test_dirty_tree_fails_closed() -> None:
    fp = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    fp["git_dirty"] = True
    with pytest.raises(ValueError):
        enforce_fingerprint_match(fp, fp, allow_dirty=False)
    with pytest.raises(ValueError):
        enforce_clean_tree(fp)


def test_materially_different_source_fails() -> None:
    fp = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    other = dict(fp)
    other["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        enforce_fingerprint_match(fp, other, allow_dirty=True)


def test_materially_different_config_fails() -> None:
    fp = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    other = dict(fp)
    other["config_sha256"] = "c"
    with pytest.raises(ValueError):
        enforce_fingerprint_match(fp, other, allow_dirty=True)


def test_clean_tree_passes() -> None:
    fp = collect_fingerprint(REPO_ROOT, config_sha256="a", data_manifest_sha256="b")
    fp["git_dirty"] = False
    enforce_clean_tree(fp)
