"""Tests for the machine-readable execution fingerprint (1B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from flashmini.fingerprint import (
    collect_fingerprint,
    enforce_clean_tree,
    enforce_fingerprint_match,
    environment_fingerprint_sha256,
    fingerprint_sha,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "flashmini"


def _config_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _abc_fingerprints() -> tuple[dict, dict, dict]:
    """Collect real fingerprints for the three actual v3 treatment configs."""
    data_manifest_sha256 = "b06f559f65e6969a9dae36d392873610330b8a2425fe27f47518c4d200d42dab"
    fp_a = collect_fingerprint(REPO_ROOT, config_sha256=_config_sha(CONFIGS / "poc_a_v3.yaml"),
                               data_manifest_sha256=data_manifest_sha256)
    fp_b = collect_fingerprint(REPO_ROOT, config_sha256=_config_sha(CONFIGS / "poc_b_v3.yaml"),
                               data_manifest_sha256=data_manifest_sha256)
    fp_c = collect_fingerprint(REPO_ROOT, config_sha256=_config_sha(CONFIGS / "poc_c_v3.yaml"),
                               data_manifest_sha256=data_manifest_sha256)
    return fp_a, fp_b, fp_c


def test_abc_full_fingerprints_differ() -> None:
    # A/B/C have different treatment configs, so their full fingerprints
    # legitimately differ.
    fp_a, fp_b, fp_c = _abc_fingerprints()
    assert fp_a["fingerprint_sha256"] != fp_b["fingerprint_sha256"]
    assert fp_b["fingerprint_sha256"] != fp_c["fingerprint_sha256"]
    assert fp_a["fingerprint_sha256"] != fp_c["fingerprint_sha256"]


def test_abc_shared_environment_fingerprints_match() -> None:
    # The treatment-neutral environment fingerprint must be identical across
    # A/B/C: it excludes the treatment-specific config hash.
    fp_a, fp_b, fp_c = _abc_fingerprints()
    env_a = environment_fingerprint_sha256(fp_a)
    env_b = environment_fingerprint_sha256(fp_b)
    env_c = environment_fingerprint_sha256(fp_c)
    assert env_a == env_b == env_c
    # The recorded field must match the recomputed value.
    assert fp_a["environment_fingerprint_sha256"] == env_a


def test_environment_fingerprint_excludes_config() -> None:
    # Mutating only the config hash must not change the environment fingerprint.
    fp_a, _, _ = _abc_fingerprints()
    mutated = dict(fp_a)
    mutated["config_sha256"] = "0" * 64
    assert environment_fingerprint_sha256(mutated) == environment_fingerprint_sha256(fp_a)


def test_environment_fingerprint_sensitive_to_shared_properties() -> None:
    # Mutating each shared property separately must change the environment
    # fingerprint.
    fp_a, _, _ = _abc_fingerprints()
    baseline = environment_fingerprint_sha256(fp_a)
    for key in ("source_sha256", "data_manifest_sha256", "python_version",
                "nvidia_driver_version", "platform"):
        mutated = dict(fp_a)
        mutated[key] = "mutated"
        assert environment_fingerprint_sha256(mutated) != baseline, key
    # PyTorch version (nested under torch).
    mutated = dict(fp_a)
    mutated["torch"] = dict(fp_a["torch"])
    mutated["torch"]["torch_version"] = "9.9.9"
    assert environment_fingerprint_sha256(mutated) != baseline
    # CUDA version (nested under torch).
    mutated = dict(fp_a)
    mutated["torch"] = dict(fp_a["torch"])
    mutated["torch"]["cuda_version"] = "99.99"
    assert environment_fingerprint_sha256(mutated) != baseline
    # GPU identity (nested under torch.devices).
    mutated = dict(fp_a)
    mutated["torch"] = dict(fp_a["torch"])
    mutated["torch"]["devices"] = [dict(d) for d in fp_a["torch"]["devices"]]
    mutated["torch"]["devices"][0]["name"] = "NVIDIA GeForce RTX 9999"
    assert environment_fingerprint_sha256(mutated) != baseline
