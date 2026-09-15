"""Tests for the pre-registered v3 gate policy (1F)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashmini.gate_policy import (
    DEFAULT_POLICY_PATH,
    gate_policy_sha256,
    load_gate_policy,
    verdict_for,
)


def test_default_policy_loads_and_is_valid() -> None:
    policy = load_gate_policy()
    assert {g["name"] for g in policy["gates"]} >= {"preflight_2p1m", "gate_100m", "final_250m"}
    for verdict in (
        "implementation_failure", "training_instability", "catastrophic_architecture_failure",
        "quality_win", "quality_parity", "ple_pass", "ple_unproven", "ple_fail",
        "needs_seed_confirmation",
    ):
        assert verdict in policy["verdicts"]


def test_gate_policy_sha256_is_stable() -> None:
    assert gate_policy_sha256() == gate_policy_sha256()
    assert len(gate_policy_sha256()) == 64


def test_verdict_for_maps_checks() -> None:
    policy = load_gate_policy()
    assert verdict_for("implementation_failure", policy) == "NO_GO"
    assert verdict_for("training_instability", policy) == "NO_GO"
    assert verdict_for("quality_win", policy) == "GO_TO_1B_SCALING"
    assert verdict_for("ple_pass", policy) == "PLE_PASS"
    assert verdict_for("ple_fail", policy) == "PLE_FAIL"


def test_verdict_for_unknown_check_fails_closed() -> None:
    policy = load_gate_policy()
    with pytest.raises(KeyError):
        verdict_for("not_a_real_check", policy)


def test_load_gate_policy_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gate_policy(tmp_path / "missing.json")


def test_load_gate_policy_rejects_missing_gate(tmp_path: Path) -> None:
    policy = load_gate_policy()
    policy["gates"] = [g for g in policy["gates"] if g["name"] != "gate_100m"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gate_policy(path)


def test_load_gate_policy_rejects_missing_verdict(tmp_path: Path) -> None:
    policy = load_gate_policy()
    del policy["verdicts"]["ple_pass"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gate_policy(path)


def test_default_policy_path_exists() -> None:
    assert DEFAULT_POLICY_PATH.is_file()
