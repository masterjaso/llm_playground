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
    assert {g["name"] for g in policy["gates"]} >= {
        "train_2p1m", "gate_2p1m", "train_100m", "gate_100m",
        "train_250m", "final_report",
    }
    for verdict in (
        "pass", "implementation_failure", "training_instability",
        "ambiguous_review_required", "catastrophic_architecture_failure",
        "catastrophic_hybrid_failure", "quality_win", "quality_parity",
        "ple_pass", "ple_unproven", "ple_fail", "needs_seed_confirmation",
    ):
        assert verdict in policy["verdicts"]


def test_policy_has_all_required_thresholds() -> None:
    policy = load_gate_policy()
    for key in (
        "max_nan", "max_nll_implementation_failure", "loss_spike_ratio",
        "loss_spike_window_updates", "ple_effective_margin", "ple_parity_tolerance",
        "quality_parity_tolerance", "router_entropy_floor",
        "expert_load_ratio_threshold", "expert_load_exceedance_max_fraction",
        "shared_clipping_ambiguity_threshold", "catastrophic_hybrid_regression",
        "efficiency_advantage_threshold",
    ):
        assert key in policy["thresholds"]


def test_policy_encodes_exact_tokens_and_updates() -> None:
    policy = load_gate_policy()
    by_name = {g["name"]: g for g in policy["gates"]}
    assert by_name["train_2p1m"]["stop_after_tokens"] == 2_097_152
    assert by_name["train_2p1m"]["optimizer_updates"] == 512
    assert by_name["gate_2p1m"]["stop_after_tokens"] == 2_097_152
    assert by_name["gate_2p1m"]["optimizer_updates"] == 512
    assert by_name["train_100m"]["stop_after_tokens"] == 100_663_296
    assert by_name["train_100m"]["optimizer_updates"] == 24_576
    assert by_name["gate_100m"]["stop_after_tokens"] == 100_663_296
    assert by_name["gate_100m"]["optimizer_updates"] == 24_576
    assert by_name["train_250m"]["stop_after_tokens"] == 250_000_000
    assert by_name["train_250m"]["optimizer_updates"] == 61_036
    assert by_name["train_250m"]["expected_padded_tokens"] == 250_000_128
    assert by_name["final_report"]["stop_after_tokens"] == 250_000_000
    assert by_name["final_report"]["optimizer_updates"] == 61_036


def test_policy_encodes_eval_slices() -> None:
    policy = load_gate_policy()
    by_name = {g["name"]: g for g in policy["gates"]}
    assert by_name["gate_2p1m"]["eval"] == {"skip_sequences": 1024, "max_sequences": 8192}
    assert by_name["gate_100m"]["eval"] == {"skip_sequences": 1024, "max_sequences": 32768}
    assert by_name["final_report"]["eval"] == {"skip_sequences": 1024, "max_sequences": None}


def test_gate_policy_sha256_is_stable() -> None:
    assert gate_policy_sha256() == gate_policy_sha256()
    assert len(gate_policy_sha256()) == 64


def test_verdict_for_maps_checks() -> None:
    policy = load_gate_policy()
    assert verdict_for("pass", policy) == "PASS"
    assert verdict_for("implementation_failure", policy) == "NO_GO"
    assert verdict_for("training_instability", policy) == "NO_GO"
    assert verdict_for("ambiguous_review_required", policy) == "AMBIGUOUS_REVIEW_REQUIRED"
    assert verdict_for("catastrophic_hybrid_failure", policy) == "NO_GO"
    assert verdict_for("quality_win", policy) == "GO_TO_1B_SCALING"
    assert verdict_for("ple_pass", policy) == "PLE_PASS"
    assert verdict_for("ple_fail", policy) == "PLE_FAIL"
    assert verdict_for("needs_seed_confirmation", policy) == "NEEDS_SEED_CONFIRMATION"


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


def test_load_gate_policy_rejects_missing_threshold(tmp_path: Path) -> None:
    policy = load_gate_policy()
    del policy["thresholds"]["router_entropy_floor"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="router_entropy_floor"):
        load_gate_policy(path)


def test_load_gate_policy_rejects_wrong_token_count(tmp_path: Path) -> None:
    policy = load_gate_policy()
    for g in policy["gates"]:
        if g["name"] == "train_2p1m":
            g["stop_after_tokens"] = 2_097_153  # off by one
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="stop_after_tokens"):
        load_gate_policy(path)


def test_default_policy_path_exists() -> None:
    assert DEFAULT_POLICY_PATH.is_file()
