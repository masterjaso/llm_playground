"""Tests for the durable resumable v3 orchestrator (1G)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import flashmini_v3_execute as orch


def _fresh_state() -> dict:
    return {"completed": {}, "failed_gates": {}, "verdicts": {}, "fingerprint": None}


def test_stage_completed_tracking() -> None:
    state = _fresh_state()
    assert orch._stage_completed(state, "A", "preflight_2p1m") is False
    orch._mark_completed(state, "A", "preflight_2p1m", {"step": 512})
    assert orch._stage_completed(state, "A", "preflight_2p1m") is True
    assert orch._stage_completed(state, "B", "preflight_2p1m") is False


def test_failed_gate_is_never_auto_continued() -> None:
    state = _fresh_state()
    orch._mark_gate_failed(state, "A", "gate_2p1m", "implementation_failure")
    assert orch._gate_failed(state, "A", "gate_2p1m") is True
    with pytest.raises(RuntimeError, match="previously FAILED"):
        orch._run_stage("A", "gate_2p1m", state, orch.load_gate_policy(), dry_run=True)


def test_state_round_trips(tmp_path: Path) -> None:
    # Redirect the module's state path to a temp location.
    original = orch.STATE_PATH
    orch.STATE_PATH = tmp_path / "state.json"
    try:
        state = orch._load_state()
        assert state == _fresh_state()
        orch._mark_completed(state, "A", "gate_2p1m", {"nll": 10.0})
        reloaded = orch._load_state()
        assert reloaded["completed"]["A:gate_2p1m"]["nll"] == 10.0
    finally:
        orch.STATE_PATH = original


def test_gate_check_nan_fails_closed() -> None:
    policy = orch.load_gate_policy()
    result = {"nll": float("nan")}
    assert orch._gate_check("A", "gate_2p1m", result, policy) == "implementation_failure"


def test_gate_check_missing_result_fails_closed() -> None:
    policy = orch.load_gate_policy()
    with pytest.raises(RuntimeError):
        orch._gate_check("A", "gate_2p1m", {"nll": None}, policy)


def test_gate_check_finite_nll_passes() -> None:
    policy = orch.load_gate_policy()
    result = {"nll": 10.0}
    assert orch._gate_check("A", "gate_2p1m", result, policy) == "ok"


def test_gate_check_nll_above_threshold_fails() -> None:
    policy = orch.load_gate_policy()
    threshold = policy["thresholds"]["max_nll_implementation_failure"]
    result = {"nll": threshold + 1.0}
    assert orch._gate_check("A", "gate_2p1m", result, policy) == "implementation_failure"


def test_checkpoint_discovery_fails_when_missing(tmp_path: Path) -> None:
    original = orch.RUN_ROOT
    orch.RUN_ROOT = tmp_path
    try:
        with pytest.raises(FileNotFoundError):
            orch._checkpoint_for("A")
    finally:
        orch.RUN_ROOT = original


def test_checkpoint_discovery_picks_latest(tmp_path: Path) -> None:
    original = orch.RUN_ROOT
    orch.RUN_ROOT = tmp_path
    run_dir = tmp_path / "treatment_A" / "checkpoints"
    run_dir.mkdir(parents=True)
    (run_dir / "step_100.pt").write_bytes(b"x")
    (run_dir / "step_200.pt").write_bytes(b"x")
    try:
        ckpt = orch._checkpoint_for("A")
        assert ckpt.name == "step_200.pt"
    finally:
        orch.RUN_ROOT = original


def test_dry_run_does_not_launch(tmp_path: Path) -> None:
    original = orch.RUN_ROOT
    orch.RUN_ROOT = tmp_path
    try:
        rc = orch.main(["--dry-run"])
        assert rc == 0
        # No state file should be written by a dry run.
        assert not (tmp_path / "state.json").exists()
    finally:
        orch.RUN_ROOT = original


def test_dry_run_reports_failed_gate(tmp_path: Path) -> None:
    original = orch.RUN_ROOT
    orch.RUN_ROOT = tmp_path
    try:
        state = _fresh_state()
        orch._mark_gate_failed(state, "A", "gate_2p1m", "implementation_failure")
        orch.STATE_PATH = tmp_path / "state.json"
        orch._save_state(state)
        rc = orch.main(["--dry-run"])
        assert rc == 0
    finally:
        orch.RUN_ROOT = original


def test_treatment_ordering_is_a_b_c() -> None:
    assert orch.TREATMENTS == ("A", "B", "C")


def test_recipe_is_frozen() -> None:
    # The frozen recipe must match the official values.
    assert orch.RECIPE["seed"] == 17
    assert orch.RECIPE["batch_size"] == 16
    assert orch.RECIPE["tokens"] == 250_000_000
    assert orch.RECIPE["lr"] == 3e-4
    assert orch.RECIPE["ple_lr_multiplier"] == 5
    assert orch.RECIPE["model_parallel_gpus"] == "1,0"
    assert orch.RECIPE["gpu_memory_gib"] == 15


def test_gate_slice_keys_map_to_official_slices() -> None:
    from flashmini.evaluation import official_slice
    assert orch.GATE_SLICE_KEY["gate_2p1m"] == "2p1m"
    assert orch.GATE_SLICE_KEY["gate_100m"] == "100m"
    assert orch.GATE_SLICE_KEY["final_report"] == "250m"
    assert official_slice("2p1m") == (1024, 8192)
    assert official_slice("100m") == (1024, 32768)
    assert official_slice("250m") == (1024, None)
