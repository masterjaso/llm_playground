"""Tests for the durable resumable v3 orchestrator.

Part 6: every orchestrator test runs against an isolated tmp_path execution
root. No test may write under the real production state
(``runs/flashmini/v3_execution``). A module-level regression test snapshots
the production state path before the suite and asserts it is byte-for-byte
unchanged afterward.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import flashmini_v3_execute as orch

# The real production execution root. Tests must never write here.
PRODUCTION_RUN_ROOT = ROOT / "runs" / "flashmini" / "v3_execution"
PRODUCTION_STATE_PATH = PRODUCTION_RUN_ROOT / "state.json"


def _snapshot_production_state() -> tuple[bool, bytes | None]:
    """Return (exists, bytes) for the production state path."""
    if PRODUCTION_STATE_PATH.is_file():
        return True, PRODUCTION_STATE_PATH.read_bytes()
    return False, None


def _fresh_state() -> dict:
    return {"completed": {}, "failed_gates": {}, "verdicts": {}, "fingerprint": None}


@pytest.fixture
def isolated_orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every orchestrator persistence path to an isolated tmp_path.

    Patches RUN_ROOT, STATE_PATH, and FREEZE_MANIFEST so no test can touch the
    real production state. monkeypatch restores the originals automatically.
    """
    monkeypatch.setattr(orch, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(orch, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(orch, "FREEZE_MANIFEST", tmp_path / "freeze_manifest.json")
    return tmp_path


def _write_metrics(isolated_orch: Path, treatment: str, records: list[dict]) -> None:
    metrics_path = isolated_orch / f"treatment_{treatment}" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_stage_completed_tracking() -> None:
    state = _fresh_state()
    assert orch._stage_completed(state, "A", "train_2p1m") is False
    orch._mark_completed(state, "A", "train_2p1m", {"step": 512})
    assert orch._stage_completed(state, "A", "train_2p1m") is True
    assert orch._stage_completed(state, "B", "train_2p1m") is False


def test_failed_gate_is_never_auto_continued(isolated_orch: Path) -> None:
    state = _fresh_state()
    orch._mark_gate_failed(state, "ALL", "gate_2p1m", "implementation_failure")
    assert orch._gate_failed(state, "ALL", "gate_2p1m") is True
    with pytest.raises(RuntimeError, match="previously FAILED"):
        orch._run_stage("ALL", "gate_2p1m", state, orch.load_gate_policy(), dry_run=True)


def test_state_round_trips(isolated_orch: Path) -> None:
    state = orch._load_state()
    assert state == _fresh_state()
    orch._mark_completed(state, "ALL", "gate_2p1m", {"nll": 10.0})
    reloaded = orch._load_state()
    assert reloaded["completed"]["ALL:gate_2p1m"]["nll"] == 10.0


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


def test_checkpoint_discovery_fails_when_missing(isolated_orch: Path) -> None:
    with pytest.raises(FileNotFoundError):
        orch._checkpoint_for("A")


def test_checkpoint_discovery_fails_closed_on_ambiguity(isolated_orch: Path) -> None:
    """Two checkpoints for one treatment must be rejected, not silently picked."""
    run_dir = isolated_orch / "treatment_A" / "checkpoints"
    run_dir.mkdir(parents=True)
    (run_dir / "step_512.pt").write_bytes(b"x")
    (run_dir / "step_24576.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="ambiguous checkpoint set"):
        orch._checkpoint_for("A")


def test_checkpoint_discovery_returns_single(isolated_orch: Path) -> None:
    run_dir = isolated_orch / "treatment_A" / "checkpoints"
    run_dir.mkdir(parents=True)
    (run_dir / "step_512.pt").write_bytes(b"x")
    ckpt = orch._checkpoint_for("A")
    assert ckpt.name == "step_512.pt"


def test_dry_run_does_not_launch(isolated_orch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from the real freeze check (dirty tree in the test environment).
    monkeypatch.setattr(orch, "_verify_freeze_and_environment", lambda state: state)
    rc = orch.main(["--dry-run"])
    assert rc == 0
    # A dry run must not write any production state: no state file, no logs.
    assert not (isolated_orch / "state.json").exists()
    assert not (isolated_orch / "logs").exists()


def test_dry_run_reports_failed_gate(isolated_orch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orch, "_verify_freeze_and_environment", lambda state: state)
    state = _fresh_state()
    orch._mark_gate_failed(state, "ALL", "gate_2p1m", "implementation_failure")
    orch._save_state(state)
    rc = orch.main(["--dry-run"])
    assert rc == 0


def test_dry_run_plan_is_stage_first(isolated_orch: Path) -> None:
    """The dry-run plan must be stage-first, not treatment-first."""
    plan = orch.full_plan()
    expected = [
        ("train_2p1m", "A"), ("train_2p1m", "B"), ("train_2p1m", "C"),
        ("gate_2p1m", "ALL"),
        ("train_100m", "A"), ("train_100m", "B"), ("train_100m", "C"),
        ("gate_100m", "ALL"),
        ("train_250m", "A"), ("train_250m", "B"), ("train_250m", "C"),
        ("final_report", "ALL"),
    ]
    assert plan == expected
    # A collective stage must reject --treatment.
    with pytest.raises(ValueError, match="does not accept --treatment"):
        orch.plan_for("gate_2p1m", "A")
    # A training stage with a single treatment returns just that pair.
    assert orch.plan_for("train_2p1m", "B") == [("train_2p1m", "B")]


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


def test_stage_expected_invariants() -> None:
    assert orch.STAGE_EXPECTED["train_2p1m"] == {"step": 512, "tokens_seen": 2_097_152}
    assert orch.STAGE_EXPECTED["train_100m"] == {"step": 24_576, "tokens_seen": 100_663_296}
    assert orch.STAGE_EXPECTED["train_250m"] == {"step": 61_036, "tokens_seen": 250_000_128}


def test_gate_slice_keys_map_to_official_slices() -> None:
    from flashmini.evaluation import official_slice
    assert orch.GATE_SLICE_KEY["gate_2p1m"] == "2p1m"
    assert orch.GATE_SLICE_KEY["gate_100m"] == "100m"
    assert orch.GATE_SLICE_KEY["final_report"] == "250m"
    assert official_slice("2p1m") == (1024, 8192)
    assert official_slice("100m") == (1024, 32768)
    assert official_slice("250m") == (1024, None)


def test_training_health_uses_policy_thresholds(isolated_orch: Path) -> None:
    """_training_health must consume the policy threshold values, not hardcode."""
    policy = orch.load_gate_policy()
    records = [
        {"tokens_seen": 600_000 + i * 100_000, "loss": 10.0 - i * 0.05,
         "router_entropy": 2.5, "expert_load_ratio": 1.5,
         "grad_clip_fraction_shared": 0.0}
        for i in range(20)
    ]
    _write_metrics(isolated_orch, "A", records)
    verdict, detail = orch._training_health("A", policy)
    assert verdict == "ok"
    assert detail["router_entropy_ok"] is True
    assert detail["expert_load_ok"] is True


def test_training_health_flags_instability(isolated_orch: Path) -> None:
    """A flat (non-improving) loss curve must be flagged as training_instability."""
    policy = orch.load_gate_policy()
    records = [
        {"tokens_seen": 600_000 + i * 100_000, "loss": 10.0,
         "router_entropy": 2.5, "expert_load_ratio": 1.5,
         "grad_clip_fraction_shared": 0.0}
        for i in range(20)
    ]
    _write_metrics(isolated_orch, "A", records)
    verdict, _ = orch._training_health("A", policy)
    assert verdict == "training_instability"


def test_training_health_flags_clipping_ambiguity(isolated_orch: Path) -> None:
    """Shared clipping fraction above the policy threshold -> ambiguous review."""
    policy = orch.load_gate_policy()
    records = [
        {"tokens_seen": 600_000 + i * 100_000, "loss": 10.0 - i * 0.05,
         "router_entropy": 2.5, "expert_load_ratio": 1.5,
         "grad_clip_fraction_shared": 0.99}
        for i in range(20)
    ]
    _write_metrics(isolated_orch, "A", records)
    verdict, detail = orch._training_health("A", policy)
    assert verdict == "ambiguous_review_required"
    assert detail["shared_clip_fraction"] == 0.99


def test_training_health_flags_loss_spike(isolated_orch: Path) -> None:
    """An improving curve with a single large spike must be flagged unstable."""
    policy = orch.load_gate_policy()
    records = [
        {"tokens_seen": 600_000 + i * 100_000, "loss": 10.0 - i * 0.1,
         "router_entropy": 2.5, "expert_load_ratio": 1.5,
         "grad_clip_fraction_shared": 0.0}
        for i in range(20)
    ]
    # Inject a single spike well above the loss_spike_ratio * window median.
    records[5]["loss"] = 30.0
    _write_metrics(isolated_orch, "A", records)
    verdict, detail = orch._training_health("A", policy)
    assert verdict == "training_instability"
    assert detail["loss_spike_ok"] is False
    assert detail["loss_spike_max_ratio"] > policy["thresholds"]["loss_spike_ratio"]


def test_paired_block_bootstrap_ci_deterministic() -> None:
    a = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.0]
    b = [1.05, 1.15, 0.95, 1.05, 1.25, 0.85, 1.15, 0.95, 1.05, 1.05]
    p1, lo1, hi1 = orch._paired_block_bootstrap_ci(a, b)
    p2, lo2, hi2 = orch._paired_block_bootstrap_ci(a, b)
    assert p1 == p2 and lo1 == lo2 and hi1 == hi2
    assert lo1 <= p1 <= hi1
    with pytest.raises(ValueError):
        orch._paired_block_bootstrap_ci(a, a[:5])


def test_catastrophic_hybrid_failure_decision() -> None:
    # Both hybrids regress beyond threshold AND CIs exclude zero -> catastrophic.
    assert orch._catastrophic_hybrid_failure(10.0, 10.6, 10.7, 0.01, 0.02, 0.05) is True
    # One hybrid within threshold -> not catastrophic.
    assert orch._catastrophic_hybrid_failure(10.0, 10.4, 10.7, 0.01, 0.02, 0.05) is False
    # CI includes zero -> not catastrophic even if relative regression is large.
    assert orch._catastrophic_hybrid_failure(10.0, 10.6, 10.7, -0.01, 0.02, 0.05) is False


def test_ple_classification_decision() -> None:
    # Entire CI below -parity_tolerance -> PLE helps (ple_pass).
    assert orch._ple_classification(-0.02, -0.01, 0.01, 0.005) == "ple_pass"
    # Entire CI above margin -> PLE hurts (ple_fail).
    assert orch._ple_classification(0.02, 0.03, 0.01, 0.005) == "ple_fail"
    # CI straddles zero/margin -> unproven.
    assert orch._ple_classification(-0.01, 0.02, 0.01, 0.005) == "ple_unproven"
    assert orch._ple_classification(0.0, 0.005, 0.01, 0.005) == "ple_unproven"
    # CI entirely below zero but within parity tolerance -> unproven.
    assert orch._ple_classification(-0.004, -0.002, 0.01, 0.005) == "ple_unproven"


def test_final_scaling_policy_decision() -> None:
    # Quality win (CI excludes zero in hybrid's favor) + efficiency -> GO.
    assert orch._final_scaling_policy(
        {"ci_high": -0.01}, {"ci_high": -0.02}, 0.02, 0.20, 0.15) == "GO_TO_1B_SCALING"
    # Noninferior (upper CI within tolerance) + efficiency -> GO.
    assert orch._final_scaling_policy(
        {"ci_high": 0.01}, {"ci_high": 0.01}, 0.02, 0.20, 0.15) == "GO_TO_1B_SCALING"
    # Noninferior but efficiency below threshold -> NEEDS_SEED_CONFIRMATION.
    assert orch._final_scaling_policy(
        {"ci_high": 0.01}, {"ci_high": 0.01}, 0.02, 0.10, 0.15) == "NEEDS_SEED_CONFIRMATION"
    # Inferior (upper CI beyond tolerance) -> NEEDS_SEED_CONFIRMATION.
    assert orch._final_scaling_policy(
        {"ci_high": 0.05}, {"ci_high": 0.05}, 0.02, 0.20, 0.15) == "NEEDS_SEED_CONFIRMATION"


def test_median_throughput_reads_metrics(isolated_orch: Path) -> None:
    records = [
        {"tok_per_sec": 100.0}, {"tok_per_sec": 200.0}, {"tok_per_sec": 300.0},
    ]
    _write_metrics(isolated_orch, "A", records)
    assert orch._median_throughput("A") == 200.0
    # No metrics -> None.
    assert orch._median_throughput("B") is None


def test_production_state_unchanged_by_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the orchestrator test suite must not mutate production state.

    Snapshots the real production state path before running a representative
    orchestrator operation against an isolated root, then asserts the
    production path is byte-for-byte unchanged.
    """
    # Isolate from the real freeze check (dirty tree in the test environment).
    monkeypatch.setattr(orch, "_verify_freeze_and_environment", lambda state: state)
    before_exists, before_bytes = _snapshot_production_state()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        orch.RUN_ROOT = tmp_path
        orch.STATE_PATH = tmp_path / "state.json"
        orch.FREEZE_MANIFEST = tmp_path / "freeze_manifest.json"
        try:
            orch.main(["--dry-run"])
        finally:
            # Restore module globals to the real production values.
            orch.RUN_ROOT = PRODUCTION_RUN_ROOT
            orch.STATE_PATH = PRODUCTION_STATE_PATH
            orch.FREEZE_MANIFEST = PRODUCTION_RUN_ROOT / "freeze_manifest.json"
    after_exists, after_bytes = _snapshot_production_state()
    assert after_exists == before_exists
    if before_bytes is not None:
        assert after_bytes == before_bytes
        assert hashlib.sha256(after_bytes).hexdigest() == hashlib.sha256(before_bytes).hexdigest()
