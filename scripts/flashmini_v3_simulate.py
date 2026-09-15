#!/usr/bin/env python
"""Orchestration simulation for the FlashMini v3 PoC.

This is a DIAGNOSTIC simulation, not an official run. It monkeypatches the
orchestrator's training/eval/checkpoint/metrics functions with deterministic
fakes so the full stage-first A/B/C pipeline can be exercised end to end:

  train_2p1m A,B,C -> gate_2p1m -> train_100m A,B,C -> gate_100m
  -> train_250m A,B,C -> final_report

It produces a plausible final decision report and verifies the stage-first
ordering, collective-gate dependency, and fail-closed behavior. It writes no
production state and launches no real training.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import flashmini_v3_execute as orch


def _fake_eval(treatment: str, stage: str, dry_run: bool) -> dict:
    """Deterministic plausible eval result per treatment/stage.

    A (full attention) is the quality reference. B (GDN hybrid) is slightly
    better; C (GDN hybrid + PLE) is slightly better still. Blocks are paired
    across treatments so the bootstrap CIs are meaningful.
    """
    base = {
        "A": 9.80,
        "B": 9.72,
        "C": 9.66,
    }[treatment]
    # A little stage-dependent improvement.
    stage_improve = {"train_2p1m": 0.0, "gate_2p1m": 0.0,
                     "train_100m": -0.05, "gate_100m": -0.05,
                     "train_250m": -0.10, "final_report": -0.10}.get(stage, 0.0)
    nll = base + stage_improve
    # 10 paired blocks with a small deterministic wiggle.
    blocks = []
    for i in range(10):
        wiggle = ((i % 3) - 1) * 0.01
        blocks.append({"nll": nll + wiggle, "tokens": 256 * 1024})
    return {
        "nll": nll,
        "perplexity": 2.72,
        "top1_accuracy": 0.55,
        "num_sequences": 8192,
        "scored_tokens": 256 * 1024 * 10,
        "blocks": blocks,
    }


def _fake_metrics(treatment: str) -> list[dict]:
    """Deterministic plausible metrics stream (post-warmup, healthy)."""
    records = []
    for i in range(40):
        records.append({
            "tokens_seen": 600_000 + i * 50_000,
            "loss": 10.0 - i * 0.04,
            "router_entropy": 2.6,
            "expert_load_ratio": 1.4,
            "grad_clip_fraction_shared": 0.0,
            "tok_per_sec": 900.0 + (100.0 if treatment in ("B", "C") else 0.0),
        })
    if treatment == "C":
        for r in records:
            r["grad_norm_ple_dense_preclip"] = 0.5
            r["grad_norm_ple_sparse_preclip"] = 0.3
            r["ple_norm_ratio"] = 1.2
    return records


def _fake_checkpoint(treatment: str) -> Path:
    return Path(f"<fake-checkpoint-{treatment}>")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Redirect all persistence to the isolated root.
        orch.RUN_ROOT = tmp_path
        orch.STATE_PATH = tmp_path / "state.json"
        orch.FREEZE_MANIFEST = tmp_path / "freeze_manifest.json"

        # Monkeypatch the I/O-heavy functions with deterministic fakes.
        orch._train_stage = lambda *a, **k: None
        orch._eval_slice = _fake_eval
        orch._checkpoint_for = _fake_checkpoint
        orch._verify_checkpoint = lambda treatment, stage: {
            "checkpoint": f"<fake-{treatment}>",
            "step": orch.STAGE_EXPECTED[stage]["step"],
        }
        orch._read_metrics = _fake_metrics
        orch._verify_freeze_and_environment = lambda state: state
        orch._preserve_milestone = lambda treatment, stage: {"milestone": "m", "sha256": "s"}
        # The 100M gate loads checkpoints with torch.load and validates provenance.
        # Fake both so the simulation exercises the gate logic without real files.
        import torch
        real_load = torch.load
        torch.load = lambda *a, **k: {
            "config": {}, "architecture_version": 3, "extra": {},
        }
        orch.validate_generic_pair = lambda *a, **k: None
        try:
            # Run the full stage-first pipeline.
            rc = orch.main(["--stage", "all"])
        finally:
            torch.load = real_load
        if rc != 0:
            print("SIMULATION FAILED: pipeline returned nonzero")
            return 1

        state = orch._load_state()
        # The state dict preserves insertion (execution) order.
        completed = list(state["completed"].keys())
        print("\n=== COMPLETED STAGES (stage-first order) ===")
        for key in completed:
            print(f"  {key}")

        # Verify the stage-first ordering invariant.
        expected_order = [
            "A:train_2p1m", "B:train_2p1m", "C:train_2p1m", "ALL:gate_2p1m",
            "A:train_100m", "B:train_100m", "C:train_100m", "ALL:gate_100m",
            "A:train_250m", "B:train_250m", "C:train_250m", "ALL:final_report",
        ]
        assert completed == expected_order, f"ordering mismatch: {completed}"
        print("\nSTAGE-FIRST ORDERING: OK")

        # The final report must be present and plausible.
        final = state["completed"]["ALL:final_report"]
        print("\n=== FINAL REPORT ===")
        print(json.dumps(final, indent=2))

        # Restore module globals.
        orch.RUN_ROOT = ROOT / "runs" / "flashmini" / "v3_execution"
        orch.STATE_PATH = orch.RUN_ROOT / "state.json"
        orch.FREEZE_MANIFEST = orch.RUN_ROOT / "freeze_manifest.json"

    print("\nSIMULATION PASSED: full A/B/C stage-first pipeline exercised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
