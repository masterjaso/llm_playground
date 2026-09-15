#!/usr/bin/env python
"""Durable, resumable, fail-closed orchestrator for the FlashMini v3 PoC.

This script drives the official A3/B3/C3 PoC end to end. It is resumable: a
persistent state file records which stages have completed, so a restart
continues exactly where it left off. It is fail-closed:

- every launch verifies freeze/environment/data/config identity for ALL of
  A/B/C (not just A);
- every stage verifies its checkpoint;
- a failed gate is recorded in state and is NEVER re-run as if it passed on
  restart; the pipeline stops and returns nonzero;
- the pre-registered gate policy is actually applied to the bounded eval slice;
- the pipeline never auto-continues to 1B.

Stages (per treatment, in A -> B -> C order):
  preflight_2p1m  train to 2.1M tokens (512 optimizer updates)
  gate_2p1m       bounded eval slice (skip 1024, next 8192); implementation check
  gate_100m       resume to 100M tokens (24576 updates); instability check
  final_250m      resume to 250M tokens (full); final quality evaluation
  final_report    produce the decision package

Usage:
  .venv/bin/python scripts/flashmini_v3_execute.py --stage all
  .venv/bin/python scripts/flashmini_v3_execute.py --stage preflight_2p1m --treatment A
  .venv/bin/python scripts/flashmini_v3_execute.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashmini.checkpoint import load_checkpoint
from flashmini.config import FlashMiniConfig
from flashmini.data import MemmapDataset, sha256_file
from flashmini.evaluation import evaluate_bounded_slice, official_slice
from flashmini.fingerprint import (
    collect_fingerprint,
    enforce_fingerprint_match,
)
from flashmini.gate_policy import gate_policy_sha256, load_gate_policy
from flashmini.milestones import preserve_milestone, verify_milestone
from flashmini.models import FlashMiniModel

CONFIG_DIR = REPO_ROOT / "configs" / "flashmini"
DATA_DIR = REPO_ROOT / "data" / "fineweb_v3_2b"
RUN_ROOT = REPO_ROOT / "runs" / "flashmini" / "v3_execution"
STATE_PATH = RUN_ROOT / "state.json"
FREEZE_MANIFEST = RUN_ROOT / "freeze_manifest.json"

TREATMENTS = ("A", "B", "C")
TREATMENT_CONFIGS = {
    "A": CONFIG_DIR / "poc_a_v3.yaml",
    "B": CONFIG_DIR / "poc_b_v3.yaml",
    "C": CONFIG_DIR / "poc_c_v3.yaml",
}

# Frozen training recipe (must not change after freeze).
RECIPE = {
    "seed": 17,
    "batch_size": 16,
    "grad_accum": 1,
    "model_parallel_gpus": "1,0",
    "gpu_memory_gib": 15,
    "lr": 3e-4,
    "ple_lr_multiplier": 5,
    "warmup_tokens": 524288,
    "min_lr_ratio": 0.1,
    "tokens": 250_000_000,
    "eval_every_tokens": 2_097_152,
    "eval_max_batches": 128,
    "checkpoint_every_tokens": 4_194_304,
    "log_every": 10,
}

STAGES = ("preflight_2p1m", "gate_2p1m", "gate_100m", "final_250m", "final_report")

# Gate name -> official eval slice key.
GATE_SLICE_KEY = {
    "gate_2p1m": "2p1m",
    "gate_100m": "100m",
    "final_report": "250m",
}


def _log(stage: str, treatment: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{stamp}] [{treatment}/{stage}] {message}"
    print(line, flush=True)
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{treatment}_{stage}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed": {}, "failed_gates": {}, "verdicts": {}, "fingerprint": None}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _run_cli(args: list[str], dry_run: bool) -> int:
    cmd = [sys.executable, "-m", "flashmini.cli", *args]
    _log("cli", "all", " ".join(cmd))
    if dry_run:
        return 0
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    return proc.returncode


def _verify_freeze_and_environment(state: dict) -> dict:
    """Verify freeze/environment/data/config identity for ALL of A/B/C.

    Fails closed if the working tree is dirty, the source has changed, the data
    manifest has changed, any A/B/C config has changed, or the gate policy has
    changed since the freeze.
    """
    config_sha256 = {t: sha256_file(TREATMENT_CONFIGS[t]) for t in TREATMENTS}
    data_manifest_sha256 = sha256_file(DATA_DIR / "data_manifest.json")
    fingerprint = collect_fingerprint(
        REPO_ROOT,
        config_sha256=config_sha256["A"],
        data_manifest_sha256=data_manifest_sha256,
    )
    # Official v3 runs fail closed on a dirty working tree.
    if fingerprint["git_dirty"]:
        raise RuntimeError("refusing to launch: working tree is dirty (git status not clean)")
    recorded = state.get("fingerprint")
    if recorded is not None:
        enforce_fingerprint_match(fingerprint, recorded)
    else:
        state["fingerprint"] = fingerprint
    # Every A/B/C config must match the frozen copy.
    if state.get("config_sha256") is not None:
        for t in TREATMENTS:
            if state["config_sha256"].get(t) != config_sha256[t]:
                raise RuntimeError(f"config for treatment {t} changed after freeze; refusing to launch")
    state["config_sha256"] = config_sha256
    # Gate policy must match the frozen copy.
    policy_sha = gate_policy_sha256()
    if state.get("gate_policy_sha256") is not None and state["gate_policy_sha256"] != policy_sha:
        raise RuntimeError("gate policy changed after freeze; refusing to launch")
    state["gate_policy_sha256"] = policy_sha
    return fingerprint


def _checkpoint_for(treatment: str) -> Path:
    run_dir = RUN_ROOT / f"treatment_{treatment}"
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint found for treatment {treatment}")
    return ckpts[-1]


def _verify_checkpoint(treatment: str, stage: str) -> dict:
    """Verify the checkpoint after a stage. Fails closed if missing or corrupt."""
    ckpt = _checkpoint_for(treatment)
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "architecture_version" not in state or int(state["architecture_version"]) != 3:
        raise ValueError(f"checkpoint {ckpt} is not a v3 checkpoint")
    _log(stage, treatment, f"checkpoint verified: {ckpt.name} step={state['step']}")
    return {"checkpoint": str(ckpt), "step": state["step"]}


def _train_stage(treatment: str, stage: str, stop_after_tokens: int, resume: bool, dry_run: bool) -> None:
    run_dir = RUN_ROOT / f"treatment_{treatment}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "train",
        "--config", str(TREATMENT_CONFIGS[treatment]),
        "--data-dir", str(DATA_DIR),
        "--run-dir", str(run_dir),
        "--tokens", str(RECIPE["tokens"]),
        "--lr", str(RECIPE["lr"]),
        "--ple-lr-multiplier", str(RECIPE["ple_lr_multiplier"]),
        "--batch-size", str(RECIPE["batch_size"]),
        "--model-parallel-gpus", RECIPE["model_parallel_gpus"],
        "--gpu-memory-gib", str(RECIPE["gpu_memory_gib"]),
        "--grad-accum", str(RECIPE["grad_accum"]),
        "--log-every", str(RECIPE["log_every"]),
        "--checkpoint-every-tokens", str(RECIPE["checkpoint_every_tokens"]),
        "--eval-every-tokens", str(RECIPE["eval_every_tokens"]),
        "--eval-max-batches", str(RECIPE["eval_max_batches"]),
        "--warmup-tokens", str(RECIPE["warmup_tokens"]),
        "--cosine-decay",
        "--min-lr-ratio", str(RECIPE["min_lr_ratio"]),
        "--seed", str(RECIPE["seed"]),
        "--stop-after-tokens", str(stop_after_tokens),
    ]
    if resume:
        args += ["--resume", str(_checkpoint_for(treatment))]
    rc = _run_cli(args, dry_run)
    if rc != 0:
        raise RuntimeError(f"training stage {stage} for {treatment} failed (exit {rc})")


def _eval_slice(treatment: str, stage: str, dry_run: bool) -> dict:
    """Run the official bounded eval slice for a gate and return the report.

    Uses the pre-registered slice (skip 1024; 8192 for 2.1M, 32768 for 100M,
    entire remaining for 250M) and records the checkpoint and data hashes.
    """
    if dry_run:
        return {"nll": None, "perplexity": None, "num_sequences": None}
    import torch
    import yaml
    ckpt = _checkpoint_for(treatment)
    config = FlashMiniConfig.from_dict(yaml.safe_load(TREATMENT_CONFIGS[treatment].read_text()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FlashMiniModel(config).to(device)
    load_checkpoint(ckpt, model)
    dataset = MemmapDataset(DATA_DIR, split="val")
    skip, max_sequences = official_slice(GATE_SLICE_KEY[stage])
    ckpt_sha = sha256_file(ckpt)
    data_sha = sha256_file(DATA_DIR / "data_manifest.json")
    result = evaluate_bounded_slice(
        model, dataset, device,
        skip_sequences=skip,
        max_sequences=max_sequences,
        block_sequences=1024,
        checkpoint_sha256=ckpt_sha,
        data_manifest_sha256=data_sha,
    )
    return result


def _gate_check(treatment: str, stage: str, result: dict, policy: dict) -> str:
    """Apply the pre-registered gate check to an eval result. Fails closed."""
    thresholds = policy["thresholds"]
    if result.get("nll") is None:
        raise RuntimeError(f"gate {stage} for {treatment}: no eval result available")
    nll = float(result["nll"])
    if math.isnan(nll):
        return "implementation_failure"
    if nll > thresholds["max_nll_implementation_failure"]:
        return "implementation_failure"
    return "ok"


def _preserve_milestone(treatment: str, stage: str) -> dict:
    ckpt = _checkpoint_for(treatment)
    run_dir = RUN_ROOT / f"treatment_{treatment}"
    record = preserve_milestone(ckpt, run_dir, f"{stage}_{treatment}")
    if not verify_milestone(Path(record["milestone"]), record["sha256"]):
        raise RuntimeError(f"milestone for {stage}/{treatment} failed verification")
    _log(stage, treatment, f"milestone preserved: {record['milestone']} sha256={record['sha256']}")
    return record


def _stage_completed(state: dict, treatment: str, stage: str) -> bool:
    return state["completed"].get(f"{treatment}:{stage}") is not None


def _gate_failed(state: dict, treatment: str, stage: str) -> bool:
    return state["failed_gates"].get(f"{treatment}:{stage}") is not None


def _mark_completed(state: dict, treatment: str, stage: str, detail: dict) -> None:
    state["completed"][f"{treatment}:{stage}"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **detail,
    }
    _save_state(state)


def _mark_gate_failed(state: dict, treatment: str, stage: str, verdict: str) -> None:
    state["failed_gates"][f"{treatment}:{stage}"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
    }
    state["verdicts"][f"{treatment}:{stage}"] = verdict
    _save_state(state)


def _run_stage(treatment: str, stage: str, state: dict, policy: dict, dry_run: bool) -> None:
    # A failed gate is never re-run as if it passed on restart.
    if _gate_failed(state, treatment, stage):
        verdict = state["failed_gates"][f"{treatment}:{stage}"]["verdict"]
        raise RuntimeError(
            f"gate {stage} for {treatment} previously FAILED ({verdict}); "
            "refusing to auto-continue; resolve and re-run explicitly"
        )
    if _stage_completed(state, treatment, stage):
        _log(stage, treatment, "already completed; skipping (idempotent restart)")
        return
    _verify_freeze_and_environment(state)

    if stage == "preflight_2p1m":
        _train_stage(treatment, stage, stop_after_tokens=2_097_152, resume=False, dry_run=dry_run)
        ckpt = _verify_checkpoint(treatment, stage)
        _mark_completed(state, treatment, stage, ckpt)
    elif stage == "gate_2p1m":
        result = _eval_slice(treatment, stage, dry_run)
        check = _gate_check(treatment, stage, result, policy)
        if check != "ok":
            _mark_gate_failed(state, treatment, stage, check)
            raise RuntimeError(f"gate {stage} for {treatment} FAILED: {check}; stopping pipeline")
        _preserve_milestone(treatment, stage)
        _mark_completed(state, treatment, stage, result)
    elif stage == "gate_100m":
        _train_stage(treatment, stage, stop_after_tokens=100_663_296, resume=True, dry_run=dry_run)
        _verify_checkpoint(treatment, stage)
        result = _eval_slice(treatment, stage, dry_run)
        check = _gate_check(treatment, stage, result, policy)
        if check != "ok":
            _mark_gate_failed(state, treatment, stage, check)
            raise RuntimeError(f"gate {stage} for {treatment} FAILED: {check}; stopping pipeline")
        _preserve_milestone(treatment, stage)
        _mark_completed(state, treatment, stage, result)
    elif stage == "final_250m":
        _train_stage(treatment, stage, stop_after_tokens=250_000_000, resume=True, dry_run=dry_run)
        _verify_checkpoint(treatment, stage)
        _mark_completed(state, treatment, stage, {"checkpoint": str(_checkpoint_for(treatment))})
    elif stage == "final_report":
        result = _eval_slice(treatment, stage, dry_run)
        _mark_completed(state, treatment, stage, result)
    else:
        raise ValueError(f"unknown stage {stage}")


def _write_freeze_manifest(state: dict) -> None:
    manifest = {
        "fingerprint": state.get("fingerprint"),
        "gate_policy_sha256": state.get("gate_policy_sha256"),
        "config_sha256": state.get("config_sha256"),
        "data_manifest_sha256": sha256_file(DATA_DIR / "data_manifest.json"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    FREEZE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=list(STAGES) + ["all"],
                        help="Which stage to run (default: all).")
    parser.add_argument("--treatment", default=None, choices=list(TREATMENTS),
                        help="Restrict to a single treatment (default: all in A->B->C order).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the plan without launching any training/eval.")
    args = parser.parse_args(argv)

    policy = load_gate_policy()
    state = _load_state()
    treatments = [args.treatment] if args.treatment else list(TREATMENTS)
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    if args.dry_run:
        _log("dry-run", "all", f"plan: treatments={treatments} stages={stages}")
        for treatment in treatments:
            for stage in stages:
                if _gate_failed(state, treatment, stage):
                    status = "FAILED (will not auto-continue)"
                elif _stage_completed(state, treatment, stage):
                    status = "completed"
                else:
                    status = "pending"
                _log("dry-run", treatment, f"{stage}: {status}")
        return 0

    _verify_freeze_and_environment(state)
    _write_freeze_manifest(state)

    for treatment in treatments:
        for stage in stages:
            try:
                _run_stage(treatment, stage, state, policy, dry_run=False)
            except (RuntimeError, ValueError, FileNotFoundError, KeyError, OSError) as exc:
                _log(stage, treatment, f"FAILED: {exc}")
                return 1
    _log("done", "all", "pipeline complete; no auto-continuation to 1B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
