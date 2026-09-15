#!/usr/bin/env python
"""Durable, resumable, fail-closed orchestrator for the FlashMini v3 PoC.

This script drives the official A3/B3/C3 PoC end to end. It is resumable: a
persistent state file records which stages have completed, so a restart
continues exactly where it left off. It is fail-closed:

- every launch verifies freeze/environment/data/config identity for ALL of
  A/B/C (not just A);
- every stage verifies its checkpoint with exact token/step/provenance checks;
- a failed gate is recorded in state and is NEVER re-run as if it passed on
  restart; the pipeline stops and returns nonzero;
- the pre-registered gate policy is actually applied to the bounded eval slice;
- the pipeline never auto-continues to 1B.

Stages are ordered STAGE-FIRST (not treatment-first):
  train_2p1m  A, B, C          (train each treatment to 2.1M tokens)
  gate_2p1m   ALL              (collective 2.1M health gate over A/B/C)
  train_100m  A, B, C          (resume each treatment to 100M tokens)
  gate_100m   ALL              (collective 100M gate over A/B/C)
  train_250m  A, B, C          (resume each treatment to 250M tokens)
  final_report ALL              (collective final decision package)

A collective gate refuses to run unless all three treatment checkpoints exist
and satisfy the exact stage invariants. A collective gate rejects ``--treatment``.

Usage:
  .venv/bin/python scripts/flashmini_v3_execute.py --stage all
  .venv/bin/python scripts/flashmini_v3_execute.py --stage train_2p1m --treatment A
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
from flashmini.comparison import validate_generic_pair
from flashmini.config import FlashMiniConfig
from flashmini.data import MemmapDataset, sha256_file
from flashmini.evaluation import evaluate_bounded_slice, official_slice
from flashmini.fingerprint import (
    collect_fingerprint,
    enforce_fingerprint_match,
    environment_fingerprint_sha256,
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

TRAINING_STAGES = ("train_2p1m", "train_100m", "train_250m")
COLLECTIVE_STAGES = ("gate_2p1m", "gate_100m", "final_report")
STAGES = TRAINING_STAGES + COLLECTIVE_STAGES

# Gate name -> official eval slice key.
GATE_SLICE_KEY = {
    "gate_2p1m": "2p1m",
    "gate_100m": "100m",
    "final_report": "250m",
}

# Exact stage invariants (step, tokens_seen) for checkpoint verification.
STAGE_TOKENS = {
    "train_2p1m": 2_097_152,
    "train_100m": 100_663_296,
    "train_250m": 250_000_000,
}
STAGE_EXPECTED = {
    "train_2p1m": {"step": 512, "tokens_seen": 2_097_152},
    "train_100m": {"step": 24_576, "tokens_seen": 100_663_296},
    "train_250m": {"step": 61_036, "tokens_seen": 250_000_128},
}

# Deterministic bootstrap seed for paired block-bootstrap CIs.
BOOTSTRAP_SEED = 17
BOOTSTRAP_RESAMPLES = 10_000

# The efficiency dimension under test is training throughput (tokens/sec).
# The GDN hybrid (B/C) is expected to be faster than full attention (A) because
# GatedDeltaNet uses linear-time recurrence. The advantage is MEASURED from the
# logged tok_per_sec, not declared.


def full_plan() -> list[tuple[str, str]]:
    """Return the stage-first orchestration plan as (stage, treatment) pairs.

    Stage-first: each training stage runs for A, B, C in order, then the
    collective gate for that stage runs over ALL. The final report is collective.
    """
    return [
        ("train_2p1m", "A"), ("train_2p1m", "B"), ("train_2p1m", "C"),
        ("gate_2p1m", "ALL"),
        ("train_100m", "A"), ("train_100m", "B"), ("train_100m", "C"),
        ("gate_100m", "ALL"),
        ("train_250m", "A"), ("train_250m", "B"), ("train_250m", "C"),
        ("final_report", "ALL"),
    ]


def plan_for(stage: str, treatment: str | None) -> list[tuple[str, str]]:
    """Return the plan for a single named stage.

    Collective stages reject ``--treatment``; training stages run for the
    given treatment (or all three in A->B->C order).
    """
    if stage in COLLECTIVE_STAGES:
        if treatment is not None:
            raise ValueError(f"collective stage {stage} does not accept --treatment")
        return [(stage, "ALL")]
    if stage in TRAINING_STAGES:
        treatments = [treatment] if treatment else list(TREATMENTS)
        return [(stage, t) for t in treatments]
    raise ValueError(f"unknown stage {stage}")


def _log(stage: str, treatment: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{stamp}] [{treatment}/{stage}] {message}"
    print(line, flush=True)
    # Dry-run is diagnostic only and must not mutate production state.
    if stage == "dry-run":
        return
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
    """Discover the checkpoint for a treatment. Fails closed on ambiguity.

    0 checkpoints -> FileNotFoundError. 1 checkpoint -> return it.
    >1 checkpoints -> RuntimeError (fail closed; print every candidate).
    """
    run_dir = RUN_ROOT / f"treatment_{treatment}"
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint found for treatment {treatment}")
    if len(ckpts) > 1:
        candidates = [str(p) for p in ckpts]
        raise RuntimeError(
            f"ambiguous checkpoint set for treatment {treatment}: "
            f"{len(ckpts)} candidates {candidates}; refusing to choose one"
        )
    return ckpts[0]


def _verify_checkpoint(treatment: str, stage: str) -> dict:
    """Verify the checkpoint after a stage with exact token/step/provenance.

    Fails closed if the checkpoint is missing, corrupt, not v3, has the wrong
    step or tokens_seen for the stage, or has a provenance mismatch (data
    manifest, source SHA, config SHA, environment fingerprint, optimizer state,
    RNG state, clipping counters).
    """
    ckpt = _checkpoint_for(treatment)
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "architecture_version" not in state or int(state["architecture_version"]) != 3:
        raise ValueError(f"checkpoint {ckpt} is not a v3 checkpoint")
    expected = STAGE_EXPECTED[stage]
    if state["step"] != expected["step"]:
        raise ValueError(
            f"checkpoint step mismatch for {stage}/{treatment}: "
            f"expected {expected['step']}, got {state['step']}"
        )
    extra = state.get("extra", {})
    if extra.get("tokens_seen") != expected["tokens_seen"]:
        raise ValueError(
            f"checkpoint tokens_seen mismatch for {stage}/{treatment}: "
            f"expected {expected['tokens_seen']}, got {extra.get('tokens_seen')}"
        )
    # Provenance checks.
    training = extra.get("training", {})
    run_metadata = training.get("run_metadata", {})
    data_manifest_sha256 = extra.get("data_manifest_sha256")
    if data_manifest_sha256 != sha256_file(DATA_DIR / "data_manifest.json"):
        raise ValueError(f"checkpoint data manifest mismatch for {stage}/{treatment}")
    config_sha256 = run_metadata.get("config_sha256")
    if config_sha256 != sha256_file(TREATMENT_CONFIGS[treatment]):
        raise ValueError(f"checkpoint config SHA mismatch for {stage}/{treatment}")
    fingerprint = run_metadata.get("execution_fingerprint")
    if fingerprint is None:
        raise ValueError(f"checkpoint missing execution fingerprint for {stage}/{treatment}")
    env_sha = environment_fingerprint_sha256(fingerprint)
    current_fp = collect_fingerprint(
        REPO_ROOT,
        config_sha256=sha256_file(TREATMENT_CONFIGS[treatment]),
        data_manifest_sha256=data_manifest_sha256,
    )
    if env_sha != environment_fingerprint_sha256(current_fp):
        raise ValueError(f"checkpoint environment fingerprint mismatch for {stage}/{treatment}")
    if state.get("optimizer_state_dict") is None:
        raise ValueError(f"checkpoint missing optimizer state for {stage}/{treatment}")
    if "rng_state" not in extra:
        raise ValueError(f"checkpoint missing RNG state for {stage}/{treatment}")
    if "clipping_counts" not in extra:
        raise ValueError(f"checkpoint missing clipping counters for {stage}/{treatment}")
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
    """Apply the pre-registered NLL implementation check to an eval result."""
    thresholds = policy["thresholds"]
    if result.get("nll") is None:
        raise RuntimeError(f"gate {stage} for {treatment}: no eval result available")
    nll = float(result["nll"])
    # max_nan is the maximum number of NaN observations allowed; 0 means any
    # NaN is an implementation failure.
    if math.isnan(nll) and thresholds["max_nan"] < 1:
        return "implementation_failure"
    if nll > thresholds["max_nll_implementation_failure"]:
        return "implementation_failure"
    return "ok"


def _read_metrics(treatment: str) -> list[dict]:
    """Read the metrics.jsonl for a treatment. Returns a list of records."""
    run_dir = RUN_ROOT / f"treatment_{treatment}"
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _post_warmup_records(records: list[dict]) -> list[dict]:
    """Return logged records after the warmup boundary (tokens_seen > warmup)."""
    warmup = RECIPE["warmup_tokens"]
    return [r for r in records if r.get("tokens_seen", 0) > warmup]


def _training_health(treatment: str, policy: dict) -> tuple[str, dict]:
    """Compute the 2.1M training health verdict for one treatment.

    Returns (verdict, detail). verdict is "ok", "implementation_failure",
    "training_instability", or "ambiguous_review_required".
    """
    thresholds = policy["thresholds"]
    records = _read_metrics(treatment)
    if not records:
        return "implementation_failure", {"reason": "no metrics recorded"}
    losses = [r["loss"] for r in records if "loss" in r and math.isfinite(r["loss"])]
    if not losses:
        return "implementation_failure", {"reason": "no finite losses"}
    # Loss improvement: median(last 5) <= 0.95 * median(first 5).
    first5 = sorted(losses[:5])
    last5 = sorted(losses[-5:])
    med_first = first5[len(first5) // 2]
    med_last = last5[len(last5) // 2]
    loss_improvement_ok = med_last <= 0.95 * med_first
    post_warmup = _post_warmup_records(records)
    detail = {
        "loss_median_first5": med_first,
        "loss_median_last5": med_last,
        "loss_improvement_ok": loss_improvement_ok,
    }
    if not loss_improvement_ok:
        return "training_instability", detail
    # Loss spike: within any window of loss_spike_window_updates consecutive
    # post-warmup updates, the max loss must not exceed loss_spike_ratio times
    # the window median.
    spike_window = thresholds["loss_spike_window_updates"]
    spike_ratio = thresholds["loss_spike_ratio"]
    post_warmup_losses = [r["loss"] for r in post_warmup
                          if "loss" in r and math.isfinite(r["loss"])]
    spike_ok = True
    spike_max_ratio = 0.0
    if len(post_warmup_losses) >= spike_window:
        for i in range(len(post_warmup_losses) - spike_window + 1):
            window = post_warmup_losses[i:i + spike_window]
            med = sorted(window)[len(window) // 2]
            if med > 0:
                ratio = max(window) / med
                spike_max_ratio = max(spike_max_ratio, ratio)
                if ratio > spike_ratio:
                    spike_ok = False
                    break
    detail["loss_spike_max_ratio"] = spike_max_ratio
    detail["loss_spike_ok"] = spike_ok
    if not spike_ok:
        return "training_instability", detail
    # Router health: after warmup, router entropy finite and latest > 1.0.
    router_entropies = [r["router_entropy"] for r in post_warmup
                       if "router_entropy" in r and math.isfinite(r["router_entropy"])]
    if not router_entropies:
        return "implementation_failure", {**detail, "reason": "no post-warmup router entropy"}
    latest_router = router_entropies[-1]
    router_ok = latest_router > thresholds["router_entropy_floor"]
    detail["router_entropy_latest"] = latest_router
    detail["router_entropy_ok"] = router_ok
    if not router_ok:
        return "training_instability", detail
    # Expert routing health: expert_load_ratio > 4.0 in no more than 25% of
    # post-warmup observations.
    ratios = [r["expert_load_ratio"] for r in post_warmup
              if "expert_load_ratio" in r and math.isfinite(r["expert_load_ratio"])]
    if not ratios:
        return "implementation_failure", {**detail, "reason": "no post-warmup expert load ratio"}
    exceed = sum(1 for x in ratios if x > thresholds["expert_load_ratio_threshold"])
    exceed_fraction = exceed / len(ratios)
    expert_ok = exceed_fraction <= thresholds["expert_load_exceedance_max_fraction"]
    detail["expert_load_exceed_fraction"] = exceed_fraction
    detail["expert_load_ok"] = expert_ok
    if not expert_ok:
        return "training_instability", detail
    # C-specific PLE health.
    if treatment == "C":
        ple_dense = [r["grad_norm_ple_dense_preclip"] for r in post_warmup
                     if "grad_norm_ple_dense_preclip" in r and math.isfinite(r["grad_norm_ple_dense_preclip"])]
        ple_sparse = [r["grad_norm_ple_sparse_preclip"] for r in post_warmup
                      if "grad_norm_ple_sparse_preclip" in r and math.isfinite(r["grad_norm_ple_sparse_preclip"])]
        ple_norm_ratios = [r["ple_norm_ratio"] for r in post_warmup
                          if "ple_norm_ratio" in r and math.isfinite(r["ple_norm_ratio"])]
        if not ple_dense or not ple_sparse or not ple_norm_ratios:
            return "implementation_failure", {**detail, "reason": "missing PLE metrics"}
        if not any(x > 0 for x in ple_dense) or not any(x > 0 for x in ple_sparse):
            return "training_instability", {**detail, "reason": "PLE gradients all zero"}
        detail["ple_health_ok"] = True
    # Clipping: if cumulative/shared clipping fraction > 98% -> AMBIGUOUS_REVIEW_REQUIRED.
    shared_clip = [r["grad_clip_fraction_shared"] for r in records
                   if "grad_clip_fraction_shared" in r and math.isfinite(r["grad_clip_fraction_shared"])]
    if shared_clip:
        latest_shared_clip = shared_clip[-1]
        if latest_shared_clip > thresholds["shared_clipping_ambiguity_threshold"]:
            return "ambiguous_review_required", {**detail, "shared_clip_fraction": latest_shared_clip}
    return "ok", detail


def _paired_block_bootstrap_ci(blocks_a: list[float], blocks_b: list[float],
                              seed: int = BOOTSTRAP_SEED,
                              resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Paired block-bootstrap 95% CI for (B - A) NLL.

    Returns (point_estimate, ci_low, ci_high). Uses a deterministic seed.
    """
    import random
    if len(blocks_a) != len(blocks_b):
        raise ValueError("paired blocks must have equal length")
    n = len(blocks_a)
    if n == 0:
        raise ValueError("no blocks for bootstrap")
    deltas = [b - a for a, b in zip(blocks_a, blocks_b)]
    point = sum(deltas) / n
    rng = random.Random(seed)
    block_size = max(1, n // 10)
    # Precompute the mean of each contiguous block.
    block_means: list[float] = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_means.append(sum(deltas[start:end]) / (end - start))
    n_blocks = len(block_means)
    samples = []
    for _ in range(resamples):
        # Resample blocks with replacement and average their means.
        chosen = [block_means[rng.randrange(n_blocks)] for _ in range(n_blocks)]
        samples.append(sum(chosen) / n_blocks)
    samples.sort()
    ci_low = samples[int(0.025 * resamples)]
    ci_high = samples[int(0.975 * resamples)]
    return point, ci_low, ci_high


def _catastrophic_hybrid_failure(a_nll: float, b_nll: float, c_nll: float,
                                 ba_low: float, ca_low: float,
                                 threshold: float) -> bool:
    """Pure decision: both hybrids regress beyond ``threshold`` AND the paired
    CIs exclude zero in the hybrid's disfavor."""
    rel_b = (b_nll - a_nll) / a_nll
    rel_c = (c_nll - a_nll) / a_nll
    return rel_b > threshold and rel_c > threshold and ba_low > 0 and ca_low > 0


def _ple_classification(cb_low: float, cb_high: float, margin: float,
                        parity_tolerance: float) -> str:
    """Pure PLE decision from the C-B paired CI.

    ple_pass: the entire CI is below -parity_tolerance (C meaningfully better
    than B by at least the pre-registered parity tolerance).
    ple_fail: the entire CI is above the pre-registered effective margin
    (C meaningfully worse than B).
    ple_unproven: otherwise.
    """
    if cb_high < -parity_tolerance:
        return "ple_pass"
    if cb_low > margin:
        return "ple_fail"
    return "ple_unproven"


def _final_scaling_policy(ba: dict, ca: dict, tolerance: float,
                          efficiency_advantage: float,
                          efficiency_threshold: float) -> str:
    """Pure final scaling decision from the B-A and C-A paired CIs.

    GO_TO_1B_SCALING if the hybrid is at least as good as full attention in
    quality (a quality win, or noninferiority within the parity tolerance) AND
    the hybrid's architectural efficiency advantage meets the pre-registered
    threshold. Otherwise NEEDS_SEED_CONFIRMATION.
    """
    quality_win = (ca["ci_high"] < 0) or (ba["ci_high"] < 0)
    noninferior = (ca["ci_high"] <= tolerance) and (ba["ci_high"] <= tolerance)
    efficiency_ok = efficiency_advantage >= efficiency_threshold
    return "GO_TO_1B_SCALING" if ((quality_win or noninferior) and efficiency_ok) else "NEEDS_SEED_CONFIRMATION"


def _median_throughput(treatment: str) -> float | None:
    """Return the median logged tok_per_sec for a treatment, or None."""
    records = _read_metrics(treatment)
    tps = [r["tok_per_sec"] for r in records
           if "tok_per_sec" in r and math.isfinite(r["tok_per_sec"])]
    if not tps:
        return None
    tps.sort()
    return tps[len(tps) // 2]


def _require_all_checkpoints(stage: str) -> None:
    """Fail closed unless all three treatment checkpoints exist and verify."""
    for treatment in TREATMENTS:
        _verify_checkpoint(treatment, stage)


def _collective_gate_2p1m(state: dict, policy: dict, dry_run: bool) -> dict:
    """Run the collective 2.1M health gate over ALL THREE treatments.

    Requires all three checkpoints to exist and verify. Runs the bounded eval
    slice (skip 1024, next 8192) for each treatment. Applies the training
    health checks per treatment. Returns the gate report.
    """
    _require_all_checkpoints("train_2p1m")
    report: dict = {"gate": "gate_2p1m", "treatments": {}}
    for treatment in TREATMENTS:
        result = _eval_slice(treatment, "gate_2p1m", dry_run)
        check = _gate_check(treatment, "gate_2p1m", result, policy)
        if check != "ok":
            report["treatments"][treatment] = {"verdict": check, "nll": result.get("nll")}
            continue
        health, detail = _training_health(treatment, policy)
        report["treatments"][treatment] = {
            "verdict": health,
            "nll": result.get("nll"),
            "perplexity": result.get("perplexity"),
            "num_sequences": result.get("num_sequences"),
            "health": detail,
        }
    # Overall verdict.
    verdicts = [t["verdict"] for t in report["treatments"].values()]
    if any(v == "implementation_failure" for v in verdicts):
        report["verdict"] = "implementation_failure"
    elif any(v == "training_instability" for v in verdicts):
        report["verdict"] = "training_instability"
    elif any(v == "ambiguous_review_required" for v in verdicts):
        report["verdict"] = "ambiguous_review_required"
    else:
        report["verdict"] = "pass"
    return report


def _collective_gate_100m(state: dict, policy: dict, dry_run: bool) -> dict:
    """Run the collective 100M gate over ALL THREE treatments.

    Requires all three checkpoints to exist and verify. Validates treatment
    provenance BEFORE statistics (fail-closed A vs B, B vs C, A vs C). Runs the
    bounded eval slice (skip 1024, next 32768) for each treatment. Computes
    paired block-bootstrap 95% CIs for B-A, C-A, C-B NLL. Classifies PLE.
    """
    _require_all_checkpoints("train_100m")
    # Validate treatment provenance BEFORE statistics.
    envelopes = {}
    for treatment in TREATMENTS:
        ckpt = _checkpoint_for(treatment)
        import torch
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        envelopes[treatment] = {
            "config": state["config"],
            "architecture_version": state["architecture_version"],
            "extra": state.get("extra", {}),
        }
    for pair in (("A", "B"), ("B", "C"), ("A", "C")):
        validate_generic_pair(envelopes[pair[0]], envelopes[pair[1]],
                              sha256_file(DATA_DIR / "data_manifest.json"))
    # Run eval slices.
    results = {}
    for treatment in TREATMENTS:
        results[treatment] = _eval_slice(treatment, "gate_100m", dry_run)
    report: dict = {"gate": "gate_100m", "treatments": {}}
    for treatment in TREATMENTS:
        result = results[treatment]
        report["treatments"][treatment] = {
            "nll": result.get("nll"),
            "perplexity": result.get("perplexity"),
            "num_sequences": result.get("num_sequences"),
            "blocks": [b["nll"] for b in result.get("blocks", [])],
        }
    # Paired block-bootstrap CIs (per-block NLL floats).
    a_blocks = [b["nll"] for b in results["A"].get("blocks", [])]
    b_blocks = [b["nll"] for b in results["B"].get("blocks", [])]
    c_blocks = [b["nll"] for b in results["C"].get("blocks", [])]
    if a_blocks and b_blocks and c_blocks:
        ba_point, ba_low, ba_high = _paired_block_bootstrap_ci(a_blocks, b_blocks)
        ca_point, ca_low, ca_high = _paired_block_bootstrap_ci(a_blocks, c_blocks)
        cb_point, cb_low, cb_high = _paired_block_bootstrap_ci(b_blocks, c_blocks)
        report["paired_cis"] = {
            "B_minus_A": {"point": ba_point, "ci_low": ba_low, "ci_high": ba_high},
            "C_minus_A": {"point": ca_point, "ci_low": ca_low, "ci_high": ca_high},
            "C_minus_B": {"point": cb_point, "ci_low": cb_low, "ci_high": cb_high},
        }
        # Catastrophic hybrid failure: both hybrids regress beyond the
        # pre-registered threshold AND the paired CIs exclude zero in the
        # hybrid's disfavor.
        a_nll = results["A"]["nll"]
        b_nll = results["B"]["nll"]
        c_nll = results["C"]["nll"]
        cat_threshold = policy["thresholds"]["catastrophic_hybrid_regression"]
        report["catastrophic_hybrid_failure"] = _catastrophic_hybrid_failure(
            a_nll, b_nll, c_nll, ba_low, ca_low, cat_threshold)
        # PLE classification.
        ple_margin = policy["thresholds"]["ple_effective_margin"]
        ple_parity = policy["thresholds"]["ple_parity_tolerance"]
        report["ple_verdict"] = _ple_classification(cb_low, cb_high, ple_margin, ple_parity)
    report["verdict"] = "no_go_at_100m" if report.get("catastrophic_hybrid_failure") else "pass"
    return report


def _final_report(state: dict, policy: dict, dry_run: bool) -> dict:
    """Produce the collective final decision package at 250M.

    Requires all three checkpoints to exist and verify. Evaluates validation
    sequences 1024 -> end identically for all three. Produces raw outcomes,
    pairwise outcomes with bootstrap CIs, C ablation, stability, efficiency,
    and the final scaling policy.
    """
    _require_all_checkpoints("train_250m")
    results = {}
    for treatment in TREATMENTS:
        results[treatment] = _eval_slice(treatment, "final_report", dry_run)
    report: dict = {"gate": "final_report", "treatments": {}}
    for treatment in TREATMENTS:
        result = results[treatment]
        report["treatments"][treatment] = {
            "nll": result.get("nll"),
            "perplexity": result.get("perplexity"),
            "top1_accuracy": result.get("top1_accuracy"),
            "num_sequences": result.get("num_sequences"),
            "scored_tokens": result.get("scored_tokens"),
            "blocks": [b["nll"] for b in result.get("blocks", [])],
        }
    # Pairwise outcomes with bootstrap CIs (per-block NLL floats).
    a_blocks = [b["nll"] for b in results["A"].get("blocks", [])]
    b_blocks = [b["nll"] for b in results["B"].get("blocks", [])]
    c_blocks = [b["nll"] for b in results["C"].get("blocks", [])]
    if a_blocks and b_blocks and c_blocks:
        ba_point, ba_low, ba_high = _paired_block_bootstrap_ci(a_blocks, b_blocks)
        ca_point, ca_low, ca_high = _paired_block_bootstrap_ci(a_blocks, c_blocks)
        cb_point, cb_low, cb_high = _paired_block_bootstrap_ci(b_blocks, c_blocks)
        report["pairwise"] = {
            "B_minus_A": {"point": ba_point, "ci_low": ba_low, "ci_high": ba_high},
            "C_minus_A": {"point": ca_point, "ci_low": ca_low, "ci_high": ca_high},
            "C_minus_B": {"point": cb_point, "ci_low": cb_low, "ci_high": cb_high},
        }
        a_nll = results["A"]["nll"]
        b_nll = results["B"]["nll"]
        c_nll = results["C"]["nll"]
        report["relative"] = {
            "B_minus_A": (b_nll - a_nll) / a_nll,
            "C_minus_A": (c_nll - a_nll) / a_nll,
            "C_minus_B": (c_nll - b_nll) / b_nll,
        }
    # C ablation: C PLE ON vs C PLE OFF (within-model memory reliance diagnostic).
    # This is NOT equivalent to B; it is a diagnostic of memory reliance.
    report["c_ablation"] = {
        "label": "within_model_memory_reliance_diagnostic",
        "note": "C PLE ON vs C PLE OFF is NOT equivalent to B; it measures memory reliance.",
    }
    # Final scaling policy.
    tolerance = policy["thresholds"]["quality_parity_tolerance"]
    efficiency_threshold = policy["thresholds"]["efficiency_advantage_threshold"]
    if "pairwise" in report:
        ca = report["pairwise"]["C_minus_A"]
        ba = report["pairwise"]["B_minus_A"]
        # Measured efficiency advantage: median B throughput vs A throughput.
        tps_a = _median_throughput("A")
        tps_b = _median_throughput("B")
        if tps_a and tps_b and tps_a > 0:
            efficiency_advantage = (tps_b - tps_a) / tps_a
        else:
            efficiency_advantage = None
        report["efficiency_advantage"] = efficiency_advantage
        if efficiency_advantage is None:
            report["final_scaling_policy"] = "NEEDS_SEED_CONFIRMATION"
        else:
            report["final_scaling_policy"] = _final_scaling_policy(
                ba, ca, tolerance, efficiency_advantage, efficiency_threshold)
    else:
        report["final_scaling_policy"] = "NEEDS_SEED_CONFIRMATION"
    return report


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
    """Run one (stage, treatment) unit from the stage-first plan.

    Training stages run for a single treatment. Collective gates run for "ALL"
    and require all three checkpoints to exist and verify. A failed gate is
    never re-run as if it passed on restart.
    """
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

    if stage in TRAINING_STAGES:
        resume = stage != "train_2p1m"
        _train_stage(treatment, stage, stop_after_tokens=STAGE_TOKENS[stage],
                     resume=resume, dry_run=dry_run)
        ckpt = _verify_checkpoint(treatment, stage)
        _mark_completed(state, treatment, stage, ckpt)
    elif stage == "gate_2p1m":
        report = _collective_gate_2p1m(state, policy, dry_run)
        if report["verdict"] != "pass":
            _mark_gate_failed(state, "ALL", stage, report["verdict"])
            raise RuntimeError(f"collective gate {stage} FAILED: {report['verdict']}; stopping pipeline")
        for t in TREATMENTS:
            _preserve_milestone(t, stage)
        _mark_completed(state, "ALL", stage, report)
    elif stage == "gate_100m":
        report = _collective_gate_100m(state, policy, dry_run)
        if report["verdict"] != "pass":
            _mark_gate_failed(state, "ALL", stage, report["verdict"])
            raise RuntimeError(f"collective gate {stage} FAILED: {report['verdict']}; stopping pipeline")
        for t in TREATMENTS:
            _preserve_milestone(t, stage)
        _mark_completed(state, "ALL", stage, report)
    elif stage == "final_report":
        report = _final_report(state, policy, dry_run)
        _mark_completed(state, "ALL", stage, report)
    else:
        raise ValueError(f"unknown stage {stage}")


def _write_freeze_manifest(state: dict) -> None:
    manifest = {
        "fingerprint": state.get("fingerprint"),
        "gate_policy_sha256": state.get("gate_policy_sha256"),
        "config_sha256": {t: sha256_file(TREATMENT_CONFIGS[t]) for t in TREATMENTS},
        "data_manifest_sha256": sha256_file(DATA_DIR / "data_manifest.json"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    FREEZE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=list(STAGES) + ["all"],
                        help="Which stage to run (default: all, stage-first order).")
    parser.add_argument("--treatment", default=None, choices=list(TREATMENTS),
                        help="Restrict to a single treatment (default: all in A->B->C order).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the plan without launching any training/eval.")
    args = parser.parse_args(argv)

    policy = load_gate_policy()
    state = _load_state()
    plan = full_plan() if args.stage == "all" else plan_for(args.stage, args.treatment)

    if args.dry_run:
        # Validate the freeze prerequisites (config hashes, data manifest, gate
        # policy, clean tree) WITHOUT launching training or writing state.
        _verify_freeze_and_environment(state)
        _log("dry-run", "all", "freeze prerequisites validated")
        _log("dry-run", "all", f"plan (stage-first): {plan}")
        for stage, treatment in plan:
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

    for stage, treatment in plan:
        try:
            _run_stage(treatment, stage, state, policy, dry_run=False)
        except (RuntimeError, ValueError, FileNotFoundError, KeyError, OSError) as exc:
            _log(stage, treatment, f"FAILED: {exc}")
            return 1
    _log("done", "all", "pipeline complete; no auto-continuation to 1B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
