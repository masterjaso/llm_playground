#!/usr/bin/env python
"""Resume the official PoC_D v3 run with the fastest verified executor.

Finds the newest *valid* checkpoint in the original run directory (never a
partial ``.tmp``), then launches the resumed run into a NEW directory so the
original evidence stays untouched:

    runs/flashmini/poc_d_v3_execution/          (original, never written)
    runs/flashmini/poc_d_v3_execution_fast/     (resume target)

The frozen C/D training schedule is carried over verbatim: seed, LR, PLE LR
multiplier, warmup, cosine decay, minimum LR ratio, validation cadence,
checkpoint cadence, batch size, sequence length and the 250M-token target.  Only
the execution engine may differ, and that change is authorized explicitly with
``--allow-pipeline-policy-transition`` and recorded in the run's checkpoints.

Usage:
    .venv/bin/python scripts/flashmini_poc_d_resume_fast.py --dry-run
    .venv/bin/python scripts/flashmini_poc_d_resume_fast.py
    .venv/bin/python scripts/flashmini_poc_d_resume_fast.py \\
        --schedule monolithic --checkpoint runs/.../step_9216.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SOURCE_RUN_DIR = REPO_ROOT / "runs/flashmini/poc_d_v3_execution"
TARGET_RUN_DIR = REPO_ROOT / "runs/flashmini/poc_d_v3_execution_fast"
CONFIG_PATH = REPO_ROOT / "configs/flashmini/poc_d_v3.yaml"
DATA_DIR = REPO_ROOT / "data/fineweb_v3_2b"

# Frozen PoC_D schedule (identical to scripts/flashmini_poc_d_gate_launch.py).
MEMORY_GIB = 15.0
SEED = 17
LR = 3e-4
BATCH = 16
SEQ_LEN = 256
PLE_LR_MULTIPLIER = 5
WARMUP_TOKENS = 524_288
MIN_LR_RATIO = 0.1
TOTAL_TOKENS = 250_000_000
EVAL_EVERY_TOKENS = 2_097_152
EVAL_MAX_BATCHES = 128
CHECKPOINT_EVERY_TOKENS = 4_194_304
LOG_EVERY = 10

# Executor selected from the measured stage-split / microbatch benchmark.
DEFAULT_SCHEDULE = "monolithic"
DEFAULT_MICROBATCH = None
DEFAULT_STAGE_SPLIT = None


def _checkpoint_step(path: Path) -> int | None:
    """Return the checkpoint's step number, or ``None`` when the name is unusable."""
    parts = path.stem.split("_")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def find_newest_valid_checkpoint(directories: list[Path]) -> tuple[Path, int, int]:
    """Return ``(path, step, tokens_seen)`` for the newest loadable checkpoint.

    Directories are searched newest-step-first across all of them, so a resumed
    run that already wrote a checkpoint wins over the original run's older one.
    Partial writes (``*.tmp``) and checkpoints that fail to load are skipped, so
    a crash at a checkpoint boundary costs only the tokens since the previous
    complete checkpoint.
    """
    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates.extend(
            path for path in directory.glob("step_*.pt") if _checkpoint_step(path) is not None
        )
    if not candidates:
        searched = ", ".join(str(directory) for directory in directories)
        raise SystemExit(f"no step_*.pt checkpoints in {searched}")
    candidates.sort(key=lambda path: _checkpoint_step(path) or -1, reverse=True)
    failures: list[str] = []
    for path in candidates:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # a bad checkpoint is a skip, not a crash
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        required = ("step", "model_state_dict", "optimizer_state_dict", "architecture_version")
        missing = [key for key in required if state.get(key) is None]
        if missing:
            failures.append(f"{path.name}: missing {', '.join(missing)}")
            continue
        extra = state.get("extra") or {}
        tokens_seen = int(extra.get("tokens_seen", int(state["step"]) * BATCH * SEQ_LEN))
        print(
            f"verified checkpoint {path.name}: step {state['step']}, "
            f"{tokens_seen:,} tokens, architecture v{state['architecture_version']}"
        )
        return path, int(state["step"]), tokens_seen
    raise SystemExit("no loadable checkpoint:\n  " + "\n  ".join(failures))


def build_command(checkpoint: Path, *, schedule: str, microbatch: int | None,
                  stage_split: int | None) -> list[str]:
    """Build the resume command with the frozen schedule and selected executor."""
    command = [
        sys.executable, "-m", "flashmini.cli", "train",
        "--config", str(CONFIG_PATH),
        "--data-dir", str(DATA_DIR),
        "--run-dir", str(TARGET_RUN_DIR),
        "--model-parallel-gpus", "1,0",
        "--gpu-memory-gib", f"{MEMORY_GIB:.1f}",
        "--tokens", str(TOTAL_TOKENS),
        "--batch-size", str(BATCH),
        "--grad-accum", "1",
        "--seed", str(SEED),
        "--lr", str(LR),
        "--ple-lr-multiplier", str(PLE_LR_MULTIPLIER),
        "--warmup-tokens", str(WARMUP_TOKENS),
        "--cosine-decay",
        "--min-lr-ratio", str(MIN_LR_RATIO),
        "--eval-every-tokens", str(EVAL_EVERY_TOKENS),
        "--eval-max-batches", str(EVAL_MAX_BATCHES),
        "--checkpoint-every-tokens", str(CHECKPOINT_EVERY_TOKENS),
        "--log-every", str(LOG_EVERY),
        "--resume", str(checkpoint),
        "--pipeline-schedule", schedule,
    ]
    if schedule != "monolithic":
        command += ["--pipeline-microbatch-size", str(microbatch)]
        if stage_split is not None:
            command += ["--pipeline-stage-split", str(stage_split)]
    # The executor changed mid-run; every training-recipe field must still match,
    # and the change is recorded in the run's checkpoints.
    command += ["--allow-pipeline-policy-transition"]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flashmini_poc_d_resume_fast")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE,
                        choices=["monolithic", "serial_microbatch_v1", "overlapped_2gpu_v1"])
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_MICROBATCH)
    parser.add_argument("--stage-split", type=int, default=DEFAULT_STAGE_SPLIT)
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from this checkpoint instead of the newest valid one")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args = parser.parse_args(argv)

    if args.schedule == "monolithic":
        microbatch, stage_split = None, None
    else:
        microbatch = args.microbatch_size or 4
        stage_split = args.stage_split
        if microbatch <= 0 or BATCH % microbatch:
            raise SystemExit(f"--microbatch-size must divide {BATCH}")
        if stage_split is not None and not 0 < stage_split < 10:
            raise SystemExit("--stage-split must leave at least one block on each GPU")

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        # Loading verifies the newest checkpoint; its CPU copy is released before
        # the training process allocates its own model and optimizer.
        checkpoint, _, _ = find_newest_valid_checkpoint(
            [TARGET_RUN_DIR / "checkpoints", SOURCE_RUN_DIR / "checkpoints"]
        )
    command = build_command(
        checkpoint, schedule=args.schedule, microbatch=microbatch, stage_split=stage_split
    )
    print("resume target:", TARGET_RUN_DIR)
    print("command:")
    print("  " + " ".join(command))
    if args.dry_run:
        return 0
    result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

