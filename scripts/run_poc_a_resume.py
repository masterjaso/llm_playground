#!/usr/bin/env python3
"""Launcher for resuming poc_a from step_24576 → 250M."""

from flashmini.cli import main as flashmini_main
import sys

sys.argv = [
    "flashmini", "train",
    "--config", "configs/flashmini/poc_a_v3.yaml",
    "--data-dir", "data/fineweb_v3_2b",
    "--run-dir", "runs/flashmini/v3_execution/treatment_A",
    "--tokens", "250000000",
    "--lr", "3e-4",
    "--ple-lr-multiplier", "5",
    "--batch-size", "16",
    "--model-parallel-gpus", "0,1",
    "--gpu-memory-gib", "15",
    "--grad-accum", "1",
    "--log-every", "10",
    "--checkpoint-every-tokens", "4194304",
    "--eval-every-tokens", "2097152",
    "--val-max-batches", "128",
    "--warmup-tokens", "524288",
    "--cosine-decay",
    "--min-lr-ratio", "0.1",
    "--seed", "17",
    "--resume", "runs/flashmini/v3_execution/treatment_A/checkpoints/step_24576.pt",
]

exit_code = flashmini_main()
sys.exit(exit_code)
