#!/usr/bin/env python
"""Throughput benchmark for the PoC_D v3 execution engines.

Measures the real D model on the local 5060 Ti pair so the resumed run can use
the fastest verified execution method.  Nothing here writes to a run directory
and no checkpoint is required (weights do not affect throughput).

Schedules
  monolithic  one full logical batch through the sharded model (no microbatches)
  serial      ``pipeline_train_step``: each microbatch runs the whole model
  overlapped  ``overlapped_pipeline_train_step``: real staged two-GPU overlap

Every reported number is measured after warm-up and excludes model
construction, checkpoint loading, validation and checkpoint writing.

Usage:
    .venv/bin/python scripts/flashmini_poc_d_bench.py --schedule monolithic
    .venv/bin/python scripts/flashmini_poc_d_bench.py --schedule serial --microbatch-size 8
    .venv/bin/python scripts/flashmini_poc_d_bench.py --schedule overlapped --stage-split 4 --microbatch-size 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashmini.config import FlashMiniConfig
from flashmini.data import MemmapDataset
from flashmini.models import FlashMiniModel
from flashmini.optim import build_optimizer
from flashmini.pipeline import overlapped_pipeline_train_step, pipeline_train_step
from flashmini.training import train_step

D_CONFIG_PATH = REPO_ROOT / "configs/flashmini/poc_d_v3.yaml"
DATA_DIR = REPO_ROOT / "data/fineweb_v3_2b"
SEED = 17
LR = 3e-4
PLE_LR_MULTIPLIER = 5.0
BATCH_SIZE = 16
AUX_COEF = 0.01
DEVICES = [torch.device("cuda:1"), torch.device("cuda:0")]


class _GpuSampler:
    """Sample per-GPU utilization/power from nvidia-smi in a background thread."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: dict[int, list[tuple[float, float]]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    index, util, power = int(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                self.samples.setdefault(index, []).append((util, power))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        result: dict[str, dict[str, float]] = {}
        for index, values in sorted(self.samples.items()):
            utils = [v[0] for v in values]
            powers = [v[1] for v in values]
            if not utils:
                continue
            result[str(index)] = {
                "util_mean_pct": statistics.mean(utils),
                "util_max_pct": max(utils),
                "power_mean_w": statistics.mean(powers),
                "power_max_w": max(powers),
                "gpu_samples": len(utils),
            }
        return result


def _load_config() -> FlashMiniConfig:
    return FlashMiniConfig.from_dict(yaml.safe_load(D_CONFIG_PATH.read_text()))


def _logical_batch(dataset, batch_size: int, device: torch.device):
    indices = torch.arange(batch_size, dtype=torch.int64).numpy()
    inputs, labels = dataset.get_batch(indices)
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(labels, dtype=torch.long, device=device),
    )


def _microbatches(input_ids, labels, microbatch_size):
    count = input_ids.shape[0] // microbatch_size
    return [
        (input_ids[i * microbatch_size:(i + 1) * microbatch_size],
         labels[i * microbatch_size:(i + 1) * microbatch_size])
        for i in range(count)
    ]


def _run_update(schedule, model, optimizer, input_ids, labels, chunks, use_amp):
    if schedule == "monolithic":
        return train_step(
            model, optimizer, input_ids, labels, aux_loss_coef=AUX_COEF, use_amp=use_amp
        )
    if schedule == "serial":
        return pipeline_train_step(
            model, optimizer, chunks, aux_loss_coef=AUX_COEF, use_amp=use_amp
        )
    return overlapped_pipeline_train_step(
        model, optimizer, chunks, aux_loss_coef=AUX_COEF, use_amp=use_amp, timing=True
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flashmini_poc_d_bench")
    parser.add_argument("--schedule", choices=["monolithic", "serial", "overlapped"], required=True)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--stage-split", type=int, default=None,
                        help="Blocks [0, split) on the first device, the rest on the second")
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device-budget-gib", type=float, default=15.0)
    parser.add_argument("--no-gpu-sampler", action="store_true",
                        help="Skip the nvidia-smi sampler thread. Each sample spawns a "
                             "process, which costs measurable CPU on a launch-bound step.")
    parser.add_argument("--gpu-sampler-interval", type=float, default=0.2)
    args = parser.parse_args(argv)

    torch.manual_seed(SEED)
    config = _load_config()
    model = FlashMiniModel(config)
    if args.stage_split is not None:
        model.parallelize(DEVICES, stage_split=args.stage_split)
    else:
        model.parallelize(DEVICES)
    optimizer = build_optimizer(model, lr=LR, ple_lr_multiplier=PLE_LR_MULTIPLIER)

    dataset = MemmapDataset(DATA_DIR, split="train")
    input_ids, labels = _logical_batch(dataset, args.batch_size, DEVICES[0])
    chunks = _microbatches(input_ids, labels, args.microbatch_size)

    for device in DEVICES:
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up: construction, allocator growth and kernel selection must not be
    # counted in the measured throughput.
    for _ in range(args.warmup):
        metrics = _run_update(args.schedule, model, optimizer, input_ids, labels, chunks, True)
    torch.cuda.synchronize()

    sampler = _GpuSampler(interval=args.gpu_sampler_interval)
    if not args.no_gpu_sampler:
        sampler.start()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(args.updates)]
    for begin, end in events:
        begin.record()
        metrics = _run_update(args.schedule, model, optimizer, input_ids, labels, chunks, True)
        end.record()
    torch.cuda.synchronize()
    gpu_stats = {} if args.no_gpu_sampler else sampler.stop()

    step_ms = [b.elapsed_time(e) for b, e in events]
    tokens_per_update = int(input_ids.numel())
    median_ms = statistics.median(step_ms)
    result = {
        "schedule": args.schedule,
        "microbatch_size": args.microbatch_size,
        "stage_split": args.stage_split,
        "batch_size": args.batch_size,
        "tokens_per_update": tokens_per_update,
        "updates": args.updates,
        "step_ms_median": median_ms,
        "step_ms_min": min(step_ms),
        "step_ms_max": max(step_ms),
        "tok_per_sec": tokens_per_update / (median_ms / 1000.0),
        "loss": float(metrics["loss"]),
        "grad_norm": float(metrics.get("grad_norm", 0.0)),
        "router_aux_loss": float(metrics.get("router_aux_loss", 0.0)),
        "gpu_metrics": gpu_stats,
        "stage_ms": metrics.get("stage_ms") if isinstance(metrics, dict) else None,
        "peak_memory": {
            str(device.index): {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            for device in DEVICES
        },
        "device_budget_gib": args.device_budget_gib,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
