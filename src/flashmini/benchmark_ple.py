"""Measure fresh-process CPU/GPU PLE training storage and short-step throughput.

Run each storage mode in a separate process. This is a training/prefill storage
probe, not a cached autoregressive decoding benchmark or a quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .accounting import count_parameters, estimate_flops_per_token
from .config import FlashMiniConfig
from .data import MemmapDataset, sha256_file
from .models import FlashMiniModel
from .optim import build_optimizer
from .training import train_step


def _tensor_bytes(value, totals=None):
    totals = {} if totals is None else totals
    if isinstance(value, torch.Tensor):
        device = value.device.type
        totals[device] = totals.get(device, 0) + value.numel() * value.element_size()
    elif isinstance(value, dict):
        for child in value.values():
            _tensor_bytes(child, totals)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _tensor_bytes(child, totals)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--offload", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.steps < 2 or args.batch_size < 1:
        raise ValueError("CUDA, at least two steps and a positive batch size are required")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cfg = FlashMiniConfig.from_dict(yaml.safe_load(Path(args.config).read_text()))
    if not cfg.use_ple:
        raise ValueError("Memory comparison requires PLE enabled")
    cfg.ple.offload = args.offload
    torch.manual_seed(17)
    device = torch.device("cuda")
    model = FlashMiniModel(cfg).to(device)
    optimizer = build_optimizer(model, lr=3e-4)
    data = MemmapDataset(Path(args.data_dir), "train")
    rng = np.random.default_rng(17)
    torch.cuda.reset_peak_memory_stats()
    losses, elapsed = [], []
    for step in range(args.steps):
        x, y = data.get_batch(rng.integers(0, len(data), args.batch_size))
        x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = train_step(model, optimizer, x, y)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        losses.append(metrics["loss"])
        if step > 0:
            elapsed.append(seconds)
    params_by_device = _tensor_bytes(list(model.parameters()))
    result = {
        "purpose": "training_storage_and_step_throughput_only",
        "offload": args.offload,
        "config": cfg.to_dict(),
        "config_sha256": sha256_file(Path(args.config)),
        "data_manifest_sha256": sha256_file(Path(args.data_dir) / "data_manifest.json"),
        "gpu": torch.cuda.get_device_name(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "counts": count_parameters(model, cfg).to_dict(),
        "matrix_flops_per_token_estimate": estimate_flops_per_token(cfg),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "parameter_bytes_by_device": params_by_device,
        "optimizer_tensor_bytes_by_device": _tensor_bytes(optimizer.state_dict()),
        "ple_table_device": str(model.ple.value_embed.weight.device),
        "ple_table_bytes": model.ple.value_embed.weight.numel() * model.ple.value_embed.weight.element_size(),
        "steady_step_seconds": elapsed,
        "tokens_per_second_excluding_first_step": (len(elapsed) * args.batch_size * cfg.max_seq_len
                                                   / sum(elapsed)),
        "losses": losses,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
