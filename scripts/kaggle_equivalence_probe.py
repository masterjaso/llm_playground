#!/usr/bin/env python
"""Migration equivalence probe for the FlashMini v3 100M -> 250M Kaggle continuation.

Runs ONE resume-step on a fixed deterministic batch (the exact next batch the
EpochSampler would consume at the 100M resume point) and captures the numerical
signature: loss, total_loss, gradient norm, router aux loss / entropy / expert
load, PLE stats, a logits hash, and representative gradient hashes.

Environment-agnostic: the same script runs locally (RTX 5060 Ti, torch 2.13,
py3.13) and on Kaggle (T4, torch 2.10, py3.12). The two manifests are then
compared to classify the migration:
  - EXACT_PORTABLE_CONTINUATION  (bit-identical)
  - MATCHED_HARDWARE_MIGRATION   (within tight BF16 tolerances)
  - KAGGLE_UNSUITABLE            (material divergence)

This probe does NOT modify the checkpoint and does NOT require the full corpus:
it consumes a pre-extracted 16-sequence batch that is exactly the resume-point
batch (epoch 0, offset 393216, seed 17).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_hash(tensor: torch.Tensor, sample: int = 4096) -> str:
    """Deterministic hash of a tensor via a strided sample of its float32 bytes."""
    t = tensor.detach().float().cpu()
    flat = t.reshape(-1)
    if flat.numel() > sample:
        idx = torch.linspace(0, flat.numel() - 1, sample).long().clamp(max=flat.numel() - 1)
        flat = flat[idx]
    return sha256_bytes(flat.numpy().tobytes())


def _restore_rng(rng_state, sampling_rng: torch.Generator, seed: int) -> None:
    """Restore checkpoint RNG state exactly as the resume path does."""
    if not isinstance(rng_state, dict):
        sampling_rng.manual_seed(seed)
        return
    if rng_state.get("sampling") is not None:
        sampling_rng.set_state(rng_state["sampling"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-input", required=True)
    ap.add_argument("--batch-labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--treatment", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    import yaml
    from flashmini.checkpoint import load_checkpoint
    from flashmini.config import FlashMiniConfig
    from flashmini.models import FlashMiniModel
    from flashmini.optim import build_optimizer
    from flashmini.training import train_step

    config = FlashMiniConfig.from_dict(yaml.safe_load(open(args.config)))
    # Exact frozen recipe: model_parallel_gpus "1,0" -> devices [cuda:1, cuda:0].
    devices = [torch.device("cuda:1"), torch.device("cuda:0")]
    torch.manual_seed(17)
    model = FlashMiniModel(config).parallelize(devices)
    optimizer = build_optimizer(model, 3e-4, ple_lr_multiplier=5)

    meta = load_checkpoint(Path(args.checkpoint), model, optimizer)
    extra = meta.get("extra") or {}
    step = int(meta["step"])
    tokens_seen = int(extra.get("tokens_seen", step * 16 * 256))

    sampling_rng = torch.Generator(device="cpu").manual_seed(17)
    _restore_rng(extra.get("rng_state"), sampling_rng, seed=17)

    input_ids = torch.as_tensor(np.load(args.batch_input), dtype=torch.long, device=devices[0])
    labels = torch.as_tensor(np.load(args.batch_labels), dtype=torch.long, device=devices[0])

    # One resume-step, exact recipe (BF16 autocast, grad-clip groups).
    metrics = train_step(model, optimizer, input_ids, labels, grad_clip=1.0, use_amp=True)

    # Post-update logits signature (same model state on both environments).
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=None)
    logits = out["logits"]
    logits_hash = tensor_hash(logits)
    logits_sample = [float(v) for v in logits.reshape(-1)[:16].float().cpu().tolist()]

    # Representative gradient hashes (post-update).
    grad_hashes: dict[str, str] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_hashes[name] = tensor_hash(p.grad)
            if len(grad_hashes) >= 12:
                break

    result = {
        "treatment": args.treatment,
        "step": step,
        "tokens_seen": tokens_seen,
        "metrics": metrics,
        "logits_hash": logits_hash,
        "logits_sample": logits_sample,
        "grad_hashes": grad_hashes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "device": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
