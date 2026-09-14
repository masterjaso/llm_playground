"""FlashMini CLI — stage orchestration.

Provides one-command entry points for the experiment stages:
  - doctor: hardware report
  - prepare-data: tokenize + pack dataset
  - overfit: micro-overfit test
  - train: run a training stage
  - eval: run evaluation suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

from .config import FlashMiniConfig
from .models import FlashMiniModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    """Seed construction-time and runtime RNGs before a model is allocated."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_doctor(args) -> int:
    from .hardware import collect

    print(json.dumps(collect(), indent=2))
    return 0


def cmd_overfit(args) -> int:
    """Run the mandatory micro-overfit test on a tiny fixed batch."""
    from .overfit import run_overfit_test

    config = FlashMiniConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=args.seq_len,
        use_ple=args.ple,
    )
    result = run_overfit_test(config, device=_device(), steps=args.steps)
    print(json.dumps(result, indent=2))
    return 0 if result.get("pass") else 1


def cmd_prepare_data(args) -> int:
    from .data import prepare_dataset

    # For CLI, accept a token stream source. Default: synthetic deterministic stream.
    def token_stream():
        rng = __import__("numpy").random.default_rng(args.seed)
        for _ in range(args.tokens):
            yield int(rng.integers(0, args.vocab_size))

    manifest = prepare_dataset(
        token_stream(),
        Path(args.out_dir),
        seq_len=args.seq_len,
        eos_id=args.eos_id,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _load_config(path: str) -> FlashMiniConfig:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    return FlashMiniConfig.from_dict(data)


def cmd_train(args) -> int:
    from .training import train

    # The seed must precede model construction. This is what lets matched B/C
    # runs share their backbone initialization while C allocates PLE afterward.
    seed = getattr(args, "seed", 0)
    _seed_everything(seed)
    config = _load_config(args.config)
    device = _device()
    gpu_ids = getattr(args, "model_parallel_gpus", None)
    devices = [torch.device(f"cuda:{int(i)}") for i in gpu_ids.split(",")] if gpu_ids else [device]
    if gpu_ids and (len(set(devices)) != len(devices) or any(
        d.index >= torch.cuda.device_count() for d in devices
    )):
        raise ValueError("model-parallel GPUs must be distinct available CUDA indices")
    budget = getattr(args, "gpu_memory_gib", None)
    if budget is not None:
        if budget <= 0 or any(d.type != "cuda" for d in devices):
            raise ValueError("gpu-memory-gib requires CUDA and a positive budget")
        for d in devices:
            free, total = torch.cuda.mem_get_info(d)
            # Account for existing allocations (including display/other jobs).
            allowance = min(budget * 2**30 - (total - free), free - 2**30)
            if allowance <= 0:
                raise ValueError(f"No memory budget remaining on {d}")
            torch.cuda.set_per_process_memory_fraction(allowance / total, d)
    device = devices[0]
    model = FlashMiniModel(config).parallelize(devices)
    from .optim import build_optimizer

    optimizer = build_optimizer(
        model,
        args.lr,
        ple_lr_multiplier=getattr(args, "ple_lr_multiplier", 1.0),
    )
    from .data import MemmapDataset

    dataset = MemmapDataset(Path(args.data_dir), split="train")
    val_dataset = None
    eval_every_tokens = getattr(args, "eval_every_tokens", 0)
    if eval_every_tokens:
        val_dataset = MemmapDataset(Path(args.data_dir), split="val")
    config_path = Path(args.config)
    run_metadata = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_parallel_devices": [str(d) for d in devices],
        "gpu_memory_gib": budget,
    }
    source_root = Path(__file__).parent
    source_files = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    run_metadata["source_sha256"] = hashlib.sha256(
        json.dumps(source_files, sort_keys=True).encode()
    ).hexdigest()
    run_metadata["source_files"] = source_files
    summary = train(
        model,
        optimizer,
        dataset,
        config,
        Path(args.run_dir),
        total_tokens=args.tokens,
        seq_len=config.max_seq_len,
        device=device,
        batch_size=args.batch_size,
        grad_accum=getattr(args, "grad_accum", 1),
        log_every=getattr(args, "log_every", 10),
        ckpt_every_tokens=getattr(args, "checkpoint_every_tokens", 25_000_000),
        resume_from=Path(args.resume) if getattr(args, "resume", None) else None,
        seed=seed,
        save_checkpoints=not getattr(args, "no_checkpoints", False),
        val_dataset=val_dataset,
        eval_every_tokens=eval_every_tokens,
        val_max_batches=getattr(args, "val_max_batches", None),
        warmup_tokens=getattr(args, "warmup_tokens", 0),
        cosine_decay=getattr(args, "cosine_decay", False),
        min_lr_ratio=getattr(args, "min_lr_ratio", 0.0),
        run_metadata=run_metadata,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args) -> int:
    """Run the fixed evaluation suite on a trained checkpoint."""
    from .checkpoint import load_checkpoint
    from .data import MemmapDataset
    from .eval import compute_validation_nll

    config = _load_config(args.config)
    model = FlashMiniModel(config).to(_device())
    ckpt = Path(args.checkpoint)
    meta = load_checkpoint(ckpt, model)
    dataset = MemmapDataset(Path(args.data_dir), split="val")
    result = compute_validation_nll(model, dataset, _device(), max_batches=args.max_batches)
    if getattr(model, "ple", None) is not None:
        result["ple_off"] = compute_validation_nll(
            model,
            dataset,
            _device(),
            max_batches=args.max_batches,
            ple_enabled=False,
        )
    result["checkpoint"] = str(ckpt)
    result["step"] = meta["step"]
    out_path = Path(args.run_dir) / "eval" / "val_nll.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="flashmini")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("overfit")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--ple", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    p.set_defaults(func=cmd_overfit)

    p = sub.add_parser("prepare-data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--eos-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ple-lr-multiplier", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model-parallel-gpus", help="Visible CUDA indices, e.g. 0,1; consecutive layer sharding")
    p.add_argument("--gpu-memory-gib", type=float, help="Per-device target including existing use; allocator guard, not a system-wide hard cap")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every-tokens", type=int, default=25_000_000)
    p.add_argument("--eval-every-tokens", type=int, default=0)
    p.add_argument("--eval-max-batches", "--val-max-batches", dest="val_max_batches", type=int, default=None)
    p.add_argument("--warmup-tokens", type=int, default=0)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-checkpoints", action="store_true")
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-batches", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
