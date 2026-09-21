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
from .data import sha256_file
from .fingerprint import collect_fingerprint, enforce_clean_tree
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


def _set_gpu_memory_budget(devices, budget):
    if budget is None:
        return
    if budget <= 0 or any(d.type != "cuda" for d in devices):
        raise ValueError("gpu-memory-gib requires CUDA and a positive budget")
    for device in devices:
        free, total = torch.cuda.mem_get_info(device)
        allowance = min(budget * 2**30 - (total - free), free - 2**30)
        if allowance <= 0:
            raise ValueError(f"No memory budget remaining on {device}")
        torch.cuda.set_per_process_memory_fraction(allowance / total, device)


def cmd_train(args) -> int:
    from .training import train

    # The seed must precede model construction. This is what lets matched B/C
    # runs share their backbone initialization while C allocates PLE afterward.
    seed = getattr(args, "seed", 0)
    _seed_everything(seed)
    config = _load_config(args.config)
    from .data import MemmapDataset
    from .experiment import validate_data_contract

    dataset = MemmapDataset(Path(args.data_dir), split="train")
    validate_data_contract(dataset, config, config.max_seq_len, args.tokens,
                           allow_repeated=getattr(args, "allow_repeated_corpus", False))
    device = _device()
    gpu_ids = getattr(args, "model_parallel_gpus", None)
    devices = [torch.device(f"cuda:{int(i)}") for i in gpu_ids.split(",")] if gpu_ids else [device]
    if gpu_ids and (len(set(devices)) != len(devices) or any(
        d.index >= torch.cuda.device_count() for d in devices
    )):
        raise ValueError("model-parallel GPUs must be distinct available CUDA indices")
    budget = getattr(args, "gpu_memory_gib", None)
    _set_gpu_memory_budget(devices, budget)
    device = devices[0]
    model = FlashMiniModel(config).parallelize(
        devices, stage_split=getattr(args, "pipeline_stage_split", None)
    )
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
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    data_manifest_sha256 = sha256_file(Path(args.data_dir) / "data_manifest.json")
    repo_root = Path(__file__).resolve().parents[2]
    fingerprint = collect_fingerprint(
        repo_root,
        config_sha256=config_sha256,
        data_manifest_sha256=data_manifest_sha256,
    )
    # Official v3 runs fail closed on a dirty working tree at launch.
    enforce_clean_tree(fingerprint)
    run_metadata = {
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "model_parallel_devices": [str(d) for d in devices],
        "gpu_memory_gib": budget,
        "source_sha256": fingerprint["source_sha256"],
        "source_files": fingerprint["source_files"],
        "data_manifest_sha256": data_manifest_sha256,
        "execution_fingerprint": fingerprint,
    }
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
        allow_repeated_corpus=getattr(args, "allow_repeated_corpus", False),
        stop_after_tokens=getattr(args, "stop_after_tokens", None),
        pipeline_microbatch_size=getattr(args, "pipeline_microbatch_size", None),
        pipeline_schedule=getattr(args, "pipeline_schedule", None),
        pipeline_stage_split=getattr(args, "pipeline_stage_split", None),
        allow_pipeline_policy_transition=getattr(args, "allow_pipeline_policy_transition", False),
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args) -> int:
    """Run the fixed evaluation suite on a trained checkpoint."""
    from .checkpoint import load_checkpoint
    from .data import MemmapDataset, sha256_file
    from .eval import compute_validation_nll
    from .experiment import validate_document_boundaries

    config = _load_config(args.config)
    dataset = MemmapDataset(Path(args.data_dir), split="val")
    validate_document_boundaries(dataset.manifest, config)
    if dataset.seq_len != config.max_seq_len:
        raise ValueError("evaluation dataset/config sequence length mismatch")
    if config.architecture_version >= 3:
        dataset.verify_integrity()
    model = FlashMiniModel(config).to(_device())
    ckpt = Path(args.checkpoint)
    meta = load_checkpoint(ckpt, model)
    manifest_hash = sha256_file(Path(args.data_dir) / "data_manifest.json")
    if config.architecture_version >= 3 and meta.get("extra", {}).get("data_manifest_sha256") != manifest_hash:
        raise ValueError("evaluation dataset differs from checkpoint provenance")
    start_sequence = getattr(args, "skip_sequences", 0)
    result = compute_validation_nll(model, dataset, _device(), max_batches=args.max_batches,
                                    start_sequence=start_sequence)
    if getattr(model, "ple", None) is not None:
        result["ple_off"] = compute_validation_nll(
            model,
            dataset,
            _device(),
            max_batches=args.max_batches,
            ple_enabled=False,
            start_sequence=start_sequence,
        )
    result["checkpoint"] = str(ckpt)
    result["step"] = meta["step"]
    result["architecture_version"] = config.architecture_version
    result["actual_seq_len"] = dataset.seq_len
    result["skipped_sequences"] = start_sequence
    result["data_manifest_sha256"] = manifest_hash
    result["long_context_validated"] = False
    result["final_go_eligible"] = False
    result["seed_scope"] = "single_checkpoint_not_across_seed_confirmation"
    if config.use_ple:
        result["ple_off_scope"] = "within_model_memory_reliance_diagnostic_not_B_baseline"
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
    p.add_argument("--pipeline-microbatch-size", type=int, default=None,
                   help="Microbatch size for staged pipeline execution; must divide --batch-size. "
                        "One optimizer update per logical batch (not gradient accumulation).")
    p.add_argument("--pipeline-schedule", default=None,
                   choices=["monolithic", "serial_microbatch_v1", "overlapped_2gpu_v1"],
                   help="Execution engine. monolithic runs the whole logical batch in one pass; "
                        "serial_microbatch_v1 runs each microbatch through the whole model; "
                        "overlapped_2gpu_v1 overlaps the two stages on explicit CUDA streams.")
    p.add_argument("--pipeline-stage-split", type=int, default=None,
                   help="Blocks [0, split) on the first model-parallel GPU, the rest on the second")
    p.add_argument("--allow-pipeline-policy-transition", action="store_true",
                   help="Authorize resuming a run whose recorded execution engine differs from the "
                        "selected one. Requires every other policy field (model config, config hash, "
                        "dataset, seed, batch size, sequence length, optimizer, base LRs, LR schedule, "
                        "token target, KVC, PLE, MoE) to still match exactly.")
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
    p.add_argument("--allow-repeated-corpus", action="store_true",
                   help="Explicit non-decisive override for repeated-corpus mechanism probes")
    p.add_argument("--stop-after-tokens", type=int,
                   help="Pause at a gate without changing --tokens or its LR schedule")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--skip-sequences", type=int, default=0,
                   help="Use the same held-out suffix as the matched B/C comparison")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
