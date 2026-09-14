"""Reproducible short v3 runtime checks; never launches a decisive training run."""

import argparse
import copy
import gc
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from flashmini.cli import _set_gpu_memory_budget
from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, MoEConfig, PLEConfig
from flashmini.context_probe import run_context_probe
from flashmini.data import MemmapDataset, prepare_streaming_documents
from flashmini.eval import compute_validation_nll
from flashmini.models import FlashMiniModel
from flashmini.optim import build_optimizer
from flashmini.overfit import run_overfit_test
from flashmini.training import train, train_step


def tiny(variant):
    return FlashMiniConfig(architecture_version=3, vocab_size=64, d_model=32,
        num_layers=4, num_heads=2, head_dim=16, max_seq_len=32,
        gdn_per_attention=0 if variant == "a" else 3, use_ple=variant == "c",
        moe=MoEConfig(num_experts=2, top_k=2, expert_intermediate=32),
        gdn=GatedDeltaNetConfig(d_state=8, chunk_size=16),
        ple=PLEConfig(ngram_vocab_size_base=101, heads_per_ngram=2,
                      embed_dim=32, eos_id=63, offload="cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--official-shape", action="store_true")
    parser.add_argument("--official-batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory-gib", type=float)
    parser.add_argument("--data-dir", help="Also run a short real frozen-corpus pilot")
    parser.add_argument("--scratch-dir", default="runs/flashmini/.validation",
                        help="Disk-backed temporary checkpoint storage; avoid small /tmp tmpfs")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = [torch.device("cuda:1"), device] if torch.cuda.device_count() >= 2 else [device]
    _set_gpu_memory_budget(devices, args.gpu_memory_gib)
    report = {"purpose": "implementation_validity_only", "variants": {},
              "devices": [str(d) for d in devices], "torch": torch.__version__,
              "gpu_memory_gib": args.gpu_memory_gib,
              "final_go": "BLOCKED_pending_fresh_training_scale_and_long_context_quality"}
    with tempfile.TemporaryDirectory(prefix="flashmini-v3-runtime-", dir=scratch) as temp:
        root = Path(temp)
        rng = np.random.default_rng(17)
        prepare_streaming_documents((rng.integers(0, 63, 127) for _ in range(80)),
            root / "data", seq_len=32, eos_id=63, seed=17, val_fraction=0.2)
        data, val = MemmapDataset(root / "data"), MemmapDataset(root / "data", "val")
        for variant in "abc":
            config = tiny(variant)
            torch.manual_seed(17)
            overfit = run_overfit_test(copy.deepcopy(config), device, steps=200)
            print(variant, "overfit", json.dumps(overfit), flush=True)
            if not overfit["pass"]:
                raise AssertionError(f"{variant} micro-overfit failed")
            torch.manual_seed(17)
            if device.type == "cuda":
                for gpu in devices:
                    torch.cuda.reset_peak_memory_stats(gpu)
            model = FlashMiniModel(copy.deepcopy(config)).parallelize(devices)
            optimizer = build_optimizer(model, 0.001, ple_lr_multiplier=5)
            kwargs = {"total_tokens": 512, "seq_len": 32, "device": devices[0], "batch_size": 2,
                      "seed": 17, "log_every": 1, "ckpt_every_tokens": 128, "warmup_tokens": 64,
                      "cosine_decay": True, "val_dataset": val, "eval_every_tokens": 256,
                      "val_max_batches": 2, "use_amp": device.type == "cuda"}
            train(model, optimizer, data, config, root / variant, stop_after_tokens=256, **kwargs)
            del model, optimizer
            model = FlashMiniModel(copy.deepcopy(config)).parallelize(devices)
            optimizer = build_optimizer(model, 0.001, ple_lr_multiplier=5)
            summary = train(model, optimizer, data, config, root / variant,
                resume_from=root / variant / "checkpoints/step_4.pt", **kwargs)
            rows = [json.loads(line) for line in (root / variant / "metrics.jsonl").read_text().splitlines()]
            last = [row for row in rows if row["event"] == "train"][-1]
            assert all(f"grad_norm_{group}_preclip" in last for group in ("shared", "ple_dense", "ple_sparse"))
            assert all(p.device.type == "cuda" for name, p in model.named_parameters()
                       if not name.startswith("ple.value_embed")) if device.type == "cuda" else True
            if model.ple is not None:
                assert model.ple.value_embed.weight.device.type == "cpu"
                assert np.isfinite(last["ple_norm_ratio"])
            report["variants"][variant] = {"overfit": overfit, "training_steps": summary["steps"],
                "tokens_seen": summary["tokens_seen"], "last_metrics": last,
                "checkpoint_resume": "PASS", "cuda_memory": summary.get("cuda_memory")}
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            report["variants"][variant]["context_probe"] = run_context_probe(config,
                seq_len=1024, device=device, seed=17)
            assert report["variants"][variant]["context_probe"]["status"] == "PASS"
            print(variant, "smoke/resume/context PASS", flush=True)
            if args.official_shape:
                if device.type == "cuda":
                    for gpu in devices:
                        torch.cuda.reset_peak_memory_stats(gpu)
                values = yaml.safe_load(Path(f"configs/flashmini/poc_{variant}_v3.yaml").read_text())
                full = FlashMiniConfig.from_dict(values)
                torch.manual_seed(17)
                model = FlashMiniModel(full).parallelize(devices)
                optimizer = build_optimizer(model, 3e-4, ple_lr_multiplier=5)
                x = torch.randint(0, full.vocab_size, (args.official_batch_size, full.max_seq_len), device=devices[0])
                for _ in range(3):
                    metrics = train_step(model, optimizer, x, x, use_amp=device.type == "cuda")
                report["variants"][variant]["official_shape_step"] = metrics
                report["variants"][variant]["official_shape_optimizer_steps"] = 3
                report["variants"][variant]["parameters"] = sum(p.numel() for p in model.parameters())
                report["variants"][variant]["official_shape_batch_size"] = args.official_batch_size
                report["variants"][variant]["official_shape_peak_allocated_gib"] = {
                    str(gpu): torch.cuda.max_memory_allocated(gpu) / 2**30 for gpu in devices
                } if device.type == "cuda" else {}
                print(variant, "official shape forward/backward/optimizer PASS", flush=True)
                del model, optimizer, x
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if args.data_dir:
        data = MemmapDataset(Path(args.data_dir))
        val = MemmapDataset(Path(args.data_dir), "val")
        integrity = data.verify_integrity()
        report["fresh_corpus_pilot"] = {"integrity": integrity, "variants": {},
            "scope": "short_instability_probe_not_PoC_quality",
            "manifest_sha256": hashlib.sha256((Path(args.data_dir) / "data_manifest.json").read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory(prefix="flashmini-v3-fresh-pilot-", dir=scratch) as temp:
            for variant in "abc":
                if args.official_shape:
                    values = yaml.safe_load(Path(f"configs/flashmini/poc_{variant}_v3.yaml").read_text())
                    values["experiment_mode"] = "screening"
                    config = FlashMiniConfig.from_dict(values)
                else:
                    config = tiny(variant)
                    config.vocab_size = 50257
                    config.max_seq_len = data.seq_len
                    config.ple.eos_id = 50256
                    config = FlashMiniConfig.from_dict(config.to_dict())
                torch.manual_seed(17)
                model = FlashMiniModel(config).parallelize(devices)
                optimizer = build_optimizer(model, 3e-4, ple_lr_multiplier=5)
                summary = train(model, optimizer, data, config, Path(temp) / variant,
                    total_tokens=32768, seq_len=data.seq_len, device=devices[0],
                    batch_size=args.official_batch_size if args.official_shape else 2,
                    seed=17, log_every=1, ckpt_every_tokens=0,
                    warmup_tokens=512, cosine_decay=True, use_amp=device.type == "cuda")
                modes = {}
                for enabled in ([True, False] if variant == "c" else [False]):
                    modes["ple_on" if enabled else "ple_off"] = compute_validation_nll(
                        model, val, devices[0], max_batches=16, ple_enabled=enabled)
                metrics_rows = [json.loads(line) for line in
                    (Path(temp) / variant / "metrics.jsonl").read_text().splitlines()]
                training_rows = [row for row in metrics_rows if row["event"] == "train"]
                report["fresh_corpus_pilot"]["variants"][variant] = {
                    "config": config.to_dict(), "steps": summary["steps"],
                    "tokens_seen": summary["tokens_seen"], "validation": modes,
                    "clipping_counts": summary["clipping_counts"],
                    "first_metrics": training_rows[0], "last_metrics": training_rows[-1],
                    "status": "PASS"}
                print(variant, "fresh corpus pilot PASS", flush=True)
                del model, optimizer
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    source_root = Path("src/flashmini")
    source = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source_root.rglob("*.py"))}
    report["source_sha256"] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    report["source_files"] = source
    report["validation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["status"] = "PASS"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
