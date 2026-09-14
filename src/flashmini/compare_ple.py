"""Evaluate frozen pilot checkpoints on a holdout excluded from LR selection."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import FlashMiniConfig
from .data import MemmapDataset, sha256_file
from .eval import compute_validation_nll
from .models import FlashMiniModel


class _Slice:
    def __init__(self, data, start, end):
        self.data, self.start, self.end = data, start, end

    def __len__(self):
        return self.end - self.start

    def get_batch(self, indices):
        return self.data.get_batch(indices + self.start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-sequences", type=int, default=128)
    parser.add_argument("--block-sequences", type=int, default=64)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    data = MemmapDataset(Path(args.data_dir), "val")
    if not 0 <= args.skip_sequences < len(data) or args.block_sequences <= 0:
        raise ValueError("Invalid held-out slice")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {}
    controls = None
    for run_path in args.run_dirs:
        run = Path(run_path)
        summary = json.loads((run / "summary.json").read_text())
        comparable = {k: summary[k] for k in ("tokens_seen", "seed", "batch_size", "seq_len", "schedule")}
        if controls is None:
            controls = comparable
        elif controls != comparable:
            raise ValueError("Runs have different token/seed/batch/schedule controls")
        checkpoint = run / "checkpoints" / f"step_{summary['steps']}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = FlashMiniConfig.from_dict(saved["config"])
        del saved
        model = FlashMiniModel(config).to(device).eval()
        load_checkpoint(checkpoint, model)
        name = run.name
        result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "modes": {}}
        for enabled in ([True, False] if config.use_ple else [False]):
            mode = "ple_on" if enabled else "ple_off"
            blocks = []
            for start in range(args.skip_sequences, len(data), args.block_sequences):
                end = min(start + args.block_sequences, len(data))
                block = compute_validation_nll(model, _Slice(data, start, end), device,
                                               ple_enabled=enabled)
                blocks.append(block)
            tokens = sum(b["tokens"] for b in blocks)
            nll = sum(b["tokens"] * b["nll"] for b in blocks) / tokens
            accuracy = sum(b["correct_tokens"] for b in blocks) / tokens
            result["modes"][mode] = {"nll": nll, "top1_accuracy": accuracy,
                                      "tokens": tokens, "blocks": blocks}
            print(json.dumps({"run": name, "mode": mode, "nll": nll,
                              "top1_accuracy": accuracy, "tokens": tokens}), flush=True)
        variants[name] = result
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline = next(iter(variants))
    base = variants[baseline]["modes"]["ple_off"]
    comparisons = {}
    rng = np.random.default_rng(17)
    for name, result in variants.items():
        if name == baseline:
            continue
        candidate = result["modes"]["ple_on"]
        weights = np.array([b["tokens"] for b in base["blocks"]])
        delta = np.array([b["nll"] - a["nll"]
                          for a, b in zip(base["blocks"], candidate["blocks"])])
        draws = rng.integers(0, len(delta), size=(2000, len(delta)))
        estimates = (delta[draws] * weights[draws]).sum(1) / weights[draws].sum(1)
        comparisons[name] = {
            "nll_delta_vs_baseline": candidate["nll"] - base["nll"],
            "accuracy_delta_vs_baseline": candidate["top1_accuracy"] - base["top1_accuracy"],
            "paired_block_bootstrap_nll_delta_95pct": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "ple_off_minus_on_nll": result["modes"]["ple_off"]["nll"] - candidate["nll"],
        }
    report = {"scope": "small_model_pilot; conditional holdout uncertainty, not across-seed assurance",
              "data_manifest_sha256": sha256_file(Path(args.data_dir) / "data_manifest.json"),
              "skipped_tuning_sequences": args.skip_sequences, "block_sequences": args.block_sequences,
              "controls": controls, "baseline": baseline, "variants": variants,
              "comparisons": comparisons}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
