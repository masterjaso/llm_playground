"""Run the identical layer-0 budget for the frozen architecture finalists."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dense2moe.config import load_config
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import train_torch_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/20260815-184644-windows-real-d2m-v4-streaming")
    parser.add_argument("--source-dir", default="runs/20260815-030931-windows/source")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    run = Path(args.run_dir)
    checkpoint_root = run / "layer-checkpoints"
    baseline = checkpoint_root / "layer-0000.json"
    if baseline.exists():
        baseline_dir = checkpoint_root / "baseline-p8s14-top2"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline, baseline_dir / baseline.name)
        tensor = checkpoint_root / "layer-0000.safetensors"
        if tensor.exists():
            shutil.copy2(tensor, baseline_dir / tensor.name)
    train_manifest = run / "capture/layer-0000.json"
    source = Path(args.source_dir)
    dev_payload = json.loads((run / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = [int(value) for value in dev_payload["selected_global_indices"]]
    search = json.loads((run / "reports/architecture-search.json").read_text(encoding="utf-8"))
    finalists = []
    for selected in search["finalists_selected_on_dev"]:
        profile_name = str(selected["profile"])
        top_k = int(selected["top_k"])
        config_name = {"p8": "qwen38_p8s1", "p16": "qwen38_p16s1", "p32": "qwen38_p32s1"}[profile_name] + f"_top{top_k}"
        finalists.append((config_name, run / "partitions" / f"layer-0000-{profile_name}-top{top_k}-architecture-finalist.json"))
    results = []
    for profile_name, partition in finalists:
        profile = load_config(Path("configs") / f"{profile_name}.yaml")
        output_dir = checkpoint_root / "architecture-finalists" / profile_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {profile_name} with {args.epochs} epoch, microbatch={args.microbatch}, lr={args.learning_rate}", flush=True)
        result = train_torch_layer(
            source_dir=source,
            activation_manifest=train_manifest,
            output_dir=output_dir,
            layer=0,
            profile=profile,
            partition_path=partition,
            epochs=args.epochs,
            microbatch=args.microbatch,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=17,
            source_revision=profile.revision,
            code_commit=current_git_commit(),
            selection_indices=validation_indices,
            fit_exclude_indices=validation_indices,
            selection_identity_hash=dev_payload["selected_row_key_hash"],
            evaluate_holdout=False,
        )
        results.append({"profile": profile_name, "partition": str(partition), "output_dir": str(output_dir), "result": result})
        print(json.dumps({"profile": profile_name, "status": result.get("status"), "holdout_metrics": result.get("holdout_metrics")}, indent=2), flush=True)
    report = {
        "schema_version": 1,
        "status": "ARCHITECTURE_FINALIST_VALIDATION_TRAINING_COMPLETE",
        "classification": "IDENTICAL_LAYER0_FINALIST_BUDGET_TRUE_VALIDATION",
        "budget": {"epochs": args.epochs, "microbatch": args.microbatch, "learning_rate": args.learning_rate, "device": args.device, "seed": 17},
        "train_manifest": str(train_manifest),
        "validation_count": len(validation_indices),
        "validation_identity_hash": dev_payload["selected_row_key_hash"],
        "holdout_reserved_for_confirmation": str(run / "capture/layer-0000-holdout.json"),
        "results": results,
        "code_commit": current_git_commit(),
    }
    (run / "reports/architecture-finalist-training.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
