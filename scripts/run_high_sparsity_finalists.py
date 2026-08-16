"""Train equal-budget high-sparsity finalists on TRAIN/dev only.

The p16/top4 schedule winner is already recorded separately.  This runner
applies the same winning warm->joint schedule, seed, optimizer budget, and
development selection set to p16/top3 and p32/top6 so equal active width is
compared fairly before any holdout confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dense2moe.config import load_config
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import train_torch_layer


RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
SOURCE = Path("runs/20260815-030931-windows/source")

SCHEDULE = [
    {"name": "router_warm_start", "epochs": 1, "train_scales": False, "train_experts": False, "use_oracle_targets": True},
    {
        "name": "joint_expert_router",
        "epochs": 2,
        "train_scales": True,
        "train_experts": True,
        "use_oracle_targets": False,
        "learning_rates": {"selection_router": 5e-5, "amplitude_router": 5e-5, "expert_scales": 5e-5, "experts": 1e-4},
    },
]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run_dir)
    source = Path(args.source_dir)
    dev_payload = json.loads((run / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    selection_indices = [int(value) for value in dev_payload["selected_global_indices"]]
    finalists = (
        ("qwen38_p16s1_top3", "p16", 3),
        ("qwen38_p32s1_top6", "p32", 6),
    )
    results: list[dict[str, Any]] = []
    for config_name, profile_name, top_k in finalists:
        profile = load_config(Path("configs") / f"{config_name}.yaml")
        partition = run / f"partitions/high-sparsity-{profile_name}-top{top_k}.json"
        if not partition.exists():
            raise FileNotFoundError(partition)
        output_dir = run / "layer-checkpoints/high-sparsity-finalists-train-dev" / config_name
        print(f"training {config_name} with equal-budget schedule A on {len(selection_indices)} TRAIN/dev rows", flush=True)
        result = train_torch_layer(
            source_dir=source,
            activation_manifest=run / "capture/layer-0000.json",
            output_dir=output_dir,
            layer=0,
            profile=profile,
            partition_path=partition,
            epochs=1,
            microbatch=args.microbatch,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=17,
            source_revision=profile.revision,
            code_commit=current_git_commit(),
            stage_schedule=SCHEDULE,
            selection_indices=selection_indices,
            evaluate_holdout=False,
        )
        row = {
            "profile": config_name,
            "partition": str(partition),
            "output_dir": str(output_dir),
            "stage_schedule": SCHEDULE,
            "status": result["status"],
            "initial_selection": result["initial_selection"],
            "final_selection": result["final_selection"],
            "holdout_metrics": result["holdout_metrics"],
            "training_config": result["training_config"],
            "code_commit": result["code_commit"],
        }
        results.append(row)
        print(json.dumps({"profile": config_name, "status": row["status"], "final_selection": row["final_selection"]}, indent=2), flush=True)
    payload = {
        "schema_version": 1,
        "status": "HIGH_SPARSITY_EQUAL_COMPUTE_FINALISTS_TRAIN_DEV_COMPLETE",
        "classification": "TRAIN_DEV_SELECTION_HOLDOUT_DEFERRED",
        "selection_split": "train_dev",
        "selection_count": len(selection_indices),
        "selection_hash": dev_payload["selected_row_key_hash"],
        "budget": {"microbatch": args.microbatch, "base_learning_rate": args.learning_rate, "seed": 17, "total_stage_epochs": 3, "device": args.device},
        "fairness": "p16/top3 and p32/top6 share active width 4096, partition/oracle search policy, schedule, seed, optimizer budget, and dev selection set",
        "holdout_policy": "full holdout is deferred until finalist list is frozen",
        "results": results,
        "code_commit": current_git_commit(),
    }
    report = run / "reports/high-sparsity-equal-compute-finalists.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    payload = _run(args)
    print(json.dumps({"status": payload["status"], "results": [{"profile": row["profile"], "final_selection": row["final_selection"]} for row in payload["results"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
