"""Fair layer-0 p16/top4 schedule ablations on a fixed TRAIN/dev subset.

The three schedules share the same source layer, activation split, seed,
microbatch, and three stage-epoch budget.  They do not read the full holdout;
the selected schedule is confirmed on the holdout only after the high-sparsity
candidate list is frozen.
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


SCHEDULES: dict[str, list[dict[str, Any]]] = {
    "A_warm_joint": [
        {"name": "router_warm_start", "epochs": 1, "train_scales": False, "train_experts": False, "use_oracle_targets": True},
        {
            "name": "joint_expert_router",
            "epochs": 2,
            "train_scales": True,
            "train_experts": True,
            "use_oracle_targets": False,
            "learning_rates": {"selection_router": 5e-5, "amplitude_router": 5e-5, "expert_scales": 5e-5, "experts": 1e-4},
        },
    ],
    "B_small_router_scale": [
        {"name": "router_warm_start", "epochs": 1, "train_scales": False, "train_experts": False, "use_oracle_targets": True},
        {
            "name": "router_plus_scale_small_lr",
            "epochs": 1,
            "train_scales": True,
            "train_experts": False,
            "use_oracle_targets": False,
            "learning_rates": {"selection_router": 2e-5, "amplitude_router": 2e-5, "expert_scales": 2e-5},
        },
        {
            "name": "joint_expert_router",
            "epochs": 1,
            "train_scales": True,
            "train_experts": True,
            "use_oracle_targets": False,
            "learning_rates": {"selection_router": 5e-5, "amplitude_router": 5e-5, "expert_scales": 5e-5, "experts": 1e-4},
        },
    ],
    "C_frozen_router_adapt": [
        {"name": "router_warm_start", "epochs": 1, "train_scales": False, "train_experts": False, "use_oracle_targets": True},
        {
            "name": "frozen_router_expert_adaptation",
            "epochs": 1,
            "train_scales": True,
            "train_experts": True,
            "train_selection_router": False,
            "train_amplitude_router": False,
            "use_oracle_targets": False,
            "learning_rates": {"expert_scales": 5e-5, "experts": 1e-4},
        },
        {
            "name": "joint_expert_router",
            "epochs": 1,
            "train_scales": True,
            "train_experts": True,
            "use_oracle_targets": False,
            "learning_rates": {"selection_router": 5e-5, "amplitude_router": 5e-5, "expert_scales": 5e-5, "experts": 1e-4},
        },
    ],
}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run_dir)
    source = Path(args.source_dir)
    dev_payload = json.loads((run / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    selection_indices = [int(value) for value in dev_payload["selected_global_indices"]]
    partition = run / "partitions/high-sparsity-p16-top4.json"
    if not partition.exists():
        raise FileNotFoundError(f"high-sparsity partition search must finish first: {partition}")
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    results: list[dict[str, Any]] = []
    for name, schedule in SCHEDULES.items():
        output_dir = run / "layer-checkpoints/high-sparsity-schedule-ablation" / name
        print(f"training p16/top4 schedule {name} on {len(selection_indices)} TRAIN/dev rows", flush=True)
        result = train_torch_layer(
            source_dir=source,
            # The trainer validates both fixed splits; pass the explicit
            # aggregate wrapper while selection_indices restricts stage
            # comparisons to the deterministic TRAIN/dev rows.
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
            stage_schedule=schedule,
            selection_indices=selection_indices,
            fit_exclude_indices=selection_indices,
            selection_identity_hash=dev_payload["selected_row_key_hash"],
            evaluate_holdout=False,
        )
        results.append(
            {
                "schedule": name,
                "stage_schedule": schedule,
                "output_dir": str(output_dir),
                "status": result["status"],
                "initial_selection": result["initial_selection"],
                "final_selection": result["final_selection"],
                "holdout_metrics": result["holdout_metrics"],
                "training_config": result["training_config"],
                "code_commit": result["code_commit"],
            }
        )
        print(json.dumps({"schedule": name, "status": result["status"], "final_selection": result["final_selection"]}, indent=2), flush=True)
    payload = {
        "schema_version": 1,
        "status": "P16_TOP4_SCHEDULE_ABLATIONS_COMPLETE_VALIDATION_ONLY",
        "classification": "TRUE_VALIDATION_SCHEDULE_SELECTION_HOLDOUT_DEFERRED",
        "profile": profile.name,
        "partition": str(partition),
        "selection_split": "validation",
        "selection_count": len(selection_indices),
        "selection_hash": dev_payload["selected_row_key_hash"],
        "budget": {"microbatch": args.microbatch, "base_learning_rate": args.learning_rate, "seed": 17, "total_stage_epochs": 3, "device": args.device},
        "holdout_policy": "full holdout is not opened by these schedule ablations; confirm only the selected finalists",
        "results": results,
        "code_commit": current_git_commit(),
    }
    report = run / "reports/p16-top4-schedule-ablations.json"
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
    print(json.dumps({"status": payload["status"], "results": [{"schedule": row["schedule"], "final_selection": row["final_selection"]} for row in payload["results"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
