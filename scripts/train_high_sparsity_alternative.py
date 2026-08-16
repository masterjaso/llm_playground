"""Train one alternate high-sparsity layer-0 partition on TRAIN/dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dense2moe.config import load_config
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import train_torch_layer


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _schedule(
    cosine_weight: float,
    *,
    oracle_routing: bool = False,
    load_balance_weight: float = 0.1,
) -> list[dict[str, Any]]:
    warm_start: dict[str, Any] = {
        "name": "router_warm_start",
        "epochs": 1,
        "train_scales": False,
        "train_experts": False,
        "train_shared": False,
        "use_oracle_targets": True,
        "loss_coefficients": {
            "mse": 1.0,
            "cosine": 0.05,
            "load_balance": float(load_balance_weight),
            "router_z_loss": 0.001,
            "oracle": 0.1,
        },
    }
    joint: dict[str, Any] = {
        "name": "joint_shared_basis_cosine_balanced",
        "epochs": 4,
        "train_scales": True,
        "train_experts": True,
        "train_shared": True,
        "use_oracle_targets": False,
        "learning_rates": {
            "selection_router": 2e-5,
            "amplitude_router": 2e-5,
            "expert_scales": 2e-5,
            "experts": 1e-4,
            "shared": 5e-5,
        },
        "loss_coefficients": {
            "mse": 1.0,
            "cosine": float(cosine_weight),
            "load_balance": float(load_balance_weight),
            "router_z_loss": 0.001,
            "oracle": 0.0,
        },
    }
    if oracle_routing:
        warm_start["oracle_target_mode"] = "residual_correlation"
        warm_start["loss_coefficients"]["oracle_amplitude"] = 0.05
        joint["use_oracle_targets"] = True
        joint["oracle_target_mode"] = "residual_correlation"
        joint["loss_coefficients"].update({"oracle": 0.02, "oracle_amplitude": 0.02})
    return [warm_start, joint]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run_dir)
    source = Path(args.source_dir)
    dev_payload = json.loads((run / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    selection_indices = None if args.full_train_selection else [int(value) for value in dev_payload["selected_global_indices"]]
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    partition = run / "partitions" / args.partition_name
    if not partition.exists():
        raise FileNotFoundError(partition)
    output_dir = run / "layer-checkpoints/high-sparsity-basis-training" / args.output_name
    result = train_torch_layer(
        source_dir=source,
        activation_manifest=run / "capture/layer-0000.json",
        output_dir=output_dir,
        layer=0,
        profile=profile,
        partition_path=partition,
        epochs=1,
        microbatch=args.microbatch,
        learning_rate=1e-4,
        device=args.device,
        seed=17,
        source_revision=profile.revision,
        code_commit=current_git_commit(),
        stage_schedule=_schedule(
            args.cosine_weight,
            oracle_routing=args.oracle_routing,
            load_balance_weight=args.load_balance_weight,
        ),
        selection_indices=selection_indices,
        fit_exclude_indices=selection_indices,
        selection_identity_hash=None if selection_indices is None else dev_payload["selected_row_key_hash"],
        evaluate_holdout=False,
    )
    payload = {
        "schema_version": 1,
        "status": "ALTERNATIVE_HIGH_SPARSITY_VALIDATION_COMPLETE",
        "classification": "TRUE_VALIDATION_PARTITION_COMPARISON_HOLDOUT_DEFERRED",
        "profile": profile.name,
        "partition": str(partition),
        "partition_strategy": args.partition_name,
        "selection_split": "train_split" if selection_indices is None else "validation",
        "selection_count": None if selection_indices is None else len(selection_indices),
        "selection_hash": None if selection_indices is None else dev_payload["selected_row_key_hash"],
        "output_dir": str(output_dir),
        "stage_schedule": _schedule(
            args.cosine_weight,
            oracle_routing=args.oracle_routing,
            load_balance_weight=args.load_balance_weight,
        ),
        "oracle_routing": args.oracle_routing,
        "result": result,
        "holdout_policy": "full holdout remains closed until this candidate is selected for confirmation",
        "code_commit": current_git_commit(),
    }
    report = run / "reports" / f"{args.report_name}.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--partition-name", default="high-sparsity-p16-top4-activation-magnitude.json")
    parser.add_argument("--output-name", default="p16-top4-activation-magnitude-cosine-balanced")
    parser.add_argument("--report-name", default="p16-top4-activation-magnitude-cosine-balanced-training")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--full-train-selection", action="store_true")
    parser.add_argument("--oracle-routing", action="store_true")
    parser.add_argument("--load-balance-weight", type=float, default=0.1)
    args = parser.parse_args()
    payload = _run(args)
    result = payload["result"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_dir": payload["output_dir"],
                "initial_selection": result["initial_selection"],
                "final_selection": result["final_selection"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
