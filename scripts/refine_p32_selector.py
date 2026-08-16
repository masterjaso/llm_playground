"""Run a bounded p32 selector/load-aware refinement from a frozen finalist.

Only router and positive-amplitude parameters are updated.  The p32 basis and
expert scales are loaded strictly from the supplied checkpoint, FIT excludes
validation-A, validation is selection-only, and holdout is never opened.
"""

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


def _schedule(epochs: int, learning_rate: float, load_balance: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "p32_selector_load_aware_refinement",
            "epochs": int(epochs),
            "train_scales": False,
            "train_experts": False,
            "train_shared": False,
            "train_selection_router": True,
            "train_amplitude_router": True,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation",
            "oracle_loss_mode": "repeated_cross_entropy",
            "oracle_amplitude_mode": "student_selected",
            "teacher_forcing_ratio": 0.0,
            "learning_rates": {
                "selection_router": float(learning_rate),
                "amplitude_router": float(learning_rate),
            },
            "loss_coefficients": {
                "mse": 1.0,
                "cosine": 0.50,
                "load_balance": float(load_balance),
                "oracle": 0.15,
                "oracle_amplitude": 0.05,
                "router_z_loss": 0.001,
            },
        }
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs") / f"{args.config_name}.yaml")
    partition = run_dir / "partitions" / args.partition_name
    initial_checkpoint = Path(args.initial_checkpoint)
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = tuple(int(value) for value in dev_payload["selected_global_indices"])
    validation_identity_hash = str(dev_payload["selected_row_key_hash"])
    schedule = _schedule(args.epochs, args.learning_rate, args.load_balance)
    output_dir = run_dir / "layer-checkpoints" / "clean-validation" / args.output_name
    result = train_torch_layer(
        source_dir=source_dir,
        activation_manifest=run_dir / "capture/layer-0000.json",
        output_dir=output_dir,
        layer=0,
        profile=profile,
        partition_path=partition,
        epochs=1,
        microbatch=args.microbatch,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        source_revision=profile.revision,
        code_commit=current_git_commit(),
        stage_schedule=schedule,
        selection_indices=validation_indices,
        fit_exclude_indices=validation_indices,
        selection_identity_hash=validation_identity_hash,
        evaluate_holdout=False,
        initial_checkpoint_dir=initial_checkpoint,
        router_hidden_size=None,
        router_feature_mode="none",
    )
    config = result["training_config"]
    report = {
        "schema_version": 1,
        "status": "P32_SELECTOR_REFINEMENT_COMPLETE",
        "classification": "FIT_VALIDATION_ONLY_P32_SELECTOR_REFINEMENT_NO_HOLDOUT",
        "hypothesis": (
            f"A load-aware router-only refinement of the {profile.name} finalist can "
            "reduce validation load CV toward 0.50 without sacrificing its near-target "
            "NMSE/cosine, while preserving the loaded FFN basis exactly."
        ),
        "falsifier": (
            "No validation checkpoint reaches the fixed gate, or the refinement "
            "regresses cosine/NMSE versus the initial finalist, or strict reload/basis "
            "provenance fails."
        ),
        "decision_enabled": (
            "retain the initial p32 finalist when refinement is not a Pareto improvement; "
            "authorize holdout confirmation only for a genuinely green validation checkpoint."
        ),
        "budget": {
            "epochs": int(args.epochs),
            "microbatch": int(args.microbatch),
            "learning_rate": float(args.learning_rate),
            "load_balance": float(args.load_balance),
            "device": args.device,
            "holdout_opened": False,
        },
        "profile": profile.name,
        "routing_mode": profile.routing_mode,
        "partition": str(partition),
        "initial_checkpoint": str(initial_checkpoint),
        "output_dir": str(output_dir),
        "code_commit": current_git_commit(),
        "split_contract": {
            "fit": {
                "count": config["fit_count"],
                "index_hash": config["fit_index_hash"],
                "excluded_validation_hash": config["fit_excluded_indices_hash"],
                "opened_for_gradients": True,
                "opened_for_selection": False,
                "opened_for_confirmation": False,
            },
            "validation": {
                "count": config["validation_count"],
                "identity_hash": validation_identity_hash,
                "selection_indices_hash": config["selection_indices_hash"],
                "opened_for_gradients": False,
                "opened_for_selection": True,
                "opened_for_confirmation": False,
            },
            "holdout": {
                "count": 16598,
                "opened_for_gradients": False,
                "opened_for_selection": False,
                "opened_for_confirmation": False,
                "status": "CLOSED",
            },
        },
        "stage_schedule": schedule,
        "initial_selection": result["initial_selection"],
        "final_selection": result["final_selection"],
        "final_fit": result["final_fit"],
        "result": result,
        "gate_rule": config["checkpoint_selection_rule"],
    }
    report_path = run_dir / "reports" / args.report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--partition-name", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--load-balance", type=float, default=0.50)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
