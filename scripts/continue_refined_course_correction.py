"""Bounded continuation of the refined p16/top4 course-correction checkpoint."""

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


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    selection_indices = [int(value) for value in dev_payload["selected_global_indices"]]
    partition = run_dir / "partitions/high-sparsity-p16-top4.json"
    initial_checkpoint = run_dir / "layer-checkpoints/clean-validation/p16-top4-refined-course-correction"
    schedule = [
        {
            "name": "refined_joint_basis_cosine_balanced_continuation",
            "epochs": args.epochs,
            "train_scales": True,
            "train_experts": True,
            "train_shared": True,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation",
            "oracle_loss_mode": "repeated_cross_entropy",
            "oracle_amplitude_mode": "student_selected",
            "learning_rates": {"selection_router": 1e-5, "amplitude_router": 1e-5, "expert_scales": 1e-5, "experts": 5e-5, "shared": 2.5e-5},
            "loss_coefficients": {"mse": 1.0, "cosine": args.cosine_weight, "load_balance": 0.25, "oracle": 0.02, "oracle_amplitude": 0.02, "router_z_loss": 0.001},
        },
    ]
    output_dir = run_dir / "layer-checkpoints/clean-validation" / args.output_name
    result = train_torch_layer(
        source_dir=source_dir,
        activation_manifest=run_dir / "capture/layer-0000.json",
        output_dir=output_dir,
        layer=0,
        profile=profile,
        partition_path=partition,
        epochs=args.epochs,
        microbatch=args.microbatch,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=31,
        source_revision=profile.revision,
        code_commit=current_git_commit(),
        stage_schedule=schedule,
        selection_indices=selection_indices,
        fit_exclude_indices=selection_indices,
        selection_identity_hash=str(dev_payload["selected_row_key_hash"]),
        evaluate_holdout=False,
        initial_checkpoint_dir=initial_checkpoint,
    )
    report = {
        "schema_version": 1,
        "status": "REFINED_BASIS_COURSE_CORRECTION_CONTINUATION_VALIDATION_ONLY",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "A low-rate continuation of the still-rising refined-basis schedule clears the remaining validation cosine margin without violating NMSE/load gates.",
        "falsifier": "No continuation checkpoint reaches normalized MSE <= 0.05, cosine >= 0.98, dead experts = 0, and load CV <= 0.50.",
        "code_commit": current_git_commit(),
        "profile": profile.name,
        "partition": str(partition),
        "initial_checkpoint": str(initial_checkpoint),
        "selection_split": "validation",
        "selection_count": len(selection_indices),
        "selection_hash": str(dev_payload["selected_row_key_hash"]),
        "fit_exclusion": "validation rows excluded from all optimizer updates",
        "holdout_policy": "full holdout is closed; confirmation is deferred until a green validation checkpoint exists",
        "stage_schedule": schedule,
        "result": result,
    }
    report_path = run_dir / "reports" / args.report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cosine-weight", type=float, default=0.75)
    parser.add_argument("--output-name", default="p16-top4-refined-course-correction-continue")
    parser.add_argument("--report-name", default="p16-top4-refined-course-correction-continuation.json")
    args = parser.parse_args()
    report = run(args)
    result = report["result"]
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "final_selection": result["final_selection"], "training_config": result["training_config"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
