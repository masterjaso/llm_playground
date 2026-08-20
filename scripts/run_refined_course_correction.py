"""Run one evidence-backed residual-correlation schedule from the refined basis.

This is deliberately a single validation-only experiment.  It reuses the
course-correction recipe that previously produced the strongest p16/top4
validation result, but starts from ``p16-top4-hard-token-weighted`` so the
newly adapted basis is tested without reopening the holdout.
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    selection_indices = [int(value) for value in dev_payload["selected_global_indices"]]
    partition = run_dir / "partitions/high-sparsity-p16-top4.json"
    initial_checkpoint = run_dir / "layer-checkpoints/clean-validation/p16-top4-hard-token-weighted"
    schedule = [
        {
            "name": "refined_router_warm_start",
            "epochs": 1,
            "train_scales": False,
            "train_experts": False,
            "train_shared": False,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation",
            "oracle_loss_mode": "repeated_cross_entropy",
            "oracle_amplitude_mode": "student_selected",
            "loss_coefficients": {"mse": 1.0, "cosine": 0.05, "load_balance": 0.25, "oracle": 0.1, "oracle_amplitude": 0.05, "router_z_loss": 0.001},
        },
        {
            "name": "refined_joint_basis_cosine_balanced",
            "epochs": 4,
            "train_scales": True,
            "train_experts": True,
            "train_shared": True,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation",
            "oracle_loss_mode": "repeated_cross_entropy",
            "oracle_amplitude_mode": "student_selected",
            "learning_rates": {"selection_router": 2e-5, "amplitude_router": 2e-5, "expert_scales": 2e-5, "experts": 1e-4, "shared": 5e-5},
            "loss_coefficients": {"mse": 1.0, "cosine": 0.75, "load_balance": 0.25, "oracle": 0.02, "oracle_amplitude": 0.02, "router_z_loss": 0.001},
        },
    ]
    output_dir = run_dir / "layer-checkpoints/clean-validation/p16-top4-refined-course-correction"
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
        seed=29,
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
        "status": "REFINED_BASIS_COURSE_CORRECTION_VALIDATION_ONLY",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "The proven residual-correlation router/basis schedule will improve angular fidelity when initialized from the hard-token refined p16/top4 basis.",
        "falsifier": "No validation checkpoint reaches normalized MSE <= 0.05, cosine >= 0.98, dead experts = 0, and load CV <= 0.50.",
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
    report_path = run_dir / "reports/p16-top4-refined-course-correction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    report = run(args)
    result = report["result"]
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "final_selection": result["final_selection"], "training_config": result["training_config"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
