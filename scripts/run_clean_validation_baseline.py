"""Run a clean p16/top4 layer-0 protocol with FIT/VALIDATION/HOLDOUT receipts.

The deterministic architecture-dev rows are the validation set.  They are
opened for checkpoint selection only and are explicitly excluded from every
optimizer batch.  The full holdout is never opened by this command.
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


def _schedule(oracle_loss_mode: str) -> list[dict[str, Any]]:
    """Shared 5-epoch budget used by the CE and unordered-set comparisons."""

    common_oracle = {
        "oracle_target_mode": "residual_correlation",
        "oracle_loss_mode": oracle_loss_mode,
        "loss_coefficients": {
            "mse": 1.0,
            "cosine": 0.45,
            "load_balance": 0.25,
            "router_z_loss": 0.001,
            "oracle": 0.10,
            "oracle_amplitude": 0.05,
        },
    }
    return [
        {
            "name": "oracle_teacher_forced_router",
            "epochs": 1,
            "train_scales": False,
            "train_experts": False,
            "train_shared": False,
            "use_oracle_targets": True,
            "oracle_amplitude_mode": "teacher_forced",
            "teacher_forcing_ratio": 1.0,
            **common_oracle,
        },
        {
            "name": "mixed_oracle_joint_shared_basis",
            "epochs": 2,
            "train_scales": True,
            "train_experts": True,
            "train_shared": True,
            "use_oracle_targets": True,
            "oracle_amplitude_mode": "mixed",
            "teacher_forcing_ratio": 0.5,
            "learning_rates": {
                "selection_router": 2e-5,
                "amplitude_router": 2e-5,
                "expert_scales": 2e-5,
                "experts": 1e-4,
                "shared": 5e-5,
            },
            **common_oracle,
        },
        {
            "name": "student_joint_shared_basis",
            "epochs": 2,
            "train_scales": True,
            "train_experts": True,
            "train_shared": True,
            "use_oracle_targets": False,
            "oracle_loss_mode": oracle_loss_mode,
            "oracle_amplitude_mode": "student_selected",
            "teacher_forcing_ratio": 0.0,
            "learning_rates": {
                "selection_router": 2e-5,
                "amplitude_router": 2e-5,
                "expert_scales": 2e-5,
                "experts": 1e-4,
                "shared": 5e-5,
            },
            "loss_coefficients": {
                "mse": 1.0,
                "cosine": 0.45,
                "load_balance": 0.25,
                "router_z_loss": 0.001,
                "oracle": 0.0,
            },
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = tuple(int(value) for value in dev_payload["selected_global_indices"])
    validation_hash = str(dev_payload["selected_row_key_hash"])
    profile = load_config(Path("configs") / f"{args.config_name}.yaml")
    partition = run_dir / "partitions" / args.partition_name
    if not partition.exists():
        raise FileNotFoundError(partition)
    output_dir = run_dir / "layer-checkpoints" / "clean-validation" / args.output_name
    stage_schedule = _schedule(args.oracle_loss_mode)
    result = train_torch_layer(
        source_dir=source_dir,
        activation_manifest=run_dir / "capture/layer-0000.json",
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
        stage_schedule=stage_schedule,
        selection_indices=validation_indices,
        fit_exclude_indices=validation_indices,
        selection_identity_hash=validation_hash,
        evaluate_holdout=False,
    )
    config = result["training_config"]
    payload = {
        "schema_version": 2,
        "status": "CLEAN_VALIDATION_TRAINING_COMPLETE",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": (
            f"The {profile.name} topology-specific partition and staged residual-correlation "
            "router training can approach or clear the fixed green layer gate on FIT/validation "
            "without opening holdout."
        ),
        "falsifier": (
            "No validation checkpoint reaches normalized MSE <= 0.05, cosine >= 0.98, "
            "dead experts = 0, and load CV <= 0.50, or the topology-specific run regresses "
            "the prior partition baseline."
        ),
        "decision_enabled": (
            "retain as a bounded p32 research finalist only if validation is near-target; "
            "authorize post-selection holdout confirmation only after a genuinely green checkpoint"
        ),
        "budget": {
            "epochs": 1,
            "microbatch": args.microbatch,
            "device": args.device,
            "oracle_loss_mode": args.oracle_loss_mode,
            "holdout_opened": False,
        },
        "profile": profile.name,
        "routing_mode": profile.routing_mode,
        "partition": str(partition),
        "output_dir": str(output_dir),
        "oracle_loss_mode": args.oracle_loss_mode,
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
                "identity_hash": validation_hash,
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
        "stage_schedule": stage_schedule,
        "result": result,
        "best_clean_validation": {
            "stage": config["best_selection_stage"],
            "epoch": config["best_selection_epoch"],
            "reason": config["best_selection_reason"],
            "metrics": result["final_selection"],
        },
        "gate_rule": config["checkpoint_selection_rule"],
        "holdout_policy": "full holdout remains closed until a frozen finalist is selected",
        "code_commit": current_git_commit(),
    }
    report = run_dir / "reports" / f"clean-validation-{args.report_name}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--partition-name", default="high-sparsity-p16-top4.json")
    parser.add_argument("--config-name", default="qwen38_p16s1_top4")
    parser.add_argument("--output-name", default="p16-top4-residual-ce")
    parser.add_argument("--report-name", default="p16-top4-residual-ce")
    parser.add_argument("--oracle-loss-mode", choices=("repeated_cross_entropy", "multilabel_bce"), default="repeated_cross_entropy")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    args = parser.parse_args()
    payload = run(args)
    result = payload["result"]
    print(json.dumps({
        "status": payload["status"],
        "output_dir": payload["output_dir"],
        "oracle_loss_mode": payload["oracle_loss_mode"],
        "fit_count": payload["split_contract"]["fit"]["count"],
        "validation_count": payload["split_contract"]["validation"]["count"],
        "best_clean_validation": payload["best_clean_validation"],
        "final_fit": result["final_fit"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
