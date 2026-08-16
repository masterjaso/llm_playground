"""Run a selector-only p16/top4 cross-validation experiment.

The p16/top4 basis is frozen at the current best FIT/validation checkpoint;
only selection and positive-amplitude router parameters are updated.  The
historical validation-A rows and a deterministic selector-only validation-B
shadow set are both excluded from optimizer updates.  The combined A+B set is
used for gate-aware checkpoint selection, while B is reported independently as
the generalization check.  The holdout split is never opened.

Every invocation records an explicit hypothesis, falsifier, budget, split
hashes, and decision enabled in a report so the result can be compared with
the prior selector experiments without silently changing the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Keep checkout-local execution usable before an editable install exists.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_config
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset, train_torch_layer


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
DEFAULT_INITIAL = DEFAULT_RUN / "layer-checkpoints/clean-validation/p16-top4-refined-course-correction-continue"


def _hash_indices(indices: list[int]) -> str:
    return hashlib.sha256("\n".join(str(int(index)) for index in indices).encode()).hexdigest()


def _load_indices(path: Path, key: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty {key} list")
    return sorted({int(value) for value in values})


def _schedule(variant: str, *, epochs: int, learning_rate: float) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    common: dict[str, Any] = {
        "name": f"selector_only_{variant}",
        "epochs": int(epochs),
        "train_scales": False,
        "train_experts": False,
        "train_shared": False,
        "use_oracle_targets": True,
        "oracle_target_mode": "residual_correlation",
        "oracle_amplitude_mode": "student_selected",
        "learning_rates": {
            "selection_router": float(learning_rate),
            "amplitude_router": float(learning_rate),
        },
        "loss_coefficients": {
            "mse": 1.0,
            "cosine": 0.50,
            "load_balance": 0.15,
            "oracle": 0.10,
            "oracle_amplitude": 0.05,
            "router_z_loss": 0.001,
        },
    }
    if variant == "residual_ce":
        common["oracle_loss_mode"] = "repeated_cross_entropy"
    elif variant == "residual_bce":
        common["oracle_loss_mode"] = "multilabel_bce"
        common["loss_coefficients"] = {
            **common["loss_coefficients"],
            "load_balance": 0.20,
        }
    else:
        raise ValueError(f"unknown selector-only variant: {variant!r}")
    return [common]


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    dev_path = run_dir / "capture/architecture-dev.json"
    shadow_path = Path(args.validation_b) if args.validation_b else run_dir / "capture/validation-b-selector-only.json"
    validation_a = _load_indices(dev_path, "selected_global_indices")
    validation_b = _load_indices(shadow_path, "selected_global_indices")
    if set(validation_a).intersection(validation_b):
        raise ValueError("validation-A and validation-B must be disjoint")
    combined = sorted(set(validation_a) | set(validation_b))
    train_dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    if combined[-1] >= train_dataset.count:
        raise IndexError("validation rows exceed the explicit FIT split")
    partition = run_dir / "partitions/high-sparsity-p16-top4.json"
    initial_checkpoint = Path(args.initial_checkpoint) if args.initial_checkpoint else DEFAULT_INITIAL
    output_name = args.output_name or f"p16-top4-selector-cross-validation-{args.variant}"
    report_name = args.report_name or f"p16-top4-selector-cross-validation-{args.variant}.json"
    schedule = _schedule(args.variant, epochs=args.epochs, learning_rate=args.learning_rate)
    result = train_torch_layer(
        source_dir=source_dir,
        activation_manifest=run_dir / "capture/layer-0000.json",
        output_dir=run_dir / "layer-checkpoints/clean-validation" / output_name,
        layer=0,
        profile=profile,
        partition_path=partition,
        epochs=args.epochs,
        microbatch=args.microbatch,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        source_revision=profile.revision,
        code_commit=current_git_commit(),
        stage_schedule=schedule,
        # Select on the union so no single historical validation slice drives
        # the router-only checkpoint.  B is still emitted separately below.
        selection_indices=combined,
        fit_exclude_indices=combined,
        selection_identity_hash=_hash_indices(combined),
        validation_b_indices=validation_b,
        validation_b_identity_hash=_hash_indices(validation_b),
        evaluate_holdout=False,
        initial_checkpoint_dir=initial_checkpoint,
    )
    report = {
        "schema_version": 1,
        "status": "SELECTOR_CROSS_VALIDATION_COMPLETE",
        "classification": "FROZEN_BASIS_SELECTOR_ONLY_FIT_VALIDATION_A_B_HOLDOUT_CLOSED",
        "hypothesis": (
            "A low-rate residual-correlation selector-only update selected on the combined "
            "validation-A/B union will preserve the green gate on both slices and reduce "
            "the p16/top4 selector generalization gap without changing the frozen basis."
        ),
        "falsifier": (
            "The combined A+B checkpoint is not green, validation-B is not green, or the "
            "router-only update worsens the frozen-basis baseline on either validation slice."
        ),
        "decision_enabled": "authorize_post_selection_holdout_confirmation_only_if_combined_and_B_green",
        "code_commit": current_git_commit(),
        "profile": profile.name,
        "partition": str(partition),
        "initial_checkpoint": str(initial_checkpoint),
        "output_name": output_name,
        "variant": args.variant,
        "budget": {
            "epochs": args.epochs,
            "microbatch": args.microbatch,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "seed": args.seed,
            "trainable_scope": "selection_router_and_positive_amplitude_router_only",
        },
        "split_contract": {
            "fit": {
                "count": train_dataset.count - len(combined),
                "gradient_updates": True,
                "excluded_validation_a_and_b": True,
            },
            "validation_a": {
                "count": len(validation_a),
                "identity_hash": _hash_indices(validation_a),
                "checkpoint_selection": "combined_union",
                "gradient_updates": False,
            },
            "validation_b": {
                "count": len(validation_b),
                "identity_hash": _hash_indices(validation_b),
                "checkpoint_selection": False,
                "gradient_updates": False,
                "receipt": str(shadow_path),
            },
            "combined_selection": {
                "count": len(combined),
                "identity_hash": _hash_indices(combined),
                "checkpoint_selection": True,
            },
            "holdout": {"count": 16598, "opened": False, "status": "CLOSED"},
        },
        "stage_schedule": schedule,
        "result": result,
    }
    report_path = run_dir / "reports" / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--validation-b", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--variant", choices=("residual_ce", "residual_bce"), default="residual_ce")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    report = run(args)
    result = report["result"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "code_commit": report["code_commit"],
                "combined_selection": result["final_selection"],
                "validation_b": result["validation_b_metrics"],
                "training_config": result["training_config"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

