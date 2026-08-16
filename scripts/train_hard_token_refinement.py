"""Run one FIT-only hard-token basis refinement for p16/top4.

This is a single falsifiable training experiment after the exact oracle showed
that the current frozen basis is just below the cosine gate.  The architecture
and routing mode stay fixed.  FIT batches receive a detached weight based on
their shared-branch residual norm; validation is used only once per epoch for
checkpoint selection, and the holdout remains closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dense2moe.checkpoint.layer import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import (
    ActivationShardDataset,
    _dense_target_torch,
    _feasible_rank,
    _stream_metrics,
)

try:
    from scripts.run_exact_p16_oracle import _load_dense_mlp, _load_deployed_checkpoint, _load_plan
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import (  # type: ignore
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
    )


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _eligible(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["normalized_mse"]) <= 0.05
        and int(metrics["dead_experts"]) == 0
        and float(metrics["load_cv"]) <= 0.50
    )


def _train_epoch(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    validation_indices: np.ndarray,
    *,
    gate: Any,
    up: Any,
    down: Any,
    microbatch: int,
    device: str,
    learning_rate: float,
    residual_weight_alpha: float,
    cosine_weight: float,
    load_balance_weight: float,
) -> dict[str, Any]:
    import torch

    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    weight_means: list[float] = []
    updates = 0
    gate = gate.to(device)
    up = up.to(device)
    down = down.to(device)
    for values in dataset.iter_excluding_batches(validation_indices.tolist(), microbatch):
        inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
        target = _dense_target_torch(inputs, gate, up, down)
        prediction, info = model(inputs, return_router=True)
        with torch.no_grad():
            shared = model.shared_down_proj(torch.nn.functional.silu(model.shared_gate_proj(inputs)) * model.shared_up_proj(inputs))
            residual_norm = torch.linalg.vector_norm(target - shared, dim=1)
            residual_scale = residual_norm / residual_norm.mean().clamp_min(1e-6)
            sample_weight = (1.0 + residual_weight_alpha * residual_scale).clamp_max(8.0)
        mse_rows = (prediction - target).square().mean(dim=-1)
        cosine_rows = 1.0 - (prediction * target).sum(dim=-1) / (
            torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12
        )
        denominator = sample_weight.sum().clamp_min(1e-6)
        mse = (sample_weight * mse_rows).sum() / denominator
        cosine = (sample_weight * cosine_rows).sum() / denominator
        probabilities = torch.zeros((prediction.shape[0], model.routed_experts), device=device)
        probabilities.scatter_add_(1, info["indices"], info["weights"])
        load_balance = model.routed_experts * torch.mean(probabilities, dim=0).square().sum()
        z_loss = torch.mean(torch.logsumexp(info["logits"], dim=-1).square())
        loss = mse + cosine_weight * cosine + load_balance_weight * load_balance + 0.001 * z_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        weight_means.append(float(sample_weight.mean().cpu().item()))
        updates += 1
    return {"updates": updates, "mean_loss": float(np.mean(losses)), "mean_sample_weight": float(np.mean(weight_means))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    partition_path = run_dir / "partitions" / "high-sparsity-p16-top4.json"
    plan = _load_plan(partition_path)
    source_weights = _load_dense_mlp(source_dir)
    model, initial_metadata = _load_deployed_checkpoint(source_weights, profile, plan, run_dir / "layer-checkpoints" / "clean-validation" / args.initial_checkpoint, args.device)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    dev = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev["selected_row_key_hash"])
    gate = source_weights["gate_proj.weight"]
    up = source_weights["up_proj.weight"]
    down = source_weights["down_proj.weight"]
    initial_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    trajectory = [{"epoch": 0, "stage": "initialized", **initial_validation, "eligible_nmse_load": _eligible(initial_validation)}]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_record = trajectory[0]
    for epoch in range(1, args.epochs + 1):
        train_record = _train_epoch(
            model,
            dataset,
            validation_indices,
            gate=gate,
            up=up,
            down=down,
            microbatch=args.microbatch,
            device=args.device,
            learning_rate=args.learning_rate,
            residual_weight_alpha=args.residual_weight_alpha,
            cosine_weight=args.cosine_weight,
            load_balance_weight=args.load_balance_weight,
        )
        metrics = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
        record = {"epoch": epoch, "stage": "hard_token_weighted_basis_refinement", **train_record, **metrics, "eligible_nmse_load": _eligible(metrics)}
        trajectory.append(record)
        if (_eligible(metrics) and (not _eligible(best_record) or _feasible_rank(metrics) > _feasible_rank(best_record))) or (not _eligible(best_record) and _feasible_rank(metrics) > _feasible_rank(best_record)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_record = record
    model.load_state_dict(best_state, strict=True)
    final_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    # Strict reload of the selected state before publishing it.
    reloaded = TorchQwen35SwiGLUMoE.from_dense(
        source_weights["gate_proj.weight"],
        source_weights["up_proj.weight"],
        source_weights["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    ).to(args.device)
    reloaded.load_state_dict(best_state, strict=True)
    strict_validation = _stream_metrics(reloaded, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    output_dir = run_dir / "layer-checkpoints" / "clean-validation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in reloaded.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output_dir / "layer-0000.safetensors")
    partition_hash = hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest()
    report = {
        "schema_version": 1,
        "status": "HARD_TOKEN_WEIGHTED_REFINEMENT_COMPLETE",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "Upweighting high shared-residual FIT tokens during basis refinement improves hard-token cosine without breaking NMSE/load gates.",
        "expected_result": "Validation cosine should rise above the current 0.97297 while normalized MSE remains <=0.05 and load CV <=0.50.",
        "falsifier": "Validation cosine does not improve, or NMSE/load gates regress.",
        "code_commit": current_git_commit(),
        "dataset_hash": dataset.dataset_hash,
        "source_revision": str(profile.revision),
        "profile": profile.name,
        "partition": str(partition_path),
        "initial_checkpoint": {"name": args.initial_checkpoint, "metadata_code_commit": initial_metadata.get("code_commit"), "tensor_sha256": initial_metadata.get("tensor_sha256_observed")},
        "training": {"epochs": args.epochs, "microbatch": args.microbatch, "seed": args.seed, "learning_rate": args.learning_rate, "residual_weight_alpha": args.residual_weight_alpha, "cosine_weight": args.cosine_weight, "load_balance_weight": args.load_balance_weight, "optimizer": "AdamW", "fit_rows": int(dataset.count - len(validation_indices)), "validation_rows": len(validation_indices)},
        "split_contract": {"fit": {"count": int(dataset.count - len(validation_indices)), "gradient_updates": True, "validation_excluded": True}, "validation": {"count": len(validation_indices), "identity_hash": validation_hash, "gradient_updates": False, "checkpoint_selection": True}, "holdout": {"count": 16598, "opened": False, "status": "CLOSED"}},
        "initial_validation": initial_validation,
        "trajectory": trajectory,
        "best_validation": best_record,
        "final_validation": final_validation,
        "strict_reload_validation": strict_validation,
        "output_dir": str(output_dir),
        "tensor_sha256": tensor_hash,
        "quality_gate": {"overall": "research-candidate", "thresholds_version": "gate-aware-2026-08-16", "evaluation_scope": "validation", "metrics": strict_validation},
    }
    checkpoint = LayerCheckpoint(
        layer=0,
        profile=profile.name,
        status="TRAINED_DEV_SELECTED",
        profile_hash=profile_fingerprint(profile.as_dict()),
        source_revision=profile.revision,
        source_config_hash=hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest(),
        source_index_hash=hashlib.sha256((source_dir / "model.safetensors.index.json").read_bytes()).hexdigest(),
        dataset_hash=dataset.dataset_hash,
        partition_strategy="artifact",
        partition_hash=partition_hash,
        router_architecture="torch-linear-topk-independent_positive-hard-token-weighted-v1",
        routing_mode=profile.routing_mode,
        training_seed=args.seed,
        training_config=report["training"] | {"selection_identity_hash": validation_hash, "holdout_evaluation": "deferred"},
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"initial_validation": initial_validation, "final_validation": strict_validation},
        holdout_metrics={"split": "holdout", "status": "DEFERRED_UNTIL_FINALIST_CONFIRMATION", "normalized_mse": None, "cosine": None},
        router_metrics={"load_cv": strict_validation["load_cv"], "dead_experts": strict_validation["dead_experts"], "selected_counts": strict_validation["selected_counts"]},
        quality_gate=report["quality_gate"],
        code_commit=report["code_commit"],
    )
    save_layer_checkpoint(checkpoint, output_dir / "layer-0000.json")
    report_path = run_dir / "reports" / args.report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initial-checkpoint", default="p16-top4-residual-ce")
    parser.add_argument("--output-name", default="p16-top4-hard-token-weighted")
    parser.add_argument("--report-name", default="p16-top4-hard-token-weighted-training.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--residual-weight-alpha", type=float, default=2.0)
    parser.add_argument("--cosine-weight", type=float, default=0.45)
    parser.add_argument("--load-balance-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "initial_validation": report["initial_validation"], "strict_reload_validation": report["strict_reload_validation"], "best_validation": report["best_validation"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

