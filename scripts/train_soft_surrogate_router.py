"""Train selection/amplitude routers with a differentiable soft top-k surrogate.

Independent-positive inference has hard, non-differentiable expert selection;
this bounded experiment supplies a training-only soft top-k reconstruction
gradient while keeping the deployed checkpoint semantics unchanged.  The
refined basis is frozen and validation is the only checkpoint-selection split.
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


def _enable_router_only(model: TorchQwen35SwiGLUMoE) -> list[Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    if model.routing_mode == "independent_positive":
        for parameter in model.amplitude_router.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _train_epoch(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    validation_indices: np.ndarray,
    weights: dict[str, Any],
    *,
    microbatch: int,
    device: str,
    learning_rate: float,
    temperature: float,
    cosine_weight: float,
    load_balance_weight: float,
    sharpness_weight: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    soft_cosines: list[float] = []
    updates = 0
    model.train()
    for values in dataset.iter_excluding_batches(validation_indices.tolist(), microbatch):
        inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
        target = _dense_target_torch(inputs, gate, up, down).reshape(-1, inputs.shape[-1])
        _hard_prediction, info = model(inputs, return_router=True, return_contributions=True)
        shared = info["shared"].reshape(-1, info["shared"].shape[-1])
        routed = info["contributions"].reshape(-1, info["contributions"].shape[-2], info["contributions"].shape[-1])
        logits = info["logits"].reshape(-1, model.routed_experts)
        amplitude = F.softplus(info["amplitude_logits"]).reshape(-1, model.routed_experts)
        threshold = torch.topk(logits, model.top_k, dim=-1).values[:, -1:].detach()
        soft_gate = torch.sigmoid((logits - threshold) / temperature)
        soft_gate = soft_gate * (float(model.top_k) / soft_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        soft_prediction = shared + torch.einsum("ne,neh->nh", soft_gate * amplitude, routed)
        mse = torch.mean((soft_prediction - target).square())
        cosine_rows = 1.0 - torch.sum(soft_prediction * target, dim=-1) / (torch.linalg.vector_norm(soft_prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12)
        cosine = torch.mean(cosine_rows)
        soft_usage = soft_gate / float(model.top_k)
        load_balance = model.routed_experts * torch.mean(soft_usage, dim=0).square().sum()
        sharpness = torch.mean(soft_gate * (1.0 - soft_gate))
        loss = mse + cosine_weight * cosine + load_balance_weight * load_balance + sharpness_weight * sharpness
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        soft_cosines.append(float((1.0 - cosine).detach().cpu().item()))
        updates += 1
    return {"updates": float(updates), "mean_loss": float(np.mean(losses)), "mean_soft_cosine": float(np.mean(soft_cosines))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan = _load_plan(run_dir / "partitions/high-sparsity-p16-top4.json")
    weights = _load_dense_mlp(source_dir)
    model, _initial_metadata = _load_deployed_checkpoint(weights, profile, plan, run_dir / "layer-checkpoints/clean-validation" / args.initial_checkpoint, args.device)
    trainable = _enable_router_only(model)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev_payload["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev_payload["selected_row_key_hash"])
    gate, up, down = weights["gate_proj.weight"], weights["up_proj.weight"], weights["down_proj.weight"]
    initial_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    trajectory: list[dict[str, Any]] = [{"epoch": 0, "stage": "frozen_basis_initial", **initial_validation}]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_record = trajectory[0]
    for epoch in range(1, args.epochs + 1):
        train_record = _train_epoch(model, dataset, validation_indices, weights, microbatch=args.microbatch, device=args.device, learning_rate=args.learning_rate, temperature=args.temperature, cosine_weight=args.cosine_weight, load_balance_weight=args.load_balance_weight, sharpness_weight=args.sharpness_weight)
        metrics = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
        record = {"epoch": epoch, "stage": "soft_surrogate_router", **train_record, **metrics}
        trajectory.append(record)
        feasible = metrics["normalized_mse"] <= 0.05 and metrics["cosine"] >= 0.98 and metrics["dead_experts"] == 0 and metrics["load_cv"] <= 0.50
        best_feasible = best_record["normalized_mse"] <= 0.05 and best_record["cosine"] >= 0.98 and best_record["dead_experts"] == 0 and best_record["load_cv"] <= 0.50
        if (feasible and (not best_feasible or _feasible_rank(metrics) > _feasible_rank(best_record))) or (not feasible and not best_feasible and _feasible_rank(metrics) > _feasible_rank(best_record)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_record = record
    model.load_state_dict(best_state, strict=True)
    final_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    reloaded = TorchQwen35SwiGLUMoE.from_dense(weights["gate_proj.weight"], weights["up_proj.weight"], weights["down_proj.weight"], routed_experts=plan.routed_experts, shared_intermediate_size=plan.shared_intermediate_size, top_k=profile.top_k, routing_mode=profile.routing_mode, partition=plan, learnable_scales=True).to(args.device)
    reloaded.load_state_dict(best_state, strict=True)
    strict_validation = _stream_metrics(reloaded, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    output_dir = run_dir / "layer-checkpoints/clean-validation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in reloaded.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output_dir / "layer-0000.safetensors")
    report = {
        "schema_version": 1,
        "status": "SOFT_SURROGATE_ROUTER_VALIDATION_ONLY",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "A differentiable soft top-k reconstruction surrogate improves the independent-positive selector by providing gradient through expert choice.",
        "falsifier": "Validation cosine remains below 0.98 or NMSE/load gates regress.",
        "code_commit": current_git_commit(),
        "profile": profile.name,
        "initial_checkpoint": args.initial_checkpoint,
        "training": {"epochs": args.epochs, "microbatch": args.microbatch, "learning_rate": args.learning_rate, "temperature": args.temperature, "cosine_weight": args.cosine_weight, "load_balance_weight": args.load_balance_weight, "sharpness_weight": args.sharpness_weight, "basis_frozen": True, "trainable_router_parameters": len(trainable)},
        "split_contract": {"fit": {"count": int(dataset.count - len(validation_indices)), "gradient_updates": True, "validation_excluded": True}, "validation": {"count": len(validation_indices), "identity_hash": validation_hash, "gradient_updates": False, "checkpoint_selection": True}, "holdout": {"count": 16598, "opened": False, "status": "CLOSED"}},
        "initial_validation": initial_validation,
        "trajectory": trajectory,
        "best_validation": best_record,
        "final_validation": final_validation,
        "strict_reload_validation": strict_validation,
        "output_dir": str(output_dir),
        "tensor_sha256": tensor_hash,
        "quality_gate": {"overall": "green" if strict_validation["normalized_mse"] <= 0.05 and strict_validation["cosine"] >= 0.98 and strict_validation["dead_experts"] == 0 and strict_validation["load_cv"] <= 0.50 else "research-candidate", "evaluation_scope": "validation", "metrics": strict_validation},
    }
    checkpoint = LayerCheckpoint(layer=0, profile=profile.name, status="TRAINED_DEV_SELECTED", profile_hash=profile_fingerprint(profile.as_dict()), source_revision=profile.revision, source_config_hash=hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest(), source_index_hash=hashlib.sha256((source_dir / "model.safetensors.index.json").read_bytes()).hexdigest(), dataset_hash=dataset.dataset_hash, partition_strategy="artifact", partition_hash=hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest(), router_architecture=f"torch-linear-topk-{profile.routing_mode}-v1", routing_mode=profile.routing_mode, training_seed=args.seed, training_config=report["training"] | {"selection_identity_hash": validation_hash, "holdout_evaluation": "deferred"}, tensor_file=tensor_path.name, tensor_sha256=tensor_hash, tensor_inventory=inventory, train_metrics={"initial_validation": initial_validation, "final_validation": strict_validation}, holdout_metrics={"split": "holdout", "status": "DEFERRED_UNTIL_FINALIST_CONFIRMATION", "normalized_mse": None, "cosine": None}, router_metrics={"load_cv": strict_validation["load_cv"], "dead_experts": strict_validation["dead_experts"], "selected_counts": strict_validation["selected_counts"]}, quality_gate=report["quality_gate"], code_commit=report["code_commit"])
    save_layer_checkpoint(checkpoint, output_dir / "layer-0000.json")
    (run_dir / "reports" / args.report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initial-checkpoint", default="p16-top4-refined-course-correction-continue")
    parser.add_argument("--output-name", default="p16-top4-soft-surrogate-router")
    parser.add_argument("--report-name", default="p16-top4-soft-surrogate-router.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--load-balance-weight", type=float, default=0.1)
    parser.add_argument("--sharpness-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "strict_reload_validation": report["strict_reload_validation"], "trajectory": report["trajectory"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
