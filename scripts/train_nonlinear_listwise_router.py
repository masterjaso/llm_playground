"""Train one small nonlinear selector against exact FIT oracle sets.

The p16/top4 basis is frozen from the hard-token refinement.  Only a
5120 -> 128 -> SiLU -> 16 router is optimized.  Exact all-1,820-set positive
oracle labels are generated from FIT rows; validation is measured per epoch
for checkpoint selection and the holdout stays closed.
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
    from scripts.run_exact_p16_oracle import (
        _dense_hidden_target,
        _exact_topk_precomputed,
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
        _route_reconstruction,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import (  # type: ignore
        _dense_hidden_target,
        _exact_topk_precomputed,
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
        _route_reconstruction,
    )


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _warm_start_nonlinear(
    linear: TorchQwen35SwiGLUMoE,
    profile: Any,
    plan: Any,
    weights: dict[str, Any],
    device: str,
    hidden_size: int,
    *,
    router_feature_mode: str = "none",
) -> TorchQwen35SwiGLUMoE:
    import torch

    model = TorchQwen35SwiGLUMoE.from_dense(
        weights["gate_proj.weight"],
        weights["up_proj.weight"],
        weights["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        router_hidden_size=hidden_size,
        router_feature_mode=router_feature_mode,
        partition=plan,
        learnable_scales=True,
    ).to(device)
    linear_state = linear.state_dict()
    nonlinear_state = model.state_dict()
    for name in nonlinear_state:
        if not name.startswith("router."):
            nonlinear_state[name] = linear_state[name]
    model.load_state_dict(nonlinear_state, strict=True)
    with torch.no_grad():
        u, singular, vh = torch.linalg.svd(linear.router.weight.detach(), full_matrices=False)
        rank = min(hidden_size, int(singular.shape[0]))
        input_scale = 0.02
        model.router.in_proj.weight.zero_()
        model.router.in_proj.bias.zero_()
        if router_feature_mode == "shared_output":
            # Preserve the linear warm start on x and start the shared-output
            # half at zero; subsequent FIT-only updates can discover whether
            # the already-computed nonlinear feature improves generalization.
            model.router.in_proj.weight[:rank, : linear.hidden_size].copy_(vh[:rank] * input_scale)
        else:
            model.router.in_proj.weight[:rank].copy_(vh[:rank] * input_scale)
        model.router.out_proj.weight.zero_()
        model.router.out_proj.weight[:, :rank].copy_(u[:, :rank] * singular[:rank].unsqueeze(0) * (2.0 / input_scale))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("router.")
    model.eval()
    return model


def _oracle_target(logits: Any, exact_ids: Any, correlations: Any, *, temperature: float, membership_mix: float) -> Any:
    import torch

    membership = torch.zeros_like(logits)
    membership.scatter_(1, exact_ids, 1.0)
    membership_distribution = membership / exact_ids.shape[1]
    correlation_distribution = torch.softmax(correlations / temperature, dim=-1)
    return membership_mix * membership_distribution + (1.0 - membership_mix) * correlation_distribution


def _train_epoch(model: TorchQwen35SwiGLUMoE, dataset: ActivationShardDataset, validation_indices: np.ndarray, weights: dict[str, Any], *, microbatch: int, device: str, learning_rate: float, temperature: float, membership_mix: float) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    model.train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    recalls: list[float] = []
    updates = 0
    for values in dataset.iter_excluding_batches(validation_indices.tolist(), microbatch):
        inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
        target = _dense_target_torch(inputs, gate, up, down)
        _prediction, info = model(inputs, return_router=True, return_contributions=True)
        with torch.no_grad():
            shared = info["shared"].reshape(-1, info["shared"].shape[-1])
            routed = info["contributions"].reshape(-1, info["contributions"].shape[-2], info["contributions"].shape[-1])
            target_flat = target.reshape(-1, target.shape[-1])
            exact = _exact_topk_precomputed(shared, routed, target_flat, model.top_k)
            residual = target_flat - shared
            correlations = torch.einsum("neh,nh->ne", routed, residual)
            correlation_scale = torch.linalg.vector_norm(routed, dim=-1).clamp_min(1e-6)
            correlations = correlations / correlation_scale
            target_distribution = _oracle_target(info["logits"].reshape(-1, model.routed_experts), exact["indices"], correlations, temperature=temperature, membership_mix=membership_mix)
        log_probs = torch.log_softmax(info["logits"].reshape(-1, model.routed_experts), dim=-1)
        listwise_loss = -(target_distribution * log_probs).sum(dim=-1).mean()
        probabilities = torch.zeros((inputs.shape[0], model.routed_experts), device=device)
        probabilities.scatter_add_(1, info["indices"], info["weights"])
        load_balance = model.routed_experts * torch.mean(probabilities, dim=0).square().sum()
        z_loss = torch.mean(torch.logsumexp(info["logits"], dim=-1).square())
        loss = listwise_loss + 0.02 * load_balance + 0.001 * z_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        overlap = (torch.gather(torch.nn.functional.one_hot(info["indices"], num_classes=model.routed_experts).sum(dim=1), 1, exact["indices"]) > 0).float().mean().item()
        recalls.append(float(overlap))
        updates += 1
    return {"updates": updates, "mean_loss": float(np.mean(losses)), "mean_oracle_membership_recall": float(np.mean(recalls))}


def _selector_eval(model: TorchQwen35SwiGLUMoE, dataset: ActivationShardDataset, indices: np.ndarray, weights: dict[str, Any], *, microbatch: int, device: str) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in ("target_sq", "student_sq", "exact_sq", "student_cos", "exact_cos", "residual")}
    ids: dict[str, list[np.ndarray]] = {key: [] for key in ("student", "exact")}
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(indices.tolist(), microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(device))
            student, info = model(inputs, return_router=True, return_contributions=True)
            shared = info["shared"].reshape(-1, info["shared"].shape[-1])
            routed = info["contributions"].reshape(-1, info["contributions"].shape[-2], info["contributions"].shape[-1])
            target = target.reshape(-1, target.shape[-1])
            student = student.reshape(-1, student.shape[-1])
            exact = _exact_topk_precomputed(shared, routed, target, model.top_k)
            oracle = _route_reconstruction(shared, routed, exact)
            target_sq = target.square().sum(dim=1)
            rows["target_sq"].append(target_sq.cpu().numpy())
            rows["student_sq"].append((student - target).square().sum(dim=1).cpu().numpy())
            rows["exact_sq"].append((oracle - target).square().sum(dim=1).cpu().numpy())
            rows["student_cos"].append((student * target).sum(dim=1).div(torch.linalg.vector_norm(student, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).cpu().numpy())
            rows["exact_cos"].append((oracle * target).sum(dim=1).div(torch.linalg.vector_norm(oracle, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).cpu().numpy())
            rows["residual"].append(torch.linalg.vector_norm(target - shared, dim=1).cpu().numpy())
            ids["student"].append(info["indices"].reshape(-1, model.top_k).cpu().numpy())
            ids["exact"].append(exact["indices"].cpu().numpy())
    return {"rows": {key: np.concatenate(value) for key, value in rows.items()}, "ids": {key: np.concatenate(value) for key, value in ids.items()}}


def _summary(raw: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    rows = raw["rows"]
    target = rows["target_sq"][mask]
    return {
        "tokens": int(mask.sum()),
        "student_global_nmse": float(rows["student_sq"][mask].sum() / max(target.sum(), 1e-12)),
        "student_mean_token_relative_mse": float(np.mean(rows["student_sq"][mask] / np.maximum(target, 1e-12))),
        "student_mean_cosine": float(np.mean(rows["student_cos"][mask])),
        "exact_global_nmse": float(rows["exact_sq"][mask].sum() / max(target.sum(), 1e-12)),
        "exact_mean_token_relative_mse": float(np.mean(rows["exact_sq"][mask] / np.maximum(target, 1e-12))),
        "exact_mean_cosine": float(np.mean(rows["exact_cos"][mask])),
    }


def _set_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(left, right)], dtype=np.float64)
    union = np.asarray([len(set(a.tolist()) | set(b.tolist())) for a, b in zip(left, right)], dtype=np.float64)
    return {"topk_recall": float(np.mean(overlap / left.shape[1])), "exact_set_match": float(np.mean(overlap == left.shape[1])), "mean_jaccard": float(np.mean(overlap / np.maximum(union, 1.0)))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    base_profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan = _load_plan(run_dir / "partitions" / "high-sparsity-p16-top4.json")
    weights = _load_dense_mlp(source_dir)
    linear, initial_metadata = _load_deployed_checkpoint(weights, base_profile, plan, run_dir / "layer-checkpoints" / "clean-validation" / args.initial_checkpoint, args.device)
    model = _warm_start_nonlinear(linear, base_profile, plan, weights, args.device, args.router_hidden_size, router_feature_mode=args.router_feature_mode)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    dev = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev["selected_row_key_hash"])
    gate, up, down = weights["gate_proj.weight"], weights["up_proj.weight"], weights["down_proj.weight"]
    initial_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    trajectory = [{"epoch": 0, "stage": "nonlinear_warm_start", **initial_validation}]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_record = trajectory[0]
    for epoch in range(1, args.epochs + 1):
        train_record = _train_epoch(model, dataset, validation_indices, weights, microbatch=args.microbatch, device=args.device, learning_rate=args.learning_rate, temperature=args.temperature, membership_mix=args.membership_mix)
        metrics = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
        record = {"epoch": epoch, "stage": "exact_oracle_listwise_router", **train_record, **metrics}
        trajectory.append(record)
        feasible_dims = metrics["normalized_mse"] <= 0.05 and metrics["dead_experts"] == 0 and metrics["load_cv"] <= 0.50
        best_dims = best_record["normalized_mse"] <= 0.05 and best_record["dead_experts"] == 0 and best_record["load_cv"] <= 0.50
        if (feasible_dims and (not best_dims or _feasible_rank(metrics) > _feasible_rank(best_record))) or (not feasible_dims and not best_dims and _feasible_rank(metrics) > _feasible_rank(best_record)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_record = record
    model.load_state_dict(best_state, strict=True)
    final_validation = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    raw = _selector_eval(model, dataset, validation_indices, weights, microbatch=args.microbatch, device=args.device)
    edges = np.quantile(raw["rows"]["residual"], [0.0, 0.25, 0.5, 0.75, 1.0])
    hard = raw["rows"]["residual"] >= edges[3]
    all_mask = np.ones(len(validation_indices), dtype=bool)
    # Strict reload with the nonlinear router architecture before publishing.
    reloaded = _warm_start_nonlinear(linear, base_profile, plan, weights, args.device, args.router_hidden_size, router_feature_mode=args.router_feature_mode)
    reloaded.load_state_dict(best_state, strict=True)
    strict_validation = _stream_metrics(reloaded, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    output_dir = run_dir / "layer-checkpoints" / "clean-validation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in reloaded.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output_dir / "layer-0000.safetensors")
    report = {
        "schema_version": 1,
        "status": "NONLINEAR_LISTWISE_ROUTER_COMPLETE",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "A low-rank SiLU selector trained against exact positive oracle sets improves expert-set fidelity enough to clear cosine >=0.98 with the refined p16/top4 basis.",
        "falsifier": "Validation cosine remains below 0.98 or NMSE/load gates regress under the fixed nonlinear-router budget.",
        "code_commit": current_git_commit(),
        "dataset_hash": dataset.dataset_hash,
        "source_revision": str(base_profile.revision),
        "profile": base_profile.name,
        "router": {"architecture": "shared-output-feature->5120+5120->128->SiLU->16" if args.router_feature_mode == "shared_output" else "5120->128->SiLU->16", "feature_mode": args.router_feature_mode, "hidden_size": args.router_hidden_size, "loss": "exact_set_membership_mix_plus_correlation_listwise_cross_entropy", "temperature": args.temperature, "membership_mix": args.membership_mix},
        "initial_checkpoint": {"name": args.initial_checkpoint, "metadata_code_commit": initial_metadata.get("code_commit"), "tensor_sha256": initial_metadata.get("tensor_sha256_observed")},
        "training": {"epochs": args.epochs, "microbatch": args.microbatch, "learning_rate": args.learning_rate, "seed": args.seed, "fit_rows": int(dataset.count - len(validation_indices)), "validation_rows": len(validation_indices), "basis_frozen": True},
        "split_contract": {"fit": {"count": int(dataset.count - len(validation_indices)), "gradient_updates": True, "oracle_labels": "exact_all_1820", "validation_excluded": True}, "validation": {"count": len(validation_indices), "identity_hash": validation_hash, "gradient_updates": False, "checkpoint_selection": True}, "holdout": {"count": 16598, "opened": False, "status": "CLOSED"}},
        "initial_validation": initial_validation,
        "trajectory": trajectory,
        "best_validation": best_record,
        "final_validation": final_validation,
        "strict_reload_validation": strict_validation,
        "all_validation_oracle_comparison": _summary(raw, all_mask),
        "hard_quartile_oracle_comparison": _summary(raw, hard),
        "selector_metrics": {"all_validation": _set_metrics(raw["ids"]["student"], raw["ids"]["exact"]), "hard_quartile": _set_metrics(raw["ids"]["student"][hard], raw["ids"]["exact"][hard])},
        "residual_quartile_edges": [float(v) for v in edges],
        "output_dir": str(output_dir),
        "tensor_sha256": tensor_hash,
        "quality_gate": {"overall": "research-candidate", "thresholds_version": "gate-aware-2026-08-16", "evaluation_scope": "validation", "metrics": strict_validation},
    }
    checkpoint = LayerCheckpoint(
        layer=0,
        profile=base_profile.name,
        status="TRAINED_DEV_SELECTED",
        profile_hash=profile_fingerprint(base_profile.as_dict()),
        source_revision=base_profile.revision,
        source_config_hash=hashlib.sha256((source_dir / "config.json").read_bytes()).hexdigest(),
        source_index_hash=hashlib.sha256((source_dir / "model.safetensors.index.json").read_bytes()).hexdigest(),
        dataset_hash=dataset.dataset_hash,
        partition_strategy="artifact",
        partition_hash=hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest(),
        router_architecture="torch-shared-output-feature-low-rank-silu-topk-independent_positive-v1" if args.router_feature_mode == "shared_output" else "torch-low-rank-silu-topk-independent_positive-v1",
        routing_mode=base_profile.routing_mode,
        training_seed=args.seed,
        training_config=report["training"] | report["router"] | {"selection_identity_hash": validation_hash, "holdout_evaluation": "deferred"},
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
    (run_dir / "reports" / args.report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initial-checkpoint", default="p16-top4-hard-token-weighted")
    parser.add_argument("--output-name", default="p16-top4-nonlinear-listwise")
    parser.add_argument("--report-name", default="p16-top4-nonlinear-listwise-training.json")
    parser.add_argument("--router-hidden-size", type=int, default=128)
    parser.add_argument("--router-feature-mode", choices=("none", "shared_output"), default="none")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--membership-mix", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "initial_validation": report["initial_validation"], "strict_reload_validation": report["strict_reload_validation"], "all_validation_oracle_comparison": report["all_validation_oracle_comparison"], "hard_quartile_oracle_comparison": report["hard_quartile_oracle_comparison"], "selector_metrics": report["selector_metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
