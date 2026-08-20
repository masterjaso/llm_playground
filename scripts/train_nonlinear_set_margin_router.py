"""Train a bounded nonlinear selector with exact-set ranking and soft balance.

This is a focused follow-up to ``train_nonlinear_listwise_router.py``.  The
positive oracle labels are the exact all-combinations active-face sets, while
the balance penalty is computed from the *soft* router distribution so it has
gradient with respect to the selector (the hard top-k usage receipt does not).
The refined p16/top4 basis remains frozen and the holdout remains closed.
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
        _exact_topk_precomputed,
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
    )
    from scripts.train_nonlinear_listwise_router import (
        _selector_eval,
        _set_metrics,
        _summary,
        _warm_start_nonlinear,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import (  # type: ignore
        _exact_topk_precomputed,
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
    )
    from train_nonlinear_listwise_router import (  # type: ignore
        _selector_eval,
        _set_metrics,
        _summary,
        _warm_start_nonlinear,
    )


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _set_margin_loss(
    logits: Any,
    exact_ids: Any,
    *,
    margin: float,
    positive_weight: float,
) -> tuple[Any, dict[str, Any]]:
    """Rank every exact positive above all negatives without ordering the set."""

    import torch
    import torch.nn.functional as F

    selected = torch.gather(logits, 1, exact_ids)
    membership = torch.zeros_like(logits, dtype=torch.bool)
    membership.scatter_(1, exact_ids, True)
    negatives = logits.masked_fill(membership, float("-inf"))
    # A smooth all-negative ranking loss is less sensitive to arbitrary
    # ordering than four independent slot labels, while still giving every
    # positive a gradient against the strongest competing expert.
    pairwise = F.softplus(negatives.unsqueeze(1) - selected.unsqueeze(-1) + margin)
    pairwise = pairwise.masked_fill(torch.isinf(pairwise), 0.0).mean()
    log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    positive_ce = -(selected - log_normalizer).mean()
    loss = pairwise + positive_weight * positive_ce
    return loss, {"pairwise": pairwise.detach(), "positive_ce": positive_ce.detach()}


def _train_epoch(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    validation_indices: np.ndarray,
    weights: dict[str, Any],
    *,
    microbatch: int,
    device: str,
    learning_rate: float,
    margin: float,
    positive_weight: float,
    load_balance_weight: float,
) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    pairwise_losses: list[float] = []
    positive_losses: list[float] = []
    recalls: list[float] = []
    soft_balance_values: list[float] = []
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
        logits = info["logits"].reshape(-1, model.routed_experts)
        set_loss, pieces = _set_margin_loss(logits, exact["indices"], margin=margin, positive_weight=positive_weight)
        # Unlike hard top-k usage, this soft distribution is differentiable in
        # router logits and therefore supplies an actual anti-collapse signal.
        soft_probabilities = torch.softmax(logits, dim=-1)
        soft_load_balance = model.routed_experts * torch.mean(soft_probabilities, dim=0).square().sum()
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        loss = set_loss + load_balance_weight * soft_load_balance + 0.001 * z_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        pairwise_losses.append(float(pieces["pairwise"].cpu().item()))
        positive_losses.append(float(pieces["positive_ce"].cpu().item()))
        soft_balance_values.append(float(soft_load_balance.detach().cpu().item()))
        overlap = (torch.gather(torch.nn.functional.one_hot(info["indices"], num_classes=model.routed_experts).sum(dim=1), 1, exact["indices"]) > 0).float().mean().item()
        recalls.append(float(overlap))
        updates += 1
    return {
        "updates": updates,
        "mean_loss": float(np.mean(losses)),
        "mean_pairwise_loss": float(np.mean(pairwise_losses)),
        "mean_positive_ce": float(np.mean(positive_losses)),
        "mean_soft_load_balance": float(np.mean(soft_balance_values)),
        "mean_oracle_membership_recall": float(np.mean(recalls)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    base_profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan = _load_plan(run_dir / "partitions" / "high-sparsity-p16-top4.json")
    weights = _load_dense_mlp(source_dir)
    linear, initial_metadata = _load_deployed_checkpoint(
        weights,
        base_profile,
        plan,
        run_dir / "layer-checkpoints" / "clean-validation" / args.initial_checkpoint,
        args.device,
    )
    model = _warm_start_nonlinear(linear, base_profile, plan, weights, args.device, args.router_hidden_size)
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
        train_record = _train_epoch(
            model,
            dataset,
            validation_indices,
            weights,
            microbatch=args.microbatch,
            device=args.device,
            learning_rate=args.learning_rate,
            margin=args.margin,
            positive_weight=args.positive_weight,
            load_balance_weight=args.load_balance_weight,
        )
        metrics = _stream_metrics(model, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
        record = {"epoch": epoch, "stage": "exact_set_margin_router", **train_record, **metrics}
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
    reloaded = _warm_start_nonlinear(linear, base_profile, plan, weights, args.device, args.router_hidden_size)
    reloaded.load_state_dict(best_state, strict=True)
    strict_validation = _stream_metrics(reloaded, dataset, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device, selected_indices=validation_indices.tolist())
    output_dir = run_dir / "layer-checkpoints" / "clean-validation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in reloaded.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output_dir / "layer-0000.safetensors")
    report = {
        "schema_version": 1,
        "status": "NONLINEAR_SET_MARGIN_ROUTER_COMPLETE",
        "classification": "TRUE_FIT_VALIDATION_HOLDOUT_PROTOCOL",
        "hypothesis": "Exact-set pairwise ranking plus a differentiable soft-load penalty improves selector fidelity without the hard-usage collapse seen in listwise training.",
        "falsifier": "Validation cosine remains below 0.98 or NMSE/load gates regress under the fixed nonlinear-router budget.",
        "code_commit": current_git_commit(),
        "dataset_hash": dataset.dataset_hash,
        "source_revision": str(base_profile.revision),
        "profile": base_profile.name,
        "router": {
            "architecture": "5120->128->SiLU->16",
            "hidden_size": args.router_hidden_size,
            "loss": "exact_set_pairwise_softplus_plus_positive_cross_entropy_plus_differentiable_soft_load",
            "margin": args.margin,
            "positive_weight": args.positive_weight,
            "load_balance_weight": args.load_balance_weight,
        },
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
        router_architecture="torch-low-rank-silu-topk-independent_positive-v1",
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
    parser.add_argument("--output-name", default="p16-top4-nonlinear-set-margin")
    parser.add_argument("--report-name", default="p16-top4-nonlinear-set-margin-training.json")
    parser.add_argument("--router-hidden-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument("--load-balance-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "code_commit": report["code_commit"], "initial_validation": report["initial_validation"], "strict_reload_validation": report["strict_reload_validation"], "all_validation_oracle_comparison": report["all_validation_oracle_comparison"], "hard_quartile_oracle_comparison": report["hard_quartile_oracle_comparison"], "selector_metrics": report["selector_metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
