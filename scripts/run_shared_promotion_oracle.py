"""Evaluate a FIT-scored shared-capacity topology before retraining.

The experiment promotes eight high-contribution neurons from each routed
expert into the always-active shared branch.  The dense partition remains
fixed while active width becomes 5,216 (70.04% FFN reduction).  Scores use
FIT rows only; validation is opened only for frozen oracle measurement and
the holdout remains closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import PartitionPlan
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset

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


def _fit_scores(
    dataset: ActivationShardDataset,
    validation_indices: np.ndarray,
    weights: dict[str, Any],
    *,
    tokens: int,
    microbatch: int,
    device: str,
) -> np.ndarray:
    """Score dense neurons from a bounded FIT-only prefix."""

    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    scores = torch.zeros(gate.shape[0], dtype=torch.float64, device=device)
    seen = 0
    with torch.inference_mode():
        for values in dataset.iter_excluding_batches(validation_indices.tolist(), microbatch):
            if seen >= tokens:
                break
            remaining = min(int(values.shape[0]), tokens - seen)
            inputs = torch.as_tensor(values[:remaining], dtype=torch.float32, device=device)
            hidden, _target = _dense_hidden_target(
                inputs,
                {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                torch.device(device),
            )
            scores += hidden.square().sum(dim=0, dtype=torch.float64) * down.square().sum(dim=0).to(torch.float64)
            seen += remaining
    if seen != tokens:
        raise ValueError(f"FIT scoring prefix is short: {seen} != {tokens}")
    return (scores / max(seen, 1)).cpu().numpy().astype(np.float64)


def _promoted_plan(old: PartitionPlan, scores: np.ndarray, promote_per_expert: int) -> tuple[PartitionPlan, dict[int, int]]:
    if promote_per_expert <= 0 or promote_per_expert >= old.expert_intermediate_size:
        raise ValueError("promote_per_expert must be within the old expert capacity")
    promotions: list[int] = []
    groups: list[tuple[int, ...]] = []
    origin: dict[int, int] = {}
    for expert, group in enumerate(old.expert_indices):
        ranked = sorted(group, key=lambda index: (-float(scores[index]), int(index)))
        promoted = ranked[:promote_per_expert]
        promoted_set = set(promoted)
        promotions.extend(promoted)
        groups.append(tuple(index for index in group if index not in promoted_set))
        for index in promoted:
            origin[int(index)] = expert
    shared = tuple(sorted((*old.shared_indices, *promotions)))
    new = PartitionPlan(
        old.dense_intermediate_size,
        old.routed_experts,
        old.expert_intermediate_size - promote_per_expert,
        old.shared_intermediate_size + len(promotions),
        shared,
        tuple(groups),
    )
    new.validate()
    return new, origin


def _remap_model(
    old_model: TorchQwen35SwiGLUMoE,
    new_plan: PartitionPlan,
    promoted_origin: dict[int, int],
    source_weights: dict[str, Any],
    *,
    profile: Any,
    device: str,
) -> TorchQwen35SwiGLUMoE:
    """Copy adapted old neuron parameters into the promoted topology."""

    import torch

    new = TorchQwen35SwiGLUMoE.from_dense(
        source_weights["gate_proj.weight"],
        source_weights["up_proj.weight"],
        source_weights["down_proj.weight"],
        routed_experts=new_plan.routed_experts,
        shared_intermediate_size=new_plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=new_plan,
        learnable_scales=True,
    ).to(device)
    old_shared = {int(index): offset for offset, index in enumerate(old_model.partition.shared_indices)}
    old_expert: dict[int, tuple[int, int]] = {}
    for expert, group in enumerate(old_model.partition.expert_indices):
        for offset, index in enumerate(group):
            old_expert[int(index)] = (expert, offset)
    with torch.no_grad():
        for offset, index_value in enumerate(new_plan.shared_indices):
            index = int(index_value)
            if index in old_shared:
                source_expert, source_offset, scale = 0, old_shared[index], 1.0
                gate = old_model.shared_gate_proj.weight[source_offset]
                up = old_model.shared_up_proj.weight[source_offset]
                down = old_model.shared_down_proj.weight[:, source_offset] * scale
            else:
                source_expert = promoted_origin[index]
                source_offset = old_expert[index][1]
                scale = float(old_model.expert_scales[source_expert].item())
                gate = old_model.expert_gate_proj[source_expert].weight[source_offset]
                up = old_model.expert_up_proj[source_expert].weight[source_offset]
                down = old_model.expert_down_proj[source_expert].weight[:, source_offset] * scale
            new.shared_gate_proj.weight[offset].copy_(gate)
            new.shared_up_proj.weight[offset].copy_(up)
            new.shared_down_proj.weight[:, offset].copy_(down)
        for expert, group in enumerate(new_plan.expert_indices):
            for offset, index_value in enumerate(group):
                source_expert, source_offset = old_expert[int(index_value)]
                new.expert_gate_proj[expert].weight[offset].copy_(old_model.expert_gate_proj[source_expert].weight[source_offset])
                new.expert_up_proj[expert].weight[offset].copy_(old_model.expert_up_proj[source_expert].weight[source_offset])
                new.expert_down_proj[expert].weight[:, offset].copy_(old_model.expert_down_proj[source_expert].weight[:, source_offset])
        new.router.weight.copy_(old_model.router.weight)
        new.amplitude_router.weight.copy_(old_model.amplitude_router.weight)
        new.amplitude_router.bias.copy_(old_model.amplitude_router.bias)
        new.expert_scales.copy_(old_model.expert_scales)
    new.eval()
    return new


def _evaluate(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    indices: np.ndarray,
    weights: dict[str, Any],
    *,
    microbatch: int,
    device: str,
) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in ("target_sq", "student_sq", "exact_sq", "student_cos", "exact_cos", "residual")}
    ids: dict[str, list[np.ndarray]] = {"student": [], "exact": []}
    usage = {key: np.zeros(model.routed_experts, dtype=np.int64) for key in ids}
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
            exact_prediction = _route_reconstruction(shared, routed, exact)
            target_sq = target.square().sum(dim=1)
            rows["target_sq"].append(target_sq.detach().cpu().numpy())
            rows["student_sq"].append((student - target).square().sum(dim=1).detach().cpu().numpy())
            rows["exact_sq"].append((exact_prediction - target).square().sum(dim=1).detach().cpu().numpy())
            rows["student_cos"].append((student * target).sum(dim=1).div(torch.linalg.vector_norm(student, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).detach().cpu().numpy())
            rows["exact_cos"].append((exact_prediction * target).sum(dim=1).div(torch.linalg.vector_norm(exact_prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).detach().cpu().numpy())
            rows["residual"].append(torch.linalg.vector_norm(target - shared, dim=1).detach().cpu().numpy())
            current_student_ids = info["indices"].reshape(-1, model.top_k)
            ids["student"].append(current_student_ids.detach().cpu().numpy())
            ids["exact"].append(exact["indices"].detach().cpu().numpy())
            for key, current_ids in (("student", current_student_ids), ("exact", exact["indices"])):
                for slot in range(model.top_k):
                    usage[key] += np.bincount(current_ids[:, slot].detach().cpu().numpy(), minlength=model.routed_experts)
    return {"rows": {key: np.concatenate(value) for key, value in rows.items()}, "ids": {key: np.concatenate(value) for key, value in ids.items()}, "usage": usage}


def _summary(raw: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    rows = raw["rows"]
    target = rows["target_sq"][mask]
    output: dict[str, Any] = {"tokens": int(mask.sum())}
    for name, error_key, cos_key, usage_key in (("student", "student_sq", "student_cos", "student"), ("exact_oracle", "exact_sq", "exact_cos", "exact")):
        error = rows[error_key][mask]
        usage = raw["usage"][usage_key]
        fraction = usage / max(int(mask.sum()) * 4, 1)
        output[name] = {
            "global_nmse": float(error.sum() / max(target.sum(), 1e-12)),
            "mean_token_relative_mse": float(np.mean(error / np.maximum(target, 1e-12))),
            "mean_cosine": float(np.mean(rows[cos_key][mask])),
            "expert_usage_counts": usage.tolist(),
            "expert_usage_fraction": fraction.tolist(),
            "dead_experts": int(np.sum(usage == 0)),
            "load_cv": float(usage.std() / max(usage.mean(), 1e-12)),
        }
    return output


def _set_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(left, right)], dtype=np.float64)
    union = np.asarray([len(set(a.tolist()) | set(b.tolist())) for a, b in zip(left, right)], dtype=np.float64)
    return {"topk_recall": float(np.mean(overlap / left.shape[1])), "exact_set_match": float(np.mean(overlap == left.shape[1])), "mean_jaccard": float(np.mean(overlap / np.maximum(union, 1.0)))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dev = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev["selected_row_key_hash"])
    old_profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    new_profile = load_config(Path("configs/qwen38_p16s1152_top4.yaml"))
    old_plan = _load_plan(run_dir / "partitions" / "high-sparsity-p16-top4.json")
    source_weights = _load_dense_mlp(source_dir)
    old_model, old_checkpoint = _load_deployed_checkpoint(source_weights, old_profile, old_plan, run_dir / "layer-checkpoints" / "clean-validation" / args.checkpoint_name, args.device)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    scores = _fit_scores(dataset, validation_indices, source_weights, tokens=args.score_tokens, microbatch=args.microbatch, device=args.device)
    new_plan, promoted_origin = _promoted_plan(old_plan, scores, args.promote_per_expert)
    new_model = _remap_model(old_model, new_plan, promoted_origin, source_weights, profile=new_profile, device=args.device)
    started = time.perf_counter()
    old_eval = _evaluate(old_model, dataset, validation_indices, source_weights, microbatch=args.microbatch, device=args.device)
    new_eval = _evaluate(new_model, dataset, validation_indices, source_weights, microbatch=args.microbatch, device=args.device)
    old_edges = np.quantile(old_eval["rows"]["residual"], [0.0, 0.25, 0.5, 0.75, 1.0])
    hard_mask = old_eval["rows"]["residual"] >= old_edges[3]
    all_mask = np.ones(len(validation_indices), dtype=bool)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "SHARED_PROMOTION_FROZEN_ORACLE_COMPLETE",
        "classification": "TRUE_FIT_SCORED_VALIDATION_ONLY_NO_GRADIENTS_NO_HOLDOUT",
        "hypothesis": "Promoting high-contribution routed neurons into the shared branch improves hard-token angular fidelity while preserving >=70% reduction.",
        "falsifier": "The promoted topology exact-oracle hard-quartile cosine does not improve materially over the current basis.",
        "code_commit": current_git_commit(),
        "source_revision": str(new_profile.revision),
        "dataset_hash": dataset.dataset_hash,
        "validation": {"count": len(validation_indices), "identity_hash": validation_hash, "holdout": {"count": 16598, "opened": False, "status": "CLOSED"}},
        "score_contract": {"rows": "FIT only", "tokens": int(args.score_tokens), "method": "mean hidden-square times dense down-column norm", "validation_excluded": True},
        "architecture": {"profile": new_profile.name, "routed_experts": 16, "expert_intermediate_size": 1016, "shared_intermediate_size": 1152, "top_k": 4, "active_width": new_profile.active_intermediate_size, "ffn_reduction": new_profile.sparsity},
        "promotion": {"promote_per_expert": int(args.promote_per_expert), "promoted_indices": sorted(int(v) for v in promoted_origin), "partition": new_plan.as_dict(), "score_sha256": hashlib.sha256(scores.tobytes()).hexdigest()},
        "baseline_checkpoint": {"metadata_code_commit": old_checkpoint.get("code_commit"), "tensor_sha256": old_checkpoint.get("tensor_sha256_observed")},
        "baseline": _summary(old_eval, all_mask),
        "promoted": _summary(new_eval, all_mask),
        "baseline_hard_quartile": _summary(old_eval, hard_mask),
        "promoted_hard_quartile": _summary(new_eval, hard_mask),
        "selection_comparison": {"baseline_student_vs_exact": _set_metrics(old_eval["ids"]["student"], old_eval["ids"]["exact"]), "promoted_student_vs_exact": _set_metrics(new_eval["ids"]["student"], new_eval["ids"]["exact"])},
        "residual_quartile_edges": [float(v) for v in old_edges],
        "decision": "train_promoted_topology" if float(np.mean(new_eval["rows"]["exact_cos"][hard_mask])) > float(np.mean(old_eval["rows"]["exact_cos"][hard_mask])) + args.min_hard_gain else "reject_promoted_topology",
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    partition_path = run_dir / "partitions" / args.partition_name
    partition_payload = {**new_plan.as_dict(), "schema_version": 1, "status": "FIT_SCORED_SHARED_PROMOTION_READY", "profile": new_profile.as_dict(), "code_commit": report["code_commit"], "score_contract": report["score_contract"], "promotion": report["promotion"]}
    partition_path.write_text(json.dumps(partition_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = run_dir / "reports" / args.report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.with_suffix(".md").write_text(
        "# p16/top4 shared-promotion frozen oracle\n\n"
        f"- Code commit: `{report['code_commit']}`; FIT score rows: {args.score_tokens}; holdout opened: **no**.\n"
        f"- Geometry: shared 1152, expert 1016, top-4; active width {new_profile.active_intermediate_size}; reduction {new_profile.sparsity:.4%}.\n\n"
        "| scope | baseline student cosine | promoted student cosine | baseline exact cosine | promoted exact cosine |\n|---|---:|---:|---:|---:|\n"
        f"| all validation | {report['baseline']['student']['mean_cosine']:.6f} | {report['promoted']['student']['mean_cosine']:.6f} | {report['baseline']['exact_oracle']['mean_cosine']:.6f} | {report['promoted']['exact_oracle']['mean_cosine']:.6f} |\n"
        f"| hard quartile | {report['baseline_hard_quartile']['student']['mean_cosine']:.6f} | {report['promoted_hard_quartile']['student']['mean_cosine']:.6f} | {report['baseline_hard_quartile']['exact_oracle']['mean_cosine']:.6f} | {report['promoted_hard_quartile']['exact_oracle']['mean_cosine']:.6f} |\n\n"
        f"Decision: **{report['decision']}**.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint-name", default="p16-top4-residual-ce")
    parser.add_argument("--partition-name", default="p16-top4-shared1152-promotion.json")
    parser.add_argument("--report-name", default="p16-top4-shared1152-promotion-oracle.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=2048)
    parser.add_argument("--promote-per-expert", type=int, default=8)
    parser.add_argument("--min-hard-gain", type=float, default=0.001)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "decision": report["decision"], "baseline_hard_quartile": report["baseline_hard_quartile"], "promoted_hard_quartile": report["promoted_hard_quartile"], "code_commit": report["code_commit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
