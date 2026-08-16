"""Run the strong p16/top4 frozen-basis oracle on TRUE validation only.

The existing bounded residual-correlation beam is useful for architecture
search, but it is not a sufficient capacity test for p16/top4: there are only
``C(16, 4) = 1820`` routed sets.  This driver evaluates every set and every
non-negative active face using the deployed checkpoint's shared/routed basis.
It never opens the holdout manifest and does not perform optimizer updates.

The default scope is the hardest residual-norm validation quartile (4,096
rows), which is the falsification-critical population identified by the
router diagnosis.  ``--scope all`` extends the same exact calculation to all
16,384 validation rows when the runtime budget permits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    from scripts.run_topk_architecture_search import (
        _dense_hidden_target,
        _exact_topk,
        _residual_correlation_beam_topk,
        _route_reconstruction,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_topk_architecture_search import (  # type: ignore
        _dense_hidden_target,
        _exact_topk,
        _residual_correlation_beam_topk,
        _route_reconstruction,
    )


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _load_dense_mlp(source: Path, layer: int = 0) -> dict[str, Any]:
    from safetensors import safe_open  # type: ignore

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"layer {layer} MLP inventory is incomplete: {sorted(names)}")
    values: dict[str, Any] = {}
    for name, shard in names.items():
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source / shard), framework="pt", device="cpu", **kwargs) as handle:
            values[name] = handle.get_tensor(prefix + name).float()
    return values


def _load_plan(path: Path) -> PartitionPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = payload.get("plan", payload)
    plan = PartitionPlan(
        int(payload["dense_intermediate_size"]),
        int(payload["routed_experts"]),
        int(payload["expert_intermediate_size"]),
        int(payload["shared_intermediate_size"]),
        tuple(int(value) for value in payload["shared_indices"]),
        tuple(tuple(int(value) for value in group) for group in payload["expert_indices"]),
    )
    plan.validate()
    return plan


def _load_deployed_checkpoint(
    source_weights: dict[str, Any],
    profile: Any,
    plan: PartitionPlan,
    checkpoint_dir: Path,
    device: str,
) -> tuple[TorchQwen35SwiGLUMoE, dict[str, Any]]:
    from safetensors.torch import load_file  # type: ignore

    model = TorchQwen35SwiGLUMoE.from_dense(
        source_weights["gate_proj.weight"],
        source_weights["up_proj.weight"],
        source_weights["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    ).to(device)
    tensor_path = checkpoint_dir / "layer-0000.safetensors"
    metadata_path = checkpoint_dir / "layer-0000.json"
    if not tensor_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"deployed checkpoint is incomplete: {checkpoint_dir}")
    raw = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix) :]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise ValueError(f"checkpoint tensor namespace mismatch: {sorted(raw)[:3]}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    model.eval()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tensor_sha256_observed"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    return model, metadata


def _selected_positions(
    dataset: ActivationShardDataset,
    indices: np.ndarray,
    model: TorchQwen35SwiGLUMoE,
    weights: dict[str, Any],
    *,
    microbatch: int,
    device: str,
) -> np.ndarray:
    """Return residual norms in the same order as ``indices``."""

    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in dataset.iter_selected_batches(indices.tolist(), microbatch):
            inputs = torch.as_tensor(batch, dtype=torch.float32, device=device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(device))
            _prediction, info = model(inputs, return_router=True, return_contributions=True)
            residual_norm = torch.linalg.vector_norm(target - info["shared"], dim=1)
            values.append(residual_norm.detach().cpu().numpy())
    result = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    if result.shape[0] != indices.shape[0]:
        raise ValueError(f"residual-norm count mismatch: {result.shape[0]} != {indices.shape[0]}")
    return result


def _new_metric() -> dict[str, Any]:
    return {
        "tokens": 0,
        "squared_error_sum": 0.0,
        "target_squared_sum": 0.0,
        "cosine_sum": 0.0,
        "token_relative_mse": [],
        "ids": [],
        "usage": None,
        "elapsed_seconds": 0.0,
    }


def _update_metric(
    metric: dict[str, Any],
    prediction: Any,
    target: Any,
    *,
    ids: Any,
    expert_count: int,
    elapsed: float = 0.0,
) -> None:
    import torch

    error = (prediction - target).square().sum(dim=1)
    target_norm = target.square().sum(dim=1)
    cosine = torch.sum(prediction * target, dim=1) / (
        torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12
    )
    metric["tokens"] += int(target.shape[0])
    metric["squared_error_sum"] += float(error.sum().item())
    metric["target_squared_sum"] += float(target_norm.sum().item())
    metric["cosine_sum"] += float(cosine.sum().item())
    metric["token_relative_mse"].append((error / target_norm.clamp_min(1e-12)).detach().cpu().numpy())
    metric["ids"].append(ids.detach().cpu().numpy())
    if metric["usage"] is None:
        metric["usage"] = np.zeros(expert_count, dtype=np.int64)
    for slot in range(ids.shape[1]):
        metric["usage"] += np.bincount(ids[:, slot].detach().cpu().numpy(), minlength=expert_count)
    metric["elapsed_seconds"] += float(elapsed)


def _finish_metric(metric: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(metric["tokens"]), 1)
    relative = np.concatenate(metric["token_relative_mse"]) if metric["token_relative_mse"] else np.empty(0)
    usage = np.asarray(metric["usage"] if metric["usage"] is not None else [], dtype=np.int64)
    usage_fraction = usage / max(tokens * 4, 1)
    return {
        "tokens": int(metric["tokens"]),
        "global_nmse": float(metric["squared_error_sum"] / max(metric["target_squared_sum"], 1e-12)),
        "mean_token_relative_mse": float(relative.mean()) if relative.size else None,
        "mean_cosine": float(metric["cosine_sum"] / tokens),
        "expert_usage_counts": usage.tolist(),
        "expert_usage_fraction": usage_fraction.tolist(),
        "dead_experts": int(np.sum(usage == 0)) if usage.size else None,
        "load_cv": float(usage.std() / max(usage.mean(), 1e-12)) if usage.size else None,
        "elapsed_seconds": float(metric["elapsed_seconds"]),
    }


def _set_metrics(left: list[np.ndarray], right: list[np.ndarray], top_k: int) -> dict[str, float]:
    a = np.concatenate(left) if left else np.empty((0, top_k), dtype=np.int64)
    b = np.concatenate(right) if right else np.empty((0, top_k), dtype=np.int64)
    if a.shape != b.shape or not a.size:
        return {"topk_recall": float("nan"), "exact_set_match": float("nan"), "mean_jaccard": float("nan")}
    overlap = np.asarray([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)], dtype=np.float64)
    union = np.asarray([len(set(x.tolist()) | set(y.tolist())) for x, y in zip(a, b)], dtype=np.float64)
    return {
        "topk_recall": float(np.mean(overlap / top_k)),
        "exact_set_match": float(np.mean(overlap == top_k)),
        "mean_jaccard": float(np.mean(overlap / np.maximum(union, 1.0))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dev_payload = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev_payload["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev_payload["selected_row_key_hash"])
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan_path = run_dir / "partitions" / args.partition_name
    plan = _load_plan(plan_path)
    source_weights = _load_dense_mlp(source_dir)
    model, checkpoint_metadata = _load_deployed_checkpoint(
        source_weights,
        profile,
        plan,
        run_dir / "layer-checkpoints" / "clean-validation" / args.checkpoint_name,
        args.device,
    )
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)

    started = time.perf_counter()
    residual_norms = _selected_positions(
        dataset,
        validation_indices,
        model,
        source_weights,
        microbatch=args.microbatch,
        device=args.device,
    )
    hard_count = int(validation_indices.shape[0] // 4)
    hard_positions = np.argsort(residual_norms, kind="stable")[-hard_count:]
    hard_positions.sort()
    if args.scope == "hard":
        scope_positions = hard_positions
        scope_name = "hardest_residual_norm_quartile"
    else:
        scope_positions = np.arange(validation_indices.shape[0], dtype=np.int64)
        scope_name = "all_validation"
    scope_indices = validation_indices[scope_positions]
    scope_residual_norms = residual_norms[scope_positions]
    residual_edges = np.quantile(residual_norms, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

    metrics = {name: _new_metric() for name in ("student", "bounded_oracle", "exact_oracle")}
    gate = source_weights["gate_proj.weight"].to(args.device)
    up = source_weights["up_proj.weight"].to(args.device)
    down = source_weights["down_proj.weight"].to(args.device)
    with torch.inference_mode():
        for batch in dataset.iter_selected_batches(scope_indices.tolist(), args.microbatch):
            inputs = torch.as_tensor(batch, dtype=torch.float32, device=args.device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(args.device))
            student_prediction, info = model(inputs, return_router=True, return_contributions=True)
            shared = info["shared"]
            routed = info["contributions"]
            student_ids = info["indices"].reshape(-1, model.top_k)
            student_ids_for_metrics = student_ids
            _update_metric(metrics["student"], student_prediction.reshape(-1, student_prediction.shape[-1]), target.reshape(-1, target.shape[-1]), ids=student_ids_for_metrics, expert_count=model.routed_experts)

            tic = time.perf_counter()
            bounded = _residual_correlation_beam_topk(
                shared.reshape(-1, shared.shape[-1]),
                routed.reshape(-1, routed.shape[-2], routed.shape[-1]),
                target.reshape(-1, target.shape[-1]),
                model.top_k,
                simplex=False,
                beam_width=args.beam_width,
                pool_size=args.pool_size,
            )
            bounded_prediction = _route_reconstruction(
                shared.reshape(-1, shared.shape[-1]),
                routed.reshape(-1, routed.shape[-2], routed.shape[-1]),
                bounded,
            )
            _update_metric(
                metrics["bounded_oracle"],
                bounded_prediction,
                target.reshape(-1, target.shape[-1]),
                ids=bounded["indices"],
                expert_count=model.routed_experts,
                elapsed=time.perf_counter() - tic,
            )
            tic = time.perf_counter()
            exact = _exact_topk(
                shared.reshape(-1, shared.shape[-1]),
                routed.reshape(-1, routed.shape[-2], routed.shape[-1]),
                target.reshape(-1, target.shape[-1]),
                model.top_k,
                simplex=False,
            )
            exact_prediction = _route_reconstruction(
                shared.reshape(-1, shared.shape[-1]),
                routed.reshape(-1, routed.shape[-2], routed.shape[-1]),
                exact,
            )
            _update_metric(
                metrics["exact_oracle"],
                exact_prediction,
                target.reshape(-1, target.shape[-1]),
                ids=exact["indices"],
                expert_count=model.routed_experts,
                elapsed=time.perf_counter() - tic,
            )
            del inputs, target, student_prediction, info, shared, routed, bounded_prediction, exact_prediction
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize()

    result_metrics = {name: _finish_metric(value) for name, value in metrics.items()}
    # The stored ID streams are ordered identically for each variant.  The
    # exact route is in the right-hand stream of both comparisons.
    student_ids_stream = metrics["student"]["ids"]
    bounded_ids_stream = metrics["bounded_oracle"]["ids"]
    exact_ids_stream = metrics["exact_oracle"]["ids"]
    decision = (
        "capacity_sufficient_focus_router"
        if result_metrics["exact_oracle"]["mean_cosine"] >= 0.98
        else "capacity_not_green_move_to_basis_topology"
    )
    payload = {
        "schema_version": 1,
        "status": "EXACT_P16_TOP4_ORACLE_COMPLETE",
        "classification": "TRUE_VALIDATION_ONLY_NO_GRADIENTS_NO_HOLDOUT",
        "hypothesis": "The deployed p16/top4 basis can clear cosine >= 0.98 if expert selection is exact.",
        "expected_result": "The exact all-1820-set positive oracle should materially exceed the learned linear selector, especially on the hardest residual quartile.",
        "falsifier": "Exact oracle mean cosine below 0.98 on the evaluated scope means selector learning alone cannot establish the product gate for this basis.",
        "compute_budget": {"scope": scope_name, "microbatch": int(args.microbatch), "beam_width": int(args.beam_width), "pool_size": int(args.pool_size)},
        "decision_enabled": decision,
        "code_commit": current_git_commit(),
        "source_revision": str(profile.revision),
        "dataset_hash": dataset.dataset_hash,
        "validation": {
            "count": int(validation_indices.shape[0]),
            "identity_hash": validation_hash,
            "scope_count": int(scope_indices.shape[0]),
            "scope": scope_name,
            "scope_global_indices_sha256": hashlib.sha256("\n".join(str(int(v)) for v in scope_indices).encode()).hexdigest(),
            "residual_norm_quartile_edges": [float(v) for v in residual_edges],
            "scope_residual_norm_min": float(scope_residual_norms.min()),
            "scope_residual_norm_max": float(scope_residual_norms.max()),
            "holdout": {"count": 16598, "opened": False, "status": "CLOSED"},
        },
        "architecture": {
            "profile": profile.name,
            "routed_experts": int(plan.routed_experts),
            "expert_intermediate_size": int(plan.expert_intermediate_size),
            "shared_intermediate_size": int(plan.shared_intermediate_size),
            "top_k": int(profile.top_k),
            "routing_mode": str(profile.routing_mode),
            "ffn_reduction": float(1.0 - (plan.shared_intermediate_size + profile.top_k * plan.expert_intermediate_size) / plan.dense_intermediate_size),
            "partition": str(plan_path),
        },
        "checkpoint": {
            "directory": str(run_dir / "layer-checkpoints" / "clean-validation" / args.checkpoint_name),
            "metadata_code_commit": checkpoint_metadata.get("code_commit"),
            "tensor_sha256": checkpoint_metadata.get("tensor_sha256_observed"),
            "basis": "deployed_checkpoint_shared_and_routed_contributions",
        },
        "oracle": {
            "exact_selection": "all_1820_four_expert_combinations_with_nonnegative_active_face_solves",
            "bounded_selection": "residual_correlation_beam_search_exact_final_coefficients",
            "bounded_beam_width": int(args.beam_width),
            "bounded_pool_size": int(args.pool_size),
            "all_combinations": 1820,
        },
        "metrics": result_metrics,
        "selection_comparison": {
            "student_vs_exact": _set_metrics(student_ids_stream, exact_ids_stream, model.top_k),
            "bounded_vs_exact": _set_metrics(bounded_ids_stream, exact_ids_stream, model.top_k),
        },
        "elapsed_seconds_total": float(time.perf_counter() - started),
    }
    report_path = run_dir / "reports" / args.report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(
        "# Exact p16/top4 validation oracle\n\n"
        f"- Scope: `{scope_name}` ({scope_indices.shape[0]:,} tokens); validation identity `{validation_hash}`.\n"
        f"- Code commit: `{payload['code_commit']}`; holdout opened: **no**.\n"
        f"- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.\n\n"
        "| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {value['global_nmse']:.6f} | {value['mean_token_relative_mse']:.6f} | {value['mean_cosine']:.6f} | {value['load_cv'] if value['load_cv'] is not None else float('nan'):.4f} | {value['dead_experts']} |"
            for name, value in result_metrics.items()
        )
        + "\n\n"
        f"Decision: **{decision}**.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--partition-name", default="high-sparsity-p16-top4.json")
    parser.add_argument("--checkpoint-name", default="p16-top4-residual-ce")
    parser.add_argument("--report-name", default="p16-top4-exact-oracle-validation.json")
    parser.add_argument("--scope", choices=("hard", "all"), default="hard")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=10)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "decision_enabled": payload["decision_enabled"], "metrics": payload["metrics"], "selection_comparison": payload["selection_comparison"], "code_commit": payload["code_commit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
