"""Measure p16/top4 routing against a residual-correlation oracle on validation.

This report is deliberately validation-only.  It does not perform optimizer
updates and it never opens the holdout manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import PartitionPlan
from dense2moe.training.torch_distill import ActivationShardDataset

try:
    from scripts.run_topk_architecture_search import _dense_hidden_target, _residual_correlation_beam_topk
except ModuleNotFoundError:
    from run_topk_architecture_search import _dense_hidden_target, _residual_correlation_beam_topk


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _plan(path: Path) -> PartitionPlan:
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


def _load_dense_mlp(source: Path) -> dict[str, Any]:
    from safetensors import safe_open  # type: ignore

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = "model.language_model.layers.0.mlp."
    names = {key[len(prefix):]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    values: dict[str, Any] = {}
    for name, shard in names.items():
        open_kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source / shard), framework="pt", device="cpu", **open_kwargs) as handle:
            values[name] = handle.get_tensor(prefix + name).float()
    return values


def _load_checkpoint(
    source_weights: dict[str, Any],
    profile: Any,
    plan: PartitionPlan,
    checkpoint_dir: Path,
    device: str,
) -> TorchQwen35SwiGLUMoE:
    import torch
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
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    raw = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise ValueError(f"checkpoint tensor namespace mismatch: {sorted(raw)[:3]}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _conditional(shared: Any, routed: Any, ids: Any, weights: Any) -> Any:
    import torch

    rows = torch.arange(routed.shape[0], device=routed.device)
    output = shared.clone()
    for slot in range(ids.shape[1]):
        output = output + routed[rows, ids[:, slot]] * weights[:, slot].unsqueeze(1)
    return output


def _positive_coefficients(routed: Any, ids: Any, residual: Any) -> Any:
    import torch

    selected = torch.gather(routed, 1, ids.unsqueeze(-1).expand(-1, -1, routed.shape[-1]))
    gram = torch.einsum("bkh,blh->bkl", selected, selected)
    rhs = torch.einsum("bkh,bh->bk", selected, residual)
    eye = torch.eye(ids.shape[1], dtype=gram.dtype, device=gram.device).unsqueeze(0)
    return torch.linalg.solve(gram + 1e-4 * eye, rhs.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)


def _row_metrics(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    error = (prediction - target).square().mean(dim=1) / (target.square().mean(dim=1) + 1e-12)
    cosine = torch.sum(prediction * target, dim=1) / (
        torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12
    )
    return error.detach().cpu().numpy(), cosine.detach().cpu().numpy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dev = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = tuple(int(value) for value in dev["selected_global_indices"])
    validation_hash = str(dev["selected_row_key_hash"])
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    partition_path = run_dir / "partitions" / args.partition_name
    plan = _plan(partition_path)
    weights = _load_dense_mlp(source_dir)
    model = _load_checkpoint(
        weights,
        profile,
        plan,
        run_dir / "layer-checkpoints" / "clean-validation" / args.checkpoint_name,
        args.device,
    )
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    gate = weights["gate_proj.weight"].to(args.device)
    up = weights["up_proj.weight"].to(args.device)
    down = weights["down_proj.weight"].to(args.device)

    set_intersection: list[np.ndarray] = []
    amplitude_error: list[np.ndarray] = []
    amplitude_relative_error: list[np.ndarray] = []
    row_values: dict[str, list[np.ndarray]] = {
        "student_nmse": [],
        "student_cosine": [],
        "student_ids_oracle_amplitudes_nmse": [],
        "student_ids_oracle_amplitudes_cosine": [],
        "oracle_ids_student_amplitudes_nmse": [],
        "oracle_ids_student_amplitudes_cosine": [],
        "oracle_ids_oracle_amplitudes_nmse": [],
        "oracle_ids_oracle_amplitudes_cosine": [],
        "teacher_norm": [],
        "residual_norm": [],
        "selector_margin": [],
        "oracle_amplitude_sum": [],
        "oracle_amplitude_max_fraction": [],
    }
    confusion = np.zeros((model.routed_experts, model.routed_experts), dtype=np.int64)
    target_norm_sum = 0.0
    prediction_error_sum = 0.0
    oracle_error_sum = 0.0
    oracle_target_norm_sum = 0.0
    cosine_sums = {key: 0.0 for key in ("student", "student_ids_oracle_amplitudes", "oracle_ids_student_amplitudes", "oracle_ids_oracle_amplitudes")}
    count = 0
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(validation_indices, args.microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=args.device)
            hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(args.device))
            prediction, info = model(inputs, return_router=True, return_contributions=True)
            shared = info["shared"]
            routed = info["contributions"]
            oracle = _residual_correlation_beam_topk(
                shared,
                routed,
                target,
                model.top_k,
                simplex=False,
                beam_width=args.beam_width,
                pool_size=args.pool_size,
            )
            ids = info["indices"]
            student_amp_all = torch.nn.functional.softplus(info["amplitude_logits"])
            oracle_amp = oracle["weights"]
            oracle_ids_batch = oracle["indices"]
            residual = target - shared
            student_ids_oracle_amp = _positive_coefficients(routed, ids, residual)
            oracle_amp_on_oracle = torch.gather(student_amp_all, 1, oracle_ids_batch)
            # The conditional variants intentionally differ only in their
            # selector and amplitude source.
            oracle_ids_student = _conditional(shared, routed, oracle_ids_batch, oracle_amp_on_oracle)
            oracle_ids_oracle = _conditional(shared, routed, oracle_ids_batch, oracle_amp)
            student_ids_student = prediction
            student_ids_oracle = _conditional(shared, routed, ids, torch.gather(student_amp_all, 1, ids))
            student_ids_oracle_amplitudes = _conditional(shared, routed, ids, student_ids_oracle_amp)
            batch_metrics = {
                "student": student_ids_student,
                "student_ids_oracle_amplitudes": student_ids_oracle_amplitudes,
                "oracle_ids_student_amplitudes": oracle_ids_student,
                "oracle_ids_oracle_amplitudes": oracle_ids_oracle,
            }
            for name, value in batch_metrics.items():
                nmse, cosine = _row_metrics(value, target)
                row_values[f"{name}_nmse"].append(nmse)
                row_values[f"{name}_cosine"].append(cosine)
                cosine_sums[name] += float(cosine.sum())
            teacher_norm = torch.linalg.vector_norm(target, dim=1)
            residual_norm = torch.linalg.vector_norm(target - shared, dim=1)
            sorted_logits = torch.sort(info["logits"], dim=1, descending=True).values
            selector_margin = sorted_logits[:, model.top_k - 1] - sorted_logits[:, model.top_k]
            oracle_sum = oracle_amp.sum(dim=1)
            oracle_max_fraction = oracle_amp.max(dim=1).values / oracle_sum.clamp_min(1e-12)
            for name, values_tensor in {
                "teacher_norm": teacher_norm,
                "residual_norm": residual_norm,
                "selector_margin": selector_margin,
                "oracle_amplitude_sum": oracle_sum,
                "oracle_amplitude_max_fraction": oracle_max_fraction,
            }.items():
                row_values[name].append(values_tensor.detach().cpu().numpy())
            amp_diff = oracle_amp_on_oracle - oracle_amp
            amplitude_error.append(amp_diff.abs().detach().cpu().numpy().reshape(-1))
            amplitude_relative_error.append((amp_diff.abs() / oracle_amp.clamp_min(1e-6)).detach().cpu().numpy().reshape(-1))
            ids_np = ids.detach().cpu().numpy()
            oracle_np = oracle_ids_batch.detach().cpu().numpy()
            set_intersection.append(np.asarray([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids_np, oracle_np)], dtype=np.float64))
            for student_row, oracle_row in zip(ids_np, oracle_np):
                for student_expert in student_row:
                    for oracle_expert in oracle_row:
                        confusion[int(student_expert), int(oracle_expert)] += 1
            target_norm_sum += float(target.square().sum().item())
            prediction_error_sum += float((prediction - target).square().sum().item())
            oracle_error_sum += float((oracle_ids_oracle - target).square().sum().item())
            oracle_target_norm_sum += float(target.square().sum().item())
            count += int(inputs.shape[0])

    arrays = {name: np.concatenate(values) for name, values in row_values.items()}
    overlap = np.concatenate(set_intersection)
    amplitude_abs = np.concatenate(amplitude_error)
    amplitude_rel = np.concatenate(amplitude_relative_error)
    agreement = {
        "exact_set_match_rate": float(np.mean(overlap == model.top_k)),
        "mean_intersection": float(np.mean(overlap)),
        "mean_jaccard": float(np.mean(overlap / (2 * model.top_k - overlap))),
        "mean_selector_recall": float(np.mean(overlap / model.top_k)),
    }
    difficulty = []
    edges = np.quantile(arrays["residual_norm"], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    for bucket in range(4):
        lower, upper = float(edges[bucket]), float(edges[bucket + 1])
        mask = (arrays["residual_norm"] >= lower) & (arrays["residual_norm"] <= upper if bucket == 3 else arrays["residual_norm"] < upper)
        difficulty.append({
            "bucket": bucket,
            "residual_norm_range": [lower, upper],
            "count": int(mask.sum()),
            "student_nmse": float(np.mean(arrays["student_nmse"][mask])),
            "student_cosine": float(np.mean(arrays["student_cosine"][mask])),
            "oracle_ids_oracle_amplitudes_nmse": float(np.mean(arrays["oracle_ids_oracle_amplitudes_nmse"][mask])),
            "oracle_ids_oracle_amplitudes_cosine": float(np.mean(arrays["oracle_ids_oracle_amplitudes_cosine"][mask])),
            "selector_recall": float(np.mean(overlap[mask] / model.top_k)),
        })
    payload = {
        "schema_version": 1,
        "status": "CLEAN_VALIDATION_ROUTER_DIAGNOSTICS_COMPLETE",
        "classification": "VALIDATION_ONLY_NO_GRADIENTS_NO_HOLDOUT",
        "profile": profile.name,
        "routing_mode": profile.routing_mode,
        "checkpoint_name": args.checkpoint_name,
        "partition": str(partition_path),
        "validation": {
            "count": count,
            "identity_hash": validation_hash,
            "selected_indices_hash": validation_hash,
            "opened_for_gradients": False,
            "opened_for_selection": False,
            "opened_for_diagnostics": True,
        },
        "holdout": {"status": "CLOSED", "opened": False},
        "oracle": {
            "selection": "residual_correlation_beam_search_exact_final_coefficients",
            "beam_width": args.beam_width,
            "pool_size": args.pool_size,
            "positive_amplitudes": True,
        },
        "set_agreement": agreement,
        "amplitude_quality_on_oracle_ids": {
            "mae": float(np.mean(amplitude_abs)),
            "rmse": float(np.sqrt(np.mean(amplitude_abs ** 2))),
            "relative_mae": float(np.mean(amplitude_rel)),
            "median_absolute_error": float(np.median(amplitude_abs)),
            "finite": bool(np.isfinite(amplitude_abs).all() and np.isfinite(amplitude_rel).all()),
        },
        "conditional_reconstruction": {
            "student_ids_student_amplitudes": {
                "normalized_mse": float(np.mean(arrays["student_nmse"])),
                "cosine": float(np.mean(arrays["student_cosine"])),
            },
            "student_ids_oracle_amplitudes": {
                "normalized_mse": float(np.mean(arrays["student_ids_oracle_amplitudes_nmse"])),
                "cosine": float(np.mean(arrays["student_ids_oracle_amplitudes_cosine"])),
            },
            "oracle_ids_student_amplitudes": {
                "normalized_mse": float(np.mean(arrays["oracle_ids_student_amplitudes_nmse"])),
                "cosine": float(np.mean(arrays["oracle_ids_student_amplitudes_cosine"])),
            },
            "oracle_ids_oracle_amplitudes": {
                "normalized_mse": float(np.mean(arrays["oracle_ids_oracle_amplitudes_nmse"])),
                "cosine": float(np.mean(arrays["oracle_ids_oracle_amplitudes_cosine"])),
            },
        },
        "difficulty_buckets": difficulty,
        "router_margin": {
            "mean": float(np.mean(arrays["selector_margin"])),
            "p10": float(np.quantile(arrays["selector_margin"], 0.1)),
            "p50": float(np.quantile(arrays["selector_margin"], 0.5)),
            "p90": float(np.quantile(arrays["selector_margin"], 0.9)),
        },
        "expert_set_confusion": confusion.tolist(),
        "counts": {"tokens": count, "target_norm": target_norm_sum, "student_squared_error": prediction_error_sum, "oracle_squared_error": oracle_error_sum, "oracle_target_norm": oracle_target_norm_sum},
    }
    report = run_dir / "reports" / args.report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--partition-name", default="high-sparsity-p16-top4.json")
    parser.add_argument("--checkpoint-name", default="p16-top4-residual-ce")
    parser.add_argument("--report-name", default="p16-top4-clean-validation-diagnostics.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=10)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "status": payload["status"],
        "validation_count": payload["validation"]["count"],
        "set_agreement": payload["set_agreement"],
        "conditional_reconstruction": payload["conditional_reconstruction"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
