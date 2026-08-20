"""Test a FIT-only global scale on the p16/top4 shared branch.

This is a bounded basis/topology diagnostic, not a training sweep.  It fits
one scalar ``shared_scale`` from FIT rows, then compares the deployed student
and all-1,820-set positive oracle on true validation.  The holdout is never
opened.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dense2moe.config import load_config
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


def _fit_shared_scale(model: Any, dataset: ActivationShardDataset, validation_indices: np.ndarray, weights: dict[str, Any], *, microbatch: int, device: str) -> float:
    import torch
    import torch.nn.functional as F

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    numerator = 0.0
    denominator = 0.0
    with torch.inference_mode():
        for values in dataset.iter_excluding_batches(validation_indices.tolist(), microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(device))
            shared = model.shared_down_proj(F.silu(model.shared_gate_proj(inputs)) * model.shared_up_proj(inputs))
            numerator += float(torch.sum(shared * target).item())
            denominator += float(torch.sum(shared.square()).item())
    if denominator <= 1e-12:
        raise ValueError("shared branch has zero FIT energy")
    return float(numerator / denominator)


def _evaluate(model: Any, dataset: ActivationShardDataset, indices: np.ndarray, weights: dict[str, Any], *, scale: float, microbatch: int, device: str) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in ("target_sq", "student_sq", "oracle_sq", "student_cos", "oracle_cos", "residual")}
    usage = np.zeros(model.routed_experts, dtype=np.int64)
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(indices.tolist(), microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(device))
            student, info = model(inputs, return_router=True, return_contributions=True)
            shared = info["shared"].reshape(-1, info["shared"].shape[-1])
            routed = info["contributions"].reshape(-1, info["contributions"].shape[-2], info["contributions"].shape[-1])
            target = target.reshape(-1, target.shape[-1])
            scaled_shared = shared * float(scale)
            exact = _exact_topk_precomputed(scaled_shared, routed, target, model.top_k)
            oracle = _route_reconstruction(scaled_shared, routed, exact)
            rows["target_sq"].append(target.square().sum(dim=1).cpu().numpy())
            rows["student_sq"].append((student.reshape(-1, student.shape[-1]) - target).square().sum(dim=1).cpu().numpy())
            rows["oracle_sq"].append((oracle - target).square().sum(dim=1).cpu().numpy())
            rows["student_cos"].append((student.reshape(-1, student.shape[-1]) * target).sum(dim=1).div(torch.linalg.vector_norm(student.reshape(-1, student.shape[-1]), dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).cpu().numpy())
            rows["oracle_cos"].append((oracle * target).sum(dim=1).div(torch.linalg.vector_norm(oracle, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12).cpu().numpy())
            rows["residual"].append(torch.linalg.vector_norm(target - shared, dim=1).cpu().numpy())
            ids = info["indices"].reshape(-1, model.top_k)
            for slot in range(model.top_k):
                usage += np.bincount(ids[:, slot].cpu().numpy(), minlength=model.routed_experts)
    return {"rows": {key: np.concatenate(value) for key, value in rows.items()}, "usage": usage}


def _summary(raw: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    rows = raw["rows"]
    target = rows["target_sq"][mask]
    usage = raw["usage"]
    return {
        "tokens": int(mask.sum()),
        "global_nmse": float(rows["oracle_sq"][mask].sum() / max(target.sum(), 1e-12)),
        "mean_token_relative_mse": float(np.mean(rows["oracle_sq"][mask] / np.maximum(target, 1e-12))),
        "mean_cosine": float(np.mean(rows["oracle_cos"][mask])),
        "student_mean_cosine_unscaled": float(np.mean(rows["student_cos"][mask])),
        "load_cv": float(usage.std() / max(usage.mean(), 1e-12)),
        "dead_experts": int(np.sum(usage == 0)),
        "expert_usage_counts": usage.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dev = json.loads((run_dir / "capture/architecture-dev.json").read_text(encoding="utf-8"))
    validation_indices = np.asarray(dev["selected_global_indices"], dtype=np.int64)
    validation_hash = str(dev["selected_row_key_hash"])
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan = _load_plan(run_dir / "partitions" / "high-sparsity-p16-top4.json")
    weights = _load_dense_mlp(source_dir)
    model, checkpoint = _load_deployed_checkpoint(weights, profile, plan, run_dir / "layer-checkpoints" / "clean-validation" / args.checkpoint_name, args.device)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="train", microbatch=args.microbatch)
    scale = _fit_shared_scale(model, dataset, validation_indices, weights, microbatch=args.microbatch, device=args.device)
    started = time.perf_counter()
    raw = _evaluate(model, dataset, validation_indices, weights, scale=scale, microbatch=args.microbatch, device=args.device)
    edges = np.quantile(raw["rows"]["residual"], [0.0, 0.25, 0.5, 0.75, 1.0])
    hard = raw["rows"]["residual"] >= edges[3]
    all_mask = np.ones(len(validation_indices), dtype=bool)
    report = {
        "schema_version": 1,
        "status": "SHARED_SCALE_FROZEN_ORACLE_COMPLETE",
        "classification": "TRUE_FIT_ONLY_SCALE_VALIDATION_ONLY_NO_GRADIENTS_NO_HOLDOUT",
        "hypothesis": "A single FIT-fitted shared-branch scale can recover the residual angular gap without changing active geometry.",
        "falsifier": "The exact positive oracle with the FIT-fitted scale remains below the current unscaled ceiling or does not improve the hard quartile.",
        "code_commit": current_git_commit(),
        "source_revision": str(profile.revision),
        "dataset_hash": dataset.dataset_hash,
        "validation": {"count": len(validation_indices), "identity_hash": validation_hash, "holdout": {"count": 16598, "opened": False, "status": "CLOSED"}},
        "fit_scale": {"value": scale, "rows": 115124, "method": "least_squares_shared_to_dense_target", "validation_excluded": True},
        "checkpoint": {"metadata_code_commit": checkpoint.get("code_commit"), "tensor_sha256": checkpoint.get("tensor_sha256_observed")},
        "all_validation": _summary(raw, all_mask),
        "hard_quartile": _summary(raw, hard),
        "residual_quartile_edges": [float(v) for v in edges],
        "decision": "train_shared_scale_variant" if float(np.mean(raw["rows"]["oracle_cos"][hard])) > 0.961 else "reject_shared_scale_variant",
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = run_dir / "reports" / args.report_name
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(
        "# p16/top4 FIT-fitted shared scale oracle\n\n"
        f"- Code commit: `{report['code_commit']}`; fitted scale `{scale:.8f}`; holdout opened: **no**.\n"
        f"- All-validation exact cosine: `{report['all_validation']['mean_cosine']:.6f}`; hard-quartile exact cosine: `{report['hard_quartile']['mean_cosine']:.6f}`.\n\n"
        f"Decision: **{report['decision']}**.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint-name", default="p16-top4-residual-ce")
    parser.add_argument("--report-name", default="p16-top4-shared-scale-oracle.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=256)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "decision": report["decision"], "fit_scale": report["fit_scale"], "all_validation": report["all_validation"], "hard_quartile": report["hard_quartile"], "code_commit": report["code_commit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
