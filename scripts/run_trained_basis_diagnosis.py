"""Diagnose a trained p16/top4 or p32/top5 basis on fresh validation-A.

This driver is intentionally bounded.  It opens only the fresh layer-0 train
activation manifest and the frozen A index list, computes actual checkpoint
shared/expert contributions, and then runs the load-aware oracle.  Validation-B
is never opened and no optimizer update is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_config
from dense2moe.partition import frozen_slice_load_aware_oracle
from dense2moe.partition.contributions import (
    basis_outputs_from_state,
    canonical_partition_sha256,
    load_partition_plan,
    load_trained_basis_state,
    sha256_file,
)
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset

DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_FRESH = DEFAULT_RUN / "fresh-selector-layer0"
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _hash_indices(indices: list[int] | np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(int(value)) for value in indices).encode()).hexdigest()


def _load_ab(path: Path, count: int) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_SELECTOR_AB_FROZEN" or payload.get("historical_holdout_opened") is not False:
        raise ValueError("fresh A/B receipt is not a closed, frozen validation receipt")
    a = sorted({int(value) for value in payload["validation_a"]["indices"]})
    b = sorted({int(value) for value in payload["validation_b"]["indices"]})
    if len(a) != int(payload["validation_a"]["count"]) or len(b) != int(payload["validation_b"]["count"]):
        raise ValueError("fresh A/B receipt contains duplicate rows")
    if set(a) & set(b) or any(index < 0 or index >= count for index in (*a, *b)):
        raise ValueError("fresh A/B rows overlap or are outside the train capture")
    if _hash_indices(a) != str(payload["validation_a"]["indices_hash"]):
        raise ValueError("fresh validation-A identity hash mismatch")
    if _hash_indices(b) != str(payload["validation_b"]["indices_hash"]):
        raise ValueError("fresh validation-B identity hash mismatch")
    return a, b


def _metric_update(metric: dict[str, Any], prediction: np.ndarray, target: np.ndarray, ids: np.ndarray) -> None:
    error = np.sum((prediction - target) ** 2, axis=1, dtype=np.float64)
    target_norm = np.sum(target * target, axis=1, dtype=np.float64)
    cosine = np.sum(prediction * target, axis=1, dtype=np.float64) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    metric["tokens"] += int(target.shape[0])
    metric["error"] += float(np.sum(error))
    metric["target_norm"] += float(np.sum(target_norm))
    metric["cosine"] += float(np.sum(cosine))
    metric["relative"].append(error / np.maximum(target_norm, 1e-12))
    metric["cosine_rows"].append(cosine)
    metric["hardness"].append(metric.pop("_hardness_batch"))
    metric["ids"].append(ids)
    metric["usage"] += np.bincount(ids.reshape(-1), minlength=metric["usage"].shape[0])


def _finish_metric(metric: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(metric["tokens"]), 1)
    cosine_rows = np.concatenate(metric["cosine_rows"])
    hardness = np.concatenate(metric["hardness"])
    hard_count = max(1, int(np.ceil(tokens * 0.25)))
    hard = np.argsort(-hardness, kind="stable")[:hard_count]
    usage = np.asarray(metric["usage"], dtype=np.int64)
    return {
        "tokens": int(metric["tokens"]),
        "global_nmse": float(metric["error"] / max(metric["target_norm"], 1e-12)),
        "mean_token_relative_mse": float(np.mean(np.concatenate(metric["relative"]))),
        "cosine": float(metric["cosine"] / tokens),
        "hard_quartile_cosine": float(np.mean(cosine_rows[hard])),
        "expert_usage_counts": usage.tolist(),
        "load_cv": float(usage.std() / max(usage.mean(), 1e-12)),
        "dead_experts": int(np.sum(usage == 0)),
        "ids": np.concatenate(metric["ids"], axis=0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    topology = str(args.topology)
    profile_name = "qwen38_p16s1_top4.yaml" if topology == "p16/top4" else "qwen38_p32s1_top5.yaml"
    profile = load_config(Path("configs") / profile_name)
    plan_path = Path(args.partition)
    plan = load_partition_plan(plan_path)
    expected_experts = 16 if topology.startswith("p16/") else 32
    expected_top_k = int(topology.rsplit("top", 1)[1])
    if plan.routed_experts != expected_experts or expected_top_k != profile.top_k:
        raise ValueError("partition/profile topology mismatch")
    basis_state, checkpoint_metadata = load_trained_basis_state(args.checkpoint, plan)
    dataset = ActivationShardDataset(Path(args.activation_manifest), split="train", microbatch=args.microbatch)
    a_indices, b_indices = _load_ab(Path(args.validation_ab), dataset.count)
    if args.max_rows > len(a_indices):
        raise ValueError("max_rows exceeds validation-A count")
    # Stable evenly-spaced sampling keeps the bounded receipt representative
    # without turning A into a tuning loop.
    positions = np.linspace(0, len(a_indices) - 1, args.max_rows, dtype=np.int64)
    selected_indices = np.asarray(a_indices, dtype=np.int64)[positions]

    from scripts.run_exact_p16_oracle import _dense_hidden_target, _load_dense_mlp
    from scripts.train_fresh_p16_selector import _load_checkpoint_model

    source_weights = _load_dense_mlp(Path(args.source_dir))
    model = _load_checkpoint_model(source_weights, profile, plan_path, Path(args.checkpoint), args.device)
    gate = source_weights["gate_proj.weight"].to(args.device)
    up = source_weights["up_proj.weight"].to(args.device)
    down = source_weights["down_proj.weight"].to(args.device)
    state_hidden = int(np.asarray(basis_state["shared_gate_proj.weight"]).shape[1])
    if state_hidden != int(profile.hidden_size):
        raise ValueError("checkpoint hidden size does not match profile")

    student_metric = {
        "tokens": 0,
        "error": 0.0,
        "target_norm": 0.0,
        "cosine": 0.0,
        "relative": [],
        "cosine_rows": [],
        "hardness": [],
        "ids": [],
        "usage": np.zeros(plan.routed_experts, dtype=np.int64),
    }
    shared_parts: list[np.ndarray] = []
    routed_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    equivalence_max = 0.0
    equivalence_mean_sum = 0.0
    equivalence_squared_sum = 0.0
    equivalence_dot_sum = 0.0
    equivalence_direct_norm_sum = 0.0
    equivalence_store_norm_sum = 0.0
    equivalence_count = 0
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(selected_indices.tolist(), args.microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=args.device)
            _hidden, target_t = _dense_hidden_target(
                inputs,
                {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                torch.device(args.device),
            )
            student, info = model(inputs, return_router=True, return_contributions=True)
            direct_shared = info["shared"].reshape(-1, state_hidden).detach().cpu().numpy()
            direct_routed = info["contributions"].reshape(-1, plan.routed_experts, state_hidden).detach().cpu().numpy()
            values_np = values.float().cpu().numpy() if hasattr(values, "float") else np.asarray(values, dtype=np.float32)
            store_shared, store_routed = basis_outputs_from_state(
                values_np, basis_state, plan, batch_size=args.microbatch
            )
            store_shared = np.asarray(store_shared).reshape(-1, state_hidden)
            store_routed = np.asarray(store_routed).reshape(-1, plan.routed_experts, state_hidden)
            delta = np.abs(direct_shared - store_shared)
            equivalence_max = max(equivalence_max, float(np.max(delta)))
            equivalence_mean_sum += float(np.sum(delta))
            equivalence_squared_sum += float(np.sum((direct_shared - store_shared) ** 2))
            equivalence_dot_sum += float(np.sum(direct_shared * store_shared))
            equivalence_direct_norm_sum += float(np.sum(direct_shared * direct_shared))
            equivalence_store_norm_sum += float(np.sum(store_shared * store_shared))
            equivalence_count += int(delta.size)
            delta_r = np.abs(direct_routed - store_routed)
            equivalence_max = max(equivalence_max, float(np.max(delta_r)))
            equivalence_mean_sum += float(np.sum(delta_r))
            equivalence_squared_sum += float(np.sum((direct_routed - store_routed) ** 2))
            equivalence_dot_sum += float(np.sum(direct_routed * store_routed))
            equivalence_direct_norm_sum += float(np.sum(direct_routed * direct_routed))
            equivalence_store_norm_sum += float(np.sum(store_routed * store_routed))
            equivalence_count += int(delta_r.size)
            target = target_t.reshape(-1, state_hidden).detach().cpu().numpy()
            student_np = student.reshape(-1, state_hidden).detach().cpu().numpy()
            ids = info["indices"].reshape(-1, model.top_k).detach().cpu().numpy()
            student_metric["_hardness_batch"] = np.linalg.norm(target - direct_shared, axis=1)
            _metric_update(student_metric, student_np, target, ids)
            shared_parts.append(store_shared)
            routed_parts.append(store_routed)
            target_parts.append(target)

    shared = np.concatenate(shared_parts, axis=0).astype(np.float32, copy=False)
    routed = np.concatenate(routed_parts, axis=0).astype(np.float32, copy=False)
    target = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
    oracle = frozen_slice_load_aware_oracle(
        shared,
        routed,
        target,
        top_k=expected_top_k,
        target_load_cv=0.50,
        candidate_pool_size=(16 if topology == "p16/top4" else args.candidate_pool_size),
        max_combinations=4096,
        iterations=args.iterations,
        penalty_grid=tuple(args.penalties),
        batch_size=args.oracle_batch_size,
        max_in_memory_bytes=args.max_in_memory_bytes,
        materialize_outputs=False,
    )
    student = _finish_metric(student_metric)
    unconstrained = oracle["unconstrained"]
    basis_quality_green = bool(float(unconstrained["cosine"]) >= 0.98 and float(unconstrained["global_nmse"]) <= 0.05)
    joint_green = bool(oracle["green_gate"])
    blocker = "SELECTOR" if joint_green else "BASIS_LOAD_GEOMETRY" if basis_quality_green else "BASIS_QUALITY"
    student_ids = student.pop("ids")
    oracle_ids = np.asarray(oracle["indices"], dtype=np.int64)
    overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(student_ids, oracle_ids)])
    return {
        "schema_version": 1,
        "status": "TRAINED_BASIS_DIAGNOSIS_COMPLETE",
        "topology": topology,
        "basis_source": "trained_checkpoint",
        "checkpoint_path": checkpoint_metadata["checkpoint_path"],
        "checkpoint_tensor_sha256": checkpoint_metadata["checkpoint_tensor_sha256"],
        "partition_path": str(plan_path),
        "partition_sha256": sha256_file(plan_path),
        "partition_canonical_sha256": canonical_partition_sha256(plan),
        "source_revision": checkpoint_metadata.get("source_revision"),
        "activation_manifest": str(args.activation_manifest),
        "fresh_validation_receipt": str(args.validation_ab),
        "split": {
            "fit_rows": int(dataset.count - len(a_indices) - len(b_indices)),
            "validation_a_rows": len(a_indices),
            "validation_b_rows": len(b_indices),
            "selected_a_rows": len(selected_indices),
            "selected_a_indices_sha256": _hash_indices(selected_indices),
            "validation_a_indices_sha256": _hash_indices(a_indices),
            "validation_b_indices_sha256": _hash_indices(b_indices),
            "fit_a_overlap": 0,
            "fit_b_overlap": 0,
            "a_b_overlap": 0,
        },
        "equivalence": {
            "direct_checkpoint_basis_vs_contribution_store": True,
            "max_absolute_error": equivalence_max,
            "mean_absolute_error": equivalence_mean_sum / max(equivalence_count, 1),
            "mse": equivalence_squared_sum / max(equivalence_count, 1),
            "cosine_agreement": equivalence_dot_sum
            / (equivalence_direct_norm_sum**0.5 * equivalence_store_norm_sum**0.5 + 1e-12),
        },
        "student": student,
        "unconstrained_trained_basis_oracle": {
            "cosine": unconstrained["cosine"],
            "global_nmse": unconstrained["global_nmse"],
            "hard_quartile_cosine": unconstrained["hard_quartile_cosine"],
            "load_cv": unconstrained["load_cv"],
            "dead_experts": unconstrained["dead_experts"],
            "expert_usage_counts": unconstrained["expert_usage_counts"],
        },
        "load_constrained_trained_basis_oracle": {
            "cosine": oracle["cosine"],
            "global_nmse": oracle["global_nmse"],
            "hard_quartile_cosine": oracle["hard_quartile_cosine"],
            "load_cv": oracle["load_cv"],
            "dead_experts": oracle["dead_experts"],
            "expert_usage_counts": oracle["expert_usage_counts"],
            "gate_feasible": oracle["gate_feasible"],
            "pricing_frontier": oracle["pareto_all_points"],
        },
        "student_oracle_topk_recall": float(np.mean(overlap / expected_top_k)),
        "basis_quality_green": basis_quality_green,
        "joint_quality_load_green": joint_green,
        "identified_blocker": blocker,
        "refinement_state": "RESEARCH_CONTINUATION_REQUIRED",
        "holdout_opened": False,
        "representative_replay_started": False,
        "full64_replay_started": False,
        "code_commit": current_git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--validation-ab", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--candidate-pool-size", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--oracle-batch-size", type=int, default=32)
    parser.add_argument("--max-in-memory-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    if args.activation_manifest is None:
        args.activation_manifest = args.fresh_dir / "capture/layer-0000-train.json"
    if args.validation_ab is None:
        args.validation_ab = args.fresh_dir / "capture/fresh-selector-validation-ab.json"
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "identified_blocker": payload["identified_blocker"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
