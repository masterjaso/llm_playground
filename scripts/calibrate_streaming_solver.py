"""Calibrate the streaming projected-positive p16 solver against exact refits.

This is a FIT-only diagnostic.  It uses a deterministic stratified subset of
the layer-0 train capture, computes the frozen p16/top4 contribution basis
from the pinned dense MLP, and compares the exhaustive active-face solver with
the load-aware streaming scorer.  The script intentionally never opens the
holdout manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:  # ``resource`` is POSIX-only; keep the guarded entry point Windows-safe.
    import resource
except ImportError:  # pragma: no cover - exercised by native Windows smoke
    resource = None  # type: ignore[assignment]

import numpy as np

from dense2moe.partition import PartitionPlan
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset

try:
    from scripts.run_exact_p16_oracle import (
        _dense_hidden_target,
        _exact_positive_from_gram,
        _exact_topk_precomputed,
        _load_dense_mlp,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import (  # type: ignore
        _dense_hidden_target,
        _exact_positive_from_gram,
        _exact_topk_precomputed,
        _load_dense_mlp,
    )

from dense2moe.partition.oracle import _stream_selected_weights, frozen_slice_load_aware_oracle

DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


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


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(int(value)) for value in indices).encode()).hexdigest()


def _load_train_prefix(dataset: ActivationShardDataset, count: int, microbatch: int) -> np.ndarray:
    indices = np.arange(min(int(count), dataset.count), dtype=np.int64)
    batches = list(dataset.iter_selected_batches(indices.tolist(), microbatch))
    if not batches:
        raise ValueError("train activation capture is empty")
    return np.concatenate([np.asarray(batch, dtype=np.float32) for batch in batches], axis=0)


def _dense_residual_norms(
    inputs: np.ndarray,
    weights: dict[str, Any],
    plan: PartitionPlan,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    device_obj = torch.device(device)
    gate = weights["gate_proj.weight"].to(device_obj)
    up = weights["up_proj.weight"].to(device_obj)
    down = weights["down_proj.weight"].to(device_obj)
    shared_index = torch.as_tensor(plan.shared_indices, dtype=torch.long, device=device_obj)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device_obj)
            _hidden, target = _dense_hidden_target(
                x,
                {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                device_obj,
            )
            hidden = torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
            shared = hidden.index_select(1, shared_index) @ down.index_select(1, shared_index).T
            values.append(torch.linalg.vector_norm(target - shared, dim=1).cpu().numpy())
    return np.concatenate(values, axis=0)


def _stratified_indices(residual_norms: np.ndarray, sample_count: int) -> np.ndarray:
    if sample_count < 12:
        raise ValueError("sample_count must be at least 12 for three strata")
    order = np.argsort(residual_norms, kind="stable")
    low = order[: max(1, len(order) // 4)]
    high = order[-max(1, len(order) // 4) :]
    middle = order[max(1, len(order) // 4) : -max(1, len(order) // 4)]
    counts = (sample_count // 4, sample_count - 2 * (sample_count // 4), sample_count // 4)
    selected = np.concatenate(
        [
            low[np.linspace(0, len(low) - 1, counts[0], dtype=np.int64)],
            middle[np.linspace(0, len(middle) - 1, counts[1], dtype=np.int64)],
            high[np.linspace(0, len(high) - 1, counts[2], dtype=np.int64)],
        ]
    )
    return np.unique(selected.astype(np.int64))


def _contributions(
    inputs: np.ndarray,
    weights: dict[str, Any],
    plan: PartitionPlan,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    device_obj = torch.device(device)
    gate = weights["gate_proj.weight"].to(device_obj)
    up = weights["up_proj.weight"].to(device_obj)
    down = weights["down_proj.weight"].to(device_obj)
    shared_index = torch.as_tensor(plan.shared_indices, dtype=torch.long, device=device_obj)
    expert_indices = [torch.as_tensor(group, dtype=torch.long, device=device_obj) for group in plan.expert_indices]
    shared_values: list[np.ndarray] = []
    routed_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    hardness: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device_obj)
            hidden, target = _dense_hidden_target(
                x,
                {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                device_obj,
            )
            shared = hidden.index_select(1, shared_index) @ down.index_select(1, shared_index).T
            routed = torch.stack(
                [hidden.index_select(1, indices) @ down.index_select(1, indices).T for indices in expert_indices],
                dim=1,
            )
            shared_values.append(shared.cpu().numpy().astype(np.float32, copy=False))
            routed_values.append(routed.cpu().numpy().astype(np.float32, copy=False))
            target_values.append(target.cpu().numpy().astype(np.float32, copy=False))
            hardness.append(torch.linalg.vector_norm(target - shared, dim=1).cpu().numpy())
    return (
        np.concatenate(shared_values, axis=0),
        np.concatenate(routed_values, axis=0),
        np.concatenate(target_values, axis=0),
        np.concatenate(hardness, axis=0),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_contribution_store(
    store_dir: Path,
    shared: np.ndarray,
    routed: np.ndarray,
    target: np.ndarray,
    *,
    dataset_hash: str,
    partition_hash: str,
    indices_hash: str,
    basis_source: str = "raw_dense_partition",
    checkpoint_path: str | None = None,
    checkpoint_tensor_sha256: str | None = None,
    partition_path: str | None = None,
    topology: dict[str, Any] | None = None,
    source_revision: str | None = None,
    capture_identity: dict[str, Any] | None = None,
    split: str = "train",
    row_count: int | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Persist a read-only mmap contribution store consumed by the CLI."""

    store_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for name, values in (("shared", shared), ("routed", routed), ("target", target)):
        path = store_dir / f"{name}.npy"
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=values.shape)
        mapped[...] = np.asarray(values, dtype=np.float32)
        mapped.flush()
        del mapped
        arrays[name] = {
            "path": path.name,
            "shape": list(values.shape),
            "dtype": "float32",
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "format": "dense2moe-contribution-store-v1",
        "arrays": arrays,
        "dataset_hash": dataset_hash,
        "partition_hash": partition_hash,
        "indices_sha256": indices_hash,
        "basis_source": basis_source,
        "checkpoint_path": checkpoint_path,
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
        "partition_path": partition_path,
        "partition_sha256": partition_hash,
        "topology_manifest": topology or {},
        "source_revision": source_revision,
        "capture_identity": capture_identity or {},
        "split": split,
        "row_count": int(row_count if row_count is not None else len(target)),
        "dtype": dtype,
        "code_commit": current_git_commit(),
        "classification": "FIT_ONLY_SOLVER_CALIBRATION_NO_HOLDOUT",
        "holdout_opened": False,
    }
    manifest_path = store_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(store_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "format": manifest["format"],
        "arrays": arrays,
        "classification": manifest["classification"],
        "holdout_opened": False,
    }


def _metric(shared: np.ndarray, routed: np.ndarray, target: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    prediction = shared + np.sum(np.take_along_axis(routed, ids[:, :, None], axis=1) * weights[:, :, None], axis=1)
    error = np.sum((prediction - target) ** 2, axis=1, dtype=np.float64)
    target_norm = np.sum(target**2, axis=1, dtype=np.float64)
    cosine = np.sum(prediction * target, axis=1, dtype=np.float64) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    usage = np.bincount(ids.reshape(-1), minlength=routed.shape[1])
    hard = np.argsort(-np.linalg.norm(target - shared, axis=1), kind="stable")[: max(1, len(target) // 4)]
    return {
        "tokens": len(target),
        "global_nmse": float(error.sum() / max(target_norm.sum(), 1e-12)),
        "mean_cosine": float(cosine.mean()),
        "hard_quartile_cosine": float(cosine[hard].mean()),
        "expert_usage_counts": usage.tolist(),
        "dead_experts": int(np.sum(usage == 0)),
        "load_cv": float(usage.std() / max(usage.mean(), 1e-12)),
        "mean_token_relative_mse": float(np.mean(error / np.maximum(target_norm, 1e-12))),
    }


def _exact_batches(
    shared: np.ndarray,
    routed: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    ids_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    device_obj = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(target), batch_size):
            stop = min(len(target), start + batch_size)
            result = _exact_topk_precomputed(
                torch.as_tensor(shared[start:stop], dtype=torch.float32, device=device_obj),
                torch.as_tensor(routed[start:stop], dtype=torch.float32, device=device_obj),
                torch.as_tensor(target[start:stop], dtype=torch.float32, device=device_obj),
                4,
            )
            ids_parts.append(result["indices"].cpu().numpy())
            weight_parts.append(result["weights"].cpu().numpy())
    return np.concatenate(ids_parts, axis=0), np.concatenate(weight_parts, axis=0)


def _exact_refit_selected(
    routed: np.ndarray,
    shared: np.ndarray,
    target: np.ndarray,
    ids: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    device_obj = torch.device(device)
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(target), batch_size):
            stop = min(len(target), start + batch_size)
            rv = torch.as_tensor(routed[start:stop], dtype=torch.float32, device=device_obj)
            selected_ids = torch.as_tensor(ids[start:stop], dtype=torch.long, device=device_obj)
            gather_ids = selected_ids[:, None, :, None].expand(-1, 1, -1, rv.shape[-1])
            selected = torch.gather(rv[:, None, :, :], 2, gather_ids)[:, 0]
            residual = torch.as_tensor(target[start:stop] - shared[start:stop], dtype=torch.float32, device=device_obj)
            gram = torch.bmm(selected, selected.transpose(1, 2))
            rhs = torch.bmm(selected, residual.unsqueeze(-1)).squeeze(-1)
            residual_squared = (residual * residual).sum(dim=1)
            errors, fitted = _exact_positive_from_gram(
                gram[:, None],
                rhs[:, None],
                residual_squared[:, None],
            )
            del errors
            parts.append(fitted[:, 0].cpu().numpy())
    return np.concatenate(parts, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    dataset = ActivationShardDataset(run_dir / "capture/layer-0000-train.json", split="train", microbatch=args.microbatch)
    plan_path = run_dir / "partitions" / args.partition_name
    plan = _load_plan(plan_path)
    weights = _load_dense_mlp(source_dir, layer=0)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    prefix_inputs = _load_train_prefix(dataset, args.stratify_pool, args.microbatch)
    residual_norms = _dense_residual_norms(prefix_inputs, weights, plan, device=device, batch_size=args.microbatch)
    local = _stratified_indices(residual_norms, min(args.sample_count, len(prefix_inputs)))
    inputs = prefix_inputs[local]
    indices = local.astype(np.int64)
    selection_started = time.perf_counter()
    shared, routed, target, hardness = _contributions(inputs, weights, plan, device=device, batch_size=args.microbatch)
    partition_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    contribution_store = (
        _write_contribution_store(
            Path(args.store_dir),
            shared,
            routed,
            target,
            dataset_hash=dataset.dataset_hash,
            partition_hash=partition_hash,
            indices_hash=_hash_indices(indices),
            basis_source="raw_dense_partition",
            partition_path=str(plan_path),
            topology={
                "expert_count": int(plan.routed_experts),
                "expert_width": int(plan.expert_intermediate_size),
                "shared_width": int(plan.shared_intermediate_size),
                "top_k": 4,
            },
            capture_identity={"manifest": str(run_dir / "capture/layer-0000-train.json")},
            split="train",
            row_count=len(indices),
            dtype="float32",
        )
        if args.store_dir
        else None
    )
    exact_ids, exact_weights = _exact_batches(shared, routed, target, device=device, batch_size=args.microbatch)
    exact_elapsed = time.perf_counter() - selection_started
    exact_metric = _metric(shared, routed, target, exact_ids, exact_weights)

    streaming_started = time.perf_counter()
    projected = frozen_slice_load_aware_oracle(
        shared,
        routed,
        target,
        top_k=4,
        target_load_cv=1.0,
        candidate_pool_size=None,
        max_combinations=4096,
        iterations=1,
        penalty_grid=(0.0,),
        batch_size=args.microbatch,
        max_in_memory_bytes=args.max_in_memory_bytes,
        storage_dir=run_dir / "evidence" / "streaming-solver-calibration-work",
        materialize_outputs=False,
    )
    projected_elapsed = time.perf_counter() - streaming_started
    projected_ids = np.asarray(projected["unconstrained"]["indices"], dtype=np.int64)
    # The load-aware scorer deliberately prices candidates with the bounded
    # float32 projected fit.  The production result then refits the selected
    # route exactly on its small active-face system.  Record both stages so a
    # calibration cannot accidentally label the final refit as a scorer pass.
    candidate_score_weights = _stream_selected_weights(
        routed,
        target,
        shared,
        projected_ids,
        simplex=False,
        batch_size=args.microbatch,
        exact_positive=False,
    )
    candidate_score_metric = _metric(shared, routed, target, projected_ids, candidate_score_weights)
    projected_weights = np.asarray(projected["unconstrained"]["weights"], dtype=np.float32)
    final_metric = _metric(shared, routed, target, projected_ids, projected_weights)
    exact_projected_weights = _exact_refit_selected(
        routed,
        shared,
        target,
        projected_ids,
        device=device,
        batch_size=args.microbatch,
    )
    candidate_score_coefficient_error = np.linalg.norm(candidate_score_weights - exact_projected_weights, axis=1) / np.maximum(
        np.linalg.norm(exact_projected_weights, axis=1), 1e-8
    )
    final_coefficient_error = np.linalg.norm(projected_weights - exact_projected_weights, axis=1) / np.maximum(
        np.linalg.norm(exact_projected_weights, axis=1), 1e-8
    )
    overlap = np.asarray(
        [len(set(left.tolist()) & set(right.tolist())) / 4.0 for left, right in zip(exact_ids, projected_ids)],
        dtype=np.float64,
    )
    exact_set_match = float(np.mean(overlap == 1.0))
    hard = np.argsort(-hardness, kind="stable")[: max(1, len(target) // 4)]
    if resource is None:
        rss = 0
    else:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if os.name == "posix" else 1))
    cuda_peak = int(torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() and device.startswith("cuda") else 0)
    payload = {
        "schema_version": 1,
        "status": "STREAMING_SOLVER_CALIBRATION_COMPLETE",
        "classification": "FIT_ONLY_SOLVER_CALIBRATION_NO_HOLDOUT",
        "code_commit": current_git_commit(),
        "dataset_hash": dataset.dataset_hash,
        "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "split": "train",
        "holdout_opened": False,
        "partition": str(plan_path),
        "partition_hash": partition_hash,
        "sample": {
            "stratify_pool": len(prefix_inputs),
            "count": len(indices),
            "indices_sha256": _hash_indices(indices),
            "strata": {"easy": int(len(indices) // 4), "medium": int(len(indices) - 2 * (len(indices) // 4)), "hard": int(len(indices) // 4)},
            "hard_quartile_count": len(hard),
        },
        "exact_solver": {
            "method": "all_1820_candidate_sets_nonnegative_active_face_refit",
            "runtime_seconds": float(exact_elapsed),
            **exact_metric,
        },
        "streaming_solver": {
            "method": "exhaustive_candidate_set_batched_projected_float32_scoring_with_exact_selected_route_refit",
            "runtime_seconds": float(projected_elapsed),
            "candidate_assurance": projected["assurance"],
            "coefficient_solver": projected["coefficient_solver"],
            "candidate_fit_exact": projected["candidate_fit_exact"],
            "selected_fit_exact": projected["selected_fit_exact"],
            "candidate_batch_size": projected["candidate_batch_size"],
            "input_batch_bytes": projected["input_batch_bytes"],
            **final_metric,
        },
        "candidate_score": {
            "method": "batched_projected_float32_candidate_fit_before_selected_refit",
            **candidate_score_metric,
        },
        "discrepancy": {
            "candidate_score_mean_cosine_delta_minus_exact": float(candidate_score_metric["mean_cosine"] - exact_metric["mean_cosine"]),
            "candidate_score_global_nmse_delta_minus_exact": float(candidate_score_metric["global_nmse"] - exact_metric["global_nmse"]),
            "candidate_score_hard_quartile_cosine_delta_minus_exact": float(candidate_score_metric["hard_quartile_cosine"] - exact_metric["hard_quartile_cosine"]),
            "final_refit_mean_cosine_delta_minus_exact": float(final_metric["mean_cosine"] - exact_metric["mean_cosine"]),
            "final_refit_global_nmse_delta_minus_exact": float(final_metric["global_nmse"] - exact_metric["global_nmse"]),
            "final_refit_hard_quartile_cosine_delta_minus_exact": float(final_metric["hard_quartile_cosine"] - exact_metric["hard_quartile_cosine"]),
            "selected_set_mean_overlap": float(overlap.mean()),
            "selected_set_exact_match": exact_set_match,
            "candidate_score_coefficient_relative_l2_error_mean": float(candidate_score_coefficient_error.mean()),
            "candidate_score_coefficient_relative_l2_error_p95": float(np.quantile(candidate_score_coefficient_error, 0.95)),
            "final_refit_coefficient_relative_l2_error_mean": float(final_coefficient_error.mean()),
            "final_refit_coefficient_relative_l2_error_p95": float(np.quantile(final_coefficient_error, 0.95)),
            "acceptance": {
                "final_refit_mean_cosine_abs_le_2e-4": bool(abs(final_metric["mean_cosine"] - exact_metric["mean_cosine"]) <= 2e-4),
                "final_refit_global_nmse_abs_le_5e-4": bool(abs(final_metric["global_nmse"] - exact_metric["global_nmse"]) <= 5e-4),
            },
        },
        "memory": {"peak_rss_bytes": rss, "peak_cuda_allocated_bytes": cuda_peak, "max_in_memory_bytes": int(args.max_in_memory_bytes)},
    }
    if contribution_store is not None:
        payload["contribution_store"] = contribution_store
    output = run_dir / "reports" / args.output_name
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        "# Streaming solver calibration\n\n"
        f"- FIT-only train subset: `{len(indices):,}` rows; holdout opened: **no**.\n"
        f"- Exact cosine `{exact_metric['mean_cosine']:.8f}`; candidate-score cosine `{candidate_score_metric['mean_cosine']:.8f}`; final-refit cosine `{final_metric['mean_cosine']:.8f}`.\n"
        f"- Exact global NMSE `{exact_metric['global_nmse']:.8f}`; candidate-score `{candidate_score_metric['global_nmse']:.8f}`; final-refit `{final_metric['global_nmse']:.8f}`.\n"
        f"- Mean route overlap `{overlap.mean():.4f}`; exact set match `{exact_set_match:.4f}`.\n"
        f"- Acceptance after selected-route refit: cosine `{payload['discrepancy']['acceptance']['final_refit_mean_cosine_abs_le_2e-4']}`, NMSE `{payload['discrepancy']['acceptance']['final_refit_global_nmse_abs_le_5e-4']}`.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--partition-name", default="high-sparsity-p16-top4.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--stratify-pool", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--max-in-memory-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--store-dir", type=Path, default=None, help="optionally persist the FIT-only contribution store")
    parser.add_argument("--output-name", default="streaming-solver-calibration.json")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "exact_solver": payload["exact_solver"],
                "streaming_solver": payload["streaming_solver"],
                "discrepancy": payload["discrepancy"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
