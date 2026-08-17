"""Bounded TRAIN/dev-only search for the >=70% FFN-reduction candidates.

This search deliberately never opens the holdout manifest.  It compares
partition/basis choices for p16 and p32 with positive-amplitude oracle routing,
uses a stronger residual-correlation beam than the original search, and then
performs a small FIT-only residual swap refinement on the best p16/top4 and
p32/top5 plans.  Both p32/top5 and p32/top4 are emitted as explicit product
targets; the artifacts are inputs to subsequent fair layer-0 training and
load-aware oracle diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dense2moe.partition import PartitionPlan, partition_indices
from dense2moe.provenance import current_git_commit

try:
    from scripts.run_topk_architecture_search import (
        _accumulate_scales,
        _dense_hidden_target,
        _exact_topk,
        _fit_scales,
        _load_mlp,
        _materialize_selected,
        _plan_contributions,
        _residual_correlation_beam_topk,
        _route_reconstruction,
        _select_dev_rows,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_topk_architecture_search import (
        _accumulate_scales,
        _dense_hidden_target,
        _exact_topk,
        _fit_scales,
        _load_mlp,
        _materialize_selected,
        _plan_contributions,
        _residual_correlation_beam_topk,
        _route_reconstruction,
        _select_dev_rows,
    )


RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _profile(name: str, experts: int, expert_width: int, shared_width: int) -> dict[str, Any]:
    return {
        "name": name,
        "routed_experts": experts,
        "expert_intermediate_size": expert_width,
        "shared_intermediate_size": shared_width,
        "dense_intermediate_size": shared_width + experts * expert_width,
        "hidden_size": 5120,
    }


def _active_params(profile: dict[str, Any], top_k: int) -> dict[str, Any]:
    hidden = int(profile["hidden_size"])
    width = int(profile["shared_intermediate_size"] + top_k * profile["expert_intermediate_size"])
    dense_width = int(profile["dense_intermediate_size"])
    return {
        "active_intermediate_width": width,
        "active_ffn_parameters": 3 * hidden * width,
        "dense_ffn_parameters": 3 * hidden * dense_width,
        "active_ffn_parameter_ratio": width / dense_width,
        "ffn_reduction": 1.0 - width / dense_width,
        "expert_dispatches_per_token": int(top_k),
    }


def _ranked_plan(
    profile: dict[str, Any],
    scores: np.ndarray,
    *,
    shared_indices: np.ndarray | None = None,
) -> PartitionPlan:
    """Create a capacity-exact plan from a deterministic neuron ranking."""

    dense = int(profile["dense_intermediate_size"])
    experts = int(profile["routed_experts"])
    expert_width = int(profile["expert_intermediate_size"])
    shared_width = int(profile["shared_intermediate_size"])
    if shared_indices is None:
        order = sorted(range(dense), key=lambda index: (-float(scores[index]), index))
        shared = tuple(order[:shared_width])
        remaining = order[shared_width:]
    else:
        shared_set = {int(value) for value in np.asarray(shared_indices).tolist()}
        if len(shared_set) != shared_width:
            raise ValueError("shared_indices has the wrong capacity")
        shared = tuple(sorted(shared_set))
        remaining = sorted(
            (index for index in range(dense) if index not in shared_set),
            key=lambda index: (-float(scores[index]), index),
        )
    groups = tuple(
        tuple(remaining[offset * expert_width : (offset + 1) * expert_width])
        for offset in range(experts)
    )
    plan = PartitionPlan(dense, experts, expert_width, shared_width, shared, groups)
    plan.validate()
    return plan


def _signature_plan(profile: dict[str, Any], contribution_scores: np.ndarray, down: np.ndarray, seed: int) -> PartitionPlan:
    """Group neurons by deterministic low-dimensional output signatures."""

    dense = int(profile["dense_intermediate_size"])
    experts = int(profile["routed_experts"])
    expert_width = int(profile["expert_intermediate_size"])
    shared_width = int(profile["shared_intermediate_size"])
    shared = np.argsort(-contribution_scores, kind="stable")[:shared_width]
    shared_set = {int(value) for value in shared.tolist()}
    remaining = [index for index in range(dense) if index not in shared_set]
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((down.shape[0], 16), dtype=np.float32)
    signatures = down[:, remaining].T @ projection
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    signatures = signatures / np.maximum(norms, 1e-12)
    order = sorted(range(len(remaining)), key=lambda position: (-float(norms[position, 0]), remaining[position]))
    groups: list[list[int]] = [[] for _ in range(experts)]
    centroids = np.zeros((experts, signatures.shape[1]), dtype=np.float64)
    for position in order:
        choices = [expert for expert in range(experts) if len(groups[expert]) < expert_width]
        if not choices:
            raise RuntimeError("signature grouping exhausted expert capacity")
        if not any(groups[expert] for expert in choices):
            expert = min(choices)
        else:
            scores = []
            signature = signatures[position]
            for candidate in choices:
                if groups[candidate]:
                    centroid = centroids[candidate] / max(np.linalg.norm(centroids[candidate]), 1e-12)
                    scores.append((float(np.dot(signature, centroid)), -len(groups[candidate]), -candidate, candidate))
                else:
                    scores.append((-1.0, -len(groups[candidate]), -candidate, candidate))
            expert = max(scores)[-1]
        groups[expert].append(int(remaining[position]))
        centroids[expert] += signatures[position]
    plan = PartitionPlan(dense, experts, expert_width, shared_width, tuple(sorted(shared_set)), tuple(tuple(group) for group in groups))
    plan.validate()
    return plan


def _compute_dev_scores(
    inputs: np.ndarray,
    weights_cpu: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
    shared_width: int,
) -> dict[str, np.ndarray]:
    """Compute activation, contribution, and residual-correlation scores."""

    import torch
    import torch.nn.functional as F

    device_obj = torch.device(device)
    gate = torch.as_tensor(weights_cpu["gate_proj.weight"], dtype=torch.float32, device=device_obj)
    up = torch.as_tensor(weights_cpu["up_proj.weight"], dtype=torch.float32, device=device_obj)
    down = torch.as_tensor(weights_cpu["down_proj.weight"], dtype=torch.float32, device=device_obj)
    dense_width = int(gate.shape[0])
    activation_abs = torch.zeros(dense_width, dtype=torch.float64, device=device_obj)
    activation_sq = torch.zeros(dense_width, dtype=torch.float64, device=device_obj)
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device_obj)
            hidden = F.silu(x @ gate.T) * (x @ up.T)
            activation_abs += hidden.abs().sum(dim=0, dtype=torch.float64)
            activation_sq += hidden.square().sum(dim=0, dtype=torch.float64)
            del x, hidden
    count = max(int(inputs.shape[0]), 1)
    activation = (activation_abs / count).cpu().numpy().astype(np.float64)
    contribution = (activation_sq / count).cpu().numpy().astype(np.float64) * down.square().sum(dim=0).cpu().numpy().astype(np.float64)
    shared_indices = np.argsort(-contribution, kind="stable")[:shared_width]
    residual_score = torch.zeros(dense_width, dtype=torch.float64, device=device_obj)
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device_obj)
            hidden = F.silu(x @ gate.T) * (x @ up.T)
            target = hidden @ down.T
            shared_idx = torch.as_tensor(shared_indices, dtype=torch.long, device=device_obj)
            shared = hidden.index_select(1, shared_idx) @ down.index_select(1, shared_idx).T
            residual = target - shared
            projection = residual @ down
            residual_score += (hidden * projection).abs().sum(dim=0, dtype=torch.float64)
            del x, hidden, target, shared, residual, projection
    residual_score_np = (residual_score / count).cpu().numpy().astype(np.float64)
    return {
        "activation_magnitude": activation,
        "contribution_magnitude": contribution,
        "residual_aware_greedy": residual_score_np,
        "shared_indices": shared_indices.astype(np.int64),
    }


def _evaluate_positive_oracle(
    profile: dict[str, Any],
    inputs: np.ndarray,
    weights_cpu: dict[str, np.ndarray],
    plan: PartitionPlan,
    *,
    top_k: int,
    device: str,
    batch_size: int,
    exact: bool = False,
    beam_width: int = 8,
    pool_size: int | None = None,
    return_route_assignments: bool = False,
) -> dict[str, Any]:
    """Evaluate positive coefficients and a fitted global-scale diagnostic."""

    import torch

    device_obj = torch.device(device)
    weights = {name: torch.as_tensor(value, dtype=torch.float32, device=device_obj) for name, value in weights_cpu.items()}
    expert_count = int(profile["routed_experts"])
    error_sum = 0.0
    target_norm_sum = 0.0
    cosine_sum = 0.0
    usage = np.zeros(expert_count, dtype=np.int64)
    coefficient_values: list[np.ndarray] = []
    route_cache: list[tuple[np.ndarray, np.ndarray]] = []
    scale_fit = {"gram": np.zeros((expert_count, expert_count), dtype=np.float64), "rhs": np.zeros(expert_count, dtype=np.float64)}
    elapsed = 0.0
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size]
            hidden, target = _dense_hidden_target(batch, weights, device_obj)
            shared, routed = _plan_contributions(hidden, weights["down_proj.weight"], plan)
            tic = time.perf_counter()
            if exact:
                route = _exact_topk(shared, routed, target, top_k, simplex=False)
            else:
                route = _residual_correlation_beam_topk(
                    shared,
                    routed,
                    target,
                    top_k,
                    simplex=False,
                    beam_width=beam_width,
                    pool_size=pool_size,
                )
            elapsed += time.perf_counter() - tic
            prediction = _route_reconstruction(shared, routed, route)
            error_sum += float(torch.sum((prediction - target).square()).item())
            target_norm_sum += float(torch.sum(target.square()).item())
            cosine_sum += float(
                torch.sum(
                    torch.sum(prediction * target, dim=1)
                    / (torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12)
                ).item()
            )
            route_cache.append((route["indices"].cpu().numpy(), route["weights"].cpu().numpy()))
            coefficient_values.append(route["weights"].cpu().numpy().reshape(-1))
            for slot in range(top_k):
                usage += np.bincount(route["indices"][:, slot].cpu().numpy(), minlength=expert_count)
            _accumulate_scales(scale_fit, routed, target, shared, route)
            del hidden, target, shared, routed, prediction
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
    scales = _fit_scales(scale_fit)
    scaled_error = 0.0
    scaled_norm = 0.0
    scaled_cosine = 0.0
    cursor = 0
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size]
            hidden, target = _dense_hidden_target(batch, weights, device_obj)
            shared, routed = _plan_contributions(hidden, weights["down_proj.weight"], plan)
            ids, coeff = route_cache[cursor]
            cursor += 1
            route = {
                "indices": torch.as_tensor(ids, dtype=torch.long, device=device_obj),
                "weights": torch.as_tensor(coeff, dtype=torch.float32, device=device_obj),
            }
            prediction = _route_reconstruction(shared, routed, route, scales=scales)
            scaled_error += float(torch.sum((prediction - target).square()).item())
            scaled_norm += float(torch.sum(target.square()).item())
            scaled_cosine += float(
                torch.sum(
                    torch.sum(prediction * target, dim=1)
                    / (torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1) + 1e-12)
                ).item()
            )
            del hidden, target, shared, routed, prediction
    tokens = max(int(inputs.shape[0]), 1)
    usage_fraction = usage / max(tokens * top_k, 1)
    coefficients = np.concatenate(coefficient_values) if coefficient_values else np.empty(0, dtype=np.float64)
    finite = bool(np.isfinite(coefficients).all() and np.isfinite(scales).all())
    result = {
        "profile": profile["name"],
        "top_k": int(top_k),
        "formulation": "independent_positive_oracle",
        "selection_method": "exact_all_combinations_active_faces" if exact else "residual_correlation_beam_search_exact_final_coefficients",
        "beam_width": None if exact else int(beam_width),
        "candidate_pool_size": None if exact else int(pool_size or min(expert_count, max(8, 2 * top_k + 2))),
        "tokens": tokens,
        "normalized_mse": error_sum / max(target_norm_sum, 1e-12),
        "cosine": cosine_sum / tokens,
        "learned_scale_normalized_mse": scaled_error / max(scaled_norm, 1e-12),
        "learned_scale_cosine": scaled_cosine / tokens,
        "learned_global_scales": [float(value) for value in scales],
        "expert_usage_fraction": [float(value) for value in usage_fraction],
        "expert_usage_cv": float(np.std(usage_fraction) / max(np.mean(usage_fraction), 1e-12)),
        "dead_experts": int(np.sum(usage == 0)),
        "coefficient_min": float(coefficients.min()) if coefficients.size else None,
        "coefficient_max": float(coefficients.max()) if coefficients.size else None,
        "coefficient_mean": float(coefficients.mean()) if coefficients.size else None,
        "coefficient_sum_mean": float(coefficients.reshape(-1, top_k).sum(axis=1).mean()) if coefficients.size else None,
        "finite_coefficients_and_scales": finite,
        "elapsed_seconds": float(elapsed),
    }
    if return_route_assignments:
        # Keep this opt-in because a full 128k screen can otherwise inflate a
        # receipt with one Python object per token.  The candidate runner uses
        # the assignments only long enough to compute the declared p32 route
        # stabilization statistic, then removes them before publishing.
        result["route_assignments"] = [
            [sorted(int(value) for value in ids_row) for ids_row in ids]
            for ids, _weights in route_cache
            for ids_row in ids
        ]
    result.update(_active_params(profile, top_k))
    return result


def _swap_plan(plan: PartitionPlan, first: int, second: int) -> PartitionPlan:
    groups = [list(group) for group in plan.expert_indices]
    locations: dict[int, tuple[int, int]] = {}
    for expert, group in enumerate(groups):
        for offset, index in enumerate(group):
            locations[int(index)] = (expert, offset)
    first_location = locations[int(first)]
    second_location = locations[int(second)]
    groups[first_location[0]][first_location[1]], groups[second_location[0]][second_location[1]] = (
        groups[second_location[0]][second_location[1]],
        groups[first_location[0]][first_location[1]],
    )
    candidate = PartitionPlan(
        plan.dense_intermediate_size,
        plan.routed_experts,
        plan.expert_intermediate_size,
        plan.shared_intermediate_size,
        plan.shared_indices,
        tuple(tuple(group) for group in groups),
    )
    candidate.validate()
    return candidate


def _refine_plan(
    plan: PartitionPlan,
    residual_scores: np.ndarray,
    inputs: np.ndarray,
    weights_cpu: dict[str, np.ndarray],
    profile: dict[str, Any],
    *,
    top_k: int,
    device: str,
    batch_size: int,
    max_candidates: int = 4,
) -> tuple[PartitionPlan, list[dict[str, Any]]]:
    """Try bounded expert-neuron swaps and keep the lowest exact oracle error."""

    shared = set(plan.shared_indices)
    ranked = [int(index) for index in np.argsort(-residual_scores, kind="stable") if int(index) not in shared]
    locations: dict[int, int] = {}
    for expert, group in enumerate(plan.expert_indices):
        for index in group:
            locations[int(index)] = expert
    candidates: list[tuple[int, int]] = []
    for elite in ranked[: max_candidates]:
        expert = locations[elite]
        victim_expert = (expert + 1) % plan.routed_experts
        victim = min(plan.expert_indices[victim_expert], key=lambda index: (float(residual_scores[index]), int(index)))
        pair = tuple(sorted((elite, int(victim))))
        if pair not in candidates:
            candidates.append(pair)
    history: list[dict[str, Any]] = []
    best_plan = plan
    best_score = float("inf")
    refine_batch = inputs[: min(inputs.shape[0], 4096)]
    for first, second in candidates:
        candidate = _swap_plan(plan, first, second)
        result = _evaluate_positive_oracle(
            profile,
            refine_batch,
            weights_cpu,
            candidate,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
            exact=False,
            beam_width=8,
            pool_size=min(int(profile["routed_experts"]), 20),
        )
        row = {"swap": [first, second], "refine_tokens": int(refine_batch.shape[0]), "normalized_mse": result["normalized_mse"], "cosine": result["cosine"]}
        history.append(row)
        if float(result["normalized_mse"]) < best_score:
            best_score = float(result["normalized_mse"])
            best_plan = candidate
    return best_plan, history


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run = Path(args.run_dir)
    train_manifest = run / "capture/layer-0000-train.json"
    selected, dev_meta = _select_dev_rows(train_manifest, args.dev_tokens, args.seed)
    _json(run / "capture/architecture-dev.json", dev_meta)
    inputs = _materialize_selected(train_manifest, selected)
    weights_cpu = _load_mlp(Path(args.source_dir), 0)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    profiles = {
        "p8": _profile("p8", 8, 2048, 1024),
        "p16": _profile("p16", 16, 1024, 1024),
        "p32": _profile("p32", 32, 512, 1024),
    }
    score = _compute_dev_scores(inputs, weights_cpu, device=device, batch_size=args.batch_size, shared_width=1024)
    down = weights_cpu["down_proj.weight"]
    plans_by_profile: dict[str, dict[str, PartitionPlan]] = {}
    for name, profile in profiles.items():
        plans_by_profile[name] = {
            "activation_magnitude": partition_indices(
                profile["dense_intermediate_size"],
                profile["routed_experts"],
                profile["expert_intermediate_size"],
                profile["shared_intermediate_size"],
                strategy="activation_magnitude",
                scores=score["activation_magnitude"],
            ),
            "contribution_magnitude": _ranked_plan(profile, score["contribution_magnitude"]),
            "residual_aware_greedy": _ranked_plan(profile, score["residual_aware_greedy"], shared_indices=score["shared_indices"]),
            "signature_grouping": _signature_plan(profile, score["contribution_magnitude"], down, args.seed),
        }
    # Keep the proof-of-method p16/top4 track while screening both high-
    # sparsity p32 product targets.  All three are evaluated on the same
    # FIT/dev rows before selector training or any holdout read.
    candidate_ks = {"p16": (3, 4), "p32": (4, 5, 6)}
    all_results: list[dict[str, Any]] = []
    for profile_name in ("p16", "p32"):
        profile = profiles[profile_name]
        for strategy, plan in plans_by_profile[profile_name].items():
            for top_k in candidate_ks[profile_name]:
                print(f"evaluating {profile_name}/{strategy}/top{top_k} on {inputs.shape[0]} TRAIN/dev rows", flush=True)
                row = _evaluate_positive_oracle(
                    profile,
                    inputs,
                    weights_cpu,
                    plan,
                    top_k=top_k,
                    device=device,
                    batch_size=args.batch_size,
                    beam_width=8,
                    pool_size=min(int(profile["routed_experts"]), 20),
                )
                row.update({"partition_strategy": strategy, "partition": plan.as_dict(), "split": "architecture_dev"})
                all_results.append(row)
    # Refine the proof target and p32/top5's topology-specific basis.  The
    # resulting p32 plan is evaluated for top4/top5/top6 so the quality/load
    # curve is not inferred from a partition tuned for a different k.
    refinements: dict[str, Any] = {}
    for profile_name, target_k in (("p16", 4), ("p32", 5)):
        profile = profiles[profile_name]
        candidates = [row for row in all_results if row["profile"] == profile_name and row["top_k"] == target_k]
        seed_row = min(candidates, key=lambda row: float(row["normalized_mse"]))
        seed_plan = PartitionPlan(
            int(profile["dense_intermediate_size"]),
            int(profile["routed_experts"]),
            int(profile["expert_intermediate_size"]),
            int(profile["shared_intermediate_size"]),
            tuple(seed_row["partition"]["shared_indices"]),
            tuple(tuple(group) for group in seed_row["partition"]["expert_indices"]),
        )
        print(f"refining {profile_name}/top{target_k} from {seed_row['partition_strategy']}", flush=True)
        refined_plan, history = _refine_plan(
            seed_plan,
            score["residual_aware_greedy"],
            inputs,
            weights_cpu,
            profile,
            top_k=target_k,
            device=device,
            batch_size=args.batch_size,
        )
        refinements[profile_name] = {
            "target_top_k": target_k,
            "seed_strategy": seed_row["partition_strategy"],
            "seed_normalized_mse": seed_row["normalized_mse"],
            "refinement_method": "bounded_residual_ranked_neuron_swaps_exact_positive_oracle",
            "refine_tokens": min(int(inputs.shape[0]), 4096),
            "history": history,
            "selected_plan": refined_plan.as_dict(),
        }
        plans_by_profile[profile_name]["residual_swap_refined"] = refined_plan
        for top_k in candidate_ks[profile_name]:
            row = _evaluate_positive_oracle(
                profile,
                inputs,
                weights_cpu,
                refined_plan,
                top_k=top_k,
                device=device,
                batch_size=args.batch_size,
                beam_width=8,
                pool_size=min(int(profile["routed_experts"]), 20),
            )
            row.update({"partition_strategy": "residual_swap_refined", "partition": refined_plan.as_dict(), "split": "architecture_dev"})
            all_results.append(row)
    # Reconfirm the p8 control on this same current-HEAD dev subset.  The full
    # p8 k=1..6 curve remains in the earlier report; this row is the decisive
    # top6 control receipt for the present provenance.
    p8 = profiles["p8"]
    p8_plan = plans_by_profile["p8"]["activation_magnitude"]
    p8_control = _evaluate_positive_oracle(
        p8,
        inputs,
        weights_cpu,
        p8_plan,
        top_k=6,
        device=device,
        batch_size=args.batch_size,
        exact=True,
    )
    p8_control.update({"partition_strategy": "activation_magnitude", "partition": p8_plan.as_dict(), "split": "architecture_dev", "classification": "TRAINABILITY_ROUTER_QUALITY_CONTROL"})
    all_results.append(p8_control)
    selected_rows: list[dict[str, Any]] = []
    selected_specs = (
        ("p16", 4, "PROOF_OF_METHOD_NEAR_TERM_PRODUCTION"),
        ("p16", 3, "AGGRESSIVE_EQUAL_COMPUTE_CANDIDATE"),
        ("p32", 5, "HIGH_SPARSITY_PRODUCT_TARGET"),
        ("p32", 4, "HIGH_SPARSITY_PRODUCT_TARGET"),
        ("p32", 6, "AGGRESSIVE_EQUAL_COMPUTE_CANDIDATE"),
    )
    for profile_name, top_k, classification in selected_specs:
        rows = [row for row in all_results if row["profile"] == profile_name and row["top_k"] == top_k]
        chosen = min(rows, key=lambda row: float(row["normalized_mse"]))
        selected_rows.append({
            "profile": profile_name,
            "top_k": top_k,
            "classification": classification,
            "partition_strategy": chosen["partition_strategy"],
            "normalized_mse": chosen["normalized_mse"],
            "cosine": chosen["cosine"],
            "learned_scale_normalized_mse": chosen["learned_scale_normalized_mse"],
            **_active_params(profiles[profile_name], top_k),
        })
        partition_payload = chosen["partition"] | {
            "schema_version": 3,
            "status": "PARTITION_READY_HIGH_SPARSITY_FINALIST_SEARCH",
            "layer": 0,
            "profile": profiles[profile_name],
            "profile_name": f"{profile_name}_top{top_k}",
            "top_k": top_k,
            "routing_mode": "independent_positive",
            "strategy": chosen["partition_strategy"],
            "initial_expert_scales": [max(float(value), 1e-3) for value in chosen["learned_global_scales"]],
            "oracle_learned_global_scales": chosen["learned_global_scales"],
            "architecture_dev_selection_hash": dev_meta["selected_row_key_hash"],
            "architecture_dev_tokens": int(inputs.shape[0]),
            "selection_contract": "TRAIN/dev only; full holdout not opened by this script",
            "source_manifest": str(train_manifest),
            "oracle_result": chosen,
            "code_commit": current_git_commit(),
        }
        _json(run / "partitions" / f"high-sparsity-{profile_name}-top{top_k}.json", partition_payload)
    payload = {
        "schema_version": 1,
        "status": "HIGH_SPARSITY_TRAIN_DEV_SEARCH_COMPLETE",
        "classification": "TRAIN_ONLY_PARTITION_AND_ORACLE_SELECTION",
        "run_dir": str(run),
        "source_dir": str(args.source_dir),
        "layer": 0,
        "dev_subset": dev_meta,
        "dev_tokens": int(inputs.shape[0]),
        "selection_policy": "deterministic TRAIN development subset; no holdout access",
        "product_priority_tracks": [
            {"profile": "p16", "top_k": 4, "active_intermediate_width": 5120, "ffn_reduction": 1.0 - 5120 / 17408, "classification": "PROOF_OF_METHOD_NEAR_TERM_PRODUCTION"},
            {"profile": "p32", "top_k": 5, "active_intermediate_width": 3584, "ffn_reduction": 1.0 - 3584 / 17408, "classification": "HIGH_SPARSITY_PRODUCT_TARGET"},
            {"profile": "p32", "top_k": 4, "active_intermediate_width": 3072, "ffn_reduction": 1.0 - 3072 / 17408, "classification": "HIGH_SPARSITY_PRODUCT_TARGET"},
        ],
        "load_aware_oracle": {
            "status": "PENDING_CONTRIBUTION_ARRAY_MATERIALIZATION",
            "command": "scripts/run_load_aware_oracle.py",
            "target_load_cv": 0.50,
            "holdout_opened": False,
        },
        "strategies": ["activation_magnitude", "contribution_magnitude", "residual_aware_greedy", "signature_grouping", "residual_swap_refined"],
        "beam_policy": {"p16": {"beam_width": 8, "candidate_pool_size": 16}, "p32": {"beam_width": 8, "candidate_pool_size": 20}},
        "scores": {name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "max": float(np.max(values))} for name, values in score.items() if isinstance(values, np.ndarray) and name != "shared_indices"},
        "results": all_results,
        "refinements": refinements,
        "p8_top6_control": p8_control,
        "selected_candidates": selected_rows,
        "code_commit": current_git_commit(),
    }
    _json(run / "reports/high-sparsity-partition-search.json", payload)
    _json(run / "reports/p8-top6-current-head-control.json", {"dev_subset": dev_meta, "result": p8_control, "code_commit": current_git_commit()})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dev-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({"status": result["status"], "selected_candidates": result["selected_candidates"], "p8_top6": result["p8_top6_control"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
