#!/usr/bin/env python3
"""Run the sealed FIT-TRAIN/FIT-DEV candidate-development loop.

The default invocation retains the inexpensive contract/readiness check used
by orchestration discovery. ``--execute`` performs the real layer-0 science
loop: four independent basis families, exact p16 oracle screening, bounded
p32 pool expansion, frozen-basis router/amplitude training for seeds
17/29/41, and strict checkpoint reload receipts. No promotion tier is read or
opened by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import iter_activation_shards
from dense2moe.config import load_config
from dense2moe.data import sha256_file, write_immutable_json
from dense2moe.partition import PartitionPlan, partition_indices
from dense2moe.provenance import current_git_commit

try:
    from scripts.run_high_sparsity_search import (
        _compute_dev_scores,
        _evaluate_positive_oracle,
        _profile,
        _refine_plan,
        _signature_plan,
    )
    from scripts.run_topk_architecture_search import _load_mlp
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_high_sparsity_search import (  # type: ignore
        _compute_dev_scores,
        _evaluate_positive_oracle,
        _profile,
        _refine_plan,
        _signature_plan,
    )
    from run_topk_architecture_search import _load_mlp  # type: ignore


METHOD_VERSION_DEFAULT = "moe-v22-m01"
THRESHOLD_FINGERPRINT_DEFAULT = "sealed-qwen38-promotion-v1"
SEEDS = (17, 29, 41)
P32_POOL_SIZES = (1024, 2048, 4096, 8192)


def _p32_expert_pool_size(*, routed_experts: int, top_k: int, candidate_budget: int) -> int:
    """Map a bounded combination budget to a real correlation expert pool.

    The oracle's ``pool_size`` parameter is a number of expert IDs, whereas
    the science protocol freezes p32 budgets as candidate route combinations.
    Mapping via ``C(pool, top_k)`` keeps the receipt's 1024/2048/4096/8192
    rounds meaningful instead of silently clipping every round to 32 experts.
    """

    if candidate_budget <= 0:
        raise ValueError("p32 candidate budget must be positive")
    for expert_pool in range(top_k, routed_experts + 1):
        if math.comb(expert_pool, top_k) >= candidate_budget:
            return expert_pool
    return routed_experts


def _record_count(path: Path) -> int:
    if path.suffix.casefold() == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("count", "rows", "records", "selected_rows", "shards"):
            value = payload.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, list):
                return len(value)
    return 0


def _manifest_identity(path: Path) -> str:
    """Hash the manifest and every referenced shard, not only JSON metadata."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = payload.get("train_manifest")
    if isinstance(reference, str) and reference not in {"", "pending"}:
        child = Path(reference)
        path = child if child.is_absolute() else path.parent / child
        payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    for shard in payload.get("shards", []):
        shard_path = Path(str(shard["path"]))
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        digest.update(str(shard_path).encode())
        digest.update(sha256_file(shard_path).encode())
    return digest.hexdigest()


def _resolve_train_manifest(path: Path) -> tuple[Path, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = payload.get("train_manifest")
    if not reference and isinstance(payload.get("splits"), dict):
        reference = payload["splits"].get("train")
    if isinstance(reference, str) and reference not in {"", "pending"}:
        candidate = Path(reference)
        path = candidate if candidate.is_absolute() else path.parent / candidate
        payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split")
    return path, str(split) if split in {"train", "holdout"} else None


def _materialize_manifest(path: Path, *, limit: int | None = None) -> np.ndarray:
    """Materialize only the bounded screen prefix from an activation manifest."""

    manifest, expected_split = _resolve_train_manifest(path)
    values: list[np.ndarray] = []
    seen = 0
    for shard in iter_activation_shards(manifest, expected_split=expected_split):
        array = np.asarray(shard, dtype=np.float32)
        if limit is not None:
            remaining = max(0, int(limit) - seen)
            if remaining <= 0:
                break
            array = array[:remaining]
        if array.size:
            values.append(array)
            seen += int(array.shape[0])
        if limit is not None and seen >= int(limit):
            break
    if not values:
        raise ValueError(f"activation manifest is empty: {path}")
    result = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    return result[: int(limit)] if limit is not None else result


def _plan_from_payload(payload: dict[str, Any]) -> PartitionPlan:
    value = payload.get("plan", payload)
    plan = PartitionPlan(
        int(value["dense_intermediate_size"]),
        int(value["routed_experts"]),
        int(value["expert_intermediate_size"]),
        int(value["shared_intermediate_size"]),
        tuple(int(item) for item in value["shared_indices"]),
        tuple(tuple(int(item) for item in group) for group in value["expert_indices"]),
    )
    plan.validate()
    return plan


def _flatten_routes(value: Any) -> list[tuple[int, ...]]:
    """Normalize batched route receipts to one top-k tuple per token."""

    result: list[tuple[int, ...]] = []

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)):
            if all(isinstance(value, (int, np.integer)) for value in item):
                result.append(tuple(int(value) for value in item))
                return
            for child in item:
                visit(child)

    visit(value)
    return result


def _route_jaccard(left: list[list[int]], right: list[list[int]]) -> float:
    left_rows = _flatten_routes(left)
    right_rows = _flatten_routes(right)
    if not left_rows or not right_rows or len(left_rows) != len(right_rows):
        return 0.0
    total = 0.0
    for first, second in zip(left_rows, right_rows):
        a, b = set(first), set(second)
        total += len(a & b) / max(1, len(a | b))
    return total / len(left_rows)


def _pareto_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    frontier: list[dict[str, Any]] = []
    for candidate in values:
        dominated = False
        for other in values:
            if other is candidate:
                continue
            other_load = float(other.get("expert_usage_cv", other.get("load_cv", 999.0)))
            candidate_load = float(candidate.get("expert_usage_cv", candidate.get("load_cv", 999.0)))
            no_worse = float(other["cosine"]) >= float(candidate["cosine"]) and float(other["normalized_mse"]) <= float(candidate["normalized_mse"]) and other_load <= candidate_load
            strictly = float(other["cosine"]) > float(candidate["cosine"]) or float(other["normalized_mse"]) < float(candidate["normalized_mse"]) or other_load < candidate_load
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (-float(row["cosine"]), float(row["normalized_mse"]), float(row.get("expert_usage_cv", row.get("load_cv", 999.0))), str(row["partition_strategy"])))


def _plan_hash(plan: PartitionPlan) -> str:
    return hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_partition(path: Path, *, profile: dict[str, Any], topology: str, plan: PartitionPlan, oracle: dict[str, Any], train_hash: str, dev_hash: str, method_version: str) -> None:
    payload = {
        "schema_version": 4,
        "status": "FROZEN_DEVELOPMENT_BASIS",
        "method_version": method_version,
        "topology": topology,
        "profile": profile,
        "profile_name": f"{profile['name']}_top{profile['top_k']}",
        "top_k": int(profile["top_k"]),
        "routing_mode": "independent_positive",
        "strategy": oracle["partition_strategy"],
        "plan": plan.as_dict(),
        "initial_expert_scales": [max(float(value), 1e-3) for value in oracle.get("learned_global_scales", [])],
        "oracle_result": {key: value for key, value in oracle.items() if key != "route_assignments"},
        "fit_train_dataset_hash": train_hash,
        "fit_dev_dataset_hash": dev_hash,
        "basis_frozen": True,
        "external_data_used": False,
        "code_commit": current_git_commit(),
    }
    write_immutable_json(path, payload)


def _strict_reload_checkpoint(*, source_dir: Path, profile: Any, partition_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Reload a trained layer into the exact architecture and inspect keys."""

    import torch
    from safetensors.torch import load_file

    from dense2moe.models import TorchQwen35SwiGLUMoE

    metadata_path = checkpoint_dir / "layer-0000.json"
    tensor_path = checkpoint_dir / "layer-0000.safetensors"
    if not metadata_path.exists() or not tensor_path.exists():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint_dir}")
    raw_weights = _load_mlp(source_dir, 0)
    plan = _plan_from_payload(json.loads(partition_path.read_text(encoding="utf-8")))
    model = TorchQwen35SwiGLUMoE.from_dense(raw_weights["gate_proj.weight"], raw_weights["up_proj.weight"], raw_weights["down_proj.weight"], routed_experts=int(profile.routed_experts), shared_intermediate_size=int(profile.shared_intermediate_size), top_k=int(profile.top_k), routing_mode=str(profile.routing_mode), partition=plan, learnable_scales=True)
    raw_state = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix):]: value for key, value in raw_state.items() if key.startswith(prefix)}
    if len(state) != len(raw_state):
        raise ValueError("checkpoint tensor namespace mismatch")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    return {"status": "RELOAD_GREEN", "metadata": str(metadata_path), "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(), "tensor_count": len(raw_state), "strict": True, "fresh_model": type(model).__name__, "torch_version": str(torch.__version__)}


def _train_finalist_seeds(*, run_dir: Path, source_dir: Path, train_manifest: Path, dev_manifest: Path, profile_dict: dict[str, Any], partition_path: Path, method_version: str, seeds: tuple[int, ...], epochs: int, microbatch: int, learning_rate: float, device: str) -> list[dict[str, Any]]:
    from dense2moe.training import train_torch_layer

    profile_name = "qwen38_p16s1_top4" if profile_dict["name"] == "p16" else "qwen38_p32s1_top5"
    profile = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{profile_name}.yaml")
    train_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    train_dataset_hash = str(train_payload.get("dataset_hash", ""))
    results: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_dir = run_dir / "development" / "checkpoints" / profile_dict["name"] / partition_path.stem / f"seed-{seed:02d}"
        metadata_path = checkpoint_dir / "layer-0000.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            plan = _plan_from_payload(json.loads(partition_path.read_text(encoding="utf-8")))
            expected_partition_hash = hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode()).hexdigest()
            if metadata.get("dataset_hash") == train_dataset_hash and int(metadata.get("training_seed", -1)) == seed and metadata.get("partition_hash") == expected_partition_hash:
                reload_receipt = _strict_reload_checkpoint(source_dir=source_dir, profile=profile, partition_path=partition_path, checkpoint_dir=checkpoint_dir)
                results.append({"seed": seed, "status": "REUSED", "checkpoint_dir": str(checkpoint_dir), "reload": reload_receipt, "metrics": metadata.get("holdout_metrics")})
                continue
        trained = train_torch_layer(source_dir=source_dir, activation_manifest=train_manifest, selection_manifest=dev_manifest, selection_split="FIT-DEV", output_dir=checkpoint_dir, layer=0, profile=profile, partition_path=partition_path, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate, device=device, seed=seed, source_revision=profile.revision, stage_schedule=[{"name": "frozen_basis_selection_amplitude", "epochs": epochs, "train_scales": True, "train_experts": False, "train_shared": False, "use_oracle_targets": True, "train_selection_router": True, "train_amplitude_router": True}], evaluate_holdout=False)
        reload_receipt = _strict_reload_checkpoint(source_dir=source_dir, profile=profile, partition_path=partition_path, checkpoint_dir=checkpoint_dir)
        results.append({"seed": seed, "status": trained["status"], "checkpoint_dir": str(checkpoint_dir), "reload": reload_receipt, "metrics": trained.get("holdout_metrics"), "validation_b_metrics": trained.get("validation_b_metrics"), "method_version": method_version})
    return results


def _execute_candidate_search(*, run_dir: Path, activation_manifest: Path, dev_manifest: Path, topology: str, source_dir: Path, device: str, screen_tokens: int, batch_size: int, epochs: int, microbatch: int, learning_rate: float, method_version: str, seed: int) -> dict[str, Any]:
    import torch

    topology_profile = {"p16/top4": ("p16", 16, 1024, 1024, 4), "p32/top5": ("p32", 32, 512, 1024, 5)}
    name, experts, expert_width, shared_width, top_k = topology_profile[topology]
    profile = _profile(name, experts, expert_width, shared_width)
    profile["top_k"] = top_k
    train_hash = _manifest_identity(activation_manifest)
    dev_hash = _manifest_identity(dev_manifest)
    source_hash_digest = hashlib.sha256()
    for source_file in (source_dir / "model.safetensors.index.json", source_dir / "config.json"):
        if source_file.exists():
            source_hash_digest.update(source_file.read_bytes())
    source_hash = source_hash_digest.hexdigest()
    identity = hashlib.sha256(json.dumps({"method_version": method_version, "topology": topology, "train": train_hash, "dev": dev_hash, "source": source_hash, "screen_tokens": screen_tokens, "batch_size": batch_size, "epochs": epochs, "microbatch": microbatch, "learning_rate": learning_rate, "seed": seed}, sort_keys=True).encode()).hexdigest()
    stem = "p16-exhaustive-receipt" if topology == "p16/top4" else "p32-bounded-pool-receipt"
    canonical = run_dir / "development" / f"{stem}.json"
    if canonical.exists():
        existing = json.loads(canonical.read_text(encoding="utf-8"))
        if existing.get("science_identity") == identity:
            return existing if existing.get("status") == "BLOCKED" else {**existing, "status": "REUSED"}
        canonical = run_dir / "development" / f"{stem}-{identity[:16]}.json"
    actual_device = device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu"
    train_inputs = _materialize_manifest(activation_manifest, limit=screen_tokens)
    dev_inputs = _materialize_manifest(dev_manifest, limit=screen_tokens)
    weights_cpu = _load_mlp(source_dir, 0)
    scores = _compute_dev_scores(train_inputs, weights_cpu, device=actual_device, batch_size=batch_size, shared_width=shared_width)
    down = weights_cpu["down_proj.weight"]
    plans: dict[str, PartitionPlan] = {
        "activation_magnitude": partition_indices(profile["dense_intermediate_size"], experts, expert_width, shared_width, strategy="activation_magnitude", scores=scores["activation_magnitude"]),
        "output_contribution": partition_indices(profile["dense_intermediate_size"], experts, expert_width, shared_width, strategy="output_contribution", scores=scores["contribution_magnitude"]),
        "balanced_signature": _signature_plan(profile, scores["contribution_magnitude"], down, seed),
    }
    initial_rows: list[dict[str, Any]] = []
    pool_rounds: list[dict[str, Any]] = []
    pool_values = (1024, 2048, 4096, 8192) if topology == "p32/top5" else (None,)
    previous_routes: dict[str, list[list[int]]] = {}
    previous_finalists: tuple[str, ...] | None = None
    stable_rounds = 0
    for requested_pool in pool_values:
        round_rows: list[dict[str, Any]] = []
        effective_expert_pool = (
            _p32_expert_pool_size(routed_experts=experts, top_k=top_k, candidate_budget=int(requested_pool))
            if topology == "p32/top5"
            else None
        )
        for strategy, plan in list(plans.items()):
            result = _evaluate_positive_oracle(profile, dev_inputs, weights_cpu, plan, top_k=top_k, device=actual_device, batch_size=batch_size, exact=topology == "p16/top4", beam_width=8, pool_size=effective_expert_pool if effective_expert_pool is not None else requested_pool, return_route_assignments=topology == "p32/top5")
            routes = result.pop("route_assignments", [])
            result.update({"partition_strategy": strategy, "partition": plan.as_dict(), "requested_pool_size": requested_pool, "effective_pool_size": result.get("candidate_pool_size"), "effective_candidate_combinations": math.comb(int(result.get("candidate_pool_size", 0)), top_k) if topology == "p32/top5" else None, "split": "FIT-DEV"})
            result["route_fingerprint"] = hashlib.sha256(json.dumps(routes, separators=(",", ":")).encode()).hexdigest() if routes else None
            result["_routes"] = routes
            round_rows.append(result)
        if topology == "p32/top5":
            frontier = _pareto_rows(round_rows)[:2]
            finalists = tuple(sorted(str(row["partition_strategy"]) for row in frontier))
            jaccards = []
            for row in frontier:
                strategy = str(row["partition_strategy"])
                if strategy in previous_routes:
                    jaccards.append(_route_jaccard(previous_routes[strategy], row["_routes"]))
                previous_routes[strategy] = row["_routes"]
            movement = 0.0
            if previous_finalists is not None:
                current_by_strategy = {str(row["partition_strategy"]): row for row in round_rows}
                prior_by_strategy = {str(row["partition_strategy"]): row for row in pool_rounds[-1]["rows"]}
                for strategy in set(previous_finalists) & set(finalists):
                    old = float(prior_by_strategy[strategy]["normalized_mse"])
                    new = float(current_by_strategy[strategy]["normalized_mse"])
                    movement = max(movement, abs(new - old) / max(abs(old), 1e-12))
            stable = previous_finalists == finalists and (not jaccards or min(jaccards) >= 0.98) and movement <= 0.01
            stable_rounds = stable_rounds + 1 if stable else 0
            pool_rounds.append({"requested_pool_size": requested_pool, "effective_expert_pool_size": effective_expert_pool, "rows": [{key: value for key, value in row.items() if key != "_routes"} for row in round_rows], "pareto_finalists": list(finalists), "route_jaccard": min(jaccards) if jaccards else None, "metric_movement": movement, "stable": stable, "stable_rounds": stable_rounds})
            previous_finalists = finalists
            if stable_rounds >= 2:
                break
        else:
            initial_rows.extend(round_rows)
    if topology == "p32/top5" and stable_rounds < 2:
        receipt = {"status": "BLOCKED", "blocker_code": "P32_POOL_UNSTABLE", "topology": topology, "pool_rounds": pool_rounds, "science_identity": identity, "opened_evaluation_tiers": []}
        write_immutable_json(canonical, receipt)
        return receipt
    if topology == "p32/top5":
        final_rows = []
        for strategy, plan in plans.items():
            result = _evaluate_positive_oracle(profile, dev_inputs, weights_cpu, plan, top_k=top_k, device=actual_device, batch_size=batch_size, exact=False, beam_width=8, pool_size=pool_rounds[-1]["effective_expert_pool_size"])
            result.update({"partition_strategy": strategy, "partition": plan.as_dict(), "requested_pool_size": pool_rounds[-1]["requested_pool_size"], "split": "FIT-DEV"})
            final_rows.append(result)
        initial_rows = final_rows
    best_seed = min(initial_rows, key=lambda row: (float(row["normalized_mse"]), -float(row["cosine"])))
    refined, refine_history = _refine_plan(plans[best_seed["partition_strategy"]], scores["residual_aware_greedy"], train_inputs, weights_cpu, profile, top_k=top_k, device=actual_device, batch_size=batch_size)
    plans["residual_swap_refined"] = refined
    refined_pool = _p32_expert_pool_size(routed_experts=experts, top_k=top_k, candidate_budget=P32_POOL_SIZES[-1]) if topology == "p32/top5" else None
    refined_result = _evaluate_positive_oracle(profile, dev_inputs, weights_cpu, refined, top_k=top_k, device=actual_device, batch_size=batch_size, exact=topology == "p16/top4", pool_size=refined_pool)
    refined_result.update({"partition_strategy": "residual_swap_refined", "partition": refined.as_dict(), "split": "FIT-DEV", "refinement_history": refine_history})
    rows = initial_rows + [refined_result]
    finalists = _pareto_rows(rows)[:2]
    finalist_entries: list[dict[str, Any]] = []
    profile_name = "qwen38_p16s1_top4" if topology == "p16/top4" else "qwen38_p32s1_top5"
    profile_obj = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{profile_name}.yaml")
    for rank, row in enumerate(finalists, start=1):
        plan = _plan_from_payload(row["partition"])
        partition_path = run_dir / "development" / "partitions" / f"{name}-top{top_k}-{identity[:12]}-pareto-{rank}.json"
        _write_partition(partition_path, profile=profile, topology=topology, plan=plan, oracle=row, train_hash=train_hash, dev_hash=dev_hash, method_version=method_version)
        seed_results = _train_finalist_seeds(run_dir=run_dir, source_dir=source_dir, train_manifest=activation_manifest, dev_manifest=dev_manifest, profile_dict=profile, partition_path=partition_path, method_version=method_version, seeds=SEEDS, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate, device=actual_device)
        checkpoint_sha = hashlib.sha256(json.dumps(seed_results, sort_keys=True, default=str).encode()).hexdigest()
        finalist_entries.append({"status": "DEV_FINALIST", "rank": rank, "profile": profile_name, "topology": topology, "partition_strategy": row["partition_strategy"], "partition_path": str(partition_path), "partition_sha256": sha256_file(partition_path), "checkpoint_sha256": checkpoint_sha, "dataset_hash": train_hash, "dev_dataset_hash": dev_hash, "oracle_metrics": {key: value for key, value in row.items() if key not in {"partition", "_routes"}}, "seeds": list(SEEDS), "checkpoints": seed_results, "basis_frozen": True, "external_data_used": False})
    receipt = {"schema_version": 3, "receipt_type": "dense2moe-development-candidate-search", "status": "DEV_FINALISTS", "topology": topology, "method_version": method_version, "science_identity": identity, "partition": "FIT-TRAIN", "ranking_partition": "FIT-DEV", "activation_manifest": {"path": str(activation_manifest), "sha256": train_hash, "records": _record_count(activation_manifest)}, "dev_manifest": {"path": str(dev_manifest), "sha256": dev_hash, "records": _record_count(dev_manifest)}, "search_class": "EXHAUSTIVE_C(16,4)" if topology == "p16/top4" else "BOUNDED_CORRELATION_POOL_WITH_STABILIZATION", "exhaustive": topology == "p16/top4", "expected_combinations": 1820 if topology == "p16/top4" else None, "evaluated_combinations": 1820 if topology == "p16/top4" else None, "candidate_pool_sizes": list(P32_POOL_SIZES) if topology == "p32/top5" else None, "pool_rounds": pool_rounds if topology == "p32/top5" else [], "partition_families": ["activation_magnitude", "output_contribution", "balanced_signature", "residual_swap_refined"], "profiles": {profile_name: {"status": "DEV_FINALIST", "profile": profile_name, "topology": topology, "finalists": finalist_entries, "checkpoint_sha256": hashlib.sha256(json.dumps(finalist_entries, sort_keys=True, default=str).encode()).hexdigest(), "dataset_hash": train_hash}}, "finalists": finalist_entries, "selector_seeds": list(SEEDS), "amplitude_router_required": True, "opened_evaluation_tiers": [], "promotion_status": "DEV_FINALIST", "threshold_fingerprint": THRESHOLD_FINGERPRINT_DEFAULT, "source_dir": str(source_dir), "source_revision": profile_obj.revision, "code_commit": current_git_commit()}
    write_immutable_json(canonical, receipt)
    return receipt


def run_candidate_search(*, run_dir: Path, activation_manifest: Path, dev_manifest: Path, topology: str, exhaustive: bool = False, bounded: bool = False, expected_combinations: int | None = None, candidate_pool_size: int | None = None, execute: bool = False, source_dir: Path | None = None, device: str = "cpu", screen_tokens: int = 4096, batch_size: int = 256, epochs: int = 1, microbatch: int = 8, learning_rate: float = 1e-3, method_version: str = METHOD_VERSION_DEFAULT, seed: int = 17) -> dict[str, Any]:
    if topology not in {"p16/top4", "p32/top5"}:
        raise ValueError("only p16/top4 and p32/top5 are active")
    if not activation_manifest.exists() or not dev_manifest.exists():
        return {"status": "BLOCKED", "blocker_code": "DEVELOPMENT_ACTIVATIONS_REQUIRED", "topology": topology, "message": "FIT-TRAIN and FIT-DEV activation manifests are required; evaluation data is not a fallback"}
    if topology == "p16/top4":
        total = sum(1 for _ in itertools.combinations(range(16), 4))
        if not exhaustive or expected_combinations != total:
            raise ValueError(f"p16 requires explicit exhaustive C(16,4)={total} search")
    elif not bounded or not candidate_pool_size or candidate_pool_size <= 0:
        raise ValueError("p32 requires an explicitly bounded positive candidate pool")
    if not execute:
        pool_size = total if topology == "p16/top4" else int(candidate_pool_size)
        receipt = {"schema_version": 2, "receipt_type": "dense2moe-development-candidate-search", "status": "CANDIDATE_SEARCH_READY", "topology": topology, "partition": "FIT-TRAIN", "ranking_partition": "FIT-DEV", "activation_manifest": {"path": str(activation_manifest), "sha256": sha256_file(activation_manifest), "records": _record_count(activation_manifest)}, "dev_manifest": {"path": str(dev_manifest), "sha256": sha256_file(dev_manifest), "records": _record_count(dev_manifest)}, "search_class": "EXHAUSTIVE_C(16,4)" if topology == "p16/top4" else "BOUNDED_CORRELATION_POOL_NOT_EXHAUSTIVE", "exhaustive": topology == "p16/top4", "candidate_pool_size": pool_size, "basis_freeze_required": True, "selector_seeds_required": list(SEEDS), "amplitude_router_required": True, "opened_evaluation_tiers": [], "promotion_status": "DEV_FINALIST_NOT_YET_EVALUATED", "code_commit": current_git_commit()}
        output = run_dir / "development" / ("p16-exhaustive-receipt.json" if topology == "p16/top4" else "p32-bounded-pool-receipt.json")
        write_immutable_json(output, receipt)
        return receipt
    if source_dir is None:
        raise ValueError("--source-dir is required with --execute")
    return _execute_candidate_search(run_dir=run_dir, activation_manifest=activation_manifest, dev_manifest=dev_manifest, topology=topology, source_dir=source_dir, device=device, screen_tokens=screen_tokens, batch_size=batch_size, epochs=epochs, microbatch=microbatch, learning_rate=learning_rate, method_version=method_version, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), required=True)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--expected-combinations", type=int)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--screen-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--method-version", default=METHOD_VERSION_DEFAULT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_candidate_search(run_dir=args.run_dir, activation_manifest=args.activation_manifest, dev_manifest=args.dev_manifest, topology=args.topology, exhaustive=args.exhaustive, bounded=args.bounded, expected_combinations=args.expected_combinations, candidate_pool_size=args.candidate_pool_size, execute=args.execute, source_dir=args.source_dir, device=args.device, screen_tokens=args.screen_tokens, batch_size=args.batch_size, epochs=args.epochs, microbatch=args.microbatch, learning_rate=args.learning_rate, method_version=args.method_version, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] not in {"BLOCKED", "VALIDATION_FAILED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
