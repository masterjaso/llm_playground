#!/usr/bin/env python3
"""Open sealed promotion tiers and evaluate frozen finalists without fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    new_contamination_ledger,
    open_evaluation_tier,
    retire_evaluation_tier,
    validate_contamination_ledger,
    write_immutable_json,
)
from dense2moe.evaluation import evaluate_promotion_metrics
from dense2moe.provenance import current_git_commit


def _load_finalist_lock(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Load the development lock, with a compatibility-only old location."""

    candidates = (run_dir / "development" / "finalist-lock.json", run_dir / "promotion" / "finalist-lock.json")
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return path, payload
    return None


def _manifest_for_tier(root: Path, tier: str) -> Path | None:
    candidates = (root / f"{tier}.json", root / tier / "layer-0000.json", root / tier / "layer-0000-train.json", root / f"layer-0000-{tier}.json")
    return next((path for path in candidates if path.exists()), None)


def _direct_metrics(*, lock: dict[str, Any], source_dir: Path, manifest: Path, profile_name: str | None, max_tokens: int = 4096) -> dict[str, Any]:
    """Compute promotion metrics from frozen checkpoints, never optimizer data."""

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from dense2moe.capture import iter_activation_shards
    from dense2moe.config import load_config
    from dense2moe.models import TorchQwen35SwiGLUMoE
    from dense2moe.training.torch_distill import _dense_target_torch

    profile_entries = dict(lock.get("finalists", lock.get("profiles", {})))
    if profile_name is not None:
        profile_entries = {profile_name: profile_entries[profile_name]} if profile_name in profile_entries else {}
    if not profile_entries:
        raise ValueError("method lock contains no frozen finalist profiles")
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    split = raw_manifest.get("split") if raw_manifest.get("split") in {"train", "holdout"} else None
    values: list[np.ndarray] = []
    count = 0
    for shard in iter_activation_shards(manifest, expected_split=split):
        remaining = max_tokens - count
        if remaining <= 0:
            break
        values.append(np.asarray(shard[:remaining], dtype=np.float32))
        count += min(int(shard.shape[0]), remaining)
    if not values:
        raise ValueError(f"promotion activation manifest is empty: {manifest}")
    inputs = np.concatenate(values, axis=0)[:max_tokens]
    from scripts.run_topk_architecture_search import _load_mlp

    source_weights = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in _load_mlp(source_dir, 0).items()}
    metric_rows: list[dict[str, Any]] = []
    for name, entry in profile_entries.items():
        if not isinstance(entry, dict):
            continue
        candidates = entry.get("finalists") if isinstance(entry.get("finalists"), list) else [entry]
        for candidate in candidates:
            partition_path = Path(str(candidate.get("partition_path", "")))
            if not partition_path.exists():
                raise FileNotFoundError(f"frozen finalist partition is missing: {partition_path}")
            partition_payload = json.loads(partition_path.read_text(encoding="utf-8"))
            plan_payload = partition_payload.get("plan", partition_payload)
            from dense2moe.partition import PartitionPlan

            plan = PartitionPlan(int(plan_payload["dense_intermediate_size"]), int(plan_payload["routed_experts"]), int(plan_payload["expert_intermediate_size"]), int(plan_payload["shared_intermediate_size"]), tuple(plan_payload["shared_indices"]), tuple(tuple(group) for group in plan_payload["expert_indices"]))
            plan.validate()
            config_path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
            profile = load_config(config_path)
            seed_rows: list[dict[str, Any]] = []
            checkpoints = candidate.get("checkpoints", [])
            if not checkpoints:
                raise ValueError(f"finalist {name} has no frozen checkpoints")
            for checkpoint in checkpoints:
                checkpoint_dir = Path(str(checkpoint.get("checkpoint_dir", "")))
                tensor_path = checkpoint_dir / "layer-0000.safetensors"
                if not tensor_path.exists():
                    raise FileNotFoundError(f"frozen finalist checkpoint is missing: {tensor_path}")
                model = TorchQwen35SwiGLUMoE.from_dense(source_weights["gate_proj.weight"], source_weights["up_proj.weight"], source_weights["down_proj.weight"], routed_experts=profile.routed_experts, shared_intermediate_size=profile.shared_intermediate_size, top_k=profile.top_k, routing_mode=profile.routing_mode, partition=plan, learnable_scales=True)
                raw_state = load_file(str(tensor_path), device="cpu")
                state = {key[len("model.layers.0."):]: value for key, value in raw_state.items() if key.startswith("model.layers.0.")}
                missing, unexpected = model.load_state_dict(state, strict=True)
                if missing or unexpected:
                    raise ValueError(f"strict promotion checkpoint reload failed: {missing}, {unexpected}")
                model.eval()
                gate, up, down = (source_weights["gate_proj.weight"], source_weights["up_proj.weight"], source_weights["down_proj.weight"])
                errors = []
                cosines = []
                ratios = []
                counts = np.zeros(profile.routed_experts, dtype=np.int64)
                with torch.inference_mode():
                    for start in range(0, inputs.shape[0], 32):
                        x = torch.as_tensor(inputs[start:start + 32], dtype=torch.float32)
                        target = _dense_target_torch(x, gate, up, down)
                        prediction, info = model(x, return_router=True)
                        errors.append(float(torch.sum((prediction - target).square()).item()))
                        cosines.append(float(torch.sum(torch.sum(prediction * target, dim=-1) / (torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12)).item()))
                        ratios.extend((torch.linalg.vector_norm(prediction, dim=-1) / (torch.linalg.vector_norm(target, dim=-1) + 1e-12)).cpu().numpy().tolist())
                        counts += np.bincount(info["indices"].cpu().numpy().reshape(-1), minlength=profile.routed_experts)
                target_norm = float(np.sum(np.square(np.concatenate([np.asarray(_dense_target_torch(torch.as_tensor(inputs[start:start + 32], dtype=torch.float32), gate, up, down)).numpy() for start in range(0, inputs.shape[0], 32)]))))
                nmse = float(sum(errors) / max(target_norm, 1e-12))
                cosine = float(sum(cosines) / inputs.shape[0])
                ratio_array = np.asarray(ratios, dtype=np.float64)
                from scripts.run_high_sparsity_search import _evaluate_positive_oracle

                oracle_profile = {"name": "p16" if profile.routed_experts == 16 else "p32", "routed_experts": profile.routed_experts, "expert_intermediate_size": profile.expert_intermediate_size, "shared_intermediate_size": profile.shared_intermediate_size, "dense_intermediate_size": profile.dense_intermediate_size, "hidden_size": profile.hidden_size}
                oracle = _evaluate_positive_oracle(oracle_profile, inputs, {key: value.numpy() for key, value in source_weights.items()}, plan, top_k=profile.top_k, device="cpu", batch_size=32, exact=profile.routed_experts == 16)
                seed_rows.append({"nmse": nmse, "cosine": cosine, "loadcv": float(counts.std() / max(counts.mean(), 1e-12)), "dead_experts": int(np.sum(counts == 0)), "oracle_regret": max(0.0, nmse - float(oracle["normalized_mse"])), "repeat_variation": 0.0, "median_norm_ratio_error": abs(float(np.median(ratio_array)) - 1.0), "p95_relative_norm_error": float(np.quantile(np.abs(ratio_array - 1.0), 0.95))})
            mean_nmse = float(np.mean([row["nmse"] for row in seed_rows]))
            repeat_variation = max(abs(float(row["nmse"]) - mean_nmse) for row in seed_rows) / max(mean_nmse, 1e-12)
            for row in seed_rows:
                row["repeat_variation"] = repeat_variation
            aggregate = {key: (max(row[key] for row in seed_rows) if key in {"nmse", "loadcv", "dead_experts", "oracle_regret", "repeat_variation", "median_norm_ratio_error", "p95_relative_norm_error"} else min(row[key] for row in seed_rows)) for key in seed_rows[0]}
            metric_rows.append(aggregate)
    if not metric_rows:
        raise ValueError("no direct finalist metrics were produced")
    return {key: max(float(row[key]) for row in metric_rows) if key != "cosine" else min(float(row[key]) for row in metric_rows) for key in metric_rows[0]}


def run_promotion(*, run_dir: Path, method_version: str, tiers: list[str], execute: bool = False, source_dir: Path | None = None, activation_root: Path | None = None, profile: str | None = None, max_tokens: int = 4096) -> dict[str, Any]:
    lock_result = _load_finalist_lock(run_dir)
    if lock_result is None:
        return {"status": "BLOCKED", "blocker_code": "FINALIST_LOCK_REQUIRED", "message": "frozen development finalists and thresholds must be locked before opening promotion tiers"}
    finalist_lock_path, finalist_lock = lock_result
    if finalist_lock.get("status") not in {"METHOD_LOCKED", "METHOD_LOCKS_FROZEN", "FROZEN", "GREEN"}:
        return {"status": "BLOCKED", "blocker_code": "FINALIST_LOCK_NOT_FROZEN", "path": str(finalist_lock_path)}
    if str(finalist_lock.get("method_version", method_version)) != method_version:
        return {"status": "BLOCKED", "blocker_code": "METHOD_VERSION_MISMATCH", "path": str(finalist_lock_path)}
    if finalist_lock.get("external_tuning_forbidden") is not True:
        return {"status": "BLOCKED", "blocker_code": "EXTERNAL_TUNING_NOT_FORBIDDEN", "path": str(finalist_lock_path)}
    order = ("GATE-A", "SHADOW-B", "SHADOW-C", "G1", "G2")
    if len(set(tiers)) != len(tiers) or any(tier not in order for tier in tiers):
        return {"status": "BLOCKED", "blocker_code": "INVALID_PROMOTION_TIER_ORDER", "tiers": tiers}
    positions = [order.index(tier) for tier in tiers]
    if positions != sorted(positions):
        return {"status": "BLOCKED", "blocker_code": "INVALID_PROMOTION_TIER_ORDER", "tiers": tiers}
    external_positions = [position for position in positions if position >= order.index("G1")]
    if external_positions:
        first_external = min(external_positions)
        required_internal = set(order[:first_external])
        existing_open = set()
        existing_ledger_path = run_dir / "promotion" / "contamination-ledger.json"
        if existing_ledger_path.exists():
            existing_payload = json.loads(existing_ledger_path.read_text(encoding="utf-8"))
            existing_open = {name for name, value in dict(existing_payload.get("tiers", {})).items() if value.get("status") == "OPENED"}
        if not required_internal.issubset(set(tiers) | existing_open):
            return {"status": "BLOCKED", "blocker_code": "EXTERNAL_BEFORE_INTERNAL_GREEN", "tiers": tiers}
    ledger_path = run_dir / "promotion" / "contamination-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = new_contamination_ledger(
            method_version=method_version,
            code_commit=current_git_commit(),
            thresholds_fingerprint="sealed-qwen38-promotion-v1",
            runtime_lock_sha256="",
            corpus_hashes={tier: "" for tier in tiers},
        )
    results: dict[str, Any] = {}
    halted = False
    halt_reason: str | None = None
    for index, tier in enumerate(tiers):
        if halted:
            results[tier] = {"status": "SEALED", "tier": tier, "message": "later tier remained sealed after an earlier promotion failure"}
            continue
        metrics_path = run_dir / "promotion" / f"{tier}.json"
        if execute and not metrics_path.exists():
            if source_dir is None or activation_root is None:
                results[tier] = {"status": "BLOCKED", "blocker_code": "DIRECT_EVALUATION_INPUTS_REQUIRED", "message": "--source-dir and --activation-root are required with --execute"}
                halted = True
                halt_reason = f"direct evaluation inputs missing for {tier}"
                continue
            manifest = _manifest_for_tier(activation_root, tier)
            if manifest is None:
                results[tier] = {"status": "BLOCKED", "blocker_code": "PROMOTION_ACTIVATIONS_REQUIRED", "tier": tier}
                halted = True
                halt_reason = f"promotion activations missing for {tier}"
                continue
            try:
                direct = _direct_metrics(lock=finalist_lock, source_dir=source_dir, manifest=manifest, profile_name=profile, max_tokens=max_tokens)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                write_immutable_json(metrics_path, {"schema_version": 1, "receipt_type": "dense2moe-direct-promotion-metrics", "tier": tier, "dataset_hash": str(json.loads(manifest.read_text(encoding="utf-8")).get("dataset_hash", "")), "metrics": direct, "source_manifest": str(manifest), "external_data_untouched": True})
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                results[tier] = {"status": "BLOCKED", "blocker_code": "DIRECT_EVALUATION_FAILED", "tier": tier, "message": str(exc)}
                halted = True
                halt_reason = f"direct evaluation failed for {tier}"
                continue
        if not metrics_path.exists():
            results[tier] = {"status": "BLOCKED", "blocker_code": "FROZEN_METRICS_REQUIRED", "message": f"no frozen metric receipt for {tier}; no tier opening occurred"}
            halted = True
            halt_reason = f"missing frozen metrics for {tier}"
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_hash = str(payload.get("dataset_hash", ""))
        metrics = payload.get("metrics")
        if not dataset_hash or not isinstance(metrics, dict):
            results[tier] = {"status": "BLOCKED", "blocker_code": "INVALID_METRIC_RECEIPT", "message": f"{metrics_path} must contain dataset_hash and metrics"}
            halted = True
            halt_reason = f"invalid frozen metrics for {tier}"
            continue
        try:
            ledger = open_evaluation_tier(
                ledger,
                tier=tier,
                dataset_hash=dataset_hash,
                method_version=method_version,
                code_commit=current_git_commit(),
                thresholds_fingerprint="sealed-qwen38-promotion-v1",
            )
            gate = evaluate_promotion_metrics(metrics, domain_slices=payload.get("domain_slices"), development_metrics=payload.get("development_metrics"))
            result = {"status": "GREEN" if gate["overall"] == "green" else "REJECTED", "tier": tier, "gate": gate}
            results[tier] = result
            if result["status"] != "GREEN":
                ledger = retire_evaluation_tier(ledger, tier=tier, reason="frozen finalist failed promotion gate")
                halted = True
                halt_reason = f"promotion gate failed for {tier}"
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            results[tier] = {"status": "BLOCKED", "tier": tier, "message": str(exc)}
            halted = True
            halt_reason = f"promotion could not open {tier}"
    write_immutable_json(ledger_path, ledger)
    green = bool(results) and all(item.get("status") == "GREEN" for item in results.values())
    return {"status": "PROMOTION_GREEN" if green else "PROMOTION_REJECTED", "method_version": method_version, "tiers": results, "halted": halted, "halt_reason": halt_reason, "contamination_ledger": str(ledger_path), "finalist_lock": str(finalist_lock_path), "ledger_validation": validate_contamination_ledger(ledger)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--tier", action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--activation-root", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_promotion(run_dir=args.run_dir, method_version=args.method_version, tiers=args.tier, execute=args.execute, source_dir=args.source_dir, activation_root=args.activation_root, profile=args.profile, max_tokens=args.max_tokens)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PROMOTION_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
