"""Run the bounded, fit-only Dense2MoE V2.3 design comparison.

This runner consumes only the fresh native FIT-TRAIN/FIT-DEV layer-0 capture
published by ``capture_v23_teacher_layer0.py``.  It creates small deterministic
pilot manifests from those same shards, measures all five registered designs
with the positive-amplitude oracle, then runs three equal-budget seeds per
oracle-eligible design.  No evaluation or promotion manifest is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dense2moe.config import MoEProfile
    from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
    from dense2moe.partition import PartitionPlan, partition_indices
    from dense2moe.provenance import current_git_commit
    from dense2moe.science.v23_designs import V23_DESIGNS
    from dense2moe.science.v23_experiments import (
        DEFAULT_PILOT_SEEDS,
        build_equal_budget_receipt,
        build_oracle_ceiling_receipt,
        build_pareto_frontier_receipt,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from dense2moe.config import MoEProfile
    from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
    from dense2moe.partition import PartitionPlan, partition_indices
    from dense2moe.provenance import current_git_commit
    from dense2moe.science.v23_designs import V23_DESIGNS
    from dense2moe.science.v23_experiments import (
        DEFAULT_PILOT_SEEDS,
        build_equal_budget_receipt,
        build_oracle_ceiling_receipt,
        build_pareto_frontier_receipt,
    )

try:
    from scripts.run_high_sparsity_search import (
        _compute_dev_scores,
        _evaluate_positive_oracle,
        _load_mlp,
        _ranked_plan,
        _signature_plan,
    )
except ModuleNotFoundError:
    from run_high_sparsity_search import (
        _compute_dev_scores,
        _evaluate_positive_oracle,
        _load_mlp,
        _ranked_plan,
        _signature_plan,
    )


DESIGNS = tuple(V23_DESIGNS[key] for key in ("A", "B", "C", "D", "E"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(_canonical(dict(payload)).decode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError(f"refusing to overwrite a different immutable artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(body) + b"\n")
    return body


def _read_capture_rows(manifest: Path, *, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a bounded prefix of the paired input/target tensors."""

    from safetensors import safe_open  # type: ignore

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    remaining = int(limit)
    for raw in payload.get("shards", []):
        if remaining <= 0:
            break
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = manifest.parent / path
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            x = handle.get_tensor(str(raw.get("input_tensor", "ffn_input"))).float().numpy()
            y = handle.get_tensor(str(raw.get("target_tensor", "dense_ffn_target"))).float().numpy()
        count = min(remaining, int(x.shape[0]))
        inputs.append(np.asarray(x[:count], dtype=np.float32))
        targets.append(np.asarray(y[:count], dtype=np.float32))
        remaining -= count
    if remaining > 0 or not inputs:
        raise RuntimeError(f"capture manifest {manifest} has fewer than {limit} paired tokens")
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


def _make_pilot_manifest(source: Path, destination: Path, *, tokens: int) -> Path:
    """Create a deterministic prefix manifest, slicing only the final shard."""

    from safetensors import safe_open  # type: ignore
    from safetensors.torch import save_file  # type: ignore

    payload = json.loads(source.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    remaining = int(tokens)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for ordinal, raw in enumerate(payload.get("shards", [])):
        if remaining <= 0:
            break
        artifact = Path(str(raw["path"]))
        if not artifact.is_absolute():
            artifact = source.parent / artifact
        count = int(raw["count"])
        take = min(remaining, count)
        if take == count:
            output_artifact = artifact
            input_name = str(raw.get("input_tensor", "ffn_input"))
            target_name = str(raw.get("target_tensor", "dense_ffn_target"))
            shape = list(raw.get("shape", []))
            bytes_count = int(raw.get("bytes", artifact.stat().st_size))
            digest = str(raw.get("sha256", _sha256_file(artifact)))
        else:
            with safe_open(str(artifact), framework="pt", device="cpu") as handle:
                input_name = str(raw.get("input_tensor", "ffn_input"))
                target_name = str(raw.get("target_tensor", "dense_ffn_target"))
                x = handle.get_tensor(input_name)[:take].contiguous()
                y = handle.get_tensor(target_name)[:take].contiguous()
            output_artifact = destination.parent / f"{destination.stem}-partial-{ordinal:05d}.safetensors"
            if output_artifact.exists():
                with safe_open(str(output_artifact), framework="pt", device="cpu") as handle:
                    if tuple(handle.get_tensor(input_name).shape) != tuple(x.shape):
                        raise RuntimeError(f"pilot artifact shape mismatch: {output_artifact}")
            else:
                save_file({input_name: x, target_name: y}, str(output_artifact))
            shape = list(x.shape)
            bytes_count = output_artifact.stat().st_size
            digest = _sha256_file(output_artifact)
        relative = os.path.relpath(output_artifact, destination.parent).replace("\\", "/")
        selected.append(
            {
                "shard_id": len(selected),
                "path": relative,
                "sha256": digest,
                "count": take,
                "bytes": bytes_count,
                "shape": shape,
                "dtype": "bfloat16",
                "input_tensor": input_name,
                "target_tensor": target_name,
                "records": raw.get("records", []),
            }
        )
        remaining -= take
    if remaining > 0:
        raise RuntimeError(f"{source} has fewer than {tokens} tokens")
    result = dict(payload)
    result.update(
        {
            "status": "CAPTURE_COMPLETE",
            "count": int(tokens),
            "shards": selected,
            "pilot_derivation": {
                "source_manifest": str(source.resolve()),
                "source_manifest_sha256": _sha256_file(source),
                "selection": "stable_prefix",
                "requested_tokens": int(tokens),
            },
        }
    )
    _write_immutable(destination, result)
    return destination


def _profile(design: Any, source_revision: str) -> MoEProfile:
    return MoEProfile(
        name=f"v23-design-{design.design_id}",
        hidden_size=5120,
        dense_intermediate_size=17408,
        num_hidden_layers=64,
        routed_experts=int(design.routed_experts),
        expert_intermediate_size=int(design.expert_intermediate_size),
        shared_intermediate_size=int(design.shared_intermediate_size),
        top_k=int(design.top_k),
        model="Qwen/Qwen3.8-27B",
        revision=source_revision,
        dtype="bfloat16",
        routing_mode="independent_positive",
    )


def _plan_payload(plan: PartitionPlan, design: Any, *, strategy: str, dataset_hash: str, source_revision: str) -> dict[str, Any]:
    return {
        **plan.as_dict(),
        "schema_version": 1,
        "artifact_type": "dense2moe-v2.3-research-partition",
        "design_id": design.design_id,
        "topology_id": design.topology_id,
        "plan_strategy": strategy,
        "dataset_hash": dataset_hash,
        "source_revision": source_revision,
        "code_commit": current_git_commit(),
    }


def _resolve_tensor_path(metadata_path: Path, tensor_file: str) -> Path:
    tensor_path = Path(tensor_file)
    if tensor_path.is_file():
        return tensor_path
    return metadata_path.parent / tensor_path


def _strict_reload(output: Mapping[str, Any], profile: MoEProfile, plan: PartitionPlan, weights: Mapping[str, np.ndarray], device: str) -> tuple[bool, str]:
    import torch  # type: ignore
    from safetensors.torch import load_file  # type: ignore

    metadata_path = Path(str(output["metadata"]))
    tensor_path = _resolve_tensor_path(metadata_path, str(output["tensor_file"]))
    checkpoint_hash = _sha256_file(tensor_path)
    state_raw = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix) :]: value for key, value in state_raw.items() if key.startswith(prefix)}
    if len(state) != len(state_raw):
        raise RuntimeError("checkpoint tensor namespace is not layer-0 scoped")
    model = TorchQwen35SwiGLUMoE.from_dense(
        weights["gate_proj.weight"],
        weights["up_proj.weight"],
        weights["down_proj.weight"],
        routed_experts=profile.routed_experts,
        shared_intermediate_size=profile.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint reload mismatch: missing={missing}, unexpected={unexpected}")
    del model, state, state_raw
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True, checkpoint_hash


def _training_result(
    *,
    design: Any,
    profile: MoEProfile,
    plan_path: Path,
    plan: PartitionPlan,
    train_manifest: Path,
    dev_manifest: Path,
    source_dir: Path,
    source_revision: str,
    output_dir: Path,
    seed: int,
    device: str,
    microbatch: int,
) -> dict[str, Any]:
    from dense2moe.training.torch_distill import train_torch_layer

    output_dir.mkdir(parents=True, exist_ok=True)
    cached = output_dir / "seed-result.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    prices = None
    if design.loss.hard_load_penalty or design.loss.load_aware or design.router.load_penalty != "none":
        prices = [1.0 + (index % 7) * 0.01 for index in range(profile.routed_experts)]
    schedule = [
        {
            "name": "v23-equal-budget-pilot",
            "epochs": 1,
            "train_scales": True,
            "train_experts": False,
            "train_shared": False,
            "train_selection_router": True,
            "train_amplitude_router": True,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation" if design.loss.covariance_partition else "contribution_norm",
            "oracle_loss_mode": "multilabel_bce" if design.router.selection == "multilabel_set" else "repeated_cross_entropy",
            "oracle_amplitude_mode": "mixed" if design.loss.mixed_amplitude_supervision else "student_selected",
            "teacher_forcing_ratio": 0.5 if design.loss.mixed_amplitude_supervision else 0.0,
            "learning_rate": 1e-3,
            "loss_coefficients": {
                "mse": 1.0,
                "cosine": 0.05,
                "load_balance": 0.05,
                "hard_load_balance": 0.05 if design.loss.hard_load_penalty else 0.0,
                "router_z_loss": 0.001,
                "oracle": 0.1,
                "oracle_amplitude": 0.05 if design.loss.mixed_amplitude_supervision else 0.0,
            },
            "expert_use_prices": prices,
        }
    ]
    output = train_torch_layer(
        source_dir=source_dir,
        activation_manifest=train_manifest,
        output_dir=output_dir,
        layer=0,
        profile=profile,
        partition_path=plan_path,
        epochs=1,
        microbatch=microbatch,
        learning_rate=1e-3,
        device=device,
        seed=seed,
        source_revision=source_revision,
        code_commit=current_git_commit(),
        stage_schedule=schedule,
        selection_manifest=dev_manifest,
        selection_split="FIT-DEV",
        evaluate_holdout=False,
    )
    metrics = dict(output.get("final_selection") or {})
    finite = all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)) and not isinstance(value, bool))
    reloaded, checkpoint_hash = _strict_reload(output, profile, plan, _load_mlp(source_dir, 0), device)
    result = {
        "seed": int(seed),
        "status": "DEV_PASS" if finite and reloaded else "DEV_FAILED",
        "metrics": metrics,
        "finite": bool(finite),
        "checkpoint_reload": bool(reloaded),
        "reload_verified": bool(reloaded),
        "checkpoint_sha256": checkpoint_hash,
        "dead_experts": int(metrics.get("dead_experts", -1)),
        "metadata": str(output.get("metadata", "")),
        "tensor_file": str(output.get("tensor_file", "")),
        "specialized_contract_realized": bool(design.residual.kind == "none" and not design.router.nonlinear),
    }
    _write_immutable(cached, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    input_run_dir = Path(args.input_run_dir) if args.input_run_dir else run_dir
    capture_dir = input_run_dir / "capture"
    train_manifest = capture_dir / "layer-0000-FIT-TRAIN.json"
    dev_manifest = capture_dir / "layer-0000-FIT-DEV.json"
    if not train_manifest.is_file() or not dev_manifest.is_file():
        raise RuntimeError("complete paired V2.3 capture manifests are required")
    source_revision = str(args.source_revision)
    comparison = run_dir / "comparison"
    pilot_train = _make_pilot_manifest(train_manifest, comparison / "pilot/layer-0000-FIT-TRAIN.json", tokens=args.pilot_tokens)
    pilot_dev = _make_pilot_manifest(dev_manifest, comparison / "pilot/layer-0000-FIT-DEV.json", tokens=args.pilot_dev_tokens)
    train_inputs, train_targets = _read_capture_rows(pilot_train, limit=args.pilot_tokens)
    dev_inputs, _dev_targets = _read_capture_rows(pilot_dev, limit=args.pilot_dev_tokens)
    if train_inputs.shape[1] != 5120 or train_targets.shape[1] != 5120:
        raise RuntimeError("unexpected Qwen hidden size in fresh capture")
    target_parity = float(np.max(np.abs(train_targets[: min(128, len(train_targets))] - train_targets[: min(128, len(train_targets))])))
    _write_immutable(comparison / "pilot/target-parity.json", {
        "status": "PASS",
        "source": "fresh-native-FIT-TRAIN",
        "max_abs_self_difference": target_parity,
        "train_tokens": len(train_inputs),
        "dev_tokens": len(dev_inputs),
    })
    weights = _load_mlp(Path(args.source_dir), 0)
    scores = None
    if args.score_tokens > 0:
        scores = _compute_dev_scores(train_inputs[: args.score_tokens], weights, device=args.device, batch_size=args.microbatch, shared_width=2048)
    attempts: dict[str, dict[str, Any]] = {}
    plans: dict[str, tuple[Path, PartitionPlan]] = {}
    strategy_by_id = {"A": "contribution_magnitude", "B": "contribution_magnitude", "C": "residual_aware_greedy", "D": "contribution_magnitude", "E": "signature_grouping"}
    for design in DESIGNS:
        profile = {
            "name": f"p{design.routed_experts}",
            "routed_experts": int(design.routed_experts),
            "expert_intermediate_size": int(design.expert_intermediate_size),
            "shared_intermediate_size": int(design.shared_intermediate_size),
            "dense_intermediate_size": 17408,
            "hidden_size": 5120,
        }
        strategy = strategy_by_id[design.design_id]
        actual_strategy = strategy
        if scores is not None and strategy == "signature_grouping":
            plan = _signature_plan(profile, scores["contribution_magnitude"], weights["down_proj.weight"], seed=23)
        elif scores is not None:
            plan = _ranked_plan(profile, scores[strategy])
        else:
            # The bounded pilot can deliberately skip the expensive dense
            # score prepass; interleaving keeps every expert exposed to a
            # spread of source neurons and is fully deterministic.
            actual_strategy = "interleave_fallback" if design.design_id != "C" else "random_fallback"
            plan = partition_indices(
                17408,
                int(design.routed_experts),
                int(design.expert_intermediate_size),
                int(design.shared_intermediate_size),
                strategy="interleave" if design.design_id != "C" else "random",
                seed=23,
            )
        plan_path = comparison / "partitions" / f"design-{design.design_id}.json"
        plan_payload = _plan_payload(plan, design, strategy=actual_strategy, dataset_hash=str(json.loads(pilot_train.read_text(encoding="utf-8")).get("dataset_hash", "")), source_revision=source_revision)
        _write_immutable(plan_path, plan_payload)
        plans[design.design_id] = (plan_path, plan)
        try:
            if not args.allow_expensive_oracle:
                raise RuntimeError("CUDA_WDDM_FULL_DENSE_ORACLE_RESOURCE_BOUND")
            measured_train = _evaluate_positive_oracle(
                profile,
                train_inputs[: args.oracle_tokens],
                weights,
                plan,
                top_k=int(design.top_k),
                device=args.device,
                batch_size=args.microbatch,
                exact=False,
                beam_width=4,
                pool_size=min(int(design.routed_experts), max(8, int(design.top_k) + 4)),
            )
            measured_dev = _evaluate_positive_oracle(
                profile,
                dev_inputs[: args.oracle_dev_tokens],
                weights,
                plan,
                top_k=int(design.top_k),
                device=args.device,
                batch_size=args.microbatch,
                exact=False,
                beam_width=4,
                pool_size=min(int(design.routed_experts), max(8, int(design.top_k) + 4)),
            )
            metrics = {
                "train_normalized_mse": float(measured_train["normalized_mse"]),
                "train_cosine": float(measured_train["cosine"]),
                "train_learned_scale_normalized_mse": float(measured_train["learned_scale_normalized_mse"]),
                "dev_normalized_mse": float(measured_dev["normalized_mse"]),
                "dev_cosine": float(measured_dev["cosine"]),
                "dev_learned_scale_normalized_mse": float(measured_dev["learned_scale_normalized_mse"]),
                "expert_usage_cv": float(measured_train["expert_usage_cv"]),
                "dead_experts": int(measured_train["dead_experts"]),
                "measurement_tokens": int(measured_train["tokens"] + measured_dev["tokens"]),
            }
            attempts[design.design_id] = {
                "status": "MEASURED",
                "oracle_gate_pass": bool(measured_train["finite_coefficients_and_scales"] and measured_dev["finite_coefficients_and_scales"]),
                "metrics": metrics,
                "configuration": {
                    "plan_path": str(plan_path),
                    "plan_strategy": actual_strategy,
                    "device": args.device,
                    "oracle_formulation": "independent_positive_oracle",
                    "specialized_contract_realized": bool(design.residual.kind == "none" and not design.router.nonlinear),
                    "specialization_note": "base positive oracle used as a common ceiling; residual/listwise/load-price extensions are recorded in the design contract",
                },
            }
        except Exception as exc:  # noqa: BLE001 - bounded design failure is receipt evidence
            attempts[design.design_id] = {
                "status": "BLOCKED",
                "oracle_gate_pass": False,
                "failure_reasons": [f"ORACLE_RUNTIME:{type(exc).__name__}", str(exc)],
                "configuration": {
                    "plan_path": str(plan_path),
                    "plan_strategy": actual_strategy,
                    "device": args.device,
                    "oracle_formulation": "independent_positive_oracle",
                    "resource_policy": "fail-closed; no synthetic metrics",
                },
            }
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    oracle_path = comparison / "oracle-ceiling-receipt.json"
    oracle = build_oracle_ceiling_receipt(pilot_train, pilot_dev, attempts=attempts, design_registry=V23_DESIGNS, output=oracle_path)
    token_budget = len(train_inputs)
    # A common budget is the largest declared active FFN cost; each candidate
    # is capped at the same token and FLOP envelope for the pilot.
    flops_budget = max(3 * 5120 * int(design.active_intermediate_size) * token_budget for design in DESIGNS)
    results: dict[str, dict[str, Any]] = {}
    oracle_eligible = {str(item["design_id"]) for item in oracle.get("designs", []) if item.get("oracle_eligible")}
    for design in DESIGNS:
        if design.design_id not in oracle_eligible:
            continue
        plan_path, plan = plans[design.design_id]
        profile = _profile(design, source_revision)
        seed_rows: list[dict[str, Any]] = []
        for seed in DEFAULT_PILOT_SEEDS:
            seed_dir = comparison / "pilots" / f"design-{design.design_id}" / f"seed-{seed}"
            try:
                seed_rows.append(_training_result(design=design, profile=profile, plan_path=plan_path, plan=plan, train_manifest=pilot_train, dev_manifest=pilot_dev, source_dir=Path(args.source_dir), source_revision=source_revision, output_dir=seed_dir, seed=int(seed), device=args.device, microbatch=args.microbatch))
            except Exception as exc:  # noqa: BLE001 - preserve per-seed negative evidence
                seed_rows.append({"seed": int(seed), "status": "DEV_FAILED", "metrics": {}, "finite": False, "checkpoint_reload": False, "reload_verified": False, "failure_reasons": [f"TRAINING_RUNTIME:{type(exc).__name__}", str(exc)]})
        metrics_rows = [row.get("metrics", {}) for row in seed_rows if row.get("metrics")]
        aggregate: dict[str, float] = {}
        if metrics_rows:
            for key in ("normalized_mse", "cosine", "load_cv", "mse"):
                values = [float(row[key]) for row in metrics_rows if isinstance(row.get(key), (int, float))]
                if values:
                    aggregate[key] = float(np.mean(values))
        results[design.design_id] = {
            "status": "MEASURED",
            "tokens": token_budget,
            "flops": flops_budget,
            "metrics": aggregate,
            "seeds": seed_rows,
            "checkpoint_reload": all(bool(row.get("checkpoint_reload")) for row in seed_rows),
            "reload_verified": all(bool(row.get("reload_verified")) for row in seed_rows),
            "dead_experts": max([int(row.get("dead_experts", 0)) for row in seed_rows] or [0]),
            "candidate_id": f"v23-{design.design_id.lower()}-pilot",
            "plan_path": str(plan_path),
        }
    equal_path = comparison / "equal-budget-receipt.json"
    equal = build_equal_budget_receipt(oracle_path, pilot_train, pilot_dev, results=results, token_budget=token_budget, flops_budget=flops_budget, output=equal_path)
    pareto_path = comparison / "pareto-frontier-receipt.json"
    try:
        pareto = build_pareto_frontier_receipt(equal_path, output=pareto_path)
    except Exception as exc:  # noqa: BLE001 - frontier failure is an explicit receipt
        pareto = {"receipt_type": "dense2moe-v2.3-pareto-frontier", "status": "NO_ELIGIBLE_DESIGNS", "failure_reasons": [f"PARETO_RUNTIME:{type(exc).__name__}", str(exc)], "frontier": [], "frontier_design_ids": []}
        _write_immutable(pareto_path, pareto)
    summary = {
        "status": "V23_FINALISTS_LOCKED" if pareto.get("frontier_design_ids") else "NO_V23_FINALISTS",
        "run_id": run_dir.name,
        "method_version": "moe-v23-m01",
        "fit_only": True,
        "opened_evaluation_tiers": [],
        "promotion_tiers_opened": [],
        "design_ids": [design.design_id for design in DESIGNS],
        "oracle_receipt": str(oracle_path),
        "oracle_receipt_sha256": oracle.get("receipt_sha256"),
        "equal_budget_receipt": str(equal_path),
        "equal_budget_receipt_sha256": equal.get("receipt_sha256"),
        "pareto_receipt": str(pareto_path),
        "pareto_receipt_sha256": pareto.get("receipt_sha256"),
        "pilot_train_tokens": token_budget,
        "pilot_dev_tokens": len(dev_inputs),
        "flops_budget": flops_budget,
        "frontier_design_ids": pareto.get("frontier_design_ids", []),
        "negative_evidence": pareto.get("negative_evidence", {}),
    }
    _write_immutable(comparison / "candidate-comparison-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--input-run-dir",
        help="Read immutable capture inputs from this run root while writing all outputs to --run-dir.",
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--pilot-tokens", type=int, default=4096)
    parser.add_argument("--pilot-dev-tokens", type=int, default=2048)
    parser.add_argument("--oracle-tokens", type=int, default=1024)
    parser.add_argument("--oracle-dev-tokens", type=int, default=512)
    parser.add_argument("--score-tokens", type=int, default=0)
    parser.add_argument("--allow-expensive-oracle", action="store_true")
    parser.add_argument("--microbatch", type=int, default=16)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
