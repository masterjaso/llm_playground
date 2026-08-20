"""Bounded decomposition of the p16/top6 50% FIT-DEV failure.

This is a diagnostic-only reader.  It loads the already-produced capacity
checkpoints, opens FIT-DEV, and compares the trained route/amplitudes with a
teacher-derived positive route/amplitude oracle over the *trained* basis.  It
does not update parameters, open protected tiers, or change any selection
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dense2moe.evaluation.structural import compute_structural_metrics
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import PartitionPlan
from dense2moe.training.torch_distill import ActivationShardDataset

try:
    from scripts.run_topk_architecture_search import (
        _dense_hidden_target,
        _residual_correlation_beam_topk,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from run_topk_architecture_search import (  # type: ignore
        _dense_hidden_target,
        _residual_correlation_beam_topk,
    )


DEFAULT_PREREG = REPO_ROOT / ".nsp" / "artifacts" / "runs" / "d2m-p16-top6-50-selection" / "preregistration.json"
DEFAULT_SOURCE = REPO_ROOT / "runs" / "20260815-030931-windows" / "source"
DEFAULT_CAPACITY_ROOT = REPO_ROOT / ".nsp" / "artifacts" / "runs" / "d2m-p16-top6-50-selection-rerun-router-lr"
DEFAULT_RUN_ROOT = REPO_ROOT / ".nsp" / "artifacts" / "runs" / "d2m-p16-top6-50-decomposition"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_dense_layer(source_dir: Path) -> tuple[Any, Any, Any]:
    from safetensors import safe_open  # type: ignore

    index = _json(source_dir / "model.safetensors.index.json")
    prefix = "model.language_model.layers.0.mlp."
    names = {key[len(prefix):]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    if set(names) != {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
        raise ValueError("source layer-0 MLP inventory mismatch")
    values: dict[str, Any] = {}
    for short, shard in names.items():
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **kwargs) as handle:
            values[short] = handle.get_tensor(prefix + short).float()
    return values["gate_proj.weight"], values["up_proj.weight"], values["down_proj.weight"]


def _load_plan(path: Path) -> PartitionPlan:
    payload = _json(path)
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


def _load_checkpoint(
    source_weights: tuple[Any, Any, Any],
    plan: PartitionPlan,
    checkpoint_dir: Path,
    candidate: Mapping[str, Any],
    device: str,
) -> TorchQwen35SwiGLUMoE:
    import torch
    from safetensors.torch import load_file  # type: ignore

    metadata = _json(checkpoint_dir / "layer-0000.json")
    training = metadata.get("training_config", {})
    router_hidden_size = training.get("router_hidden_size")
    if router_hidden_size is not None:
        router_hidden_size = int(router_hidden_size)
    model = TorchQwen35SwiGLUMoE.from_dense(
        *source_weights,
        routed_experts=int(candidate["routed_experts"]),
        shared_intermediate_size=int(candidate["shared_width"]),
        top_k=int(candidate["top_k"]),
        routing_mode="independent_positive",
        router_hidden_size=router_hidden_size,
        router_feature_mode=str(training.get("router_feature_mode", "none")),
        partition=plan,
        learnable_scales=True,
        residual_intermediate_size=int(candidate["residual_width"]),
        residual_scope=str(candidate["residual_scope"]),
        fallback_mode="none",
        fallback_rate_budget=0.0,
    )
    raw = load_file(str(checkpoint_dir / "layer-0000.safetensors"), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise RuntimeError("checkpoint namespace is not layer-0 compatible")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model


def _conditional(base: Any, routed: Any, ids: Any, weights: Any) -> Any:
    import torch

    rows = torch.arange(routed.shape[0], device=routed.device)
    output = base.clone()
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


def _metric_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    q4 = metrics.get("target_norm_buckets", {}).get("q4", {})
    return {
        "cosine": metrics.get("cosine_similarity"),
        "normalized_mse": metrics.get("normalized_mse"),
        "target_relative_norm_error": metrics.get("target_relative_norm_error"),
        "mean_prediction_to_target_norm_ratio": metrics.get("mean_prediction_to_target_norm_ratio"),
        "p95_abs_relative_norm_error": metrics.get("p95_abs_relative_norm_error"),
        "q4_cosine": q4.get("cosine_similarity"),
        "q4_normalized_mse": q4.get("normalized_mse"),
        "token_count": metrics.get("scored_token_count"),
        "load_cv": metrics.get("learned_load_cv"),
        "dead_experts": metrics.get("dead_expert_count"),
    }


def _gate_failures(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> list[str]:
    q4 = metrics.get("target_norm_buckets", {}).get("q4", {})
    checks = {
        "global_cosine": (metrics.get("cosine_similarity"), float(gates["global_cosine_min"]), lambda a, b: a >= b),
        "global_normalized_mse": (metrics.get("normalized_mse"), float(gates["global_normalized_mse_max"]), lambda a, b: a <= b),
        "target_relative_norm_error": (metrics.get("target_relative_norm_error"), float(gates["target_relative_norm_error_max"]), lambda a, b: a <= b),
        "mean_norm_ratio": (metrics.get("mean_prediction_to_target_norm_ratio"), float(gates["mean_prediction_target_norm_ratio_min"]), lambda a, b: a >= b),
        "p95_abs_relative_norm_error": (metrics.get("p95_abs_relative_norm_error"), float(gates["p95_abs_relative_norm_error_max"]), lambda a, b: a <= b),
        "q4_cosine": (q4.get("cosine_similarity"), float(gates["q4_cosine_min"]), lambda a, b: a >= b),
        "q4_normalized_mse": (q4.get("normalized_mse"), float(gates["q4_normalized_mse_max"]), lambda a, b: a <= b),
    }
    failures: list[str] = []
    for name, (observed, threshold, predicate) in checks.items():
        if observed is None:
            failures.append(f"{name}=missing")
            continue
        if not np.isfinite(float(observed)) or not predicate(float(observed), threshold):
            failures.append(f"{name}={float(observed):.9g} threshold={threshold:.9g}")
    return failures


def _classify(variants: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "student_ids_student_amplitudes",
        "student_ids_oracle_amplitudes",
        "oracle_ids_student_amplitudes",
        "oracle_ids_oracle_amplitudes",
    )
    failures = {name: _gate_failures(variants[name], gates) for name in names}
    ceiling_green = not failures[names[-1]]
    if not ceiling_green:
        label = "BASIS_CAPACITY_LIMITED"
        statement = "The trained shared/expert/residual basis misses the FIT-DEV gates even when route IDs and positive amplitudes are teacher-optimized."
    elif failures[names[2]]:
        label = "AMPLITUDE_LIMITED"
        statement = "The trained basis and oracle route set are sufficient, but the learned amplitude predictor is the remaining gate failure."
    elif failures[names[1]]:
        label = "ROUTING_LIMITED"
        statement = "The trained basis and positive amplitudes are sufficient for oracle routes, but the learned route selector chooses the wrong experts."
    elif failures[names[0]]:
        label = "JOINT_OPTIMIZATION_LIMITED"
        statement = "The separately controlled route and amplitude paths clear, but their jointly learned combination does not."
    else:
        label = "NO_STRUCTURAL_FAILURE_REPRODUCED"
        statement = "The controlled variants clear the declared structural gates; the original failure is not reproduced by this checkpoint decomposition."
    return {"label": label, "statement": statement, "gate_failures": failures}


def _parse_checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.checkpoint:
        output: list[tuple[str, Path]] = []
        for item in args.checkpoint:
            if "=" not in item:
                raise ValueError("--checkpoint must be RECIPE_ID=CHECKPOINT_DIRECTORY")
            recipe, path = item.split("=", 1)
            output.append((recipe, _repo_path(path)))
        return output
    return [
        ("capacity-standard-a", args.capacity_root / "checkpoints" / "capacity-standard-a" / "seed-17"),
        ("capacity-quantile-balanced-b", args.capacity_root / "checkpoints" / "capacity-quantile-balanced-b" / "seed-17"),
    ]


def _run_checkpoint(
    *,
    recipe_id: str,
    checkpoint_dir: Path,
    prereg: Mapping[str, Any],
    source_weights: tuple[Any, Any, Any],
    plan: PartitionPlan,
    dataset: ActivationShardDataset,
    receipt: Mapping[str, Any] | None,
    microbatch: int,
    device: str,
    beam_width: int,
    pool_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    import torch

    candidate = prereg["candidate"]
    model = _load_checkpoint(source_weights, plan, checkpoint_dir, candidate, device)
    gate, up, down = (value.to(device) for value in source_weights)
    variant_names = (
        "student_ids_student_amplitudes",
        "student_ids_oracle_amplitudes",
        "oracle_ids_student_amplitudes",
        "oracle_ids_oracle_amplitudes",
    )
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in variant_names}
    assignments: dict[str, list[np.ndarray]] = {name: [] for name in variant_names}
    probabilities: dict[str, list[np.ndarray]] = {name: [] for name in variant_names[:2]}
    targets: list[np.ndarray] = []
    overlaps: list[np.ndarray] = []
    amplitude_abs: list[np.ndarray] = []
    amplitude_rel: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    selector_margins: list[np.ndarray] = []
    count = 0
    batches = 0
    baseline_max_abs = 0.0
    with torch.inference_mode():
        for values in dataset.iter_batches(microbatch):
            if max_batches is not None and batches >= max_batches:
                break
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            _hidden, target = _dense_hidden_target(inputs, {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}, torch.device(device))
            prediction, info = model(inputs, return_router=True, return_contributions=True)
            shared = info["shared"]
            routed = info["contributions"]
            residual_output = model.residual_corrector(inputs) if model.residual_corrector is not None else torch.zeros_like(shared)
            base = shared + residual_output
            residual = target - base
            student_ids = info["indices"]
            student_weights = info["weights"]
            student_amp_all = torch.nn.functional.softplus(info["amplitude_logits"])
            oracle = _residual_correlation_beam_topk(
                base,
                routed,
                target,
                model.top_k,
                simplex=False,
                beam_width=beam_width,
                pool_size=pool_size,
            )
            oracle_ids = oracle["indices"]
            oracle_weights = oracle["weights"]
            student_oracle_weights = _positive_coefficients(routed, student_ids, residual)
            oracle_student_weights = torch.gather(student_amp_all, 1, oracle_ids)
            variants = {
                "student_ids_student_amplitudes": prediction,
                "student_ids_oracle_amplitudes": _conditional(base, routed, student_ids, student_oracle_weights),
                "oracle_ids_student_amplitudes": _conditional(base, routed, oracle_ids, oracle_student_weights),
                "oracle_ids_oracle_amplitudes": _conditional(base, routed, oracle_ids, oracle_weights),
            }
            baseline_rebuilt = _conditional(base, routed, student_ids, student_weights)
            baseline_max_abs = max(baseline_max_abs, float((baseline_rebuilt - prediction).abs().max().item()))
            for name, value in variants.items():
                predictions[name].append(value.detach().cpu().numpy())
                assignments[name].append((student_ids if name.startswith("student_") else oracle_ids).detach().cpu().numpy())
            probs = torch.softmax(info["logits"], dim=-1).detach().cpu().numpy()
            probabilities["student_ids_student_amplitudes"].append(probs)
            probabilities["student_ids_oracle_amplitudes"].append(probs)
            targets.append(target.detach().cpu().numpy())
            overlap = (student_ids.unsqueeze(2) == oracle_ids.unsqueeze(1)).any(dim=2).sum(dim=1)
            overlaps.append(overlap.detach().cpu().numpy())
            amp_diff = oracle_student_weights - oracle_weights
            amplitude_abs.append(amp_diff.abs().detach().cpu().numpy().reshape(-1))
            amplitude_rel.append((amp_diff.abs() / oracle_weights.abs().clamp_min(1e-6)).detach().cpu().numpy().reshape(-1))
            residual_norms.append(torch.linalg.vector_norm(residual, dim=1).detach().cpu().numpy())
            sorted_logits = torch.sort(info["logits"], dim=1, descending=True).values
            selector_margins.append((sorted_logits[:, model.top_k - 1] - sorted_logits[:, model.top_k]).detach().cpu().numpy())
            count += int(inputs.shape[0])
            batches += 1
    if count <= 0:
        raise ValueError("diagnostic received no FIT-DEV rows")
    if max_batches is None and count != int(dataset.count):
        raise RuntimeError(f"FIT-DEV count mismatch: {count} != {dataset.count}")
    target_array = np.concatenate(targets, axis=0)
    variant_metrics: dict[str, dict[str, Any]] = {}
    metadata = tuple({"independent_group": f"FIT-DEV:{index}"} for index in range(count))
    for name in variant_names:
        prediction_array = np.concatenate(predictions[name], axis=0)
        assignment_array = np.concatenate(assignments[name], axis=0)
        metric = compute_structural_metrics(
            target_array,
            prediction_array,
            metadata=metadata,
            learned_assignments=assignment_array,
            learned_probabilities=np.concatenate(probabilities[name], axis=0) if name in probabilities else None,
        )
        metric["prediction_reconstruction"] = "base(shared+trained_residual)+selected_trained_expert_contributions"
        variant_metrics[name] = metric
    overlap_array = np.concatenate(overlaps)
    amplitude_abs_array = np.concatenate(amplitude_abs)
    amplitude_rel_array = np.concatenate(amplitude_rel)
    residual_array = np.concatenate(residual_norms)
    margin_array = np.concatenate(selector_margins)
    expected = None
    if receipt is not None:
        expected = receipt.get("fit_dev")
    observed = _metric_view(variant_metrics["student_ids_student_amplitudes"])
    reproduction = {"available": expected is not None, "max_abs_metric_delta": None, "checks": {}}
    if expected is not None:
        expected_view = {
            "cosine": expected.get("cosine_similarity", expected.get("cosine")),
            "normalized_mse": expected.get("normalized_mse"),
            "target_relative_norm_error": expected.get("target_relative_norm_error"),
            "mean_prediction_to_target_norm_ratio": expected.get("mean_prediction_to_target_norm_ratio"),
            "p95_abs_relative_norm_error": expected.get("p95_abs_relative_norm_error"),
        }
        deltas = {
            name: abs(float(observed[name]) - float(value))
            for name, value in expected_view.items()
            if value is not None and observed.get(name) is not None
        }
        reproduction["checks"] = {name: bool(delta <= 1e-4) for name, delta in deltas.items()}
        reproduction["max_abs_metric_delta"] = max(deltas.values()) if deltas else None
        reproduction["passed"] = bool(deltas) and all(reproduction["checks"].values())
    else:
        reproduction["passed"] = None
    result = {
        "recipe_id": recipe_id,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "tensor": str(checkpoint_dir / "layer-0000.safetensors"),
            "tensor_sha256": _sha256(checkpoint_dir / "layer-0000.safetensors"),
        },
        "split": "FIT-DEV",
        "diagnostic_only": True,
        "teacher_dependent": True,
        "opened_for_gradients": False,
        "opened_for_selection": False,
        "count": count,
        "batches": batches,
        "baseline_rebuild_max_abs": baseline_max_abs,
        "reproduction": reproduction,
        "variants": {name: _metric_view(value) | {"full_metrics": value} for name, value in variant_metrics.items()},
        "route_agreement": {
            "exact_set_match_rate": float(np.mean(overlap_array == model.top_k)),
            "mean_intersection": float(np.mean(overlap_array)),
            "mean_recall": float(np.mean(overlap_array / model.top_k)),
            "mean_jaccard": float(np.mean(overlap_array / (2 * model.top_k - overlap_array))),
        },
        "amplitude_quality_on_oracle_ids": {
            "mae": float(np.mean(amplitude_abs_array)),
            "rmse": float(np.sqrt(np.mean(amplitude_abs_array ** 2))),
            "relative_mae": float(np.mean(amplitude_rel_array)),
            "median_absolute_error": float(np.median(amplitude_abs_array)),
            "finite": bool(np.isfinite(amplitude_abs_array).all() and np.isfinite(amplitude_rel_array).all()),
        },
        "diagnostic_distributions": {
            "residual_norm": {"mean": float(np.mean(residual_array)), "p50": float(np.quantile(residual_array, 0.5)), "p95": float(np.quantile(residual_array, 0.95))},
            "selector_margin": {"mean": float(np.mean(margin_array)), "p10": float(np.quantile(margin_array, 0.1)), "p50": float(np.quantile(margin_array, 0.5)), "p90": float(np.quantile(margin_array, 0.9))},
        },
    }
    result["classification"] = _classify(variant_metrics, prereg["metric_gates"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--capacity-root", type=Path, default=DEFAULT_CAPACITY_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint", action="append", default=[], help="RECIPE_ID=CHECKPOINT_DIRECTORY; repeatable")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None, help="diagnostic smoke limit; omit for all FIT-DEV")
    args = parser.parse_args()
    args.prereg = _repo_path(args.prereg)
    args.source_dir = _repo_path(args.source_dir)
    args.capacity_root = _repo_path(args.capacity_root)
    args.run_root = _repo_path(args.run_root)
    if args.microbatch <= 0 or args.beam_width <= 0 or args.pool_size <= 0:
        raise ValueError("microbatch, beam-width, and pool-size must be positive")
    prereg = _json(args.prereg)
    if prereg.get("status") != "IMMUTABLE_PREREGISTRATION":
        raise ValueError("preregistration is not immutable")
    candidate = prereg["candidate"]
    dev_manifest = _repo_path(prereg["inputs"]["fit_dev_manifest"])
    if _sha256(dev_manifest).lower() != str(prereg["inputs"]["fit_dev_sha256"]).lower():
        raise RuntimeError("FIT-DEV manifest hash differs from frozen preregistration")
    dataset = ActivationShardDataset(dev_manifest, split="FIT-DEV", microbatch=args.microbatch)
    source_weights = _load_dense_layer(args.source_dir)
    plan_path = args.capacity_root / "planning" / "partition-p16-top6-50.json"
    plan = _load_plan(plan_path)
    specs = _parse_checkpoint_specs(args)
    hypothesis = {
        "schema_version": 1,
        "artifact_type": "dense2moe-p16-top6-50-decomposition-hypotheses",
        "status": "DIAGNOSTIC_AUTHORIZED",
        "scope": "FIT-DEV only; existing capacity checkpoints; no optimizer updates; no protected tiers",
        "reproduction": "The corrected capacity-standard-a receipt reports FIT-DEV cosine 0.9007959962 and normalized MSE 0.3097931445.",
        "hypotheses": [
            {"id": "H1", "claim": "trained basis/capacity is limiting", "falsifier": "oracle route IDs plus oracle positive amplitudes clear every structural FIT-DEV gate"},
            {"id": "H2", "claim": "route selection is limiting", "falsifier": "student route IDs plus oracle amplitudes clear while oracle route IDs are not needed"},
            {"id": "H3", "claim": "amplitude prediction is limiting", "falsifier": "oracle route IDs plus student amplitudes clear every structural FIT-DEV gate"},
            {"id": "H4", "claim": "joint optimization is limiting", "falsifier": "both controlled single-factor variants clear but the trained route/amplitude pair fails"},
        ],
        "variants": [
            "trained basis + student routes + student amplitudes",
            "trained basis + student routes + oracle amplitudes",
            "trained basis + oracle routes + student amplitudes",
            "trained basis + oracle routes + oracle amplitudes",
        ],
        "oracle": {"method": "residual_correlation_beam_search_exact_final_coefficients", "beam_width": args.beam_width, "pool_size": args.pool_size, "positive_amplitudes": True},
    }
    _write_once(args.run_root / "planning" / "hypotheses.json", hypothesis)
    results: dict[str, Any] = {}
    receipt_root = args.capacity_root / "results" / "recipes"
    for recipe_id, checkpoint_dir in specs:
        receipt_path = receipt_root / recipe_id / "seed-17.json"
        receipt = _json(receipt_path) if receipt_path.exists() else None
        results[recipe_id] = _run_checkpoint(
            recipe_id=recipe_id,
            checkpoint_dir=checkpoint_dir,
            prereg=prereg,
            source_weights=source_weights,
            plan=plan,
            dataset=dataset,
            receipt=receipt,
            microbatch=args.microbatch,
            device=args.device,
            beam_width=args.beam_width,
            pool_size=args.pool_size,
            max_batches=args.max_batches,
        )
    status = "DIAGNOSTIC_SMOKE_COMPLETE" if args.max_batches is not None else "DIAGNOSTIC_COMPLETE"
    payload = {
        "schema_version": 1,
        "artifact_type": "dense2moe-p16-top6-50-decomposition",
        "status": status,
        "run_id": args.run_root.name,
        "lineage": {
            "preregistration": str(args.prereg),
            "preregistration_sha256": _sha256(args.prereg),
            "capacity_run": str(args.capacity_root),
            "source_snapshot": str(args.source_dir),
            "fit_dev_manifest": str(dev_manifest),
            "fit_dev_manifest_sha256": _sha256(dev_manifest),
        },
        "execution": {"device": args.device, "microbatch": args.microbatch, "beam_width": args.beam_width, "pool_size": args.pool_size, "max_batches": args.max_batches, "gradient_updates": 0, "protected_tiers_opened": False},
        "results": results,
    }
    _write_once(args.run_root / "results" / "decomposition.json", payload)
    summary_lines = [
        "# p16/top6 50% failure decomposition",
        "",
        f"Status: **{status}**.",
        "",
        "This is FIT-DEV-only diagnostic evidence. It uses teacher-derived routes/amplitudes only inside the diagnostic and does not authorize deployment.",
        "",
    ]
    for recipe_id, result in results.items():
        classification = result.get("classification", {})
        summary_lines.extend([f"## {recipe_id}", "", f"Classification: **{classification.get('label')}** — {classification.get('statement')}", ""])
        for name, variant in result["variants"].items():
            summary_lines.append(f"- {name}: cosine={variant.get('cosine'):.6f}, normalized MSE={variant.get('normalized_mse'):.6f}")
        summary_lines.append("")
    summary_lines.append("No learned-router training, seed robustness, protected-tier evaluation, push, PR, or release was performed.")
    summary_path = args.run_root / "results" / "executive-summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = "\n".join(summary_lines) + "\n"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary_text:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {summary_path}")
    else:
        summary_path.write_text(summary_text, encoding="utf-8")
    print(json.dumps({"status": status, "run_root": str(args.run_root), "classifications": {key: value["classification"] for key, value in results.items()}}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
