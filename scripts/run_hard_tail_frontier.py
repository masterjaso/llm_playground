"""Resumable oracle frontier runner for the dense-to-MoE hard-tail study.

The runner is deliberately below the training and promotion boundary.  It
reads the immutable paired FIT-TRAIN/FIT-DEV manifests, constructs a
deterministic dense partition from FIT-TRAIN only, and evaluates the frozen
load-aware oracle on both splits.  Every per-configuration result and V2
structural receipt is write-once, so an interrupted invocation can be safely
resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dense2moe.evaluation.receipts import (
    build_structural_generalization_receipt,
    validate_structural_receipt,
    write_immutable_receipt,
)
from dense2moe.evaluation.replay import (
    iter_paired_activation_shards,
    sha256_file,
)
from dense2moe.evaluation.structural import compute_structural_metrics
from dense2moe.partition import PartitionPlan
from dense2moe.partition.oracle import frozen_slice_load_aware_oracle
from dense2moe.provenance import current_git_commit
from dense2moe.science.hard_tail import HardTailConfig, make_static_config, summarize_active_widths

DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
DEFAULT_RUN_ROOT = Path(".nsp/artifacts/runs/d2m-hard-tail-fallback-20260819t191734z-85863e21")
DEFAULT_FIT_MANIFEST = Path(
    ".nsp/artifacts/runs/d2m-qwen38-moe-v24-cde-retry-20260819t052059z-e2c495ba/comparison/pilot/layer-0000-FIT-TRAIN.json"
)
DEFAULT_DEV_MANIFEST = Path(
    ".nsp/artifacts/runs/d2m-qwen38-moe-v24-cde-retry-20260819t052059z-e2c495ba/comparison/pilot/layer-0000-FIT-DEV.json"
)
PRODUCT_TARGETS = {"normalized_mse": 0.05, "cosine": 0.98, "dead_experts": 0, "load_cv": 0.50}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(_canonical(dict(payload)).decode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError(f"refusing to overwrite a different immutable artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(body) + b"\n")
    return body


def _resolve(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (base / value).resolve()


def _read_manifest_rows(path: Path, *, expected_split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Read every row through the production manifest validator."""

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    count = 0
    for shard in iter_paired_activation_shards(path, expected_split=expected_split, repo_root=REPO_ROOT):
        x = shard.inputs.detach().float().cpu().numpy().astype(np.float32, copy=False)
        y = shard.targets.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError(f"paired activation shape mismatch: {x.shape} vs {y.shape}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite paired activation rows in {shard.shard_path}")
        inputs.append(x)
        targets.append(y)
        metadata.extend(dict(item) for item in shard.metadata)
        count += int(x.shape[0])
    if not inputs:
        raise ValueError(f"manifest has no rows: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("count", count)) != count:
        raise ValueError(f"manifest count mismatch for {path}: {payload.get('count')} != {count}")
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0), metadata, payload


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
        try:
            context = safe_open(str(source / shard), framework="pt", device="cpu", **kwargs)
        except TypeError:
            context = safe_open(str(source / shard), framework="pt", device="cpu")
        with context as handle:
            values[name] = handle.get_tensor(prefix + name).float()
    return values


def _dense_hidden(inputs: np.ndarray, state: Mapping[str, Any], *, device: str, batch_size: int) -> Any:
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    gate = torch.as_tensor(state["gate_proj.weight"], dtype=torch.float32, device=device)
    up = torch.as_tensor(state["up_proj.weight"], dtype=torch.float32, device=device)
    blocks: list[Any] = []
    for start in range(0, int(x.shape[0]), max(1, int(batch_size))):
        batch = x[start : start + batch_size]
        blocks.append(F.silu(batch @ gate.T) * (batch @ up.T))
    return torch.cat(blocks, dim=0)


def _residual_energy_scores(hidden: Any, residual: Any, down: Any) -> np.ndarray:
    import torch

    weight = torch.mean(residual.float() * residual.float(), dim=1, keepdim=True)
    activation_energy = torch.mean(hidden.float() * hidden.float() * weight, dim=0)
    output_energy = torch.sum(down.float() * down.float(), dim=0)
    return (activation_energy * output_energy).detach().cpu().numpy().astype(np.float64, copy=False)


def _build_partition(hidden: Any, targets: np.ndarray, down: Any, config: HardTailConfig) -> PartitionPlan:
    import torch

    dense_width = int(hidden.shape[1])
    if config.shared_width + config.routed_experts * config.expert_width != dense_width:
        raise ValueError("configuration does not exactly partition the dense width")
    target_values = torch.as_tensor(targets, dtype=torch.float32, device=hidden.device)
    scores = _residual_energy_scores(hidden, target_values, down)
    shared = np.argsort(-scores, kind="stable")[: config.shared_width]
    mask = np.ones(dense_width, dtype=bool)
    mask[shared] = False
    shared_output = hidden[:, shared] @ down[:, shared].T
    residual = target_values - shared_output
    remaining = np.flatnonzero(mask)
    residual_scores = _residual_energy_scores(hidden[:, remaining], residual, down[:, remaining])
    ordered = remaining[np.argsort(-residual_scores, kind="stable")]
    groups = tuple(
        tuple(int(value) for value in ordered[offset:: config.routed_experts][: config.expert_width])
        for offset in range(config.routed_experts)
    )
    plan = PartitionPlan(
        dense_width,
        config.routed_experts,
        config.expert_width,
        config.shared_width,
        tuple(int(value) for value in shared),
        groups,
    )
    plan.validate()
    return plan


def _partition_contributions(hidden: Any, down: Any, plan: PartitionPlan) -> tuple[np.ndarray, np.ndarray]:
    import torch

    shared = hidden[:, list(plan.shared_indices)] @ down[:, list(plan.shared_indices)].T
    routed = torch.stack([hidden[:, list(group)] @ down[:, list(group)].T for group in plan.expert_indices], dim=1)
    return shared.detach().cpu().numpy().astype(np.float32, copy=False), routed.detach().cpu().numpy().astype(np.float32, copy=False)


def _oracle(shared: np.ndarray, routed: np.ndarray, target: np.ndarray, *, top_k: int, scratch: Path, iterations: int) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=True)
    result = frozen_slice_load_aware_oracle(
        shared,
        routed,
        target,
        top_k=int(top_k),
        target_load_cv=PRODUCT_TARGETS["load_cv"],
        simplex=False,
        candidate_pool_size=16,
        max_combinations=20_000,
        iterations=int(iterations),
        price_step=0.5,
        price_decay=0.95,
        penalty_grid=(0.0, 0.10, 0.25, 0.50),
        batch_size=32,
        max_in_memory_bytes=128 * 1024 * 1024,
        storage_dir=scratch,
        materialize_outputs=False,
    )
    return result


def _prediction(shared: np.ndarray, routed: np.ndarray, result: Mapping[str, Any]) -> np.ndarray:
    indices = np.asarray(result["indices"], dtype=np.int64)
    weights = np.asarray(result["weights"], dtype=np.float32)
    output = shared.astype(np.float32, copy=True)
    for slot in range(indices.shape[1]):
        output += routed[np.arange(routed.shape[0]), indices[:, slot]] * weights[:, slot, None]
    return output


def _fit_low_rank_residual(
    inputs: np.ndarray,
    residual_targets: np.ndarray,
    *,
    width: int,
    device: str,
) -> tuple[Any, Any]:
    """Fit an input-only rank-bounded linear residual branch.

    The input PCA basis is learned on FIT-TRAIN only.  The output projection is
    then a least-squares fit to the FIT-TRAIN residual target.  This is the
    same two-projection topology as ``LowRankResidualCorrector`` and keeps the
    branch inference-realizable: no target or dense output is consulted at
    application time.
    """

    import torch

    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(residual_targets, dtype=torch.float32, device=device)
    q = min(int(width), int(x.shape[0]), int(x.shape[1]))
    if q <= 0:
        raise ValueError("residual width must be positive")
    torch.manual_seed(0)
    _, _, basis = torch.pca_lowrank(x, q=q, center=False, niter=2)
    features = x @ basis[:, :q]
    output_projection = torch.linalg.lstsq(features, y).solution
    return basis[:, :q].detach(), output_projection.detach()


def _apply_low_rank_residual(inputs: np.ndarray, factors: tuple[Any, Any], *, device: str, batch_size: int) -> np.ndarray:
    import torch

    basis, output_projection = factors
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    blocks: list[Any] = []
    for start in range(0, int(x.shape[0]), max(1, int(batch_size))):
        blocks.append((x[start : start + batch_size] @ basis) @ output_projection)
    return torch.cat(blocks, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)


def _hardness_mask(errors: np.ndarray, *, rate: float, threshold: float | None = None) -> tuple[np.ndarray, float | None]:
    """Freeze FIT-TRAIN error threshold and apply it to a split."""

    count = int(errors.shape[0])
    desired = int(np.floor(float(rate) * count + 1e-12))
    if desired <= 0:
        return np.zeros(count, dtype=bool), None
    if threshold is None:
        threshold = float(np.partition(errors, count - desired)[count - desired])
    mask = errors >= float(threshold)
    # Ties at a quantile can select more rows than the preregistered budget;
    # retain stable row order and cap the selected set deterministically.
    selected = np.flatnonzero(mask)
    if selected.size > desired:
        order = selected[np.argsort(-errors[selected], kind="stable")[:desired]]
        mask = np.zeros(count, dtype=bool)
        mask[order] = True
    return mask, threshold


def _pad_assignments(base: Mapping[str, Any], extra: Mapping[str, Any], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_ids = np.asarray(base["indices"], dtype=np.int64)
    base_weights = np.asarray(base["weights"], dtype=np.float32)
    extra_ids = np.asarray(extra["indices"], dtype=np.int64)
    extra_weights = np.asarray(extra["weights"], dtype=np.float32)
    width = max(base_ids.shape[1], extra_ids.shape[1])
    ids = np.full((base_ids.shape[0], width), -1, dtype=np.int64)
    weights = np.zeros((base_ids.shape[0], width), dtype=np.float32)
    ids[:, : base_ids.shape[1]] = base_ids
    weights[:, : base_weights.shape[1]] = base_weights
    ids[mask, :] = -1
    weights[mask, :] = 0.0
    ids[mask, : extra_ids.shape[1]] = extra_ids[mask]
    weights[mask, : extra_weights.shape[1]] = extra_weights[mask]
    return ids, weights


def _metric_payload(metrics: Mapping[str, Any], *, oracle_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(metrics),
        "oracle_load_cv": float(metrics.get("oracle_load_cv") if metrics.get("oracle_load_cv") is not None else oracle_result["load_cv"]),
        "oracle_dead_expert_count": int(metrics.get("oracle_dead_expert_count") if metrics.get("oracle_dead_expert_count") is not None else oracle_result["dead_experts"]),
        "oracle_expert_counts": [int(value) for value in (metrics.get("oracle_expert_counts") or oracle_result["expert_usage_counts"])],
        "oracle_method": str(oracle_result["method"]),
        "oracle_assurance": str(oracle_result["assurance"]),
        "oracle_gate_feasible": bool(oracle_result["gate_feasible"]),
    }


def _identity(manifest: Path, payload: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "split": split,
        "source_revision": payload.get("source_revision"),
        "tokenizer_hash": payload.get("tokenizer_hash"),
        "dataset_hash": payload.get("dataset_hash"),
    }


def _generalization(fit: Mapping[str, Any], dev: Mapping[str, Any]) -> dict[str, Any]:
    def value(metrics: Mapping[str, Any], key: str) -> float:
        return float(metrics[key])

    return {
        "classification": "ORACLE_ONLY_DIAGNOSTIC",
        "cosine_gap": value(fit, "cosine_similarity") - value(dev, "cosine_similarity"),
        "absolute_nmse_increase": value(dev, "normalized_mse") - value(fit, "normalized_mse"),
        "nmse_ratio": value(dev, "normalized_mse") / max(value(fit, "normalized_mse"), 1e-8),
        "relative_norm_error_increase": value(dev, "target_relative_norm_error") - value(fit, "target_relative_norm_error"),
        "oracle_only": True,
    }


def _config_result(
    *,
    config: HardTailConfig,
    fit_inputs: np.ndarray,
    fit_targets: np.ndarray,
    fit_metadata: Sequence[Mapping[str, Any]],
    dev_inputs: np.ndarray,
    dev_targets: np.ndarray,
    dev_metadata: Sequence[Mapping[str, Any]],
    fit_shared: np.ndarray,
    fit_routed: np.ndarray,
    dev_shared: np.ndarray,
    dev_routed: np.ndarray,
    run_root: Path,
    source_model: Mapping[str, Any],
    source_revision: str,
    fit_identity: Mapping[str, Any],
    dev_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    iterations: int,
    device: str,
    batch_size: int,
    output_root: Path,
) -> dict[str, Any]:
    scratch = output_root / "scratch" / config.configuration_id
    fit_oracle = _oracle(fit_shared, fit_routed, fit_targets, top_k=config.top_k, scratch=scratch / "fit", iterations=iterations)
    dev_oracle = _oracle(dev_shared, dev_routed, dev_targets, top_k=config.top_k, scratch=scratch / "dev", iterations=iterations)
    fit_base = _prediction(fit_shared, fit_routed, fit_oracle)
    dev_base = _prediction(dev_shared, dev_routed, dev_oracle)
    fit_prediction = fit_base.copy()
    dev_prediction = dev_base.copy()
    fit_assignments: tuple[np.ndarray, np.ndarray] = (
        np.asarray(fit_oracle["indices"], dtype=np.int64),
        np.asarray(fit_oracle["weights"], dtype=np.float32),
    )
    dev_assignments: tuple[np.ndarray, np.ndarray] = (
        np.asarray(dev_oracle["indices"], dtype=np.int64),
        np.asarray(dev_oracle["weights"], dtype=np.float32),
    )
    base_fit_error = np.sum((fit_base - fit_targets) ** 2, axis=1) / np.maximum(np.sum(fit_targets * fit_targets, axis=1), 1e-8)
    base_dev_error = np.sum((dev_base - dev_targets) ** 2, axis=1) / np.maximum(np.sum(dev_targets * dev_targets, axis=1), 1e-8)
    fallback_mask_fit = np.zeros(fit_targets.shape[0], dtype=bool)
    fallback_mask_dev = np.zeros(dev_targets.shape[0], dtype=bool)
    threshold: float | None = None
    extra_oracle_fit: Mapping[str, Any] | None = None
    extra_oracle_dev: Mapping[str, Any] | None = None
    residual_factors: tuple[Any, Any] | None = None
    if config.residual_width:
        residual_targets = fit_targets - fit_base
        residual_factors = _fit_low_rank_residual(fit_inputs, residual_targets, width=config.residual_width, device=device)
        fit_residual = _apply_low_rank_residual(fit_inputs, residual_factors, device=device, batch_size=batch_size)
        dev_residual = _apply_low_rank_residual(dev_inputs, residual_factors, device=device, batch_size=batch_size)
        if config.residual_scope == "static":
            fit_prediction += fit_residual
            dev_prediction += dev_residual
        elif config.fallback_mode == "residual":
            fallback_mask_fit, threshold = _hardness_mask(base_fit_error, rate=config.fallback_rate_budget)
            fallback_mask_dev, _ = _hardness_mask(base_dev_error, rate=config.fallback_rate_budget, threshold=threshold)
            fit_prediction[fallback_mask_fit] += fit_residual[fallback_mask_fit]
            dev_prediction[fallback_mask_dev] += dev_residual[fallback_mask_dev]
        else:  # pragma: no cover - guarded by HardTailConfig
            raise ValueError("selected residual scope requires residual fallback")
    if config.fallback_mode in {"top8", "top10"}:
        extra_k = int(config.fallback_top_k)
        extra_oracle_fit = _oracle(fit_shared, fit_routed, fit_targets, top_k=extra_k, scratch=scratch / f"fit-top{extra_k}", iterations=iterations)
        extra_oracle_dev = _oracle(dev_shared, dev_routed, dev_targets, top_k=extra_k, scratch=scratch / f"dev-top{extra_k}", iterations=iterations)
        extra_fit = _prediction(fit_shared, fit_routed, extra_oracle_fit)
        extra_dev = _prediction(dev_shared, dev_routed, extra_oracle_dev)
        fallback_mask_fit, threshold = _hardness_mask(base_fit_error, rate=config.fallback_rate_budget)
        fallback_mask_dev, _ = _hardness_mask(base_dev_error, rate=config.fallback_rate_budget, threshold=threshold)
        fit_prediction[fallback_mask_fit] = extra_fit[fallback_mask_fit]
        dev_prediction[fallback_mask_dev] = extra_dev[fallback_mask_dev]
        fit_assignments = _pad_assignments(fit_oracle, extra_oracle_fit, fallback_mask_fit)
        dev_assignments = _pad_assignments(dev_oracle, extra_oracle_dev, fallback_mask_dev)
    fit_metrics = _metric_payload(
        compute_structural_metrics(fit_targets, fit_prediction, metadata=fit_metadata, oracle_assignments=fit_assignments[0]),
        oracle_result=extra_oracle_fit or fit_oracle,
    )
    dev_metrics = _metric_payload(
        compute_structural_metrics(dev_targets, dev_prediction, metadata=dev_metadata, oracle_assignments=dev_assignments[0]),
        oracle_result=extra_oracle_dev or dev_oracle,
    )
    fit_widths = config.widths_for_mask(fallback_mask_fit.tolist())
    dev_widths = config.widths_for_mask(fallback_mask_dev.tolist())
    candidate = {
        **config.as_dict(),
        "configuration_id": config.configuration_id,
        "evidence_class": "frozen_partition_oracle",
        "oracle_only": True,
        "promotion_eligible": False,
        "source_revision": source_revision,
        "selection_policy": "FIT-TRAIN baseline reconstruction-error threshold; oracle route labels are diagnostic only",
    }
    receipt_payload = build_structural_generalization_receipt(
        source_model=source_model,
        layer=0,
        candidate=candidate,
        fit_train=fit_metrics,
        fit_dev=dev_metrics,
        generalization=_generalization(fit_metrics, dev_metrics),
        evidence_class="frozen-partition-oracle",
        source_receipt_lineage={"preregistration": "immutable", "oracle_promotion_eligible": False},
        code_science_identity={"code_commit": current_git_commit(), "runner": "scripts/run_hard_tail_frontier.py"},
        runtime_lock_identity=runtime_identity,
        lm_evaluation_eligibility={"status": "NOT_COMPUTED", "reason": "oracle-only structural phase"},
        fit_train_identity=fit_identity,
        fit_dev_identity=dev_identity,
    )
    receipt_path = output_root / "receipts" / f"{config.configuration_id}.json"
    receipt = write_immutable_receipt(receipt_payload, receipt_path)
    validation = validate_structural_receipt(receipt)
    if not validation["valid"]:
        raise RuntimeError(f"invalid V2 structural receipt: {validation}")
    result = {
        "schema_version": 1,
        "artifact_type": "dense2moe-hard-tail-frontier-result",
        "configuration": config.as_dict(),
        "configuration_id": config.configuration_id,
        "oracle_only": True,
        "promotion_eligible": False,
        "fit_train": fit_metrics,
        "fit_dev": dev_metrics,
        "receipt": {"path": str(receipt_path), "sha256": receipt["receipt_sha256"], "compatibility": validation["compatibility"]},
        "compute": {
            **config.as_dict(),
            "fallback_rate": 0.0,
            "mean_active_width": config.mean_active_width(0.0),
            "average_reduction": config.average_reduction(0.0),
            "active_width_summary": {
                "fit_train": summarize_active_widths(fit_widths).as_dict(),
                "fit_dev": summarize_active_widths(dev_widths).as_dict(),
            },
            "fit_train_fallback_rate": float(fallback_mask_fit.mean()),
            "fit_dev_fallback_rate": float(fallback_mask_dev.mean()),
            "selection_threshold": threshold,
        },
        "oracle_settings": {"iterations": int(iterations), "candidate_pool_size": 16, "max_combinations": 20_000},
    }
    return result


def _load_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "IMMUTABLE_PREREGISTRATION":
        raise ValueError("preregistration is not immutable")
    return payload


def run_frontier(
    *,
    source: Path,
    fit_manifest: Path,
    dev_manifest: Path,
    run_root: Path,
    preregistration: Path,
    device: str,
    batch_size: int,
    iterations: int,
    configuration_ids: Sequence[str] | None = None,
    resume: bool = False,
    fallback_mode: str = "none",
    fallback_rate: float = 0.0,
    shared_width: int | None = None,
    residual_width: int | None = None,
    output_subdir: str = "results",
) -> dict[str, Any]:
    prereg = _load_preregistration(preregistration)
    fit_inputs, fit_targets, fit_metadata, fit_payload = _read_manifest_rows(fit_manifest, expected_split="FIT-TRAIN")
    dev_inputs, dev_targets, dev_metadata, dev_payload = _read_manifest_rows(dev_manifest, expected_split="FIT-DEV")
    if fit_inputs.shape[1] != dev_inputs.shape[1]:
        raise ValueError("FIT-TRAIN/FIT-DEV hidden sizes differ")
    source_model = {"family": "Qwen3.5", "revision": prereg["inputs"]["source_revision"], "path": str(source)}
    runtime_path = REPO_ROOT / str(prereg["runtime"]["lock_path"])
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_sha = sha256_file(runtime_path)
    expected_runtime_sha = str(prereg["runtime"].get("lock_sha256", ""))
    amendment_candidates = sorted((run_root / "planning").glob("phase-02-runtime-amendment*.json"))
    amendment_path = amendment_candidates[-1] if amendment_candidates else run_root / "planning" / "phase-02-runtime-amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else None
    if runtime_sha != expected_runtime_sha:
        effective = (amendment or {}).get("effective_runtime", {})
        if str(effective.get("file_sha256", "")) != runtime_sha:
            raise ValueError("runtime lock differs from immutable preregistration without a matching phase amendment")
    runtime_identity = {"path": str(runtime_path), "sha256": runtime_sha, "lock": runtime_payload, "amendment": str(amendment_path) if amendment else None}
    dense_state = _load_dense_mlp(source)
    hidden_fit = _dense_hidden(fit_inputs, dense_state, device=device, batch_size=batch_size)
    hidden_dev = _dense_hidden(dev_inputs, dense_state, device=device, batch_size=batch_size)
    import torch

    down = torch.as_tensor(dense_state["down_proj.weight"], dtype=torch.float32, device=device)
    selected = set(configuration_ids or [])
    rows: list[dict[str, Any]] = []
    output_root = run_root / output_subdir
    if fallback_mode not in {"none", "top8", "top10", "residual"}:
        raise ValueError("fallback_mode must be none, top8, top10, or residual")
    if fallback_mode == "none":
        fallback_rate = 0.0
    configs = [dict(item) for item in prereg["static_grid"] if bool(item.get("compute_eligible", False))]
    if shared_width is not None:
        configs = [item for item in configs if int(item["shared_width"]) == int(shared_width)]
    if residual_width is not None:
        configs = [item for item in configs if int(item["residual_width"]) == int(residual_width)]
    if not configs:
        raise ValueError("no preregistered static base matches the requested geometry")
    for item in configs:
        config_residual_width = int(item["residual_width"] if residual_width is None else residual_width)
        config = make_static_config(
            shared_width=int(item["shared_width"]),
            residual_width=config_residual_width,
            fallback_mode=fallback_mode,
            residual_scope="selected" if fallback_mode == "residual" else "static",
            fallback_rate_budget=float(fallback_rate),
        )
        try:
            config.require_compute_budget(float(fallback_rate))
        except ValueError as exc:
            rows.append(
                {
                    "schema_version": 1,
                    "artifact_type": "dense2moe-hard-tail-frontier-result",
                    "configuration": config.as_dict(),
                    "configuration_id": config.configuration_id,
                    "status": "OVER_BUDGET_DIAGNOSTIC",
                    "promotion_eligible": False,
                    "compute": {"average_reduction": config.average_reduction(float(fallback_rate)), "reason": str(exc)},
                }
            )
            continue
        if selected and config.configuration_id not in selected:
            continue
        result_path = output_root / "per-config" / f"{config.configuration_id}.json"
        if resume and result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        plan = _build_partition(hidden_fit, fit_targets, down, config)
        fit_shared, fit_routed = _partition_contributions(hidden_fit, down, plan)
        dev_shared, dev_routed = _partition_contributions(hidden_dev, down, plan)
        result = _config_result(
            config=config,
            fit_inputs=fit_inputs,
            fit_targets=fit_targets,
            fit_metadata=fit_metadata,
            dev_inputs=dev_inputs,
            dev_targets=dev_targets,
            dev_metadata=dev_metadata,
            fit_shared=fit_shared,
            fit_routed=fit_routed,
            dev_shared=dev_shared,
            dev_routed=dev_routed,
            run_root=run_root,
            source_model=source_model,
            source_revision=str(prereg["inputs"]["source_revision"]),
            fit_identity=_identity(fit_manifest, fit_payload, split="FIT-TRAIN"),
            dev_identity=_identity(dev_manifest, dev_payload, split="FIT-DEV"),
            runtime_identity=runtime_identity,
            iterations=iterations,
            device=device,
            batch_size=batch_size,
            output_root=output_root,
        )
        _write_immutable_json(result_path, result)
        rows.append(result)
    summary = {
        "schema_version": 1,
        "artifact_type": "dense2moe-hard-tail-frontier",
        "status": "ORACLE_ONLY_DIAGNOSTIC",
        "run_id": prereg.get("run_id"),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "source_revision": prereg["inputs"]["source_revision"],
        "fit_train_rows": int(fit_inputs.shape[0]),
        "fit_dev_rows": int(dev_inputs.shape[0]),
        "results": sorted(rows, key=lambda item: str(item["configuration_id"])),
        "oracle_only": True,
        "promotion_eligible": False,
        "stop_rule": "predictor_deferred_until_oracle_clears",
    }
    _write_immutable_json(output_root / "hard-tail-frontier.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--configuration-id", action="append", dest="configuration_ids")
    parser.add_argument("--fallback-mode", choices=("none", "top8", "top10", "residual"), default="none")
    parser.add_argument("--fallback-rate", type=float, default=0.0)
    parser.add_argument("--shared-width", type=int, default=None)
    parser.add_argument("--residual-width", type=int, default=None)
    parser.add_argument("--output-subdir", default="results")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = _resolve(args.run_root)
    prereg = _resolve(args.preregistration or (run_root / "preregistration.json"))
    summary = run_frontier(
        source=_resolve(args.source),
        fit_manifest=_resolve(args.fit_manifest),
        dev_manifest=_resolve(args.dev_manifest),
        run_root=run_root,
        preregistration=prereg,
        device=str(args.device),
        batch_size=int(args.batch_size),
        iterations=int(args.iterations),
        configuration_ids=args.configuration_ids,
        resume=bool(args.resume),
        fallback_mode=str(args.fallback_mode),
        fallback_rate=float(args.fallback_rate),
        shared_width=args.shared_width,
        residual_width=args.residual_width,
        output_subdir=str(args.output_subdir),
    )
    print(json.dumps({"ok": True, "status": summary["status"], "result_count": len(summary["results"]), "path": str(run_root / str(args.output_subdir) / "hard-tail-frontier.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
