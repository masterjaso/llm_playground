"""Bounded V2.4 residual-aware partition/oracle probe for designs A and E.

This experiment stays deliberately below the training/promotion boundary.  It
uses FIT-TRAIN only to choose a deterministic shared set and balanced routed
groups, then evaluates the existing frozen load-aware oracle on the same raw
dense contribution formula for FIT-TRAIN and FIT-DEV.  A receipt is written
with ``basis_source=raw_dense_partition`` and ``promotion_eligible=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dense2moe.partition import PartitionPlan
    from dense2moe.partition.contributions import load_partition_plan
    from dense2moe.partition.oracle import frozen_slice_load_aware_oracle
    from dense2moe.provenance import current_git_commit
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from dense2moe.partition import PartitionPlan
    from dense2moe.partition.contributions import load_partition_plan
    from dense2moe.partition.oracle import frozen_slice_load_aware_oracle
    from dense2moe.provenance import current_git_commit


DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
DEFAULT_RUN = Path(".nsp/artifacts/runs/d2m-qwen38-moe-v24-residual-oracle-20260819t")
DEFAULT_COMPARISON = Path(
    ".nsp/artifacts/runs/d2m-qwen38-moe-v24-cde-retry-20260819t052059z-e2c495ba/comparison"
)
DEFAULT_SCRATCH = Path(r"E:\tmp_data\d2m_v24_residual_oracle_scratch")
PRODUCT_TARGETS = {
    "normalized_mse": 0.05,
    "cosine": 0.98,
    "dead_experts": 0,
    "load_cv": 0.50,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_capture_rows(manifest: Path, *, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a stable prefix from a capture manifest without opening eval tiers."""

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


def _load_dense_mlp(source: Path, layer: int = 0) -> dict[str, Any]:
    """Load the pinned dense layer in float32, matching the existing oracle runner."""

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
            handle_context = safe_open(str(source / shard), framework="pt", device="cpu", **kwargs)
        except TypeError:
            handle_context = safe_open(str(source / shard), framework="pt", device="cpu")
        with handle_context as handle:
            values[name] = handle.get_tensor(prefix + name).float()
    return values


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _torch_dense_state(state: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: torch.as_tensor(_numpy(value), dtype=torch.float32, device=device) for key, value in state.items()}


def _dense_hidden(inputs: np.ndarray, state: Mapping[str, Any], *, device: str, batch_size: int = 128) -> Any:
    """Evaluate SwiGLU hidden activations on the selected device."""

    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.asarray(inputs, dtype=np.float32), dtype=torch.float32, device=device)
    gate = state["gate_proj.weight"]
    up = state["up_proj.weight"]
    blocks: list[Any] = []
    for start in range(0, int(x.shape[0]), max(1, int(batch_size))):
        batch = x[start : start + batch_size]
        blocks.append(F.silu(batch @ gate.T) * (batch @ up.T))
    return torch.cat(blocks, dim=0)


def _residual_energy_scores(hidden: Any, residual: Any, down: Any) -> np.ndarray:
    """Score neurons by residual-weighted contribution energy.

    The score is ``E[||residual||^2 * hidden_j^2] * ||down_j||^2``.  It is
    intentionally a residual-aware energy proxy: it avoids fitting a router or
    using FIT-DEV while still directing shared capacity toward the dense error.
    """

    import torch

    weights = torch.mean(residual.float() * residual.float(), dim=1, keepdim=True)
    activation_energy = torch.mean(hidden.float() * hidden.float() * weights, dim=0)
    output_energy = torch.sum(down.float() * down.float(), dim=0)
    return (activation_energy * output_energy).detach().cpu().numpy().astype(np.float64, copy=False)


def _build_residual_aware_plan(
    hidden: Any,
    targets: Any,
    down: Any,
    *,
    routed_experts: int,
    expert_intermediate_size: int,
    shared_intermediate_size: int,
) -> PartitionPlan:
    """Choose a residual-aware shared set and balanced round-robin expert bins."""

    import torch

    dense_width = int(hidden.shape[1])
    capacity = int(shared_intermediate_size) + int(routed_experts) * int(expert_intermediate_size)
    if capacity != dense_width:
        raise ValueError("partition dimensions do not cover dense hidden width")
    target_values = torch.as_tensor(targets, dtype=torch.float32, device=hidden.device)
    scores = _residual_energy_scores(hidden, target_values, down)
    shared = np.argsort(-scores, kind="stable")[:shared_intermediate_size]
    mask = np.ones(dense_width, dtype=bool)
    mask[shared] = False
    shared_output = hidden[:, shared] @ down[:, shared].T
    residual = target_values - shared_output
    remaining = np.flatnonzero(mask)
    residual_scores = _residual_energy_scores(hidden[:, remaining], residual, down[:, remaining])
    ordered = remaining[np.argsort(-residual_scores, kind="stable")]
    groups = tuple(
        tuple(int(value) for value in ordered[offset::routed_experts][:expert_intermediate_size])
        for offset in range(routed_experts)
    )
    plan = PartitionPlan(
        dense_width,
        int(routed_experts),
        int(expert_intermediate_size),
        int(shared_intermediate_size),
        tuple(int(value) for value in shared),
        groups,
    )
    plan.validate()
    return plan


def _dense_partition_contributions(hidden: Any, down: Any, plan: PartitionPlan) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate raw dense partition contributions using the exact SwiGLU slices."""

    import torch

    shared = hidden[:, list(plan.shared_indices)] @ down[:, list(plan.shared_indices)].T
    routed = torch.stack(
        [hidden[:, list(group)] @ down[:, list(group)].T for group in plan.expert_indices],
        dim=1,
    )
    return (
        shared.detach().cpu().numpy().astype(np.float32, copy=False),
        routed.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _metric_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_penalty": float(result["selected_penalty"]),
        "global_nmse": float(result["global_nmse"]),
        "normalized_mse": float(result["normalized_mse"]),
        "cosine": float(result["cosine"]),
        "load_cv": float(result["load_cv"]),
        "dead_experts": int(result["dead_experts"]),
        "expert_usage_counts": [int(value) for value in result["expert_usage_counts"]],
        "gate_feasible": bool(result["gate_feasible"]),
        "hard_feasible": bool(result["hard_feasible"]),
        "assurance": str(result["assurance"]),
        "combinations_considered_per_token": int(result["combinations_considered_per_token"]),
        "candidate_fit_exact": bool(result["candidate_fit_exact"]),
        "selected_fit_exact": bool(result["selected_fit_exact"]),
    }


def _delta(new: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "normalized_mse": float(new["normalized_mse"]) - float(baseline["normalized_mse"]),
        "cosine": float(new["cosine"]) - float(baseline["cosine"]),
        "load_cv": float(new["load_cv"]) - float(baseline["load_cv"]),
    }


def _plan_payload(plan: PartitionPlan, *, design: str, source: Path, fit_manifest: Path, dev_manifest: Path) -> dict[str, Any]:
    return {
        **plan.as_dict(),
        "schema_version": 1,
        "artifact_type": "dense2moe-v2.4-residual-aware-partition",
        "design_id": design,
        "plan_strategy": "residual_weighted_energy_shared_round_robin",
        "basis_source": "raw_dense_partition",
        "promotion_eligible": False,
        "source_dir": str(source.resolve()),
        "fit_manifest": str(fit_manifest.resolve()),
        "dev_manifest": str(dev_manifest.resolve()),
        "code_commit": current_git_commit(),
    }


def _run_oracle(shared: np.ndarray, routed: np.ndarray, target: np.ndarray, *, top_k: int, scratch: Path) -> dict[str, Any]:
    result = frozen_slice_load_aware_oracle(
        shared,
        routed,
        target,
        top_k=int(top_k),
        target_load_cv=PRODUCT_TARGETS["load_cv"],
        simplex=False,
        candidate_pool_size=16,
        max_combinations=10000,
        iterations=8,
        price_step=0.5,
        price_decay=0.95,
        penalty_grid=(0.0, 0.10, 0.25, 0.50),
        batch_size=32,
        max_in_memory_bytes=128 * 1024 * 1024,
        storage_dir=scratch,
        materialize_outputs=False,
    )
    return _metric_summary(result)


def run_probe(
    *,
    source: Path,
    comparison: Path,
    run_root: Path,
    scratch: Path,
    fit_tokens: int = 128,
    dev_tokens: int = 128,
    device: str = "auto",
    label: str = "",
) -> dict[str, Any]:
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    source = Path(source)
    comparison = Path(comparison)
    run_root = Path(run_root)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    fit_manifest = comparison / "pilot/layer-0000-FIT-TRAIN.json"
    dev_manifest = comparison / "pilot/layer-0000-FIT-DEV.json"
    fit_inputs, fit_targets = _read_capture_rows(fit_manifest, limit=int(fit_tokens))
    dev_inputs, dev_targets = _read_capture_rows(dev_manifest, limit=int(dev_tokens))
    dense_cpu = _load_dense_mlp(source)
    dense = _torch_dense_state(dense_cpu, device)
    fit_hidden = _dense_hidden(fit_inputs, dense, device=device)
    dev_hidden = _dense_hidden(dev_inputs, dense, device=device)
    down = dense["down_proj.weight"]

    definitions = {
        "A": {"top_k": 6, "expert_intermediate_size": 960, "shared_intermediate_size": 2048},
        "E": {"top_k": 5, "expert_intermediate_size": 960, "shared_intermediate_size": 2048},
    }
    results: dict[str, Any] = {}
    for design, geometry in definitions.items():
        current_path = comparison / f"partitions/design-{design}.json"
        current_plan = load_partition_plan(current_path)
        residual_plan = _build_residual_aware_plan(
            fit_hidden,
            fit_targets,
            down,
            routed_experts=current_plan.routed_experts,
            expert_intermediate_size=current_plan.expert_intermediate_size,
            shared_intermediate_size=current_plan.shared_intermediate_size,
        )
        suffix = f"-{label}" if label else ""
        plan_path = run_root / f"comparison/partitions/residual-aware-design-{design}{suffix}.json"
        _write_immutable(plan_path, _plan_payload(residual_plan, design=design, source=source, fit_manifest=fit_manifest, dev_manifest=dev_manifest))

        baseline_fit_shared, baseline_fit_routed = _dense_partition_contributions(fit_hidden, down, current_plan)
        residual_fit_shared, residual_fit_routed = _dense_partition_contributions(fit_hidden, down, residual_plan)
        baseline_dev_shared, baseline_dev_routed = _dense_partition_contributions(dev_hidden, down, current_plan)
        residual_dev_shared, residual_dev_routed = _dense_partition_contributions(dev_hidden, down, residual_plan)
        baseline_fit = _run_oracle(baseline_fit_shared, baseline_fit_routed, fit_targets, top_k=geometry["top_k"], scratch=scratch / design / "baseline-fit")
        residual_fit = _run_oracle(residual_fit_shared, residual_fit_routed, fit_targets, top_k=geometry["top_k"], scratch=scratch / design / "residual-fit")
        baseline_dev = _run_oracle(baseline_dev_shared, baseline_dev_routed, dev_targets, top_k=geometry["top_k"], scratch=scratch / design / "baseline-dev")
        residual_dev = _run_oracle(residual_dev_shared, residual_dev_routed, dev_targets, top_k=geometry["top_k"], scratch=scratch / design / "residual-dev")

        dev_gate = bool(
            residual_dev["normalized_mse"] <= PRODUCT_TARGETS["normalized_mse"]
            and residual_dev["cosine"] >= PRODUCT_TARGETS["cosine"]
            and residual_dev["dead_experts"] == PRODUCT_TARGETS["dead_experts"]
            and residual_dev["load_cv"] <= PRODUCT_TARGETS["load_cv"]
        )
        results[design] = {
            "topology": f"p{current_plan.routed_experts}/top{geometry['top_k']}",
            "current_plan_sha256": hashlib.sha256(_canonical(current_plan.as_dict())).hexdigest(),
            "residual_plan_sha256": hashlib.sha256(_canonical(residual_plan.as_dict())).hexdigest(),
            "residual_plan_path": str(plan_path),
            "fit": {
                "baseline": baseline_fit,
                "residual_aware": residual_fit,
                "delta": _delta(residual_fit, baseline_fit),
            },
            "dev": {
                "baseline": baseline_dev,
                "residual_aware": residual_dev,
                "delta": _delta(residual_dev, baseline_dev),
            },
            "retrain_recommended": dev_gate,
            "product_gate": dev_gate,
        }
        del baseline_fit_shared, baseline_fit_routed, residual_fit_shared, residual_fit_routed
        del baseline_dev_shared, baseline_dev_routed, residual_dev_shared, residual_dev_routed
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    receipt = {
        "schema_version": 1,
        "artifact_type": "dense2moe-v2.4-residual-partition-oracle-probe",
        "run_id": run_root.name,
        "code_commit": current_git_commit(),
        "basis_source": "raw_dense_partition",
        "evaluation_class": "frozen_partition_oracle",
        "promotion_eligible": False,
        "device": device,
        "source_dir": str(source.resolve()),
        "fit_manifest": {"path": str(fit_manifest.resolve()), "sha256": _sha256_file(fit_manifest), "tokens": int(fit_tokens)},
        "dev_manifest": {"path": str(dev_manifest.resolve()), "sha256": _sha256_file(dev_manifest), "tokens": int(dev_tokens)},
        "formulation": {
            "name": "residual_weighted_energy_shared_round_robin",
            "shared_score": "mean(residual_squared * hidden_squared) * down_column_squared_norm",
            "routing_partition": "stable descending residual score, round-robin expert bins",
            "fit_dev_separation": "FIT-TRAIN chooses; FIT-DEV evaluates",
        },
        "oracle": {
            "coefficient_domain": "nonnegative",
            "candidate_pool_size": 16,
            "max_combinations": 10000,
            "penalty_grid": [0.0, 0.1, 0.25, 0.5],
            "iterations": 8,
            "target_load_cv": PRODUCT_TARGETS["load_cv"],
        },
        "product_targets": PRODUCT_TARGETS,
        "designs": results,
        "classification": {
            "status": "candidate_probe_complete",
            "any_product_gate": any(bool(item["product_gate"]) for item in results.values()),
            "next_action": "retrain only a design whose residual-aware DEV probe meets every product gate; otherwise stop structural exploration",
        },
    }
    report_suffix = f"-{label}" if label else ""
    report_path = run_root / f"comparison/residual-partition-oracle{report_suffix}.json"
    return _write_immutable(report_path, receipt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--fit-tokens", type=int, default=128)
    parser.add_argument("--dev-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label", default="", help="optional immutable receipt suffix, e.g. full512")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = run_probe(
        source=args.source_dir,
        comparison=args.comparison_dir,
        run_root=args.run_root,
        scratch=args.scratch_dir,
        fit_tokens=args.fit_tokens,
        dev_tokens=args.dev_tokens,
        device=args.device,
        label=args.label,
    )
    suffix = f"-{args.label}" if args.label else ""
    print(json.dumps({"ok": True, "report": str(args.run_root / f'comparison/residual-partition-oracle{suffix}.json'), "designs": receipt["designs"], "classification": receipt["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
