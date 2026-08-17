"""Numerically verify a checkpoint contribution store against direct basis math.

The receipt deliberately compares every branch before any oracle search:
shared output, each routed expert, arbitrary selected top-k sums, and the final
reconstructed FFN output.  Inputs are deterministic when ``--input`` is not
provided, which keeps the command useful as a fast Windows smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.partition.contributions import (
    canonical_partition_sha256,
    load_partition_plan,
    load_trained_basis_state,
    reconstruct_selected,
    sha256_file,
    trained_checkpoint_contributions,
)
from dense2moe.provenance import current_git_commit


def _silu(value: Any) -> Any:
    values = np.asarray(value)
    return values / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _direct_basis_outputs(inputs: np.ndarray, state: dict[str, Any], plan: Any) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference implementation (separate from store helper)."""

    x = np.asarray(inputs, dtype=np.float32).reshape(-1, inputs.shape[-1])
    shared_gate = np.asarray(state["shared_gate_proj.weight"], dtype=np.float32)
    shared_up = np.asarray(state["shared_up_proj.weight"], dtype=np.float32)
    shared_down = np.asarray(state["shared_down_proj.weight"], dtype=np.float32)
    shared_hidden = _silu(x @ shared_gate.T) * (x @ shared_up.T)
    shared = shared_hidden @ shared_down.T
    routed: list[np.ndarray] = []
    scales = np.asarray(state["expert_scales"], dtype=np.float32)
    for expert in range(plan.routed_experts):
        gate = np.asarray(state[f"expert_gate_proj.{expert}.weight"], dtype=np.float32)
        up = np.asarray(state[f"expert_up_proj.{expert}.weight"], dtype=np.float32)
        down = np.asarray(state[f"expert_down_proj.{expert}.weight"], dtype=np.float32)
        hidden = _silu(x @ gate.T) * (x @ up.T)
        routed.append((hidden @ down.T) * scales[expert])
    return shared, np.stack(routed, axis=1)


def _stats(direct: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    left = np.asarray(direct, dtype=np.float64)
    right = np.asarray(reconstructed, dtype=np.float64)
    delta = right - left
    dot = np.sum(left * right, axis=-1)
    cosine = dot / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12)
    return {
        "max_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "mse": float(np.mean(delta * delta)),
        "cosine_agreement": float(np.mean(cosine)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.partition)
    plan = load_partition_plan(plan_path)
    state, metadata = load_trained_basis_state(args.checkpoint, plan)
    hidden = int(np.asarray(state["shared_gate_proj.weight"]).shape[1])
    if args.input:
        inputs = np.asarray(np.load(args.input, allow_pickle=False), dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != hidden:
            raise ValueError(f"input must be [rows, {hidden}]")
    else:
        rng = np.random.default_rng(args.seed)
        inputs = rng.normal(size=(args.rows, hidden)).astype(np.float32)
    direct_shared, direct_routed = _direct_basis_outputs(inputs, state, plan)
    store_shared, store_routed, _ = trained_checkpoint_contributions(
        inputs,
        args.checkpoint,
        plan,
        batch_size=args.batch_size,
    )
    branch_stats = {"shared": _stats(direct_shared, store_shared)}
    branch_stats["routed_all"] = _stats(direct_routed.reshape(-1, hidden), store_routed.reshape(-1, hidden))
    for expert in range(plan.routed_experts):
        branch_stats[f"routed_expert_{expert}"] = _stats(direct_routed[:, expert], store_routed[:, expert])

    rng = np.random.default_rng(args.seed + 1)
    ids = np.stack([rng.choice(plan.routed_experts, size=args.top_k, replace=False) for _ in range(len(inputs))])
    weights = rng.uniform(0.1, 1.0, size=ids.shape).astype(np.float32)
    direct_reconstruction = reconstruct_selected(direct_shared, direct_routed, ids, weights)
    store_reconstruction = reconstruct_selected(store_shared, store_routed, ids, weights)
    branch_stats["arbitrary_topk_reconstruction"] = _stats(direct_reconstruction, store_reconstruction)
    branch_stats["final_ffn_reconstruction"] = branch_stats["arbitrary_topk_reconstruction"]
    tolerances = {
        "max_absolute_error": float(args.max_abs_error),
        "mean_absolute_error": float(args.mean_abs_error),
        "mse": float(args.mse),
        "cosine_agreement": float(args.min_cosine),
    }
    passed = all(
        stats["max_absolute_error"] <= tolerances["max_absolute_error"]
        and stats["mean_absolute_error"] <= tolerances["mean_absolute_error"]
        and stats["mse"] <= tolerances["mse"]
        and stats["cosine_agreement"] >= tolerances["cosine_agreement"]
        for stats in branch_stats.values()
    )
    return {
        "schema_version": 1,
        "status": "CHECKPOINT_CONTRIBUTION_EQUIVALENCE_PASS" if passed else "CHECKPOINT_CONTRIBUTION_EQUIVALENCE_FAIL",
        "basis_source": "trained_checkpoint",
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_tensor_sha256": metadata["checkpoint_tensor_sha256"],
        "partition_path": str(plan_path),
        "partition_sha256": sha256_file(plan_path),
        "partition_canonical_sha256": canonical_partition_sha256(plan),
        "topology": {
            "expert_count": plan.routed_experts,
            "expert_width": plan.expert_intermediate_size,
            "shared_width": plan.shared_intermediate_size,
            "top_k": args.top_k,
        },
        "rows": len(inputs),
        "hidden_size": hidden,
        "dtype": str(np.asarray(inputs).dtype),
        "metrics": branch_stats,
        "tolerances": tolerances,
        "code_commit": current_git_commit(),
        "holdout_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None, help="optional .npy [rows, hidden] input batch")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-abs-error", type=float, default=5e-5)
    parser.add_argument("--mean-abs-error", type=float, default=5e-6)
    parser.add_argument("--mse", type=float, default=1e-9)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "metrics": payload["metrics"]}, indent=2, sort_keys=True))
    if payload["status"].endswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
