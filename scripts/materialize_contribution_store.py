"""Materialize a deterministic FIT-only contribution store for bounded oracle work.

The store contains the shared contribution, every routed expert contribution,
and the dense target for a stratified prefix of the FIT activation capture.  It
never opens validation or holdout manifests.  ``run_load_aware_oracle.py`` then
consumes the resulting read-only mmap arrays with an explicit bounded candidate
pool, which is the intended p32/top4 and p32/top5 screening path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.calibrate_streaming_solver import (
        _contributions,
        _dense_residual_norms,
        _hash_indices,
        _load_plan,
        _load_train_prefix,
        _stratified_indices,
        _write_contribution_store,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from calibrate_streaming_solver import (  # type: ignore
        _contributions,
        _dense_residual_norms,
        _hash_indices,
        _load_plan,
        _load_train_prefix,
        _stratified_indices,
        _write_contribution_store,
    )

from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    plan_path = run_dir / "partitions" / args.partition_name
    plan = _load_plan(plan_path)
    expected_experts = 16 if args.topology.startswith("p16/") else 32
    expected_top_k = int(args.topology.split("/top", 1)[1])
    if plan.routed_experts != expected_experts:
        raise ValueError(f"{args.topology} expects {expected_experts} routed experts, got {plan.routed_experts}")
    if expected_top_k <= 0 or expected_top_k > plan.routed_experts:
        raise ValueError(f"invalid topology top-k: {args.topology}")
    dataset = ActivationShardDataset(
        run_dir / "capture/layer-0000-train.json",
        split="train",
        microbatch=args.microbatch,
    )
    if args.stratify_pool > dataset.count:
        raise ValueError(f"stratify_pool {args.stratify_pool} exceeds FIT count {dataset.count}")
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    prefix_inputs = _load_train_prefix(dataset, args.stratify_pool, args.microbatch)
    try:
        from scripts.run_exact_p16_oracle import _load_dense_mlp  # type: ignore
    except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
        from run_exact_p16_oracle import _load_dense_mlp  # type: ignore

    weights = _load_dense_mlp(source_dir, layer=0)
    residual_norms = _dense_residual_norms(
        prefix_inputs,
        weights,
        plan,
        device=device,
        batch_size=args.microbatch,
    )
    local = _stratified_indices(residual_norms, min(args.sample_count, len(prefix_inputs)))
    inputs = prefix_inputs[local]
    shared, routed, target, hardness = _contributions(
        inputs,
        weights,
        plan,
        device=device,
        batch_size=args.microbatch,
    )
    store_dir = Path(args.store_dir)
    store = _write_contribution_store(
        store_dir,
        shared,
        routed,
        target,
        dataset_hash=dataset.dataset_hash,
        partition_hash=_sha256(plan_path),
        indices_hash=_hash_indices(local.astype("int64")),
    )
    manifest_path = store_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "topology": args.topology,
            "top_k": expected_top_k,
            "sample_count": int(len(local)),
            "stratify_pool": int(len(prefix_inputs)),
            "hardness_min": float(hardness.min()),
            "hardness_max": float(hardness.max()),
            "code_commit": current_git_commit(),
            "hypothesis": (
                "A deterministic stratified FIT contribution store plus a bounded "
                "load-aware candidate pool can reveal whether this p32 product "
                "partition has sufficient routing capacity without holdout access."
            ),
            "falsifier": (
                "The bounded oracle remains below cosine 0.98 or cannot meet "
                "load CV <= 0.50 with dead experts = 0, or any provenance/split "
                "check indicates non-FIT data was opened."
            ),
            "budget": {
                "device": device,
                "microbatch": int(args.microbatch),
                "sample_count": int(len(local)),
                "stratify_pool": int(len(prefix_inputs)),
                "holdout_opened": False,
            },
            "decision_enabled": (
                "retain the topology as a research finalist and authorize a "
                "router/refinement follow-up only when bounded oracle evidence "
                "is near-target; never authorize holdout or replay from this step."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store["manifest_sha256"] = _sha256(manifest_path)
    output = run_dir / "reports" / args.report_name
    payload = {
        "schema_version": 1,
        "status": "FIT_CONTRIBUTION_STORE_MATERIALIZED",
        "classification": "FIT_ONLY_CONTRIBUTION_STORE_NO_HOLDOUT",
        "code_commit": current_git_commit(),
        "partition": str(plan_path),
        "partition_hash": _sha256(plan_path),
        "topology": args.topology,
        "dataset_hash": dataset.dataset_hash,
        "split": "train",
        "holdout_opened": False,
        "sample": {
            "count": int(len(local)),
            "stratify_pool": int(len(prefix_inputs)),
            "indices_sha256": _hash_indices(local.astype("int64")),
            "strata": {
                "easy": int(len(local) // 4),
                "medium": int(len(local) - 2 * (len(local) // 4)),
                "hard": int(len(local) // 4),
            },
        },
        "contribution_store": store,
        "hypothesis": manifest["hypothesis"],
        "falsifier": manifest["falsifier"],
        "budget": manifest["budget"],
        "decision_enabled": manifest["decision_enabled"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--partition-name", required=True)
    parser.add_argument("--topology", required=True, help="p16/top4, p32/top4, or p32/top5")
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--stratify-pool", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
