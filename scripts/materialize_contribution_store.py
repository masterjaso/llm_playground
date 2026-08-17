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

import numpy as np

try:
    from scripts.calibrate_streaming_solver import (
        _contributions,
        _dense_residual_norms,
        _hash_indices,
        _load_plan,
        _stratified_indices,
        _write_contribution_store,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from calibrate_streaming_solver import (  # type: ignore
        _contributions,
        _dense_residual_norms,
        _hash_indices,
        _load_plan,
        _stratified_indices,
        _write_contribution_store,
    )

from dense2moe.partition.contributions import (
    contribution_manifest,
    trained_checkpoint_contributions,
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
    activation_manifest = Path(args.activation_manifest) if args.activation_manifest else run_dir / "capture/layer-0000-train.json"
    dataset = ActivationShardDataset(
        activation_manifest,
        split="train",
        microbatch=args.microbatch,
    )
    if args.validation_ab:
        ab = json.loads(Path(args.validation_ab).read_text(encoding="utf-8"))
        if ab.get("historical_holdout_opened") is not False:
            raise ValueError("validation-A/B receipt does not prove holdout closure")
        split_key = "validation_a" if args.validation_split == "validation_a" else "validation_b"
        selected_global = np.asarray(sorted({int(value) for value in ab[split_key]["indices"]}), dtype="int64")
        expected_count = int(ab[split_key]["count"])
        if len(selected_global) != expected_count or np.any(selected_global < 0) or np.any(selected_global >= dataset.count):
            raise ValueError(f"{split_key} indices are invalid for activation manifest")
        if args.stratify_pool > len(selected_global):
            raise ValueError(f"stratify_pool {args.stratify_pool} exceeds {split_key} count {len(selected_global)}")
        pool_global = selected_global
        split_identity = {
            "name": split_key,
            "indices_sha256": _hash_indices(selected_global),
            "source_receipt": str(args.validation_ab),
        }
    else:
        if args.stratify_pool > dataset.count:
            raise ValueError(f"stratify_pool {args.stratify_pool} exceeds FIT count {dataset.count}")
        pool_global = np.arange(args.stratify_pool, dtype=np.int64)
        split_identity = {"name": "train", "indices_sha256": None, "source_receipt": str(activation_manifest)}
    if args.stratify_pool > len(pool_global):
        raise ValueError(f"stratify_pool {args.stratify_pool} exceeds FIT count {dataset.count}")
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    prefix_batches = list(dataset.iter_selected_batches(pool_global[: args.stratify_pool].tolist(), args.microbatch))
    if not prefix_batches:
        raise ValueError("selected activation split is empty")
    prefix_inputs = np.concatenate(
        [batch.float().cpu().numpy() if hasattr(batch, "float") else np.asarray(batch, dtype=np.float32) for batch in prefix_batches],
        axis=0,
    )
    try:
        from scripts.run_exact_p16_oracle import (  # type: ignore
            _dense_hidden_target,
            _load_dense_mlp,
        )
    except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
        from run_exact_p16_oracle import _dense_hidden_target, _load_dense_mlp  # type: ignore

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
    if args.checkpoint:
        # The dense teacher is used only as the reconstruction target.  The
        # stored shared/routed arrays below come exclusively from the supplied
        # trained checkpoint; raw partition reconstruction is never a fallback.
        device_obj = torch.device(device)
        gate = weights["gate_proj.weight"].to(device_obj)
        up = weights["up_proj.weight"].to(device_obj)
        down = weights["down_proj.weight"].to(device_obj)
        target_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), args.microbatch):
                batch = torch.as_tensor(inputs[start : start + args.microbatch], dtype=torch.float32, device=device_obj)
                _hidden, target_batch = _dense_hidden_target(
                    batch,
                    {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                    device_obj,
                )
                target_parts.append(target_batch.cpu().numpy().astype(np.float32, copy=False))
        target = np.concatenate(target_parts, axis=0)
        shared, routed, checkpoint_metadata = trained_checkpoint_contributions(
            inputs,
            args.checkpoint,
            plan,
            batch_size=args.microbatch,
        )
        hardness = np.linalg.norm(target - np.asarray(shared), axis=1)
        basis_source = "trained_checkpoint"
    else:
        shared, routed, target, hardness = _contributions(
            inputs, weights, plan, device=device, batch_size=args.microbatch
        )
        checkpoint_metadata = {}
        basis_source = "raw_dense_partition"
    selected_global_indices = pool_global[: args.stratify_pool][local]
    store_dir = Path(args.store_dir)
    store = _write_contribution_store(
        store_dir,
        shared,
        routed,
        target,
        dataset_hash=dataset.dataset_hash,
        partition_hash=_sha256(plan_path),
        indices_hash=_hash_indices(selected_global_indices),
        basis_source=basis_source,
        checkpoint_path=checkpoint_metadata.get("checkpoint_path") if args.checkpoint else None,
        checkpoint_tensor_sha256=checkpoint_metadata.get("checkpoint_tensor_sha256") if args.checkpoint else None,
        partition_path=str(plan_path),
        topology={
            "expert_count": int(plan.routed_experts),
            "expert_width": int(plan.expert_intermediate_size),
            "shared_width": int(plan.shared_intermediate_size),
            "top_k": expected_top_k,
        },
        source_revision=checkpoint_metadata.get("source_revision"),
        capture_identity={"manifest": str(activation_manifest), "split": split_identity},
        split=split_identity["name"],
        row_count=len(selected_global_indices),
        dtype=str(np.asarray(shared).dtype),
    )
    manifest_path = store_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "basis_source": basis_source,
            "checkpoint_path": checkpoint_metadata.get("checkpoint_path") if args.checkpoint else None,
            "checkpoint_tensor_sha256": checkpoint_metadata.get("checkpoint_tensor_sha256") if args.checkpoint else None,
            "partition_path": str(plan_path),
            "partition_sha256": _sha256(plan_path),
            "partition_canonical_sha256": contribution_manifest(
                basis_source=basis_source,
                plan=plan,
                row_count=len(selected_global_indices),
                split=split_identity["name"],
                dataset_hash=dataset.dataset_hash,
                partition_path=plan_path,
                top_k=expected_top_k,
                checkpoint=args.checkpoint,
                source_revision=checkpoint_metadata.get("source_revision"),
                capture_identity={"manifest": str(activation_manifest), "split": split_identity},
                dtype=str(np.asarray(shared).dtype),
                code_commit=current_git_commit(),
            )["partition_canonical_sha256"],
            "topology_manifest": {
                "expert_count": int(plan.routed_experts),
                "expert_width": int(plan.expert_intermediate_size),
                "shared_width": int(plan.shared_intermediate_size),
                "top_k": expected_top_k,
            },
            "source_revision": checkpoint_metadata.get("source_revision"),
            "capture_identity": {"manifest": str(activation_manifest), "split": split_identity},
            "split_identity": split_identity,
            "row_count": len(selected_global_indices),
            "dtype": str(np.asarray(shared).dtype),
            "selected_global_indices_sha256": _hash_indices(selected_global_indices),
            "topology": args.topology,
            "top_k": expected_top_k,
            "sample_count": len(local),
            "stratify_pool": len(prefix_inputs),
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
                "sample_count": len(local),
                "stratify_pool": len(prefix_inputs),
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
        "split": split_identity["name"],
        "basis_source": basis_source,
        "checkpoint_path": manifest.get("checkpoint_path"),
        "checkpoint_tensor_sha256": manifest.get("checkpoint_tensor_sha256"),
        "topology_manifest": manifest["topology_manifest"],
        "holdout_opened": False,
        "sample": {
            "count": len(local),
            "stratify_pool": len(prefix_inputs),
            "indices_sha256": _hash_indices(selected_global_indices),
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="trained MoE checkpoint directory/file; selects actual checkpoint basis contributions",
    )
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--validation-ab", type=Path, default=None, help="frozen fresh A/B receipt")
    parser.add_argument("--validation-split", choices=("validation_a", "validation_b"), default="validation_a")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--stratify-pool", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
