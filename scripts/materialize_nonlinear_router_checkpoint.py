"""Convert a linear layer checkpoint to a strict low-rank-SiLU router artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dense2moe.checkpoint.layer import publish_tensor_artifact
from dense2moe.config import load_config
from dense2moe.provenance import current_git_commit

try:
    from scripts.run_exact_p16_oracle import _load_dense_mlp, _load_deployed_checkpoint, _load_plan
    from scripts.train_nonlinear_listwise_router import _warm_start_nonlinear
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import (  # type: ignore
        _load_dense_mlp,
        _load_deployed_checkpoint,
        _load_plan,
    )
    from train_nonlinear_listwise_router import _warm_start_nonlinear  # type: ignore


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--linear-checkpoint", default="p16-top4-refined-course-correction-continue")
    parser.add_argument("--output-name", default="p16-top4-nonlinear-hidden512-init")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    plan = _load_plan(run_dir / "partitions/high-sparsity-p16-top4.json")
    weights = _load_dense_mlp(source_dir)
    linear, source_metadata = _load_deployed_checkpoint(
        weights,
        profile,
        plan,
        run_dir / "layer-checkpoints/clean-validation" / args.linear_checkpoint,
        args.device,
    )
    nonlinear = _warm_start_nonlinear(linear, profile, plan, weights, args.device, args.hidden_size)
    output_dir = run_dir / "layer-checkpoints/clean-validation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in nonlinear.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, output_dir / "layer-0000.safetensors")
    metadata = dict(source_metadata)
    metadata.update(
        {
            "status": "TRAINED_DEV_SELECTED",
            "router_architecture": "torch-low-rank-silu-topk-independent_positive-v1",
            "tensor_file": tensor_path.name,
            "tensor_sha256": tensor_hash,
            "tensor_inventory": inventory,
            "code_commit": current_git_commit(),
            "training_config": dict(source_metadata.get("training_config", {})) | {"router_hidden_size": args.hidden_size, "router_initialization": "svd_warm_start_from_linear_checkpoint", "initial_linear_checkpoint": args.linear_checkpoint},
            "quality_gate": {"overall": "research-candidate", "evaluation_scope": "validation", "metrics": {"status": "materialized_initialization_only"}},
        }
    )
    (output_dir / "layer-0000.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "NONLINEAR_ROUTER_CHECKPOINT_MATERIALIZED", "code_commit": current_git_commit(), "output_dir": str(output_dir), "tensor_sha256": tensor_hash, "hidden_size": args.hidden_size}, indent=2), flush=True)


if __name__ == "__main__":
    main()
