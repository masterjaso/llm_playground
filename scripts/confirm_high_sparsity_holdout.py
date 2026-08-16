"""Strictly reload and confirm the frozen high-sparsity finalists on holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset, _plan_from_path, _stream_metrics

try:
    from scripts.run_topk_architecture_search import _load_mlp
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_topk_architecture_search import _load_mlp


RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
SOURCE = Path("runs/20260815-030931-windows/source")


def _confirm_one(
    *,
    run: Path,
    source: Path,
    config_name: str,
    partition_name: str,
    checkpoint_dir: str,
    device: str,
    microbatch: int,
) -> dict[str, Any]:
    from safetensors.torch import load_file  # type: ignore

    profile = load_config(Path("configs") / f"{config_name}.yaml")
    profile.validate()
    checkpoint_path = run / checkpoint_dir / "layer-0000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    tensor_path = checkpoint_path.parent / str(checkpoint["tensor_file"])
    partition_path = run / "partitions" / partition_name
    plan = _plan_from_path(partition_path)
    dense = _load_mlp(source, 0)
    model = TorchQwen35SwiGLUMoE.from_dense(
        dense["gate_proj.weight"],
        dense["up_proj.weight"],
        dense["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    )
    raw_state = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {name[len(prefix) :]: value for name, value in raw_state.items() if name.startswith(prefix)}
    if len(state) != len(raw_state):
        raise ValueError(f"unexpected tensor namespace in {tensor_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict finalist reload failed: missing={missing}, unexpected={unexpected}")
    model.to(device)
    gate = torch.as_tensor(dense["gate_proj.weight"], dtype=torch.float32)
    up = torch.as_tensor(dense["up_proj.weight"], dtype=torch.float32)
    down = torch.as_tensor(dense["down_proj.weight"], dtype=torch.float32)
    holdout = ActivationShardDataset(run / "capture/layer-0000.json", split="holdout", microbatch=microbatch)
    metrics = _stream_metrics(model, holdout, gate=gate, up=up, down=down, microbatch=microbatch, device=device)
    return {
        "profile": config_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_code_commit": checkpoint.get("code_commit"),
        "evaluator_code_commit": current_git_commit(),
        "routing_mode": checkpoint.get("routing_mode"),
        "strict_reload": True,
        "partition": str(partition_path),
        "metrics": metrics,
        "checkpoint_quality_gate": checkpoint.get("quality_gate", {}),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    run = Path(args.run_dir)
    source = Path(args.source_dir)
    results = [
        _confirm_one(
            run=run,
            source=source,
            config_name="qwen38_p16s1_top4",
            partition_name="high-sparsity-p16-top4.json",
            checkpoint_dir="layer-checkpoints/high-sparsity-schedule-ablation/A_warm_joint",
            device=args.device,
            microbatch=args.microbatch,
        ),
        _confirm_one(
            run=run,
            source=source,
            config_name="qwen38_p16s1_top3",
            partition_name="high-sparsity-p16-top3.json",
            checkpoint_dir="layer-checkpoints/high-sparsity-finalists-train-dev/qwen38_p16s1_top3",
            device=args.device,
            microbatch=args.microbatch,
        ),
        _confirm_one(
            run=run,
            source=source,
            config_name="qwen38_p32s1_top6",
            partition_name="high-sparsity-p32-top6.json",
            checkpoint_dir="layer-checkpoints/high-sparsity-finalists-train-dev/qwen38_p32s1_top6",
            device=args.device,
            microbatch=args.microbatch,
        ),
    ]
    search = json.loads((run / "reports/high-sparsity-partition-search.json").read_text(encoding="utf-8"))
    search_rows = {(row["profile"], int(row["top_k"])): row for row in search["results"]}
    dev_training = {
        "qwen38_p16s1_top4": json.loads((run / "reports/p16-top4-schedule-ablations.json").read_text(encoding="utf-8"))["results"][0]["final_selection"],
        "qwen38_p16s1_top3": json.loads((run / "reports/high-sparsity-equal-compute-finalists.json").read_text(encoding="utf-8"))["results"][0]["final_selection"],
        "qwen38_p32s1_top6": json.loads((run / "reports/high-sparsity-equal-compute-finalists.json").read_text(encoding="utf-8"))["results"][1]["final_selection"],
    }
    for row in results:
        profile = "p16" if "p16" in row["profile"] else "p32"
        top_k = int(row["profile"].rsplit("top", 1)[1])
        oracle = search_rows[(profile, top_k)]
        trained = dev_training[row["profile"]]
        row["basis_investigation"] = {
            "frozen_positive_oracle_dev_cosine": oracle["cosine"],
            "trained_dev_cosine": trained["cosine"],
            "trained_dev_cosine_gain_over_oracle": float(trained["cosine"] - oracle["cosine"]),
            "holdout_cosine_gap_to_green_0.98": float(0.98 - row["metrics"]["cosine"]),
            "conclusion": "basis/expert adaptation improves angular fidelity but remains below the green cosine gate; do not start representative replay",
        }
    payload = {
        "schema_version": 1,
        "status": "HIGH_SPARSITY_FINALIST_HOLDOUT_CONFIRMATION_COMPLETE",
        "classification": "POST_SELECTION_FULL_HOLDOUT_CONFIRMATION",
        "holdout_manifest": str(run / "capture/layer-0000-holdout.json"),
        "holdout_tokens": results[0]["metrics"]["token_count"],
        "results": results,
        "selection_policy": "finalists were frozen from TRAIN/dev; this is the first full-holdout read for these trained sparse finalists",
        "replay_gate": {"status": "BLOCKED", "reason": "no >=70% candidate meets cosine >=0.98 with dead=0 and load_cv<=0.50"},
        "code_commit": current_git_commit(),
    }
    report = run / "reports/high-sparsity-finalist-holdout-confirmation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    args = parser.parse_args()
    payload = _run(args)
    print(json.dumps({"status": payload["status"], "results": [{"profile": row["profile"], "metrics": row["metrics"], "strict_reload": row["strict_reload"]} for row in payload["results"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
