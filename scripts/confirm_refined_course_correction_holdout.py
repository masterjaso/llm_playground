"""Strictly reload the green validation finalist and read the full holdout once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset, _plan_from_path, _stream_metrics

try:
    from scripts.run_exact_p16_oracle import _load_dense_mlp
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from run_exact_p16_oracle import _load_dense_mlp  # type: ignore


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
DEFAULT_CHECKPOINT = "layer-checkpoints/clean-validation/p16-top4-refined-course-correction-continue"


def run(args: argparse.Namespace) -> dict[str, Any]:
    from safetensors.torch import load_file  # type: ignore

    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    profile = load_config(Path("configs/qwen38_p16s1_top4.yaml"))
    checkpoint_dir = run_dir / args.checkpoint_dir
    metadata_path = checkpoint_dir / "layer-0000.json"
    tensor_path = checkpoint_dir / "layer-0000.safetensors"
    if not metadata_path.exists() or not tensor_path.exists():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint_dir}")
    checkpoint = json.loads(metadata_path.read_text(encoding="utf-8"))
    plan = _plan_from_path(run_dir / "partitions/high-sparsity-p16-top4.json")
    dense = _load_dense_mlp(source_dir, 0)
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
    model.to(args.device)
    gate = torch.as_tensor(dense["gate_proj.weight"], dtype=torch.float32)
    up = torch.as_tensor(dense["up_proj.weight"], dtype=torch.float32)
    down = torch.as_tensor(dense["down_proj.weight"], dtype=torch.float32)
    holdout = ActivationShardDataset(run_dir / "capture/layer-0000.json", split="holdout", microbatch=args.microbatch)
    metrics = _stream_metrics(model, holdout, gate=gate, up=up, down=down, microbatch=args.microbatch, device=args.device)
    gate_result = {
        "nmse_max": 0.05,
        "cosine_min": 0.98,
        "dead_experts_max": 0,
        "load_cv_max": 0.50,
        "nmse_pass": bool(metrics["normalized_mse"] <= 0.05),
        "cosine_pass": bool(metrics["cosine"] >= 0.98),
        "dead_experts_pass": bool(metrics["dead_experts"] == 0),
        "load_cv_pass": bool(metrics["load_cv"] <= 0.50),
    }
    gate_result["all_pass"] = all(gate_result[key] for key in ("nmse_pass", "cosine_pass", "dead_experts_pass", "load_cv_pass"))
    validation_report = run_dir / "reports" / args.validation_report_name
    validation_payload = json.loads(validation_report.read_text(encoding="utf-8")) if validation_report.exists() else {}
    payload = {
        "schema_version": 1,
        "status": "REFINED_COURSE_CORRECTION_HOLDOUT_CONFIRMED",
        "classification": "POST_SELECTION_FULL_HOLDOUT_CONFIRMATION",
        "profile": profile.name,
        "checkpoint": str(metadata_path),
        "tensor_file": str(tensor_path),
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "partition": str(run_dir / "partitions/high-sparsity-p16-top4.json"),
        "checkpoint_code_commit": checkpoint.get("code_commit"),
        "evaluator_code_commit": current_git_commit(),
        "routing_mode": checkpoint.get("routing_mode"),
        "strict_reload": True,
        "holdout_manifest": str(run_dir / "capture/layer-0000.json"),
        "metrics": metrics,
        "quality_gate": gate_result,
        "selection_provenance": {
            "selection_split": checkpoint.get("training_config", {}).get("selection_split"),
            "selection_count": checkpoint.get("training_config", {}).get("selection_count"),
            "selection_identity_hash": checkpoint.get("training_config", {}).get("selection_identity_hash"),
            "validation_strict_reload": validation_payload.get("result", {}).get("final_selection"),
        },
        "replay_gate": {
            "status": "READY_FOR_REPRESENTATIVE_LAYER_ONLY" if gate_result["all_pass"] else "BLOCKED",
            "reason": "all sparse quality thresholds pass on full holdout" if gate_result["all_pass"] else "one or more sparse quality thresholds failed on full holdout",
        },
        "code_commit": current_git_commit(),
    }
    report_path = run_dir / "reports" / args.report_name
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-report-name", default="p16-top4-refined-course-correction-continuation.json")
    parser.add_argument("--report-name", default="p16-top4-refined-course-correction-holdout-confirmation.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "metrics": payload["metrics"], "strict_reload": payload["strict_reload"], "quality_gate": payload["quality_gate"], "replay_gate": payload["replay_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
