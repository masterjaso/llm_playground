"""Run a bounded fresh-FIT basis continuation for p16/top4 or p32/top5.

The continuation deliberately freezes the selector and updates only the
shared/routed FFN basis (including learned expert scales).  It consumes a
deterministic subset of fresh FIT rows, excludes frozen validation-A/B rows,
and writes a normal layer checkpoint plus before/after telemetry.  This is a
short falsifiable pilot, not a replacement for a full production refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.checkpoint.layer import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from dense2moe.config import load_config
from dense2moe.partition.contributions import (
    load_partition_plan,
    resolve_checkpoint_tensor,
    sha256_file,
)
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset

DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")


def _hash_indices(indices: list[int] | np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(int(value)) for value in indices).encode()).hexdigest()


def _load_ab(path: Path, count: int) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_SELECTOR_AB_FROZEN" or payload.get("historical_holdout_opened") is not False:
        raise ValueError("fresh A/B receipt is not a closed, frozen validation receipt")
    a = sorted({int(value) for value in payload["validation_a"]["indices"]})
    b = sorted({int(value) for value in payload["validation_b"]["indices"]})
    if len(a) != int(payload["validation_a"]["count"]) or len(b) != int(payload["validation_b"]["count"]):
        raise ValueError("fresh A/B receipt contains duplicate rows")
    if set(a) & set(b) or any(index < 0 or index >= count for index in (*a, *b)):
        raise ValueError("fresh A/B rows overlap or are outside the train capture")
    if _hash_indices(a) != str(payload["validation_a"]["indices_hash"]):
        raise ValueError("fresh validation-A identity hash mismatch")
    if _hash_indices(b) != str(payload["validation_b"]["indices_hash"]):
        raise ValueError("fresh validation-B identity hash mismatch")
    return a, b


def _even_sample(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if count > len(values):
        raise ValueError(f"sample count {count} exceeds available rows {len(values)}")
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return np.asarray(values[positions], dtype=np.int64)


def _metrics(model: Any, dataset: ActivationShardDataset, indices: np.ndarray, weights: dict[str, Any], *, device: str, microbatch: int) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    error_sum = 0.0
    target_norm_sum = 0.0
    cosine_sum = 0.0
    count = 0
    usage = np.zeros(model.routed_experts, dtype=np.int64)
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(indices.tolist(), microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            target = (torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T
            prediction, info = model(inputs, return_router=True)
            prediction = prediction.reshape(-1, prediction.shape[-1])
            target = target.reshape(-1, target.shape[-1])
            delta = prediction - target
            error_sum += float(torch.sum(delta * delta).cpu().item())
            target_norm_sum += float(torch.sum(target * target).cpu().item())
            cosine_sum += float(
                torch.sum(
                    torch.sum(prediction * target, dim=-1)
                    / (torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12)
                ).cpu().item()
            )
            count += int(target.shape[0])
            usage += np.bincount(info["indices"].reshape(-1).detach().cpu().numpy(), minlength=model.routed_experts)
    return {
        "tokens": count,
        "global_nmse": error_sum / max(target_norm_sum, 1e-12),
        "cosine": cosine_sum / max(count, 1),
        "load_cv": float(usage.std() / max(usage.mean(), 1e-12)),
        "dead_experts": int(np.sum(usage == 0)),
        "expert_usage_counts": usage.tolist(),
    }


def _load_model(source_weights: dict[str, Any], profile: Any, plan: Any, checkpoint: Path, device: str) -> Any:
    from safetensors.torch import load_file  # type: ignore

    from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE

    model = TorchQwen35SwiGLUMoE.from_dense(
        source_weights["gate_proj.weight"],
        source_weights["up_proj.weight"],
        source_weights["down_proj.weight"],
        routed_experts=plan.routed_experts,
        shared_intermediate_size=plan.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    ).to(device)
    tensor_path = resolve_checkpoint_tensor(checkpoint)
    raw = load_file(str(tensor_path), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix) :]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise ValueError("selector checkpoint tensor namespace mismatch")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    topology = str(args.topology)
    profile_name = "qwen38_p16s1_top4.yaml" if topology == "p16/top4" else "qwen38_p32s1_top5.yaml"
    profile = load_config(Path("configs") / profile_name)
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir)
    fresh_dir = Path(args.fresh_dir)
    activation_manifest = Path(args.activation_manifest) if args.activation_manifest else fresh_dir / "capture/layer-0000-train.json"
    ab_path = Path(args.validation_ab) if args.validation_ab else fresh_dir / "capture/fresh-selector-validation-ab.json"
    dataset = ActivationShardDataset(activation_manifest, split="train", microbatch=args.microbatch)
    a_indices, b_indices = _load_ab(ab_path, dataset.count)
    excluded = np.asarray(sorted(set(a_indices) | set(b_indices)), dtype=np.int64)
    available = np.setdiff1d(np.arange(dataset.count, dtype=np.int64), excluded, assume_unique=True)
    fit_indices = _even_sample(available, args.fit_rows)
    a_sample = _even_sample(np.asarray(a_indices, dtype=np.int64), min(args.validation_rows, len(a_indices)))
    plan = load_partition_plan(args.partition)

    from scripts.run_exact_p16_oracle import _dense_hidden_target, _load_dense_mlp

    source_weights = _load_dense_mlp(source_dir)
    model = _load_model(source_weights, profile, plan, Path(args.checkpoint), args.device)
    before = _metrics(model, dataset, a_sample, source_weights, device=args.device, microbatch=args.microbatch)

    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_modules = (
        model.shared_gate_proj,
        model.shared_up_proj,
        model.shared_down_proj,
        *model.expert_gate_proj,
        *model.expert_up_proj,
        *model.expert_down_proj,
    )
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.expert_scales.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=1e-5)
    gate = source_weights["gate_proj.weight"].to(args.device)
    up = source_weights["up_proj.weight"].to(args.device)
    down = source_weights["down_proj.weight"].to(args.device)
    losses: list[float] = []
    model.train()
    for _epoch in range(args.epochs):
        for values in dataset.iter_selected_batches(fit_indices.tolist(), args.microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=args.device)
            _hidden, target = _dense_hidden_target(
                inputs,
                {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
                torch.device(args.device),
            )
            prediction, _info = model(inputs, return_router=True)
            with torch.no_grad():
                shared = model.shared_down_proj(torch.nn.functional.silu(model.shared_gate_proj(inputs)) * model.shared_up_proj(inputs))
                residual = torch.linalg.vector_norm(target - shared, dim=-1)
                weights = (1.0 + float(args.hard_weight) * residual / residual.mean().clamp_min(1e-6)).clamp_max(8.0)
            delta = prediction - target
            mse = torch.mean(delta.square(), dim=-1)
            cosine = 1.0 - torch.sum(prediction * target, dim=-1) / (
                torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(target, dim=-1) + 1e-12
            )
            loss = (weights * mse).sum() / weights.sum().clamp_min(1e-6) + float(args.cosine_weight) * (weights * cosine).sum() / weights.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    model.eval()
    after = _metrics(model, dataset, a_sample, source_weights, device=args.device, microbatch=args.microbatch)

    output_dir = run_dir / "layer-checkpoints" / "fresh-basis" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.layers.0.{name}": value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensors, output_dir / "layer-0000.safetensors")
    partition_hash = sha256_file(args.partition)
    metadata = LayerCheckpoint(
        layer=0,
        profile=profile.name,
        status="RESEARCH_CANDIDATE",
        profile_hash=profile_fingerprint(profile.as_dict()),
        source_revision=profile.revision,
        source_config_hash=sha256_file(source_dir / "config.json"),
        source_index_hash=sha256_file(source_dir / "model.safetensors.index.json"),
        dataset_hash=dataset.dataset_hash,
        partition_strategy="artifact",
        partition_hash=partition_hash,
        router_architecture=f"torch-linear-topk-{profile.routing_mode}-frozen-selector-bounded-basis-v1",
        routing_mode=profile.routing_mode,
        training_seed=args.seed,
        training_config={
            "epochs": args.epochs,
            "microbatch": args.microbatch,
            "learning_rate": args.learning_rate,
            "fit_rows": len(fit_indices),
            "fit_indices_hash": _hash_indices(fit_indices),
            "validation_a_rows": len(a_indices),
            "validation_a_indices_hash": _hash_indices(a_indices),
            "validation_b_rows": len(b_indices),
            "validation_b_indices_hash": _hash_indices(b_indices),
            "basis_parameters_only": True,
            "holdout_opened": False,
        },
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_hash,
        tensor_inventory=inventory,
        train_metrics={"loss_mean": float(np.mean(losses)) if losses else None},
        holdout_metrics={"status": "CLOSED", "opened": False},
        router_metrics={"selector_frozen": True},
        quality_gate={"overall": "research-candidate", "evaluation_scope": "fresh_validation_a", "metrics": after},
        code_commit=current_git_commit(),
    )
    save_layer_checkpoint(metadata, output_dir / "layer-0000.json")
    report = {
        "schema_version": 1,
        "status": "BOUNDED_BASIS_REFINEMENT_COMPLETE",
        "classification": "FRESH_FIT_SUBSET_BASIS_ONLY_CONTINUATION_A_SELECTION_B_CLOSED",
        "topology": topology,
        "basis_source": "trained_checkpoint",
        "initial_checkpoint": str(resolve_checkpoint_tensor(args.checkpoint)),
        "initial_checkpoint_tensor_sha256": sha256_file(resolve_checkpoint_tensor(args.checkpoint)),
        "output_checkpoint": str(tensor_path),
        "output_checkpoint_tensor_sha256": tensor_hash,
        "partition": str(args.partition),
        "partition_sha256": partition_hash,
        "dataset_hash": dataset.dataset_hash,
        "source_revision": profile.revision,
        "split": {
            "fit_rows_total": len(available),
            "fit_rows_used": len(fit_indices),
            "fit_indices_sha256": _hash_indices(fit_indices),
            "validation_a_rows": len(a_indices),
            "validation_a_indices_sha256": _hash_indices(a_indices),
            "validation_b_rows": len(b_indices),
            "validation_b_indices_sha256": _hash_indices(b_indices),
            "fit_a_overlap": 0,
            "fit_b_overlap": 0,
            "a_b_overlap": 0,
        },
        "budget": {
            "epochs": args.epochs,
            "microbatch": args.microbatch,
            "fit_rows": args.fit_rows,
            "validation_rows_measured": len(a_sample),
            "learning_rate": args.learning_rate,
            "hard_weight": args.hard_weight,
            "cosine_weight": args.cosine_weight,
            "device": args.device,
        },
        "selector_frozen": True,
        "before_validation_a": before,
        "after_validation_a": after,
        "loss_mean": float(np.mean(losses)) if losses else None,
        "updates": len(losses),
        "refinement_state": "RESEARCH_CONTINUATION_COMPLETE_BOUNDED_PILOT",
        "holdout_opened": False,
        "representative_replay_started": False,
        "full64_replay_started": False,
        "code_commit": current_git_commit(),
    }
    report_path = run_dir / "reports" / args.report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_RUN / "fresh-selector-layer0")
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--validation-ab", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--fit-rows", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.fit_rows <= 0 or args.validation_rows <= 0 or args.epochs < 0:
        raise ValueError("fit/validation rows must be positive and epochs non-negative")
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "topology": payload["topology"],
                "before_validation_a": payload["before_validation_a"],
                "after_validation_a": payload["after_validation_a"],
                "output_checkpoint_tensor_sha256": payload["output_checkpoint_tensor_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
