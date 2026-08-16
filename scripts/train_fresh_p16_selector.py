"""Train a selector-only p16/top4 continuation on the fresh layer-0 capture.

This driver is deliberately development-only: it opens the fresh train
activation shards, excludes frozen validation-A and validation-B rows from
optimizer updates, selects checkpoints on A, and evaluates B only after the
A-selected checkpoint has been restored.  It never opens the historical
holdout manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_config
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.provenance import current_git_commit
from dense2moe.training.torch_distill import ActivationShardDataset, train_torch_layer


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")
DEFAULT_SOURCE = Path("runs/20260815-030931-windows/source")
DEFAULT_FRESH = DEFAULT_RUN / "fresh-selector-layer0"


def _hash_indices(indices: list[int]) -> str:
    return hashlib.sha256("\n".join(str(int(value)) for value in indices).encode()).hexdigest()


def _load_ab(path: Path, count: int) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FRESH_SELECTOR_AB_FROZEN":
        raise ValueError(f"unexpected fresh A/B status: {payload.get('status')!r}")
    if payload.get("historical_holdout_opened") is not False or payload.get("selection_union", {}).get("enabled") is not False:
        raise ValueError("fresh A/B receipt does not prove holdout closure and A-only selection")
    a = sorted({int(value) for value in payload["validation_a"]["indices"]})
    b = sorted({int(value) for value in payload["validation_b"]["indices"]})
    if len(a) != int(payload["validation_a"]["count"]) or len(b) != int(payload["validation_b"]["count"]):
        raise ValueError("fresh A/B receipt contains duplicate or missing rows")
    if set(a) & set(b) or any(index < 0 or index >= count for index in (*a, *b)):
        raise ValueError("fresh validation-A/B rows are not disjoint and in range")
    if _hash_indices(a) != str(payload["validation_a"]["indices_hash"]):
        raise ValueError("fresh validation-A identity hash mismatch")
    if _hash_indices(b) != str(payload["validation_b"]["indices_hash"]):
        raise ValueError("fresh validation-B identity hash mismatch")
    return a, b


def _load_dense_mlp(source: Path, layer: int = 0) -> dict[str, Any]:
    from safetensors import safe_open  # type: ignore

    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    if set(names) != {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
        raise ValueError(f"layer {layer} MLP inventory mismatch: {sorted(names)}")
    values: dict[str, Any] = {}
    for name, shard in names.items():
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            values[name] = handle.get_tensor(prefix + name).float()
    return values


def _load_checkpoint_model(
    source_weights: dict[str, Any],
    profile: Any,
    partition_path: Path,
    checkpoint_dir: Path,
    device: str,
) -> TorchQwen35SwiGLUMoE:
    from safetensors.torch import load_file  # type: ignore

    partition_payload = json.loads(partition_path.read_text(encoding="utf-8"))
    partition_payload = partition_payload.get("plan", partition_payload)
    from dense2moe.partition import PartitionPlan

    plan = PartitionPlan(
        int(partition_payload["dense_intermediate_size"]),
        int(partition_payload["routed_experts"]),
        int(partition_payload["expert_intermediate_size"]),
        int(partition_payload["shared_intermediate_size"]),
        tuple(int(value) for value in partition_payload["shared_indices"]),
        tuple(tuple(int(value) for value in group) for group in partition_payload["expert_indices"]),
    )
    plan.validate()
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
    raw = load_file(str(checkpoint_dir / "layer-0000.safetensors"), device="cpu")
    prefix = "model.layers.0."
    state = {key[len(prefix) :]: value for key, value in raw.items() if key.startswith(prefix)}
    if len(state) != len(raw):
        raise ValueError("selector checkpoint tensor namespace mismatch")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"selector checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _basis_hash(checkpoint_dir: Path) -> str:
    from safetensors.torch import load_file  # type: ignore

    state = load_file(str(checkpoint_dir / "layer-0000.safetensors"), device="cpu")
    ignored = ("router.", "amplitude_router.")
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.endswith(".router.weight") or any(f".{prefix}" in name for prefix in ignored):
            continue
        if name.endswith(".router.bias"):
            continue
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _route_metrics(
    model: TorchQwen35SwiGLUMoE,
    dataset: ActivationShardDataset,
    indices: list[int],
    weights: dict[str, Any],
    *,
    microbatch: int,
    device: str,
) -> dict[str, Any]:
    import torch

    gate = weights["gate_proj.weight"].to(device)
    up = weights["up_proj.weight"].to(device)
    down = weights["down_proj.weight"].to(device)
    overlap_sum = 0.0
    exact_count = 0
    jaccard_sum = 0.0
    student_usage = np.zeros(model.routed_experts, dtype=np.int64)
    oracle_usage = np.zeros(model.routed_experts, dtype=np.int64)
    token_count = 0
    with torch.inference_mode():
        for values in dataset.iter_selected_batches(indices, microbatch):
            inputs = torch.as_tensor(values, dtype=torch.float32, device=device)
            target = (torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T
            _prediction, info = model(inputs, return_router=True, return_contributions=True)
            student_ids = info["indices"].reshape(-1, model.top_k)
            residual = target - info["shared"].reshape(-1, target.shape[-1])
            contributions = info["contributions"].reshape(-1, model.routed_experts, target.shape[-1])
            norms = torch.linalg.vector_norm(contributions, dim=-1)
            scores = torch.sum(contributions * residual.unsqueeze(1), dim=-1) / (norms + 1e-12)
            oracle_ids = torch.topk(scores, model.top_k, dim=-1).indices
            for student, oracle in zip(student_ids.detach().cpu().numpy(), oracle_ids.detach().cpu().numpy()):
                student_set = set(int(value) for value in student)
                oracle_set = set(int(value) for value in oracle)
                overlap = len(student_set & oracle_set)
                overlap_sum += overlap / model.top_k
                exact_count += int(overlap == model.top_k)
                jaccard_sum += overlap / max(len(student_set | oracle_set), 1)
            student_usage += np.bincount(student_ids.detach().cpu().reshape(-1).numpy(), minlength=model.routed_experts)
            oracle_usage += np.bincount(oracle_ids.detach().cpu().reshape(-1).numpy(), minlength=model.routed_experts)
            token_count += int(student_ids.shape[0])
    if token_count <= 0:
        raise ValueError("route metric split is empty")
    return {
        "token_count": token_count,
        "oracle_route_recall": overlap_sum / token_count,
        "oracle_exact_set_match": exact_count / token_count,
        "oracle_mean_jaccard": jaccard_sum / token_count,
        "student_selected_counts": student_usage.tolist(),
        "student_load_cv": float(student_usage.std() / max(student_usage.mean(), 1e-12)),
        "student_dead_experts": int(np.sum(student_usage == 0)),
        "oracle_selected_counts": oracle_usage.tolist(),
        "oracle_load_cv": float(oracle_usage.std() / max(oracle_usage.mean(), 1e-12)),
        "oracle_dead_experts": int(np.sum(oracle_usage == 0)),
        "oracle_definition": "top-k residual-correlation scores from frozen routed contributions; diagnostic only",
    }


class _Heartbeat:
    def __init__(self, path: Path, *, command: str) -> None:
        self.path = path
        self.command = command
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "command": self.command,
                        "timestamp": time.time(),
                        "code_commit": current_git_commit(),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self.stop.wait(30.0)

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    fresh_dir = Path(args.fresh_dir)
    # The fresh capture intentionally has no aggregate train/holdout wrapper:
    # this selector-only run consumes the explicit train manifest and keeps
    # the separately frozen A/B receipt out of the activation loader.
    activation_manifest = fresh_dir / "capture/layer-0000-train.json"
    data_plan_path = fresh_dir / "capture/fresh-selector-data-plan.json"
    ab_path = fresh_dir / "capture/fresh-selector-validation-ab.json"
    data_plan = json.loads(data_plan_path.read_text(encoding="utf-8"))
    if data_plan.get("status") != "CALIBRATION_READY":
        raise ValueError(f"fresh data plan is not ready: {data_plan.get('status')!r}")
    dataset = ActivationShardDataset(activation_manifest, split="train", microbatch=args.microbatch)
    if dataset.count != int(data_plan["train_tokens"]):
        raise ValueError("fresh train capture count disagrees with data plan")
    a_indices, b_indices = _load_ab(ab_path, dataset.count)
    fit_exclude = sorted(set(a_indices) | set(b_indices))
    profile = load_config("configs/qwen38_p16s1_top4.yaml")
    partition_path = run_dir / "partitions/high-sparsity-p16-top4.json"
    basis_dir = run_dir / "layer-checkpoints/clean-validation/p16-top4-refined-course-correction-continue"
    output_dir = run_dir / "layer-checkpoints/fresh-selector/p16-top4-hard-regret-bce"
    code_commit = current_git_commit()
    command_name = "fresh-p16-selector-hard-regret-bce"
    heartbeat = Path(args.heartbeat)
    stage_schedule = [
        {
            "name": "fresh_regret_bce_hard_dispatch",
            "epochs": int(args.epochs),
            "train_scales": False,
            "train_experts": False,
            "train_shared": False,
            "train_selection_router": True,
            "train_amplitude_router": True,
            "use_oracle_targets": True,
            "oracle_target_mode": "residual_correlation",
            "oracle_loss_mode": "multilabel_bce",
            "oracle_amplitude_mode": "student_selected",
            "learning_rates": {"selection_router": args.learning_rate, "amplitude_router": args.learning_rate},
            "loss_coefficients": {
                "mse": 1.0,
                "cosine": 0.05,
                "load_balance": 0.02,
                "hard_load_balance": args.hard_load_weight,
                "router_z_loss": 0.001,
                "oracle": args.oracle_weight,
                "oracle_amplitude": args.oracle_amplitude_weight,
            },
            "oracle_regret_weight": args.oracle_regret_weight,
            "expert_use_prices": [0.0] * profile.routed_experts,
        }
    ]
    with _Heartbeat(heartbeat, command=command_name):
        result = train_torch_layer(
            source_dir=args.source_dir,
            activation_manifest=activation_manifest,
            output_dir=output_dir,
            layer=0,
            profile=profile,
            partition_path=partition_path,
            epochs=int(args.epochs),
            microbatch=int(args.microbatch),
            learning_rate=float(args.learning_rate),
            device=args.device,
            seed=int(args.seed),
            source_revision=profile.revision,
            code_commit=code_commit,
            stage_schedule=stage_schedule,
            selection_indices=a_indices,
            fit_exclude_indices=fit_exclude,
            selection_identity_hash=str(json.loads(ab_path.read_text(encoding="utf-8"))["validation_a"]["indices_hash"]),
            validation_b_indices=b_indices,
            validation_b_identity_hash=str(json.loads(ab_path.read_text(encoding="utf-8"))["validation_b"]["indices_hash"]),
            evaluate_holdout=False,
            initial_checkpoint_dir=basis_dir,
        )
        source_weights = _load_dense_mlp(Path(args.source_dir))
        model = _load_checkpoint_model(source_weights, profile, partition_path, output_dir, args.device)
        route_a = _route_metrics(model, dataset, a_indices, source_weights, microbatch=args.microbatch, device=args.device)
        route_b = _route_metrics(model, dataset, b_indices, source_weights, microbatch=args.microbatch, device=args.device)
    basis_before = _basis_hash(basis_dir)
    basis_after = _basis_hash(output_dir)
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "status": "FRESH_P16_SELECTOR_COMPLETE",
        "classification": "FRESH_LAYER0_SELECTOR_ONLY_A_SELECTION_B_CONFIRMATION_HOLDOUT_CLOSED",
        "hypothesis": "Hard top-k dispatch regularization plus regret-weighted residual BCE can improve p16/top4 routing load without changing the frozen expert/shared basis.",
        "falsifier": "Validation-A or independent validation-B remains below the research candidate gate, hard load CV remains materially above 0.50, or basis tensors change.",
        "decision": "retain_as_fresh_selector_candidate_only_until_a_green_and_b_confirmation_are_reviewed",
        "compute_budget": {
            "epochs": int(args.epochs),
            "microbatch": int(args.microbatch),
            "learning_rate": float(args.learning_rate),
            "device": args.device,
            "hard_load_weight": float(args.hard_load_weight),
            "oracle_weight": float(args.oracle_weight),
            "oracle_amplitude_weight": float(args.oracle_amplitude_weight),
            "oracle_regret_weight": float(args.oracle_regret_weight),
            "fit_tokens": int(dataset.count - len(fit_exclude)),
            "holdout_opened": False,
        },
        "code_commit": code_commit,
        "source_revision": profile.revision,
        "dataset_hash": dataset.dataset_hash,
        "data_plan": str(data_plan_path),
        "validation_protocol": {
            "selection_split": "validation-A",
            "selection_indices_hash": _hash_indices(a_indices),
            "selection_count": len(a_indices),
            "validation_b_indices_hash": _hash_indices(b_indices),
            "validation_b_count": len(b_indices),
            "fit_excluded_indices_hash": _hash_indices(fit_exclude),
            "fit_excluded_count": len(fit_exclude),
            "selection_union": {"enabled": False, "reason": "validation-B is confirmation-only"},
            "optimizer": "AdamW",
            "validation_b_checkpoint_selection": False,
            "historical_holdout_opened": False,
        },
        "basis": {
            "initial_checkpoint": str(basis_dir),
            "trained_checkpoint": str(output_dir),
            "basis_tensor_sha256_before": basis_before,
            "basis_tensor_sha256_after": basis_after,
            "basis_unchanged": basis_before == basis_after,
            "basis_metadata_tensor_sha256": metadata.get("tensor_sha256"),
            "trained_tensor_sha256": metadata.get("tensor_sha256"),
        },
        "metrics": {
            "validation_a": result["final_selection"],
            "validation_b": result["validation_b_metrics"],
            "fit": result["final_fit"],
            "route_validation_a": route_a,
            "route_validation_b": route_b,
        },
        "soft_vs_hard_load": {
            "validation_a": {
                "hard_load_balance": result["final_selection"].get("hard_load_balance"),
                "hard_load_cv": result["final_selection"].get("load_cv"),
                "soft_load_balance": result["final_selection"].get("soft_load_balance"),
                "soft_load_cv": result["final_selection"].get("soft_load_cv"),
                "topk_logit_margin": result["final_selection"].get("topk_logit_margin"),
            },
            "validation_b": {
                "hard_load_balance": result["validation_b_metrics"].get("hard_load_balance"),
                "hard_load_cv": result["validation_b_metrics"].get("load_cv"),
                "soft_load_balance": result["validation_b_metrics"].get("soft_load_balance"),
                "soft_load_cv": result["validation_b_metrics"].get("soft_load_cv"),
                "topk_logit_margin": result["validation_b_metrics"].get("topk_logit_margin"),
            },
        },
        "training": result["training_config"],
        "checkpoint": {"metadata": result["metadata"], "tensor_file": result["tensor_file"]},
        "heartbeat": str(heartbeat),
    }
    report_path = fresh_dir / "reports/p16-top4-fresh-selector-hard-regret-bce.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--hard-load-weight", type=float, default=0.10)
    parser.add_argument("--oracle-weight", type=float, default=0.10)
    parser.add_argument("--oracle-amplitude-weight", type=float, default=0.05)
    parser.add_argument("--oracle-regret-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_FRESH / "logs/fresh-selector-heartbeat.json")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "validation_a": payload["metrics"]["validation_a"],
                "validation_b": payload["metrics"]["validation_b"],
                "basis_unchanged": payload["basis"]["basis_unchanged"],
                "code_commit": payload["code_commit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
