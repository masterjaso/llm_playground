"""Bounded, lineage-safe FIT-TRAIN/FIT-DEV runner for Dense2MoE candidate A.

This module is deliberately an orchestration and evidence seam.  Training is
delegated to :func:`dense2moe.training.torch_distill.train_torch_layer`; this
runner validates the split/partition/config contract, records immutable
receipts, and performs strict checkpoint reload verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PRODUCT_TARGETS = {
    "normalized_mse": 0.05,
    "cosine": 0.98,
    "load_cv": 0.50,
    "dead_experts": 0,
}
CONTINUATION_TARGETS = {
    "normalized_mse": 0.09,
    "cosine": 0.95,
    "load_cv": 0.50,
    "dead_experts": 0,
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(_canonical(payload).decode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError(f"refusing to overwrite a different immutable artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(body) + b"\n")
    return body


def _json_safe(value: Any) -> Any:
    """Convert framework/numpy scalars while rejecting non-finite numbers."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("receipt contains a non-finite numeric value")
        return int(value) if isinstance(value, int) else number
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _finite_metrics(metrics: Mapping[str, Any] | None) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    for value in metrics.values():
        if isinstance(value, Mapping):
            if not _finite_metrics(value):
                return False
        elif isinstance(value, (list, tuple)):
            if any(isinstance(item, float) and not math.isfinite(item) for item in value):
                return False
        elif isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _manifest_contract(path: str | Path, expected_split: str) -> dict[str, Any]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"activation manifest does not exist: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"activation manifest must be an object: {manifest}")
    split = str(payload.get("split", ""))
    aliases = {"FIT-TRAIN": {"FIT-TRAIN", "train"}, "FIT-DEV": {"FIT-DEV", "holdout"}}
    if split not in aliases[expected_split]:
        raise ValueError(f"{manifest} must declare split {expected_split}, got {split!r}")
    count = int(payload.get("count", 0))
    if count <= 0:
        raise ValueError(f"{manifest} must contain a positive count")
    dataset_hash = str(payload.get("dataset_hash", ""))
    if not dataset_hash:
        raise ValueError(f"{manifest} is missing dataset_hash")
    source_revision = str(payload.get("source_revision", ""))
    return {
        "path": str(manifest.resolve()),
        "sha256": _sha256_file(manifest),
        "split": expected_split,
        "count": count,
        "dataset_hash": dataset_hash,
        "source_revision": source_revision or None,
    }


def validate_split_manifests(fit_train: str | Path, fit_dev: str | Path) -> dict[str, Any]:
    fit = _manifest_contract(fit_train, "FIT-TRAIN")
    dev = _manifest_contract(fit_dev, "FIT-DEV")
    if fit["path"] == dev["path"]:
        raise ValueError("FIT-TRAIN and FIT-DEV must be distinct immutable manifests")
    return {"fit_train": fit, "fit_dev": dev, "opened_evaluation_tiers": []}


def _source_contract(source_dir: str | Path, source_revision: str | None, manifests: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    config = source / "config.json"
    index = source / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise FileNotFoundError("source must contain config.json and model.safetensors.index.json")
    revision = str(source_revision or manifests["fit_train"].get("source_revision") or "")
    if not revision:
        raise ValueError("source_revision must be explicit or present in FIT-TRAIN manifest")
    if revision in {"main", "master", "latest"}:
        raise ValueError("source_revision must be pinned, not mutable")
    for split in ("fit_train", "fit_dev"):
        observed = manifests[split].get("source_revision")
        if observed and observed != revision:
            raise ValueError(f"{split} source_revision does not match requested source_revision")
    return {
        "path": str(source.resolve()),
        "revision": revision,
        "config_sha256": _sha256_file(config),
        "index_sha256": _sha256_file(index),
    }


def _partition_contract(partition_path: str | Path) -> dict[str, Any]:
    from dense2moe.partition.contributions import load_partition_plan

    path = Path(partition_path)
    if not path.is_file():
        raise FileNotFoundError(f"partition plan does not exist: {path}")
    plan = load_partition_plan(path)
    plan.validate()
    expected = {
        "dense_intermediate_size": 17408,
        "routed_experts": 16,
        "expert_intermediate_size": 960,
        "shared_intermediate_size": 2048,
    }
    actual = {key: int(getattr(plan, key)) for key in expected}
    if actual != expected:
        raise ValueError(f"candidate A partition geometry mismatch: {actual}")
    plan_dict = plan.as_dict()
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "plan_sha256": hashlib.sha256(json.dumps(plan_dict, sort_keys=True).encode("utf-8")).hexdigest(),
        "canonical_plan_sha256": hashlib.sha256(_canonical(plan_dict)).hexdigest(),
        "geometry": actual,
        "plan": plan,
    }


def _number(value: Any, *, name: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def normalize_schedule(raw: Any, *, config_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Validate a train_torch_layer schedule and return a deterministic copy."""

    if isinstance(raw, Mapping):
        chosen_id = str(config_id or raw.get("config_id") or raw.get("name") or "schedule")
        stages_raw = raw.get("stages")
    else:
        chosen_id = str(config_id or "schedule")
        stages_raw = raw
    if not _SAFE_ID.fullmatch(chosen_id):
        raise ValueError("config_id must be a safe identifier")
    if isinstance(stages_raw, (str, bytes)) or not isinstance(stages_raw, Sequence) or not stages_raw:
        raise ValueError("schedule must contain a non-empty stages sequence")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(stages_raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"schedule stage {index} must be an object")
        stage = {str(key): value for key, value in item.items()}
        name = str(stage.get("name", stage.get("stage", f"stage-{index}")))
        if not name:
            raise ValueError(f"schedule stage {index} has an empty name")
        epochs = int(stage.get("epochs", 0))
        if epochs < 0:
            raise ValueError(f"schedule stage {index} epochs must be non-negative")
        stage["name"] = name
        stage["epochs"] = epochs
        if "learning_rate" in stage:
            stage["learning_rate"] = _number(stage["learning_rate"], name=f"stage {index} learning_rate", minimum=1e-12)
        if "learning_rates" in stage:
            rates = stage["learning_rates"]
            if not isinstance(rates, Mapping):
                raise TypeError(f"schedule stage {index} learning_rates must be an object")
            stage["learning_rates"] = {str(key): _number(value, name=f"stage {index} learning_rates.{key}", minimum=1e-12) for key, value in rates.items()}
        if "loss_coefficients" in stage:
            losses = stage["loss_coefficients"]
            if not isinstance(losses, Mapping):
                raise TypeError(f"schedule stage {index} loss_coefficients must be an object")
            stage["loss_coefficients"] = {str(key): _number(value, name=f"stage {index} loss_coefficients.{key}", minimum=0.0) for key, value in losses.items()}
        for key in ("train_scales", "train_experts", "train_shared", "use_oracle_targets", "train_selection_router", "train_amplitude_router"):
            if key in stage:
                stage[key] = bool(stage[key])
        if "oracle_target_mode" in stage and str(stage["oracle_target_mode"]) not in {"contribution_norm", "residual_correlation"}:
            raise ValueError(f"schedule stage {index} has unsupported oracle_target_mode")
        if "oracle_loss_mode" in stage and str(stage["oracle_loss_mode"]) not in {"repeated_cross_entropy", "multilabel_bce"}:
            raise ValueError(f"schedule stage {index} has unsupported oracle_loss_mode")
        if "oracle_amplitude_mode" in stage and str(stage["oracle_amplitude_mode"]) not in {"student_selected", "teacher_forced", "mixed"}:
            raise ValueError(f"schedule stage {index} has unsupported oracle_amplitude_mode")
        if "teacher_forcing_ratio" in stage:
            stage["teacher_forcing_ratio"] = _number(stage["teacher_forcing_ratio"], name=f"stage {index} teacher_forcing_ratio", minimum=0.0)
            if stage["teacher_forcing_ratio"] > 1.0:
                raise ValueError(f"schedule stage {index} teacher_forcing_ratio must be <= 1")
        if "oracle_regret_weight" in stage:
            stage["oracle_regret_weight"] = _number(stage["oracle_regret_weight"], name=f"stage {index} oracle_regret_weight", minimum=0.0)
        if "expert_use_prices" in stage and stage["expert_use_prices"] is not None:
            prices = stage["expert_use_prices"]
            if isinstance(prices, (str, bytes)) or not isinstance(prices, Sequence) or len(prices) != 16:
                raise ValueError("expert_use_prices must contain 16 values for candidate A")
            stage["expert_use_prices"] = [_number(value, name=f"stage {index} expert_use_prices", minimum=0.0) for value in prices]
        normalized.append(stage)
    return chosen_id, normalized


def _profile(source_revision: str) -> Any:
    from dense2moe.config import MoEProfile

    return MoEProfile(
        name="v24-a-p16-top6",
        hidden_size=5120,
        dense_intermediate_size=17408,
        num_hidden_layers=64,
        routed_experts=16,
        expert_intermediate_size=960,
        shared_intermediate_size=2048,
        top_k=6,
        model="Qwen/Qwen3.8-27B",
        revision=source_revision,
        dtype="bfloat16",
        routing_mode="independent_positive",
    )


def _resolve_device(requested: str) -> str:
    value = str(requested)
    if value == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is unavailable")
        except ImportError as exc:
            raise RuntimeError("CUDA device requested but torch is unavailable") from exc
    return value


def _load_dense_mlp(source_dir: Path, layer: int = 0) -> dict[str, Any]:
    from safetensors import safe_open  # type: ignore

    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {key[len(prefix) :]: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
    required = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(names) != required:
        raise ValueError(f"source MLP inventory mismatch: {sorted(names)}")
    values: dict[str, Any] = {}
    for name, shard in names.items():
        with safe_open(str(source_dir / shard), framework="pt", device="cpu") as handle:
            values[name] = handle.get_tensor(prefix + name).float().numpy()
    return values


def strict_reload_checkpoint(output: Mapping[str, Any], *, source_dir: str | Path, profile: Any, partition_path: str | Path, layer: int = 0) -> dict[str, Any]:
    """Strictly restore every checkpoint tensor and verify its byte hash."""

    import torch
    from safetensors.torch import load_file  # type: ignore

    from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE

    metadata_path = Path(str(output.get("metadata", "")))
    tensor_path = Path(str(output.get("tensor_file", "")))
    if not tensor_path.is_file() and metadata_path.is_file():
        tensor_path = metadata_path.parent / str(output.get("tensor_file", ""))
    if not tensor_path.is_file():
        raise FileNotFoundError(f"checkpoint tensor file is missing: {tensor_path}")
    tensor_sha = _sha256_file(tensor_path)
    raw_state = load_file(str(tensor_path), device="cpu")
    prefix = f"model.layers.{layer}."
    state = {key[len(prefix) :]: value for key, value in raw_state.items() if key.startswith(prefix)}
    if len(state) != len(raw_state):
        raise RuntimeError("checkpoint tensor namespace is not layer-scoped")
    weights = _load_dense_mlp(Path(source_dir), layer)
    model = TorchQwen35SwiGLUMoE.from_dense(
        weights["gate_proj.weight"],
        weights["up_proj.weight"],
        weights["down_proj.weight"],
        routed_experts=profile.routed_experts,
        shared_intermediate_size=profile.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=__import__("dense2moe.partition.contributions", fromlist=["load_partition_plan"]).load_partition_plan(partition_path),
        learnable_scales=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint reload mismatch: missing={missing}, unexpected={unexpected}")
    del model, state, raw_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"passed": True, "tensor_path": str(tensor_path.resolve()), "tensor_sha256": tensor_sha, "tensor_count": len(output.get("tensor_inventory", {}))}


def _candidate_gate(metrics: Mapping[str, Any]) -> bool:
    try:
        return bool(
            float(metrics["normalized_mse"]) <= PRODUCT_TARGETS["normalized_mse"]
            and float(metrics["cosine"]) >= PRODUCT_TARGETS["cosine"]
            and float(metrics["load_cv"]) <= PRODUCT_TARGETS["load_cv"]
            and int(metrics["dead_experts"]) == PRODUCT_TARGETS["dead_experts"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def run_exploration(
    *,
    source_dir: str | Path,
    fit_train_manifest: str | Path,
    fit_dev_manifest: str | Path,
    partition_path: str | Path,
    run_root: str | Path,
    seed: int,
    device: str,
    microbatch: int,
    schedule: Any,
    source_revision: str | None = None,
    config_id: str | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if int(microbatch) <= 0:
        raise ValueError("microbatch must be positive")
    manifests = validate_split_manifests(fit_train_manifest, fit_dev_manifest)
    source = _source_contract(source_dir, source_revision, manifests)
    partition = _partition_contract(partition_path)
    chosen_id, stages = normalize_schedule(schedule, config_id=config_id)
    profile = _profile(source["revision"])
    profile.validate()
    effective_device = _resolve_device(device)
    config = {
        "config_id": chosen_id,
        "profile": profile.as_dict(),
        "seed": int(seed),
        "requested_device": str(device),
        "device": effective_device,
        "microbatch": int(microbatch),
        "schedule": stages,
        "evaluate_holdout": False,
        "execution_mode": "train" if execute else "dry-run",
        "opened_evaluation_tiers": [],
    }
    config_hash = hashlib.sha256(_canonical({"source": source, "manifests": manifests, "partition": {key: value for key, value in partition.items() if key != "plan"}, "config": config})).hexdigest()
    run = Path(run_root)
    output_dir = run / "comparison" / "a-candidate-exploration" / chosen_id / f"seed-{int(seed)}"
    receipt_path = output_dir / "seed-result.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash or existing.get("inputs", {}).get("fit_train", {}).get("sha256") != manifests["fit_train"]["sha256"] or existing.get("partition", {}).get("file_sha256") != partition["file_sha256"]:
            raise RuntimeError(f"cached result lineage/config mismatch: {receipt_path}")
        if execute and existing.get("status") == "DRY_RUN":
            raise RuntimeError(f"cached dry-run cannot satisfy execute request: {receipt_path}")
        if not execute and existing.get("status") not in {"DRY_RUN", "TRAINING_FAILED"}:
            raise RuntimeError(f"cached trained result cannot satisfy dry-run request: {receipt_path}")
        return existing
    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "dense2moe-v2.4-a-candidate-exploration",
        "status": "DRY_RUN" if not execute else "PENDING",
        "run_id": run.name,
        "config_id": chosen_id,
        "config_hash": config_hash,
        "source": source,
        "inputs": manifests,
        "partition": {key: value for key, value in partition.items() if key != "plan"},
        "config": config,
        "promotion_eligible": False,
        "candidate_gate_met": False,
        "opened_evaluation_tiers": [],
        "fit_dev_gradient_contract": {
            "fit_train_manifest": manifests["fit_train"]["path"],
            "fit_train_gradients": True,
            "fit_train_route_labels": True,
            "fit_dev_manifest": manifests["fit_dev"]["path"],
            "fit_dev_gradients": False,
            "fit_dev_route_labels": False,
            "fit_dev_checkpoint_selection_only": True,
            "holdout_opened": False,
        },
    }
    if not execute:
        return _write_immutable(receipt_path, base)
    from dense2moe.provenance import current_git_commit
    from dense2moe.training.torch_distill import train_torch_layer

    try:
        epochs = max([int(stage.get("epochs", 0)) for stage in stages] or [0])
        output = train_torch_layer(
            source_dir=source_dir,
            activation_manifest=fit_train_manifest,
            output_dir=output_dir / "checkpoint",
            layer=0,
            profile=profile,
            partition_path=partition_path,
            epochs=epochs,
            microbatch=int(microbatch),
            learning_rate=1e-3,
            device=effective_device,
            seed=int(seed),
            source_revision=source["revision"],
            code_commit=current_git_commit(),
            stage_schedule=stages,
            selection_manifest=fit_dev_manifest,
            selection_split="FIT-DEV",
            evaluate_holdout=False,
        )
        reload_info = strict_reload_checkpoint(output, source_dir=source_dir, profile=profile, partition_path=partition_path)
        final_dev = _json_safe(output.get("final_selection") or {})
        initial_fit = _json_safe(output.get("initial_fit") or {})
        final_fit = _json_safe(output.get("final_fit") or {})
        initial_dev = _json_safe(output.get("initial_selection") or {})
        stage_metrics = _json_safe((output.get("training_config") or {}).get("stage_selection_metrics", []))
        stages_result = _json_safe((output.get("training_config") or {}).get("stages", []))
        base.update(
            {
                "status": "TRAINED",
                "code_commit": current_git_commit(),
                "metrics": {
                    "initial_fit": initial_fit,
                    "final_fit": final_fit,
                    "initial_dev": initial_dev,
                    "final_dev": final_dev,
                    "stage_metrics": stage_metrics,
                },
                "stages": stages_result,
                "route_statistics": {
                    "final_dev_selected_counts": final_dev.get("selected_counts", []),
                    "final_dev_load_cv": final_dev.get("load_cv"),
                    "final_dev_dead_experts": final_dev.get("dead_experts"),
                    "final_fit_selected_counts": final_fit.get("selected_counts", []),
                    "final_fit_load_cv": final_fit.get("load_cv"),
                    "final_fit_dead_experts": final_fit.get("dead_experts"),
                },
                "checkpoint": {
                    "metadata": str(output.get("metadata", "")),
                    "tensor_file": reload_info["tensor_path"],
                    "tensor_sha256": reload_info["tensor_sha256"],
                    "strict_reload": reload_info,
                },
                "finite_metrics": _finite_metrics(final_dev) and _finite_metrics(final_fit),
                "candidate_gate_met": bool(_finite_metrics(final_dev) and reload_info["passed"] and _candidate_gate(final_dev)),
            }
        )
        return _write_immutable(receipt_path, _json_safe(base))
    except Exception as exc:  # noqa: BLE001 - failure is immutable negative evidence
        base.update(
            {
                "status": "TRAINING_FAILED",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "finite_metrics": False,
                "candidate_gate_met": False,
            }
        )
        return _write_immutable(receipt_path, _json_safe(base))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", "--source", dest="source_dir", type=Path, required=True)
    parser.add_argument("--fit-train", "--fit-train-manifest", dest="fit_train_manifest", type=Path, required=True)
    parser.add_argument("--fit-dev", "--fit-dev-manifest", dest="fit_dev_manifest", type=Path, required=True)
    parser.add_argument("--partition", dest="partition_path", type=Path, required=True)
    parser.add_argument("--run-root", dest="run_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--schedule", "--config", dest="schedule", required=True, help="JSON file or inline JSON schedule")
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_schedule_spec(value: str) -> Any:
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_exploration(
        source_dir=args.source_dir,
        fit_train_manifest=args.fit_train_manifest,
        fit_dev_manifest=args.fit_dev_manifest,
        partition_path=args.partition_path,
        run_root=args.run_root,
        seed=args.seed,
        device=args.device,
        microbatch=args.microbatch,
        schedule=_load_schedule_spec(args.schedule),
        source_revision=args.source_revision,
        config_id=args.config_id,
        execute=not args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"DRY_RUN", "TRAINED"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
