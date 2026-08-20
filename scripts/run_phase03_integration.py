#!/usr/bin/env python3
"""Run the bounded, profile-aware Phase 03 layer-0 integration gate.

This is intentionally smaller than full Qwen assembly.  It reads the pinned
source index/config and one locked layer-0 checkpoint per finalist, proves the
source-backed non-FFN boundary is preserved, runs the sparse block on an
already-captured FIT-DEV activation, and asks a fresh process to reload the
same checkpoint.  It never creates a full64 queue or BF16 product checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import iter_activation_shards
from dense2moe.checkpoint import profile_fingerprint
from dense2moe.config import MoEProfile, load_active_config
from dense2moe.data import sha256_file, write_immutable_json
from dense2moe.models.qwen35_full import Qwen35DenseToMoE, replace_qwen35_ffns
from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import PartitionPlan
from dense2moe.provenance import current_git_commit

PROFILES = ("qwen38_p16s1_top4", "qwen38_p32s1_top5")
METHOD_VERSION = "moe-v22-m01"
SEEDS = (17, 29, 41)
SOURCE_MODEL_TYPE = "qwen3_5_text"
CHECKPOINT_PREFIX = "model.layers.0."
SOURCE_LAYER_PREFIX = "model.language_model.layers.0."
MLP_NAMES = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _tensor_hash(value: Any) -> str:
    import torch

    tensor = value.detach().to(device="cpu").contiguous()
    # NumPy on the pinned Windows stack cannot expose torch.bfloat16. Hash its
    # raw uint16 representation so source-backed BF16 preservation remains
    # content-addressed without changing the tensor values.
    if str(tensor.dtype) == "torch.bfloat16":
        raw = tensor.view(dtype=torch.uint16).numpy().tobytes()
    else:
        raw = tensor.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _manifest_identity(path: Path) -> str:
    """Match the locked candidate-search identity (manifest plus shards)."""

    payload = _read_json(path)
    reference = payload.get("train_manifest")
    if isinstance(reference, str) and reference not in {"", "pending"}:
        child = Path(reference)
        path = child if child.is_absolute() else path.parent / child
        payload = _read_json(path)
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    for shard in payload.get("shards", []):
        shard_path = Path(str(shard["path"]))
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        digest.update(str(shard_path).encode())
        digest.update(sha256_file(shard_path).encode())
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    direct = Path(str(value))
    if direct.exists():
        return direct.resolve()
    return (root / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _source_revision(source_dir: Path) -> str:
    metadata = source_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.exists():
        return ""
    return metadata.read_text(encoding="utf-8").splitlines()[0].strip()


def _source_identity(source_dir: Path, profile: MoEProfile) -> dict[str, Any]:
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        raise FileNotFoundError("pinned Qwen source requires config.json and model.safetensors.index.json")
    config = _read_json(config_path)
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    expected = {
        "model_type": SOURCE_MODEL_TYPE,
        "hidden_size": profile.hidden_size,
        "intermediate_size": profile.dense_intermediate_size,
        "num_hidden_layers": profile.num_hidden_layers,
    }
    mismatches = {
        key: {"expected": value, "actual": text_config.get(key)}
        for key, value in expected.items()
        if text_config.get(key) != value
    }
    if config.get("model_type") != "qwen3_5":
        mismatches["root_model_type"] = {"expected": "qwen3_5", "actual": config.get("model_type")}
    if mismatches:
        raise ValueError(f"pinned Qwen config identity mismatch: {mismatches}")
    revision = _source_revision(source_dir)
    if revision != profile.revision:
        raise ValueError(f"pinned Qwen source revision mismatch: expected {profile.revision}, got {revision!r}")
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("pinned Qwen index has no weight_map")
    tokenizer_path = source_dir / "tokenizer.json"
    return {
        "source_dir": str(source_dir),
        "source_revision": revision,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "source_index": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "tokenizer": str(tokenizer_path) if tokenizer_path.exists() else None,
        "tokenizer_sha256": sha256_file(tokenizer_path) if tokenizer_path.exists() else None,
        "model_type": str(config.get("model_type")),
        "text_model_type": str(text_config.get("model_type")),
        "weight_count": len(weight_map),
    }


def _load_source_layer(source_dir: Path, layer: int = 0) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load only layer-0 MLP tensors and hash the untouched layer-0 backbone."""

    from safetensors import safe_open

    index = _read_json(source_dir / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    selected = {str(key)[len(prefix) :]: str(shard) for key, shard in weight_map.items() if str(key).startswith(prefix)}
    mlp = {name[len("mlp.") :]: shard for name, shard in selected.items() if name.startswith("mlp.")}
    if set(mlp) != MLP_NAMES:
        raise ValueError(f"layer {layer} dense MLP inventory mismatch: {sorted(mlp)}")
    values: dict[str, Any] = {}
    for name, shard in mlp.items():
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **kwargs) as handle:
            values[name] = handle.get_tensor(prefix + "mlp." + name).float()
    preserved: dict[str, dict[str, Any]] = {}
    for name, shard in selected.items():
        if name.startswith("mlp."):
            continue
        kwargs = {"backend": "pread"} if os.name == "nt" else {}
        with safe_open(str(source_dir / shard), framework="pt", device="cpu", **kwargs) as handle:
            tensor = handle.get_tensor(prefix + name)
        preserved[prefix + name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": _tensor_hash(tensor),
        }
    return values, preserved, {"layer": layer, "source_prefix": prefix, "tensor_count": len(selected)}


def _partition_from_payload(payload: Mapping[str, Any]) -> PartitionPlan:
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else payload.get("partition")
    if not isinstance(plan, Mapping):
        plan = payload
    result = PartitionPlan(
        int(plan["dense_intermediate_size"]),
        int(plan["routed_experts"]),
        int(plan["expert_intermediate_size"]),
        int(plan["shared_intermediate_size"]),
        tuple(int(value) for value in plan["shared_indices"]),
        tuple(tuple(int(value) for value in group) for group in plan["expert_indices"]),
    )
    result.validate()
    return result


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("torch.")


def _validate_manifest_identity(manifest: Path, *, activation_expected_split: str = "FIT-DEV") -> dict[str, Any]:
    payload = _read_json(manifest)
    if int(payload.get("layer", -1)) != 0:
        raise ValueError("Phase 03 integration requires a layer-0 activation manifest")
    if str(payload.get("input_tensor")) != "ffn_input" or str(payload.get("target_tensor")) != "dense_ffn_target":
        raise ValueError("activation manifest must contain the captured ffn_input/dense_ffn_target pair")
    split = str(payload.get("split", ""))
    if split not in {activation_expected_split, "holdout"}:
        raise ValueError(f"activation manifest split mismatch: expected {activation_expected_split}, got {split!r}")
    if int(payload.get("count", 0)) <= 0 or not payload.get("shards"):
        raise ValueError("activation manifest is empty")
    return payload


def _validate_checkpoint_metadata(
    metadata_path: Path,
    *,
    profile: MoEProfile,
    source: Mapping[str, Any],
    partition: PartitionPlan,
    expected_partition_sha256: str,
    expected_method_version: str,
    expected_dataset_hash: str | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Validate a finalist metadata/tensor pair without requiring product-green quality."""

    from safetensors import safe_open

    payload = _read_json(metadata_path)
    errors: list[str] = []
    if int(payload.get("layer", -1)) != 0:
        errors.append("layer must be 0")
    if payload.get("profile") != profile.name:
        errors.append("profile mismatch")
    expected_profile_hash = profile_fingerprint(profile.as_dict())
    if payload.get("profile_hash") != expected_profile_hash:
        errors.append("profile_hash mismatch")
    if payload.get("source_revision") != profile.revision:
        errors.append("source_revision mismatch")
    if payload.get("source_config_hash") != source.get("source_config_sha256"):
        errors.append("source_config_hash mismatch")
    if payload.get("source_index_hash") != source.get("source_index_sha256"):
        errors.append("source_index_hash mismatch")
    if payload.get("routing_mode") != profile.routing_mode:
        errors.append("routing_mode mismatch")
    if str(payload.get("router_architecture", "")) != f"torch-linear-topk-{profile.routing_mode}-v1":
        errors.append("router_architecture mismatch")
    if expected_dataset_hash and not str(payload.get("dataset_hash", "")):
        errors.append("checkpoint dataset identity missing")
    tensor_value = payload.get("tensor_file")
    tensor_path = metadata_path.parent / str(tensor_value or "")
    if not tensor_path.is_file():
        errors.append("tensor artifact missing")
    elif payload.get("tensor_sha256") != sha256_file(tensor_path):
        errors.append("tensor artifact hash mismatch")
    if payload.get("status") not in {"VALIDATION_DEFERRED", "TRAINED_DEV_SELECTED", "TRAINED_VALIDATED", "RESEARCH_CANDIDATE"}:
        errors.append(f"unsupported locked checkpoint status: {payload.get('status')!r}")
    canonical_partition_hash = hashlib.sha256(json.dumps(partition.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    if payload.get("partition_hash") != canonical_partition_hash:
        errors.append("partition_hash mismatch")
    if errors:
        raise ValueError(f"checkpoint identity failed for {metadata_path}: {errors}")
    inventory = payload.get("tensor_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError(f"checkpoint tensor inventory missing: {metadata_path}")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        actual_inventory: dict[str, Any] = {}
        for name in names:
            tensor = handle.get_tensor(name)
            actual_inventory[name] = {
                "shape": list(tensor.shape),
                "dtype": _dtype_name(tensor.dtype),
                "sha256": _tensor_hash(tensor),
            }
            if not bool(torch_isfinite(tensor)):
                raise ValueError(f"checkpoint tensor is non-finite: {name}")
    expected_names = sorted(str(name) for name in inventory)
    if names != expected_names:
        raise ValueError(f"checkpoint tensor inventory names mismatch: {metadata_path}")
    for name, item in actual_inventory.items():
        expected = inventory[name]
        if list(expected.get("shape", [])) != item["shape"] or _dtype_name(expected.get("dtype", "")) != item["dtype"]:
            raise ValueError(f"checkpoint tensor inventory detail mismatch: {name}")
    return payload, tensor_path, {"tensor_count": len(names), "tensor_inventory": actual_inventory, "partition_hash": canonical_partition_hash, "partition_sha256": expected_partition_sha256}


def torch_isfinite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all().item())


class _DenseMLP:
    """Dense source MLP module used only to exercise the Qwen replacement boundary."""

    def __new__(cls, gate: Any, up: Any, down: Any) -> Any:
        import torch
        from torch import nn

        module = nn.Module()
        module.gate_proj = nn.Linear(int(gate.shape[1]), int(gate.shape[0]), bias=False, dtype=torch.float32)
        module.up_proj = nn.Linear(int(up.shape[1]), int(up.shape[0]), bias=False, dtype=torch.float32)
        module.down_proj = nn.Linear(int(down.shape[1]), int(down.shape[0]), bias=False, dtype=torch.float32)
        with torch.no_grad():
            module.gate_proj.weight.copy_(gate)
            module.up_proj.weight.copy_(up)
            module.down_proj.weight.copy_(down)
        return module


def _build_qwen_layer0_boundary(dense: Mapping[str, Any], preserved: Mapping[str, Mapping[str, Any]], profile: MoEProfile) -> tuple[Qwen35DenseToMoE, dict[str, Any]]:
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _DenseMLP(dense["gate_proj.weight"], dense["up_proj.weight"], dense["down_proj.weight"])

    class TextBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    class Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TextBackbone()
            # Digest sentinels keep the boundary's non-FFN inventory source-
            # backed without materializing a 27B full model on this gate.
            self.backbone_sentinel = nn.ParameterList(
                [nn.Parameter(torch.tensor([int(byte) for byte in hashlib.sha256(json.dumps(item, sort_keys=True).encode()).digest()], dtype=torch.uint8), requires_grad=False) for item in preserved.values()]
            )

    stub = Stub().eval()
    replacement = replace_qwen35_ffns(stub, topology=profile.name, strict_layer_count=1)
    return Qwen35DenseToMoE(stub, receipt=replacement), replacement


def _load_checkpoint_state(model: TorchQwen35SwiGLUMoE, tensor_path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    raw = load_file(str(tensor_path), device="cpu")
    state = {str(name)[len(CHECKPOINT_PREFIX) :]: value for name, value in raw.items() if str(name).startswith(CHECKPOINT_PREFIX)}
    if set(state) != set(model.state_dict()):
        raise ValueError(f"strict checkpoint namespace mismatch: missing={sorted(set(model.state_dict()) - set(state))[:8]}, unexpected={sorted(set(state) - set(model.state_dict()))[:8]}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict checkpoint reload failed: missing={missing}, unexpected={unexpected}")
    return {"state_tensor_count": len(state), "state_tensor_inventory_sha256": _canonical_hash({name: _tensor_hash(value) for name, value in sorted(state.items())})}


def _representative_inputs(manifest: Path, max_tokens: int) -> Any:
    import numpy as np
    import torch

    for shard in iter_activation_shards(manifest, expected_split="FIT-DEV"):
        array = np.asarray(shard[:max_tokens], dtype=np.float32)
        if array.size:
            return torch.as_tensor(array, dtype=torch.float32)
    raise ValueError("FIT-DEV activation manifest has no representative rows")


def _forward(model: TorchQwen35SwiGLUMoE, inputs: Any) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.inference_mode():
        output, info = model(inputs, return_router=True)
    if tuple(output.shape) != tuple(inputs.shape) or not torch.isfinite(output).all():
        raise ValueError("representative sparse forward was non-finite or shape-mismatched")
    expected_dispatches = int(inputs.shape[0]) * int(model.top_k)
    if info.get("dispatch_mode") != "sparse_token_dispatch" or info.get("dense_fallback_used") is not False:
        raise ValueError("representative forward used a dense fallback")
    if int(info.get("selected_dispatches", -1)) != expected_dispatches:
        raise ValueError("representative forward top-k dispatch count mismatch")
    return {
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "output_sha256": _tensor_hash(output),
        "dispatch_mode": info["dispatch_mode"],
        "dense_fallback_used": bool(info["dense_fallback_used"]),
        "dispatch_token_count": int(info["dispatch_token_count"]),
        "selected_dispatches": int(info["selected_dispatches"]),
        "top_k": int(model.top_k),
        "nonempty_experts": int(info["nonempty_experts"]),
        "active_intermediate_width": int(info["active_intermediate_width"]),
        "dense_intermediate_width": int(info["dense_intermediate_width"]),
    }


def _fresh_process_reload(
    *,
    source_dir: Path,
    profile_name: str,
    partition_path: Path,
    checkpoint_path: Path,
    activation_manifest: Path,
    expected_output_sha256: str,
    max_tokens: int,
) -> dict[str, Any]:
    profile, _contract = load_active_config(Path(__file__).resolve().parents[1] / "configs" / f"{profile_name}.yaml")
    source = _source_identity(source_dir, profile)
    dense, _preserved, _details = _load_source_layer(source_dir)
    partition_payload = _read_json(partition_path)
    plan = _partition_from_payload(partition_payload)
    model = TorchQwen35SwiGLUMoE.from_dense(
        dense["gate_proj.weight"],
        dense["up_proj.weight"],
        dense["down_proj.weight"],
        routed_experts=profile.routed_experts,
        shared_intermediate_size=profile.shared_intermediate_size,
        top_k=profile.top_k,
        routing_mode=profile.routing_mode,
        partition=plan,
        learnable_scales=True,
    )
    state_info = _load_checkpoint_state(model, checkpoint_path)
    inputs = _representative_inputs(activation_manifest, max_tokens)
    forward = _forward(model, inputs)
    if forward["output_sha256"] != expected_output_sha256:
        raise ValueError("fresh-process representative output digest mismatch")
    return {
        "status": "FRESH_PROCESS_RELOAD_GREEN",
        "profile": profile_name,
        "source_revision": source["source_revision"],
        "checkpoint": str(checkpoint_path),
        "strict": True,
        "state": state_info,
        "forward": forward,
    }


def _run_fresh_process(**kwargs: Any) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--fresh-process"]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        command.extend([flag, str(value)])
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=1800, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"fresh process reload failed ({completed.returncode}): {completed.stdout[-2000:]} {completed.stderr[-2000:]}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"fresh process emitted invalid JSON: {completed.stdout[-2000:]}") from exc
    if not isinstance(result, dict) or result.get("status") != "FRESH_PROCESS_RELOAD_GREEN":
        raise RuntimeError(f"fresh process reload was not green: {result}")
    return result


def _candidate_integration(
    *,
    repo_root: Path,
    source_dir: Path,
    profile: MoEProfile,
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    activation_manifest: Path,
    activation_payload: Mapping[str, Any],
    lock_method_version: str,
    max_tokens: int,
) -> dict[str, Any]:
    if str(candidate.get("profile")) != profile.name or str(candidate.get("status")) != "DEV_FINALIST":
        raise ValueError(f"candidate is not a locked finalist for {profile.name}")
    if str(candidate.get("topology")) != profile.topology_id:
        raise ValueError(f"candidate topology mismatch for {profile.name}")
    if str(candidate.get("partition_sha256", "")) == "":
        raise ValueError("locked candidate partition hash is missing")
    partition_path = _resolve(repo_root, str(candidate["partition_path"]))
    if not partition_path.is_file() or sha256_file(partition_path) != str(candidate["partition_sha256"]):
        raise ValueError(f"locked candidate partition hash mismatch: {partition_path}")
    partition_payload = _read_json(partition_path)
    if partition_payload.get("status") != "FROZEN_DEVELOPMENT_BASIS" or partition_payload.get("method_version") != lock_method_version:
        raise ValueError("partition is not the locked development basis")
    if partition_payload.get("external_data_used") is not False or partition_payload.get("basis_frozen") is not True:
        raise ValueError("partition violates the locked development-only method")
    plan = _partition_from_payload(partition_payload)
    expected_partition_hash = hashlib.sha256(json.dumps(plan.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    checkpoints = candidate.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(SEEDS):
        raise ValueError("locked finalist must contain exactly three fixed-seed checkpoints")
    if tuple(int(item.get("seed", -1)) for item in checkpoints) != SEEDS:
        raise ValueError("locked finalist seed set changed")
    expected_checkpoint_digest = hashlib.sha256(json.dumps(checkpoints, sort_keys=True, default=str).encode()).hexdigest()
    if expected_checkpoint_digest != str(candidate.get("checkpoint_sha256")):
        raise ValueError("locked finalist checkpoint digest mismatch")
    if str(candidate.get("dev_dataset_hash")) not in {sha256_file(activation_manifest), _manifest_identity(activation_manifest)}:
        raise ValueError("FIT-DEV activation manifest is not the locked candidate manifest")
    dense, preserved_inventory, source_layer = _load_source_layer(source_dir)
    boundary, replacement = _build_qwen_layer0_boundary(dense, preserved_inventory, profile)
    moe = boundary.model.model.layers[0].mlp
    if not isinstance(moe, TorchQwen35SwiGLUMoE):
        raise TypeError("Qwen layer-0 replacement did not produce a sparse MoE block")
    replacement_preserved = bool(replacement.get("non_ffn_preserved"))
    if not replacement_preserved:
        raise ValueError("Qwen non-FFN backbone inventory was not preserved")
    inputs = _representative_inputs(activation_manifest, max_tokens)
    seed_results: list[dict[str, Any]] = []
    for item in checkpoints:
        metadata_path = _resolve(repo_root, str(item.get("reload", {}).get("metadata", "")))
        metadata, tensor_path, tensor_info = _validate_checkpoint_metadata(
            metadata_path,
            profile=profile,
            source=source,
            partition=plan,
            expected_partition_sha256=str(candidate["partition_sha256"]),
            expected_method_version=lock_method_version,
            expected_dataset_hash=str(candidate.get("dataset_hash", "")),
        )
        if str(item.get("method_version")) != lock_method_version:
            raise ValueError("locked seed method version changed")
        if str(item.get("reload", {}).get("tensor_sha256")) != str(metadata.get("tensor_sha256")):
            raise ValueError("locked seed tensor hash disagrees with metadata")
        model = TorchQwen35SwiGLUMoE.from_dense(
            dense["gate_proj.weight"],
            dense["up_proj.weight"],
            dense["down_proj.weight"],
            routed_experts=profile.routed_experts,
            shared_intermediate_size=profile.shared_intermediate_size,
            top_k=profile.top_k,
            routing_mode=profile.routing_mode,
            partition=plan,
            learnable_scales=True,
        )
        state_info = _load_checkpoint_state(model, tensor_path)
        forward = _forward(model, inputs)
        fresh = _run_fresh_process(
            source_dir=source_dir,
            profile_name=profile.name,
            partition_path=partition_path,
            checkpoint_path=tensor_path,
            activation_manifest=activation_manifest,
            expected_output_sha256=forward["output_sha256"],
            max_tokens=max_tokens,
        )
        seed_results.append(
            {
                "seed": int(item["seed"]),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "tensor_file": str(tensor_path),
                "tensor_sha256": str(metadata["tensor_sha256"]),
                "tensor_count": int(tensor_info["tensor_count"]),
                "state": state_info,
                "representative_forward": forward,
                "fresh_process_reload": fresh,
                "checkpoint_status": metadata.get("status"),
            }
        )
    tensor_hashes = {str(row["tensor_sha256"]) for row in seed_results}
    return {
        "status": "PROFILE_FINALIST_INTEGRATION_GREEN",
        "profile": profile.name,
        "topology": profile.topology_id,
        "rank": int(candidate.get("rank", 0)),
        "method_version": lock_method_version,
        "partition": str(partition_path),
        "partition_sha256": str(candidate["partition_sha256"]),
        "partition_canonical_hash": expected_partition_hash,
        "source_layer": source_layer,
        "source_backbone_scope": "model.language_model.layers.0 non-MLP tensors",
        "source_backbone_tensor_count": len(preserved_inventory),
        "source_backbone_inventory_sha256": _canonical_hash(preserved_inventory),
        "backbone_preserved": replacement_preserved,
        "strict_profile_hash": profile_fingerprint(profile.as_dict()),
        "activation_manifest": str(activation_manifest),
        "activation_manifest_sha256": sha256_file(activation_manifest),
        "activation_dataset_identity": str(activation_payload.get("dataset_hash", "")),
        "seed_count": len(seed_results),
        "seed_tensor_hashes": sorted(tensor_hashes),
        "seed_results": seed_results,
        "full64_assembly": "NOT_ATTEMPTED",
    }


def run_integration(
    *,
    run_dir: Path,
    source_dir: Path,
    finalist_lock: Path,
    activation_manifest: Path,
    profiles: tuple[str, ...] = PROFILES,
    execute: bool = False,
    max_tokens: int = 32,
) -> dict[str, Any]:
    """Run both active profile gates and publish immutable receipts."""

    repo_root = Path(__file__).resolve().parents[1]
    lock = _read_json(finalist_lock)
    if lock.get("status") != "METHOD_LOCKED" or lock.get("external_tuning_forbidden") is not True:
        return {"status": "BLOCKED", "blocker_code": "FINALIST_LOCK_REQUIRED", "finalist_lock": str(finalist_lock)}
    if lock.get("opened_evaluation_tiers"):
        return {"status": "BLOCKED", "blocker_code": "EVALUATION_TIERS_ALREADY_OPEN", "opened_evaluation_tiers": lock.get("opened_evaluation_tiers")}
    if str(lock.get("method_version")) != METHOD_VERSION:
        return {"status": "BLOCKED", "blocker_code": "METHOD_VERSION_MISMATCH", "method_version": lock.get("method_version")}
    missing = [profile for profile in profiles if profile not in PROFILES]
    if missing:
        return {"status": "BLOCKED", "blocker_code": "INACTIVE_PROFILE", "profiles": missing}
    if not execute:
        return {
            "status": "PROFILE_INTEGRATION_READY",
            "profiles": list(profiles),
            "method_version": METHOD_VERSION,
            "full64_assembly": "BLOCKED_UNTIL_64_VALIDATED_LAYERS",
            "next_exact_command": "& .\\.venv\\Scripts\\python.exe scripts\\run_phase03_integration.py --run-dir <run-dir> --source-dir <pinned-qwen-source> --finalist-lock <finalist-lock> --activation-manifest <FIT-DEV-layer0> --execute --json",
        }
    if not source_dir.is_dir() or not finalist_lock.is_file() or not activation_manifest.is_file():
        return {"status": "BLOCKED", "blocker_code": "INTEGRATION_INPUTS_REQUIRED"}
    activation_payload = _validate_manifest_identity(activation_manifest)
    profile_results: dict[str, Any] = {}
    receipt_dir = run_dir / "phase-03" / "integration"
    for profile_name in profiles:
        profile, contract = load_active_config(repo_root / "configs" / f"{profile_name}.yaml")
        if contract.profile_name != profile_name:
            raise ValueError(f"active contract/profile mismatch: {profile_name}")
        source = _source_identity(source_dir, profile)
        entries = lock.get("finalists", {}).get(profile_name)
        if not isinstance(entries, Mapping) or entries.get("status") != "DEV_FINALIST":
            raise ValueError(f"locked finalist profile missing: {profile_name}")
        candidates = entries.get("finalists")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError(f"expected exactly two locked finalists for {profile_name}")
        candidate_results = [
            _candidate_integration(
                repo_root=repo_root,
                source_dir=source_dir,
                profile=profile,
                source=source,
                candidate=candidate,
                activation_manifest=activation_manifest,
                activation_payload=activation_payload,
                lock_method_version=str(lock["method_version"]),
                max_tokens=max_tokens,
            )
            for candidate in candidates
        ]
        profile_receipt = {
            "schema_version": 1,
            "receipt_type": "dense2moe-phase03-profile-integration",
            "status": "PROFILE_INTEGRATION_GREEN",
            "profile": profile_name,
            "topology": profile.topology_id,
            "method_version": str(lock["method_version"]),
            "finalist_lock": str(finalist_lock),
            "finalist_lock_sha256": sha256_file(finalist_lock),
            "source": source,
            "profile_config": profile.as_dict(),
            "candidates": candidate_results,
            "opened_evaluation_tiers": [],
            "full64_assembly": "NOT_ATTEMPTED",
            "code_commit": current_git_commit(),
        }
        profile_path = receipt_dir / f"{profile_name}.json"
        write_immutable_json(profile_path, profile_receipt)
        profile_results[profile_name] = {"path": str(profile_path), "sha256": sha256_file(profile_path), "status": profile_receipt["status"]}
    aggregate = {
        "schema_version": 1,
        "receipt_type": "dense2moe-phase03-profile-integration-aggregate",
        "status": "PROFILE_INTEGRATION_GREEN",
        "profiles": profile_results,
        "profile_count": len(profile_results),
        "method_version": str(lock["method_version"]),
        "finalist_lock": str(finalist_lock),
        "finalist_lock_sha256": sha256_file(finalist_lock),
        "opened_evaluation_tiers": [],
        "full64_assembly": "BLOCKED_UNTIL_64_VALIDATED_LAYERS",
        "code_commit": current_git_commit(),
    }
    aggregate_path = receipt_dir / "integration-receipt.json"
    write_immutable_json(aggregate_path, aggregate)
    return aggregate | {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=False)
    parser.add_argument("--source-dir", type=Path, required=False)
    parser.add_argument("--finalist-lock", type=Path, required=False)
    parser.add_argument("--activation-manifest", type=Path, required=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fresh-process", action="store_true")
    parser.add_argument("--profile-name")
    parser.add_argument("--partition-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--expected-output-sha256")
    args = parser.parse_args()
    if args.fresh_process:
        required = (args.source_dir, args.profile_name, args.partition_path, args.checkpoint_path, args.activation_manifest, args.expected_output_sha256)
        if any(value is None for value in required):
            raise SystemExit("fresh-process reload requires source, profile, partition, checkpoint, activation, and expected digest")
        result = _fresh_process_reload(
            source_dir=args.source_dir,
            profile_name=args.profile_name,
            partition_path=args.partition_path,
            checkpoint_path=args.checkpoint_path,
            activation_manifest=args.activation_manifest,
            expected_output_sha256=args.expected_output_sha256,
            max_tokens=args.max_tokens,
        )
    else:
        if args.run_dir is None or args.source_dir is None or args.finalist_lock is None or args.activation_manifest is None:
            raise SystemExit("--run-dir, --source-dir, --finalist-lock, and --activation-manifest are required")
        result = run_integration(
            run_dir=args.run_dir,
            source_dir=args.source_dir,
            finalist_lock=args.finalist_lock,
            activation_manifest=args.activation_manifest,
            execute=args.execute,
            max_tokens=args.max_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {"PROFILE_INTEGRATION_READY", "PROFILE_INTEGRATION_GREEN", "FRESH_PROCESS_RELOAD_GREEN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
