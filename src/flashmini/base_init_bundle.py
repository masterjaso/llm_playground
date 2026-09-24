"""Deterministic v4 InitSpec, tensor manifest, shard plan, and donor bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import save_file

from .base_init_accounting import build_meta_model, parameter_report
from .base_init_config import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_ID,
    MODEL_ID,
    DEFAULT_CONFIG_PATH,
    FlashMini50BConfig,
    load_config,
)
from .base_init_optimizer import classify_parameters, optimizer_contract

SCHEMA_VERSION = 1
DEFAULT_BUNDLE = Path("flashmini_50b_base_init_v1")
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        return ""


def _name_seed(name: str, global_seed: int, chunk_index: int) -> int:
    payload = f"flashmini-init-v1\0{global_seed}\0{name}\0{chunk_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little") % (2**63 - 1)


def init_law(name: str, category: str) -> dict[str, Any]:
    if name.endswith("A_log"):
        return {"kind": "log_uniform", "low": 0.01, "high": 16.0, "dtype": "float32_then_bfloat16"}
    if name.endswith("dt_bias"):
        return {"kind": "ones", "dtype": "bfloat16"}
    if name.endswith("norm_offset") or name.endswith("key_norm.offset") or name.endswith("query_norm.offset") or name.endswith("conv_norm.offset") or name.endswith("hidden_norm.offset") or name.endswith("embedding_norm.offset"):
        return {"kind": "zeros", "dtype": "bfloat16"}
    if name.endswith("conv1d.weight") and (".ple." in name or name.startswith("ple.")):
        return {"kind": "zeros", "dtype": "bfloat16"}
    if category.endswith("ple_table"):
        return {"kind": "normal", "mean": 0.0, "std": 0.02, "dtype": "bfloat16"}
    if name.endswith("shared_gate.weight"):
        return {"kind": "normal", "mean": 0.0, "std": 0.02, "dtype": "bfloat16"}
    return {"kind": "normal", "mean": 0.0, "std": 0.02, "dtype": "bfloat16"}


def build_manifest(config: FlashMini50BConfig) -> dict[str, Any]:
    """Build a complete tensor manifest from the meta model, without storage."""
    model = build_meta_model(config)
    report = parameter_report(config, model)
    global_seed = int(config.section("initialization")["global_seed"])
    tensors = []
    for item in report["tensors"]:
        name, category = item["name"], item["category"]
        operator = classify_parameters([item])[0]
        tensors.append({
            "name": name,
            "shape": item["shape"],
            "dtype": DTYPE_NAME,
            "numel": item["numel"],
            "category": category,
            "optimizer": operator.family,
            "logical_operator": operator.logical_operator,
            "init_law": init_law(name, category),
            "init_name_seed": {
                "algorithm": "flashmini-sha256-name-seed-v1",
                "global_seed": global_seed,
                "chunk_elements": 1_048_576,
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "config_sha256": config.config_sha256,
        "architecture_sha256": config.architecture_sha256,
        "tensors": tensors,
        "tensor_count": len(tensors),
        "base_tensor_count": sum(not item["name"].startswith("mtp.") for item in tensors),
        "mtp_tensor_count": sum(item["name"].startswith("mtp.") for item in tensors),
        "parameter_counts": {
            "base": report["base_total_learned_parameters"],
            "mtp": report["mtp_total_learned_parameters"],
            "total": report["checkpoint_total_learned_parameters"],
            "base_active_per_token": report["base_active_parameters_per_token"],
        },
    }


def plan_shards(manifest: dict[str, Any], *, target_shard_bytes: int = 4 * 1024**3) -> dict[str, Any]:
    target = int(target_shard_bytes)
    if target <= 0:
        raise ValueError("target_shard_bytes must be positive")
    shards: list[dict[str, Any]] = []
    current: list[str] = []
    current_bytes = 0
    for item in sorted(manifest["tensors"], key=lambda value: value["name"]):
        size = item["numel"] * 2
        if ".ple.tables." in item["name"]:
            shards.append({"shard": f"model-{len(shards) + 1:05d}.safetensors", "tensors": [item["name"]], "bytes": size, "kind": "ple_hash_head"})
            continue
        if current and current_bytes + size > target:
            shards.append({"shard": f"model-{len(shards) + 1:05d}.safetensors", "tensors": current, "bytes": current_bytes, "kind": "dense_or_expert"})
            current, current_bytes = [], 0
        current.append(item["name"])
        current_bytes += size
    if current:
        shards.append({"shard": f"model-{len(shards) + 1:05d}.safetensors", "tensors": current, "bytes": current_bytes, "kind": "dense_or_expert"})
    for shard in shards:
        if shard["bytes"] > target and shard["kind"] != "ple_hash_head":
            raise ValueError(f"tensor cannot fit target shard: {shard}")
        shard["sha256"] = None
        shard["status"] = "planned_not_materialized"
    return {"schema_version": SCHEMA_VERSION, "target_shard_bytes": target, "shards": shards, "shard_count": len(shards), "planned_bytes": sum(item["bytes"] for item in shards)}


def init_spec(config: FlashMini50BConfig, manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "source_git_sha": git_sha(repo_root),
        "config_sha256": config.config_sha256,
        "architecture_sha256": config.architecture_sha256,
        "tokenizer": config.section("tokenizer"),
        "global_initialization_seed": config.section("initialization")["global_seed"],
        "deterministic_algorithm": config.section("initialization")["identity_algorithm"],
        "dtype": DTYPE_NAME,
        "ordinary_initializer": config.section("initialization")["ordinary_matrices"],
        "special_initializers": {key: config.section("initialization")[key] for key in ("gdn", "hc", "moe", "ple", "mtp")},
        "optimizer_contract": optimizer_contract(),
        "parameter_counts": manifest["parameter_counts"],
        "shard_mapping": "model.safetensors.index.json",
        "tensor_manifest": "tensor_manifest.json",
        "materialization": "python -m flashmini.base_init_bundle materialize --bundle . --output <checkpoint-dir>",
    }


def materialize_tensor(item: dict[str, Any], *, device: str = "cpu") -> torch.Tensor:
    """Materialize one tensor deterministically by stable name and chunk."""
    shape = tuple(item["shape"])
    dtype = getattr(torch, item["dtype"])
    output = torch.empty(shape, dtype=dtype, device=device)
    law = item["init_law"]
    global_seed = int(item["init_name_seed"]["global_seed"])
    chunk_elements = int(item["init_name_seed"]["chunk_elements"])
    flat = output.reshape(-1)
    for chunk_index, start in enumerate(range(0, flat.numel(), chunk_elements)):
        stop = min(flat.numel(), start + chunk_elements)
        generator = torch.Generator(device=device).manual_seed(_name_seed(item["name"], global_seed, chunk_index))
        count = stop - start
        if law["kind"] == "zeros":
            values = torch.zeros(count, dtype=torch.float32, device=device)
        elif law["kind"] == "ones":
            values = torch.ones(count, dtype=torch.float32, device=device)
        elif law["kind"] == "log_uniform":
            values = torch.rand(count, generator=generator, dtype=torch.float32, device=device)
            values = (law["low"] + values * (law["high"] - law["low"])).log()
        else:
            values = torch.randn(count, generator=generator, dtype=torch.float32, device=device)
            values = values * law["std"] + law["mean"]
        flat[start:stop].copy_(values.to(dtype))
    return output


def materialize(bundle: Path | str, output: Path | str, *, only_shards: Iterable[int] | None = None) -> dict[str, Any]:
    """Materialize planned shards one at a time; never holds the full model."""
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle / "tensor_manifest.json").read_text())
    shard_plan = json.loads((bundle / "shard_plan.json").read_text())
    by_name = {item["name"]: item for item in manifest["tensors"]}
    output.mkdir(parents=True, exist_ok=True)
    selected = set(only_shards) if only_shards is not None else None
    emitted = []
    for index, shard in enumerate(shard_plan["shards"]):
        if selected is not None and index not in selected:
            continue
        path = output / shard["shard"]
        tensors = {}
        for name in shard["tensors"]:
            tensor = materialize_tensor(by_name[name])
            tensors[name] = tensor
            del tensor
        save_file(tensors, str(path))
        digest = sha256_file(path)
        shard["sha256"] = digest
        shard["status"] = "materialized"
        emitted.append({"shard": shard["shard"], "sha256": digest, "bytes": path.stat().st_size})
    index = {"metadata": {"total_size": sum(item["bytes"] for item in shard_plan["shards"]), "total_parameters": manifest["parameter_counts"]["total"]}, "weight_map": {name: shard["shard"] for shard in shard_plan["shards"] for name in shard["tensors"]}}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (bundle / "shard_plan.json").write_text(json.dumps(shard_plan, indent=2, sort_keys=True) + "\n")
    return {"emitted": emitted, "output": str(output), "all_shards_emitted": len(emitted) == len(shard_plan["shards"])}


def write_bundle(destination: Path | str = DEFAULT_BUNDLE, *, config_path: Path | str = DEFAULT_CONFIG_PATH, repo_root: Path | str | None = None) -> dict[str, Any]:
    destination = Path(destination)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest = build_manifest(config)
    plan = plan_shards(manifest, target_shard_bytes=config.section("storage")["target_shard_bytes"])
    spec = init_spec(config, manifest, repo_root)
    config_target = destination / "flashmini_50b_base_init_v1.yaml"
    shutil.copy2(config_path, config_target)
    (destination / "init_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (destination / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (destination / "parameter_report.json").write_text(json.dumps(parameter_report(config), indent=2, sort_keys=True) + "\n")
    (destination / "shard_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (destination / "architecture.md").write_text("""# FlashMini-50B-Base architecture v4\n\nThis bundle freezes `FlashMini-50B-Base` / `FlashMini-50B-Base-Init-v1`. The canonical YAML is the source of truth. Base and MTP namespaces are separate; MTP shares the main embedding and LM head. PLE hash heads are independent tensors. No training has been run.\n""")
    (destination / "training_recipe.md").write_text("""# Training recipe boundary\n\nThis is an initialization handoff, not a training recipe. Donor-side decisions intentionally remain open: global token batch, peak learning rate, teacher-data percentage, long-context curriculum, and optimizer hyperparameters other than the frozen Muon/Adam taxonomy. MTP auxiliary coefficient is 0.30 for the first approximately 70% of training and 0.10 for the final approximately 30%; the main loss weight is 1.0. Canonical training uses ground-truth shifted embeddings and no sampled rollout.\n""")
    (destination / "donor_handoff.md").write_text(f"""# Donor handoff\n\n**BASE PARAMETER COUNT:** {manifest['parameter_counts']['base']}  \n**MTP PARAMETER COUNT:** {manifest['parameter_counts']['mtp']}  \n**TOTAL CHECKPOINT PARAMS:** {manifest['parameter_counts']['total']}  \n\nRun the materializer first: `python -m flashmini.base_init_bundle materialize --bundle . --output checkpoint`.\n\nThen run the donor smoke test: `pytest -q tests/flashmini/test_flashmini_50b_base_init.py`.\n\nThe bundle contains a deterministic InitSpec and planned safetensors shards. No full 50B tensor set was materialized on the workstation. No optimizer state or training checkpoint is included. The tokenizer artifact is pending and must be frozen before the first optimizer step.\n""")
    (destination / "source_git_sha.txt").write_text(git_sha(repo_root) + "\n")
    (destination / "tokenizer_manifest.json").write_text(json.dumps({"status": "pending", "contract": config.section("tokenizer")}, indent=2, sort_keys=True) + "\n")
    (destination / "environment.lock").write_text("torch>=2.7\nsafetensors>=0.4\npyyaml>=6.0\n")
    (destination / "validation_report.json").write_text(json.dumps({"structural": "pending test run", "shards": "planned_not_materialized", "surrogate": "pending test run"}, indent=2, sort_keys=True) + "\n")
    (destination / "materialization_command.txt").write_text("python -m flashmini.base_init_bundle materialize --bundle . --output checkpoint\n")
    (destination / "donor_smoke_test_command.txt").write_text("pytest -q tests/flashmini/test_flashmini_50b_base_init.py\n")
    return {"bundle": str(destination), "config_sha256": config.config_sha256, "architecture_sha256": config.architecture_sha256, "shard_count": plan["shard_count"], "planned_bytes": plan["planned_bytes"]}


__all__ = ["build_manifest", "init_spec", "materialize", "materialize_tensor", "plan_shards", "write_bundle"]
