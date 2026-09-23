"""Frozen FlashMini-1B production contract and parameter accounting.

The preview run is the first ten billion tokens of one declared one-hundred
billion token trajectory.  This module is intentionally independent of Kaggle
or a particular data provider: it freezes identities and policies before a
worker consumes its first token, and rejects drift on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .accounting import count_parameters
from .config import (
    FlashMiniConfig,
    GatedDeltaNetConfig,
    KVConfig,
    MoEConfig,
    PLEConfig,
)
from .data_v4.recipes import load_recipe, recipe_hash
from .data_v4.registry import merge_source_lock, validate_source_lock
from .data_v4.tokenizer import TokenizerSpec, load_frozen_spec
from .models import FlashMiniModel

CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_VERSION = 3
TOTAL_TRAINING_TOKENS = 100_000_000_000
PREVIEW_PAUSE_TOKENS = 10_000_000_000
FOUNDATION_TOKENS = 80_000_000_000
QUALITY_TOKENS = 15_000_000_000
LONG_COHERENT_TOKENS = 5_000_000_000
CHECKPOINT_INTERVAL_TOKENS = 25_000_000
DEFAULT_SEQUENCE_LENGTH = 2048

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/flashmini/flashmini_1b_v1.yaml"
DEFAULT_RECIPE = REPO_ROOT / "training_data/recipes/flashmini_1b_full_v1.yaml"
DEFAULT_FOUNDATION_RECIPE = REPO_ROOT / "training_data/recipes/flashmini_1b_foundation_v1.yaml"
DEFAULT_TOKENIZER = REPO_ROOT / "training_data/tokenizer/production.yaml"
DEFAULT_REGISTRY = REPO_ROOT / "training_data/registry/sources.yaml"
DEFAULT_SOURCE_LOCK = REPO_ROOT / "training_data/registry/source_snapshot.lock.json"
DEFAULT_SOURCE_AUDIT = REPO_ROOT / "training_data/manifests/releases/flashmini-1b-foundation-source-audit.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def source_fingerprint(repo_root: Path = REPO_ROOT) -> str:
    """Hash the code and frozen contract inputs used by the production worker."""
    root = Path(repo_root)
    paths: list[Path] = []
    for directory in (root / "src/flashmini", root / "scripts"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*.py") if path.is_file())
    paths.extend([
        root / "configs/flashmini/flashmini_1b_v1.yaml",
        root / "training_data/tokenizer/production.yaml",
        root / "training_data/registry/sources.yaml",
        root / "training_data/registry/source_snapshot.lock.json",
        root / "training_data/recipes/flashmini_1b_foundation_v1.yaml",
        root / "training_data/recipes/flashmini_1b_full_v1.yaml",
    ])
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.is_file()}):
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_config_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"model config must be a mapping: {path}")
    nested = {
        "moe": MoEConfig,
        "gdn": GatedDeltaNetConfig,
        "ple": PLEConfig,
        "kvc": KVConfig,
    }
    for key, cls in nested.items():
        value = raw.get(key, {})
        if not isinstance(value, dict):
            raise TypeError(f"model config field {key} must be a mapping")
        raw[key] = cls(**value)
    return raw


def load_production_config(path: Path | str = DEFAULT_CONFIG) -> FlashMiniConfig:
    """Load and validate the frozen v1 1B architecture config."""
    config = FlashMiniConfig(**_load_config_mapping(Path(path)))
    if config.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError("production FlashMini-1B config must use architecture_version=3")
    if config.max_seq_len != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError("initial production run requires the static 2048 sequence shape")
    if not (config.use_ple and config.kvc.enabled and config.use_hyperconnection):
        raise ValueError("production config must keep PLE, KVC, and HyperConnections enabled")
    if config.moe.num_experts != 16 or config.moe.top_k != 2:
        raise ValueError("production config must preserve the accepted D MoE ratios")
    return config


def model_config_sha256(config: FlashMiniConfig) -> str:
    return canonical_sha256(config.to_dict())


def _module_category(name: str, module: torch.nn.Module,
                     modules: dict[str, torch.nn.Module] | None = None) -> str:
    if name.startswith(("embed", "head")):
        return "embedding"
    if name.startswith("ple"):
        return "ple"
    if ".moe." in name:
        return "moe"
    if ".mixer." in name:
        # Parameters belong to Linear/Conv children, so inspect the owning
        # mixer rather than the immediate parent module.
        mixer_name = name.split(".mixer.", 1)[0] + ".mixer"
        mixer = (modules or {}).get(mixer_name, module)
        return "attention" if type(mixer).__name__ == "CausalAttention" else "gdn"
    return "dense_shared"


def parameter_report(config: FlashMiniConfig, model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Return explicit dense/sparse parameter totals for the freeze manifest."""
    if model is None:
        with torch.device("meta"):
            model = FlashMiniModel(config)
    counts = count_parameters(model, config)
    categories = {"embedding": 0, "attention": 0, "gdn": 0, "moe": 0,
                  "ple": 0, "dense_shared": 0}
    modules = dict(model.named_modules())
    for name, parameter in model.named_parameters():
        # Module lookup is cheap at freeze time and keeps classification tied to
        # the real implementation rather than guessed formulas.
        parent_name = name.rsplit(".", 1)[0] if "." in name else name
        module = modules.get(parent_name, model)
        categories[_module_category(name, module, modules)] += int(parameter.numel())
    # KVC is a QAT transform, not a separately learned table.  Report it
    # explicitly so sparse parameter counts cannot be mistaken for dense ones.
    active_moe = 0
    for name, parameter in model.named_parameters():
        if ".moe.experts." in name:
            active_moe += int(parameter.numel() * config.moe.top_k / config.moe.num_experts)
        elif ".moe.shared_experts." in name or ".moe.router." in name:
            active_moe += int(parameter.numel())
    report = {
        "total_learned_parameters": int(counts.total),
        "core_parameters": int(counts.core),
        "embedding_head_parameters": int(counts.embedding_head),
        "active_parameters_per_token": int(counts.active_per_token),
        # Keep the embedding/head matrix separate: the contract explicitly
        # reports embedding parameters and dense/shared parameters as distinct
        # categories, while ``categories_sum`` remains the total learned count.
        "dense_shared_parameters": int(categories["dense_shared"]),
        "embedding_parameters": int(categories["embedding"]),
        "attention_parameters": int(categories["attention"]),
        "gdn_parameters": int(categories["gdn"]),
        "moe_expert_parameters": int(categories["moe"]),
        "active_moe_expert_parameters_per_token": int(active_moe),
        "ple_parameters": int(categories["ple"]),
        "kvc_parameter_contribution": 0,
        "kvc_semantics": "qat_e2m1_values_e4m3_group_scales_16_cross_layer_share_2",
        "num_experts": int(config.moe.num_experts),
        "top_k": int(config.moe.top_k),
        "shared_experts": int(config.moe.shared_experts),
        "sparse_expert_parameters": int(categories["moe"]),
        "categories_sum": int(sum(categories.values())),
        "categories": categories,
    }
    if report["categories_sum"] != report["total_learned_parameters"]:
        raise ValueError("parameter category accounting does not sum to total")
    return report


def _git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def validate_source_lock_for_recipe(
    recipe: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
) -> dict[str, Any]:
    registry = yaml.safe_load(Path(registry_path).read_text()) or {}
    lock = json.loads(Path(source_lock_path).read_text())
    health = validate_source_lock(registry, lock)
    if not health["valid"]:
        raise ValueError(f"source lock is not immutable: {health}")
    merged = merge_source_lock(registry, lock, require_immutable=True)
    source_ids = sorted({str(source) for domain in recipe["domains"].values()
                         for source in domain.get("sources", [])})
    missing = [source for source in source_ids if source not in merged.get("sources", {})]
    if missing:
        raise ValueError(f"recipe references sources missing from source lock: {missing}")
    selected = {source: merged["sources"][source] for source in source_ids}
    return {"health": health, "sources": selected}


def trajectory_contract() -> dict[str, Any]:
    return {
        "total_training_tokens": TOTAL_TRAINING_TOKENS,
        "preview_pause_tokens": PREVIEW_PAUSE_TOKENS,
        "stages": [
            {"name": "foundation", "tokens": FOUNDATION_TOKENS, "recipe": "flashmini_1b_foundation_v1"},
            {"name": "quality", "tokens": QUALITY_TOKENS, "recipe": "flashmini_1b_quality_v1"},
            {"name": "long_coherent", "tokens": LONG_COHERENT_TOKENS, "recipe": "flashmini_1b_long_v1"},
        ],
        "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
        "validation": {
            "lightweight_every_tokens": 250_000_000,
            "major_tokens": [1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000],
            "deterministic_held_out": True,
        },
        "schedule": {
            "kind": "warmup_then_cosine_full_horizon",
            "decay_horizon_tokens": TOTAL_TRAINING_TOKENS,
            "pause_is_not_schedule_end": True,
        },
    }


def freeze_manifest(
    output: Path | str,
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    recipe_path: Path | str = DEFAULT_RECIPE,
    foundation_recipe_path: Path | str = DEFAULT_FOUNDATION_RECIPE,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    registry_path: Path | str = DEFAULT_REGISTRY,
    source_lock_path: Path | str = DEFAULT_SOURCE_LOCK,
    source_audit_path: Path | str = DEFAULT_SOURCE_AUDIT,
    run_id: str = "flashmini-1b-foundation-v1",
) -> dict[str, Any]:
    """Create an atomic machine-readable run freeze manifest."""
    config_path, recipe_path = Path(config_path), Path(recipe_path)
    foundation_recipe_path = Path(foundation_recipe_path)
    tokenizer_path, registry_path, source_lock_path = map(Path, (tokenizer_path, registry_path, source_lock_path))
    source_audit_path = Path(source_audit_path)
    config = load_production_config(config_path)
    recipe = load_recipe(recipe_path)
    foundation = load_recipe(foundation_recipe_path)
    tokenizer: TokenizerSpec = load_frozen_spec(tokenizer_path)
    source_info = validate_source_lock_for_recipe(recipe, registry_path=registry_path,
                                                  source_lock_path=source_lock_path)
    # The first ten billion tokens are explicitly the prefix of foundation, not
    # a separate recipe.  Keep the source/domain contract visible in the file.
    foundation_targets = {
        name: int(PREVIEW_PAUSE_TOKENS * float(domain["weight"]))
        for name, domain in foundation["domains"].items()
    }
    source_lock_sha = sha256_file(source_lock_path)
    tokenizer_sha = sha256_file(tokenizer_path)
    source_audit_sha = sha256_file(source_audit_path) if source_audit_path.is_file() else None
    config_sha = model_config_sha256(config)
    view_fingerprint = canonical_sha256({
        "view_id": "flashmini-1b-virtual-foundation-v1",
        "recipe_hash": recipe_hash(recipe),
        "foundation_recipe_hash": recipe_hash(foundation),
        "source_lock_sha256": source_lock_sha,
        "source_audit_sha256": source_audit_sha,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer_spec_sha256": tokenizer_sha,
        "sequence_length": config.max_seq_len,
        "packing_policy": "document_mix",
        "selection": "sha256_seeded_domain_source_document_v1",
    })
    report = parameter_report(config)
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.is_file() else None
    blocked_sources = [sid for sid, result in (source_audit or {}).get("probe", {}).get("results", {}).items()
                       if result.get("status") != "OK"]
    manifest = {
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "state": f"BLOCKED_SOURCE_SCHEMA_{'_'.join(blocked_sources)}" if blocked_sources else "INFRASTRUCTURE_READY",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source_fingerprint": source_fingerprint(),
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture": config.to_dict(),
        "model_config_sha256": config_sha,
        "parameter_report": report,
        "tokenizer": {**tokenizer.as_dict(), "spec_sha256": tokenizer_sha},
        "training_recipe": {
            "name": recipe["name"], "hash": recipe_hash(recipe),
            "target_tokens": int(recipe["target_tokens"]),
            "foundation_prefix_recipe": foundation["name"],
            "foundation_prefix_hash": recipe_hash(foundation),
            "preview_pause_tokens": PREVIEW_PAUSE_TOKENS,
            "preview_domain_targets": foundation_targets,
            # Keep the exact domain/source mapping in the freeze so a worker
            # can reconstruct the virtual view without depending on a mutable
            # checkout or a hand-edited recipe at resume time.
            "foundation_recipe": foundation,
        },
        "trajectory": trajectory_contract(),
        "optimizer_policy": {
            "family": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
            "bf16": True,
            "updates_per_logical_batch": 1,
        },
        "data_view": {
            "kind": "virtual_remote_shards",
            "view_id": "flashmini-1b-virtual-foundation-v1",
            "fingerprint": view_fingerprint,
            "source_lock_sha256": source_lock_sha,
            "source_lock_path": str(source_lock_path.relative_to(REPO_ROOT))
            if source_lock_path.is_relative_to(REPO_ROOT) else str(source_lock_path),
            "sources": source_info["sources"],
            "source_audit_path": str(source_audit_path.relative_to(REPO_ROOT))
            if source_audit_path.is_relative_to(REPO_ROOT) else str(source_audit_path),
            "source_audit": source_audit,
            "source_audit_sha256": source_audit_sha,
            "blocked_sources": blocked_sources,
            "sequence_length": config.max_seq_len,
            "packing_policy": "document_mix",
            "exact_token_accounting": True,
            "bounded_host_cache": True,
            "full_corpus_materialization_required": False,
        },
        "runtime_policy": {
            "backend": "pytorch_xla",
            "topology": "TPU-v5e-8",
            "world_size": 8,
            "static_shapes": True,
            "ple_placement": "tpu_resident",
            "kvc_enabled": True,
            "checkpoint_every_tokens": CHECKPOINT_INTERVAL_TOKENS,
            "retain_full_checkpoints": 1,
            "metrics_append_only": True,
            "kaggle_hard_session_seconds": 21600,
            "kaggle_soft_stop_seconds": 18000,
            "checkpoint_upload_reserve_seconds": 3600,
        },
        "provenance": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output)
    directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def validate_freeze_manifest(manifest: dict[str, Any], *, strict_files: bool = True) -> None:
    required = {"schema_version", "checkpoint_schema_version", "run_id", "source_fingerprint", "architecture",
                "model_config_sha256", "parameter_report", "tokenizer", "training_recipe",
                "trajectory", "data_view", "runtime_policy"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"freeze manifest missing fields: {missing}")
    config = FlashMiniConfig.from_dict(manifest["architecture"])
    if model_config_sha256(config) != manifest["model_config_sha256"]:
        raise ValueError("freeze manifest model-config SHA256 mismatch")
    if manifest["trajectory"]["total_training_tokens"] != TOTAL_TRAINING_TOKENS:
        raise ValueError("freeze manifest does not declare the 100B trajectory")
    if manifest["trajectory"]["preview_pause_tokens"] != PREVIEW_PAUSE_TOKENS:
        raise ValueError("freeze manifest preview pause must be 10B")
    if manifest["data_view"].get("sequence_length") != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError("freeze manifest must use the static 2048 sequence shape")
    if strict_files and manifest["tokenizer"].get("spec_sha256"):
        tok_path = REPO_ROOT / "training_data/tokenizer/production.yaml"
        if tok_path.is_file() and sha256_file(tok_path) != manifest["tokenizer"]["spec_sha256"]:
            raise ValueError("freeze manifest tokenizer spec changed")
    if strict_files and manifest.get("source_fingerprint") and manifest["source_fingerprint"] != source_fingerprint():
        raise ValueError("freeze manifest source fingerprint changed")


def next_checkpoint_threshold(tokens_seen: int, *, interval: int = CHECKPOINT_INTERVAL_TOKENS) -> int:
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    return ((int(tokens_seen) // interval) + 1) * interval


def checkpoint_boundary(tokens_before: int, batch_tokens: int, *, interval: int = CHECKPOINT_INTERVAL_TOKENS) -> dict[str, int] | None:
    """Return the first crossed 25M threshold, with bounded overshoot."""
    before, after = int(tokens_before), int(tokens_before) + int(batch_tokens)
    if batch_tokens <= 0:
        raise ValueError("logical batch token count must be positive")
    threshold = next_checkpoint_threshold(before, interval=interval)
    if after < threshold:
        return None
    overshoot = after - threshold
    if overshoot >= batch_tokens:
        raise ValueError("checkpoint boundary overshoot exceeds one logical batch")
    return {"checkpoint_threshold_tokens": threshold,
            "actual_tokens_seen": after, "overshoot_tokens": overshoot}


def curriculum_stage(tokens_seen: int) -> str:
    """Return the declared stage at a cumulative trajectory position."""
    tokens_seen = int(tokens_seen)
    if tokens_seen < 0 or tokens_seen > TOTAL_TRAINING_TOKENS:
        raise ValueError("trajectory position is outside the 100B contract")
    if tokens_seen < FOUNDATION_TOKENS:
        return "foundation"
    if tokens_seen < FOUNDATION_TOKENS + QUALITY_TOKENS:
        return "quality"
    return "long_coherent"


def preview_is_foundation(tokens_seen: int) -> bool:
    return 0 <= int(tokens_seen) <= PREVIEW_PAUSE_TOKENS and curriculum_stage(int(tokens_seen)) == "foundation"
