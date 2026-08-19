"""Bounded replay adapters for retained V2.3/V2.4 structural candidates.

This module deliberately keeps replay separate from the historical status-only
inventory command.  It reads the immutable paired activation artifacts,
executes the stored candidate tensor state, and accumulates FIT-TRAIN/FIT-DEV
metrics without concatenating the corpus in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..partition import partition_indices
from ..provenance import current_git_commit
from ..science.v23_designs import DESIGN_REGISTRY
from .decision import decide_candidate
from .generalization import classify_generalization
from .receipts import (
    build_structural_generalization_receipt,
    validate_structural_receipt,
)
from .registry import POLICY_HASH, STRUCTURAL_POLICY_VERSION
from .structural import StructuralMetricsAccumulator


class ReplayInputError(ValueError):
    """Raised when a retained replay input fails a deterministic contract."""


class ReplayValidationError(ValueError):
    """Raised when a computed replay result cannot be published as V2."""


@dataclass(frozen=True)
class PairedActivationBatch:
    """One bounded paired activation batch and its example identities."""

    inputs: Any
    targets: Any
    metadata: tuple[dict[str, Any], ...]
    manifest_path: Path
    shard_path: Path
    shard_index: int


@dataclass
class HistoricalCandidate:
    """Loaded historical Torch MoE candidate and its immutable identity."""

    model: Any
    metadata: dict[str, Any]
    checkpoint: Path
    metadata_path: Path
    state_sha256: str
    hidden_size: int
    intermediate_size: int
    routed_experts: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    top_k: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_path(raw: str | Path) -> Path:
    value = str(raw)
    if Path().anchor != "":  # native Windows; Path already understands slashes
        return Path(value)
    return Path(value.replace("\\", "/"))


def resolve_artifact_path(raw: str | Path, *, repo_root: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve an artifact path without permitting an implicit source fallback."""

    candidate = _normalise_path(raw)
    if candidate.is_absolute():
        return candidate
    roots: list[Path] = []
    if base_dir is not None:
        roots.append(Path(base_dir))
    roots.extend((Path(repo_root), Path(repo_root) / ".nsp", Path(repo_root) / "runs"))
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (roots[0] / candidate).resolve() if roots else candidate.resolve()


def _expected_split(value: str | None) -> str | None:
    if value is None:
        return None
    upper = str(value).upper().replace("_", "-")
    if upper in {"TRAIN", "FIT-TRAIN"}:
        return "FIT-TRAIN"
    if upper in {"DEV", "FIT-DEV", "VALIDATION"}:
        return "FIT-DEV"
    return upper


def _expand_metadata(records: Sequence[Mapping[str, Any]], count: int, split: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any] | None] = [None] * count
    if not records:
        return tuple({"independent_group": f"{split}:row:{index}", "group_identity": f"{split}:row:{index}"} for index in range(count))
    cursor = 0
    for item in records:
        if cursor >= count:
            break
        try:
            length = int(item.get("length", 0))
        except (TypeError, ValueError) as exc:
            raise ReplayInputError("activation record length is not an integer") from exc
        if length <= 0:
            raise ReplayInputError(f"activation record length is not positive: {length}")
        identity = str(item.get("example_id") or item.get("source_record_index") or f"{split}:record:{cursor}")
        metadata = {"independent_group": identity, "group_identity": identity}
        for key in ("source_family", "domain", "residual_difficulty", "hard_token"):
            if key in item:
                metadata[key] = item[key]
        # Pilot manifests retain the original example lengths even when a
        # stable-prefix shard contains fewer rows than the first example.  The
        # records are ordered, so consume only the prefix represented by this
        # shard; full captures consume the complete sequence of records.
        take = min(length, count - cursor)
        for index in range(cursor, cursor + take):
            rows[index] = dict(metadata)
        cursor += take
    if any(item is None for item in rows):
        raise ReplayInputError("activation records do not cover the complete shard")
    return tuple(item for item in rows if item is not None)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"activation manifest is unreadable: {target}") from exc
    if not isinstance(payload, dict):
        raise ReplayInputError("activation manifest must be a JSON object")
    if int(payload.get("schema_version", payload.get("schemaVersion", 0)) or 0) != 2:
        raise ReplayInputError("activation manifest schema_version must be 2")
    if str(payload.get("status", "")) not in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED"}:
        raise ReplayInputError(f"activation manifest is not complete: {payload.get('status')}")
    return payload


def iter_paired_activation_shards(
    manifest_path: str | Path,
    *,
    expected_split: str | None = None,
    repo_root: str | Path | None = None,
) -> Iterator[PairedActivationBatch]:
    """Yield complete paired shards after validating hashes and identities."""

    target = Path(manifest_path).resolve()
    payload = _load_manifest(target)
    expected = _expected_split(expected_split)
    actual_split = _expected_split(str(payload.get("split", "")))
    if expected is not None and actual_split != expected:
        raise ReplayInputError(f"manifest split mismatch: expected {expected}, got {actual_split}")
    if not payload.get("quality_gate_eligible", True):
        raise ReplayInputError("activation manifest is not quality-gate eligible")
    shard_entries = payload.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ReplayInputError("activation manifest has no shards")
    root = Path(repo_root) if repo_root is not None else target.parents[0]
    total = 0
    input_name_default = str(payload.get("input_tensor", "ffn_input"))
    target_name_default = str(payload.get("target_tensor", "dense_ffn_target"))
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:  # pragma: no cover - native replay requires it
        raise ReplayInputError("safetensors is required for paired activation replay") from exc
    for shard_index, entry in enumerate(shard_entries):
        if not isinstance(entry, Mapping) or not entry.get("path"):
            raise ReplayInputError(f"activation shard {shard_index} has no path")
        shard_path = resolve_artifact_path(entry["path"], repo_root=root, base_dir=target.parent)
        if not shard_path.is_file():
            raise ReplayInputError(f"activation shard is missing: {shard_path}")
        expected_hash = str(entry.get("sha256", ""))
        if expected_hash and sha256_file(shard_path) != expected_hash:
            raise ReplayInputError(f"activation shard SHA256 mismatch: {shard_path}")
        input_name = str(entry.get("input_tensor", input_name_default))
        target_name = str(entry.get("target_tensor", target_name_default))
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if input_name not in keys or target_name not in keys:
                    raise ReplayInputError(f"paired activation keys missing in {shard_path}: {input_name}, {target_name}")
                inputs = handle.get_tensor(input_name)
                targets = handle.get_tensor(target_name)
        except ReplayInputError:
            raise
        except Exception as exc:
            raise ReplayInputError(f"paired activation shard could not be loaded: {shard_path}") from exc
        if tuple(inputs.shape) != tuple(targets.shape) or inputs.ndim != 2:
            raise ReplayInputError(f"paired activation shape mismatch in {shard_path}: {tuple(inputs.shape)} vs {tuple(targets.shape)}")
        count = int(inputs.shape[0])
        declared_shape = entry.get("shape")
        if isinstance(declared_shape, Sequence) and list(declared_shape) != list(inputs.shape):
            raise ReplayInputError(f"declared shard shape mismatch in {shard_path}")
        metadata = _expand_metadata(entry.get("records", []), count, actual_split or "UNKNOWN")
        total += count
        yield PairedActivationBatch(inputs, targets, metadata, target, shard_path, shard_index)
    declared_count = payload.get("count")
    if declared_count is not None and int(declared_count) != total:
        raise ReplayInputError(f"manifest token count mismatch: declared {declared_count}, loaded {total}")


def iter_paired_activation_batches(
    manifest_path: str | Path,
    *,
    expected_split: str | None = None,
    repo_root: str | Path | None = None,
    batch_tokens: int = 256,
) -> Iterator[PairedActivationBatch]:
    """Yield bounded slices of validated paired activation shards."""

    if int(batch_tokens) <= 0:
        raise ValueError("batch_tokens must be positive")
    for shard in iter_paired_activation_shards(manifest_path, expected_split=expected_split, repo_root=repo_root):
        count = int(shard.inputs.shape[0])
        for start in range(0, count, int(batch_tokens)):
            stop = min(count, start + int(batch_tokens))
            yield PairedActivationBatch(
                shard.inputs[start:stop],
                shard.targets[start:stop],
                shard.metadata[start:stop],
                shard.manifest_path,
                shard.shard_path,
                shard.shard_index,
            )


def _infer_top_k(metadata: Mapping[str, Any], routed_experts: int) -> int:
    if metadata.get("top_k") is not None:
        value = int(metadata["top_k"])
    else:
        profile = str(metadata.get("profile", ""))
        match = re.search(r"top(\d+)", profile)
        if match:
            value = int(match.group(1))
        else:
            design_match = re.search(r"v23-design-([A-E])", profile)
            if not design_match:
                raise ReplayInputError(f"candidate top_k is not declared for profile {profile!r}")
            design = DESIGN_REGISTRY[design_match.group(1)]
            value = int(design.top_k)
    if value <= 0 or value > routed_experts:
        raise ReplayInputError(f"candidate top_k is outside routed expert count: top_k={value}, experts={routed_experts}")
    return value


def _strip_layer_prefix(state: Mapping[str, Any], layer: int) -> dict[str, Any]:
    prefix = f"model.layers.{int(layer)}."
    if all(str(key).startswith(prefix) for key in state):
        return {str(key)[len(prefix) :]: value for key, value in state.items()}
    if all(str(key).split(".", 1)[0] in {"router", "amplitude_router", "shared_gate_proj", "shared_up_proj", "shared_down_proj", "expert_gate_proj", "expert_up_proj", "expert_down_proj", "expert_scales"} for key in state):
        return {str(key): value for key, value in state.items()}
    raise ReplayInputError("candidate tensor state does not use the supported layer-0 schema")


def load_historical_candidate(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    *,
    device: str = "cpu",
    expected_layer: int = 0,
    expected_source_revision: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    checkpoint_hash_cache: dict[str, str] | None = None,
) -> HistoricalCandidate:
    """Load and strictly validate one retained Torch MoE candidate."""

    checkpoint = Path(checkpoint_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if not checkpoint.is_file() or not metadata_file.is_file():
        raise ReplayInputError(f"candidate checkpoint or metadata is missing: {checkpoint}, {metadata_file}")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"candidate metadata is unreadable: {metadata_file}") from exc
    if not isinstance(metadata, dict):
        raise ReplayInputError("candidate metadata must be an object")
    if int(metadata.get("layer", expected_layer)) != int(expected_layer):
        raise ReplayInputError("candidate layer identity mismatch")
    if expected_source_revision is not None and str(metadata.get("source_revision", "")) != str(expected_source_revision):
        raise ReplayInputError("candidate source revision mismatch")
    if str(metadata.get("routing_mode", "")) not in {"independent_positive", "normalized_softmax"}:
        raise ReplayInputError("candidate routing mode is unsupported")
    architecture = str(metadata.get("router_architecture", ""))
    if architecture != "torch-linear-topk-independent_positive-v1":
        raise ReplayInputError(f"candidate router architecture is unsupported: {architecture}")
    cache_key = str(checkpoint)
    state_hash = (checkpoint_hash_cache or {}).get(cache_key)
    if state_hash is None:
        state_hash = sha256_file(checkpoint)
        if checkpoint_hash_cache is not None:
            checkpoint_hash_cache[cache_key] = state_hash
    expected_hash = expected_checkpoint_sha256 or metadata.get("tensor_sha256")
    if expected_hash and str(expected_hash) != state_hash:
        raise ReplayInputError(f"candidate checkpoint SHA256 mismatch: {checkpoint}")
    try:
        from safetensors.torch import load_file  # type: ignore
        import torch
        from ..models.torch_moe import TorchQwen35SwiGLUMoE
    except ImportError as exc:  # pragma: no cover - native replay requires torch
        raise ReplayInputError("PyTorch and safetensors are required for candidate replay") from exc
    raw_state = load_file(str(checkpoint), device="cpu")
    state = _strip_layer_prefix(raw_state, expected_layer)
    try:
        router_shape = tuple(state["router.weight"].shape)
        shared_shape = tuple(state["shared_gate_proj.weight"].shape)
        expert_shape = tuple(state["expert_gate_proj.0.weight"].shape)
        down_shape = tuple(state["expert_down_proj.0.weight"].shape)
    except KeyError as exc:
        raise ReplayInputError(f"candidate state is missing required tensor: {exc.args[0]}") from exc
    if len(router_shape) != 2 or len(shared_shape) != 2 or len(expert_shape) != 2 or len(down_shape) != 2:
        raise ReplayInputError("candidate projection tensors must be matrices")
    routed_experts, hidden_size = map(int, router_shape)
    shared_intermediate_size, shared_hidden = map(int, shared_shape)
    expert_intermediate_size, expert_hidden = map(int, expert_shape)
    down_hidden, down_expert = map(int, down_shape)
    if hidden_size != shared_hidden or hidden_size != expert_hidden or down_hidden != hidden_size or down_expert != expert_intermediate_size:
        raise ReplayInputError("candidate projection geometry is inconsistent")
    intermediate_size = shared_intermediate_size + routed_experts * expert_intermediate_size
    if intermediate_size != 17_408 or hidden_size != 5_120:
        raise ReplayInputError(f"candidate geometry is outside Qwen3.8 layer-0 contract: hidden={hidden_size}, intermediate={intermediate_size}")
    top_k = _infer_top_k(metadata, routed_experts)
    partition = partition_indices(intermediate_size, routed_experts, expert_intermediate_size, shared_intermediate_size)
    try:
        model = TorchQwen35SwiGLUMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            routed_experts=routed_experts,
            expert_intermediate_size=expert_intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            top_k=top_k,
            routing_mode=str(metadata["routing_mode"]),
            partition=partition,
            dtype=torch.float32,
            device="cpu",
        )
        model.load_state_dict(state, strict=True)
        model.to(device=device)
        model.eval()
    except Exception as exc:
        raise ReplayInputError(f"candidate state could not be strictly reloaded: {checkpoint}") from exc
    return HistoricalCandidate(
        model=model,
        metadata=metadata,
        checkpoint=checkpoint,
        metadata_path=metadata_file,
        state_sha256=state_hash,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routed_experts=routed_experts,
        expert_intermediate_size=expert_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        top_k=top_k,
    )


def _routing_fields(metrics: dict[str, Any], counts: Sequence[int]) -> None:
    import numpy as np

    values = np.asarray([float(item) for item in counts], dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    dead = int((values <= 0).sum()) if values.size else 0
    load_cv = float(values.std() / mean) if mean > 0 else None
    utilization = float((values > 0).mean()) if values.size else None
    metrics.update(
        {
            "learned_load_cv": load_cv,
            "dead_expert_count": dead,
            "dead_expert_rate": (dead / len(values) if len(values) else None),
            "expert_utilization": utilization,
            "expert_counts": [float(item) for item in values.tolist()],
            "learned_router_status": "COMPUTED",
            "routing_health": float(utilization * max(0.0, 1.0 - min(1.0, load_cv))) if utilization is not None and load_cv is not None else None,
        }
    )


def _score_split(candidate: HistoricalCandidate, manifest_path: Path, split: str, *, repo_root: Path, batch_tokens: int, device: str) -> dict[str, Any]:
    import torch

    accumulator = StructuralMetricsAccumulator(max_distribution_samples=100_000)
    routing_counts = [0] * candidate.routed_experts
    for batch in iter_paired_activation_batches(manifest_path, expected_split=split, repo_root=repo_root, batch_tokens=batch_tokens):
        inputs = batch.inputs.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output, info = candidate.model(inputs, return_router=True)
        indices = info["indices"].detach().to(device="cpu")
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=candidate.routed_experts).tolist()
        routing_counts = [left + int(right) for left, right in zip(routing_counts, counts)]
        accumulator.update(
            batch.targets.detach().to(device="cpu", dtype=torch.float32),
            output.detach().to(device="cpu", dtype=torch.float32),
            metadata=batch.metadata,
            learned_assignments=indices,
        )
        del inputs, output, info, indices
    metrics = accumulator.finalize()
    _routing_fields(metrics, routing_counts)
    metrics["split"] = split
    metrics["batch_tokens"] = int(batch_tokens)
    metrics["device"] = str(device)
    return metrics


def _manifest_identity(path: Path, *, repo_root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_manifest(path)
    actual = _expected_split(str(payload.get("split", "")))
    if actual != _expected_split(split):
        raise ReplayInputError(f"manifest split mismatch: expected {split}, got {actual}")
    identity = {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "dataset_hash": payload.get("dataset_hash"),
        "source_revision": payload.get("source_revision"),
        "split": actual,
        "capture_kind": payload.get("capture_kind"),
        "status": payload.get("status"),
        "count": int(payload.get("count", 0)),
        "shard_count": len(payload.get("shards", [])),
    }
    return payload, identity


def replay_candidate_record(
    record: Mapping[str, Any],
    *,
    repo_root: str | Path,
    device: str = "cuda:1",
    batch_tokens: int = 256,
    runtime_lock_identity: Mapping[str, Any] | None = None,
    code_science_identity: Mapping[str, Any] | None = None,
    checkpoint_hash_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute one retained candidate and return a receipt-bearing result."""

    root = Path(repo_root).resolve()
    checkpoint = resolve_artifact_path(str(record.get("checkpoint", "")), repo_root=root)
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        for raw in record.get("raw_replayable_inputs", []):
            candidate = resolve_artifact_path(str(raw), repo_root=root)
            if candidate.suffix.lower() == ".json" and candidate.name == f"layer-{int(record.get('layer', 0)):04d}.json":
                metadata_path = candidate
                break
    train_raw = record.get("fit_train_data_identity", {})
    dev_raw = record.get("fit_dev_data_identity", {})
    train_manifest = resolve_artifact_path(str(train_raw.get("path", "")), repo_root=root)
    dev_manifest = resolve_artifact_path(str(dev_raw.get("path", "")), repo_root=root)
    train_payload, train_identity = _manifest_identity(train_manifest, repo_root=root, split="FIT-TRAIN")
    dev_payload, dev_identity = _manifest_identity(dev_manifest, repo_root=root, split="FIT-DEV")
    source_revision = str(train_payload.get("source_revision", ""))
    if source_revision != str(dev_payload.get("source_revision", "")):
        raise ReplayInputError("FIT-TRAIN/FIT-DEV source revisions differ")
    if train_payload.get("dataset_hash") != dev_payload.get("dataset_hash"):
        raise ReplayInputError("FIT-TRAIN/FIT-DEV dataset hashes differ")
    candidate = load_historical_candidate(
        checkpoint,
        metadata_path,
        device=device,
        expected_layer=int(record.get("layer", 0)),
        expected_source_revision=source_revision,
        expected_checkpoint_sha256=str(record.get("checkpoint_hash", "")) or None,
        checkpoint_hash_cache=checkpoint_hash_cache,
    )
    identity_checks = {
        "layer": int(candidate.metadata.get("layer", -1)) == int(record.get("layer", 0)),
        "source_revision": str(candidate.metadata.get("source_revision", "")) == source_revision,
        "dataset_hash": str(candidate.metadata.get("dataset_hash", "")) == str(train_payload.get("dataset_hash", "")),
        "checkpoint_hash": not record.get("checkpoint_hash") or candidate.state_sha256 == str(record.get("checkpoint_hash")),
        "fit_train_manifest_hash": not train_raw.get("sha256") or train_identity["manifest_sha256"] == str(train_raw.get("sha256")),
        "fit_dev_manifest_hash": not dev_raw.get("sha256") or dev_identity["manifest_sha256"] == str(dev_raw.get("sha256")),
    }
    if not all(identity_checks.values()):
        raise ReplayInputError(f"candidate/data identity mismatch: {sorted(key for key, value in identity_checks.items() if not value)}")
    fit_train = _score_split(candidate, train_manifest, "FIT-TRAIN", repo_root=root, batch_tokens=batch_tokens, device=device)
    fit_dev = _score_split(candidate, dev_manifest, "FIT-DEV", repo_root=root, batch_tokens=batch_tokens, device=device)
    generalization = classify_generalization(fit_train, fit_dev)
    decision = decide_candidate(
        fit_train,
        fit_dev,
        identity_checks=identity_checks,
        evidence_checks={"protected_tiers_unopened": True, "paired_activation_capture": True},
    )
    candidate_identity = {
        "candidate_id": record.get("candidate_id"),
        "design_id": record.get("design_id"),
        "seed": record.get("seed"),
        "profile": candidate.metadata.get("profile"),
        "status": candidate.metadata.get("status"),
        "routing_mode": candidate.metadata.get("routing_mode"),
        "router_architecture": candidate.metadata.get("router_architecture"),
        "top_k": candidate.top_k,
        "routed_experts": candidate.routed_experts,
        "expert_intermediate_size": candidate.expert_intermediate_size,
        "shared_intermediate_size": candidate.shared_intermediate_size,
        "checkpoint": str(candidate.checkpoint),
        "checkpoint_sha256": candidate.state_sha256,
        "metadata": str(candidate.metadata_path),
        "metadata_sha256": sha256_file(candidate.metadata_path),
        "replay_status": "EXECUTED",
    }
    source_identity = {
        "name": "Qwen/Qwen3.8-27B",
        "revision": source_revision,
        "layer": int(record.get("layer", 0)),
        "hidden_size": candidate.hidden_size,
        "dense_intermediate_size": candidate.intermediate_size,
    }
    receipt = build_structural_generalization_receipt(
        source_model=source_identity,
        layer=int(record.get("layer", 0)),
        candidate=candidate_identity,
        fit_train=fit_train,
        fit_dev=fit_dev,
        generalization=generalization,
        source_receipt_lineage={
            "legacy_receipt_locations": list(record.get("receipt_locations", [])),
            "source_receipt_lineage": list(record.get("source_receipt_lineage", [])),
        },
        code_science_identity=dict(code_science_identity or {"code_commit": current_git_commit(), "metric_policy_version": STRUCTURAL_POLICY_VERSION, "policy_hash": POLICY_HASH}),
        runtime_lock_identity=dict(runtime_lock_identity or {}),
        lm_evaluation_eligibility={
            "status": decision.get("lm_evaluation_eligibility"),
            "reason": decision.get("veto_reason", []),
        },
        fit_train_identity=train_identity,
        fit_dev_identity=dev_identity,
    )
    validation = validate_structural_receipt(receipt)
    if not validation.get("valid"):
        raise ReplayValidationError(f"structural receipt validation failed: {validation.get('errors')}")
    return {
        "status": "REPLAY_COMPLETE",
        "candidate_id": record.get("candidate_id"),
        "design_id": record.get("design_id"),
        "seed": record.get("seed"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": candidate.state_sha256,
        "receipt": receipt,
        "receipt_validation": validation,
        "decision": decision,
        "fit_train": {key: fit_train.get(key) for key in ("status", "scored_token_count", "cosine_similarity", "normalized_mse", "learned_load_cv", "dead_expert_count")},
        "fit_dev": {key: fit_dev.get(key) for key in ("status", "scored_token_count", "cosine_similarity", "normalized_mse", "learned_load_cv", "dead_expert_count")},
        "generalization_classification": generalization.get("classification"),
        "lm_evaluation_eligibility": decision.get("lm_evaluation_eligibility"),
    }


__all__ = [
    "HistoricalCandidate",
    "PairedActivationBatch",
    "ReplayInputError",
    "ReplayValidationError",
    "iter_paired_activation_batches",
    "iter_paired_activation_shards",
    "load_historical_candidate",
    "replay_candidate_record",
    "resolve_artifact_path",
    "sha256_file",
]
