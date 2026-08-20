"""Discovery and replayability classification for V2.3/V2.4 candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TERMINAL_RERUN_STATUSES = (
    "RERUN_COMPLETE",
    "PARTIAL_METRICS_ONLY",
    "NOT_REPLAYABLE_MISSING_CHECKPOINT",
    "NOT_REPLAYABLE_MISSING_DATA",
    "NOT_REPLAYABLE_IDENTITY_MISMATCH",
    "NOT_APPLICABLE",
    "BLOCKED_RUNTIME_IDENTITY",
    "BLOCKED_RESOURCE_LIMIT",
    "FAILED_VALIDATION",
)


@dataclass(frozen=True)
class CandidateInventoryRecord:
    candidate_id: str
    design_id: str | None
    topology: str | None
    routing_mode: str | None
    seed: int | None
    stage: str | None
    checkpoint: str | None
    checkpoint_hash: str | None
    source_revision: str | None
    layer: int | None
    fit_train_data_identity: dict[str, Any]
    fit_dev_data_identity: dict[str, Any]
    existing_metric_policy_version: str | None
    raw_replayable_inputs: list[str]
    receipt_locations: list[str]
    rerun_eligibility: str
    rerun_status: str
    reason: str | None
    source_receipt_lineage: list[str]
    artifact_kind: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _infer_design(path: Path, payload: Mapping[str, Any]) -> str | None:
    for candidate in (payload.get("design_id"), payload.get("config", {}).get("design_id") if isinstance(payload.get("config"), Mapping) else None):
        if candidate is not None:
            return str(candidate)
    match = re.search(r"(?:design[-_])([A-E])(?:\b|$)", str(path), re.IGNORECASE)
    return match.group(1).upper() if match else None


def _infer_stage(payload: Mapping[str, Any], path: Path) -> str | None:
    for key in ("stage", "selected_stage", "best_stage"):
        if payload.get(key) is not None:
            return str(payload[key])
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        stages = metrics.get("stage_metrics")
        if isinstance(stages, list) and stages:
            return str(stages[-1].get("stage", stages[-1].get("name", ""))) or None
    match = re.search(r"stage[-_]([0-9]+)", str(path), re.IGNORECASE)
    return f"stage-{match.group(1)}" if match else None


def _find_capture_manifests(seed_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = seed_path.parent
    for parent in [seed_path.parent, *seed_path.parents]:
        if parent.name == "comparison" or parent.name == "development":
            root = parent
            break
    fit_train: dict[str, Any] = {}
    fit_dev: dict[str, Any] = {}
    candidates = list(root.rglob("layer-0000-FIT-TRAIN.json")) + list(root.rglob("layer-0000-FIT-DEV.json"))
    for manifest_path in candidates:
        payload = _read_json(manifest_path) or {}
        split = str(payload.get("split", ""))
        if "FIT-TRAIN" in manifest_path.name or split == "FIT-TRAIN":
            fit_train = {"path": str(manifest_path), "sha256": _sha256(manifest_path), "dataset_hash": payload.get("dataset_hash"), "split": "FIT-TRAIN", "source_revision": payload.get("source_revision")}
        elif "FIT-DEV" in manifest_path.name or split == "FIT-DEV":
            fit_dev = {"path": str(manifest_path), "sha256": _sha256(manifest_path), "dataset_hash": payload.get("dataset_hash"), "split": "FIT-DEV", "source_revision": payload.get("source_revision")}
    return fit_train, fit_dev


def _record_from_seed(path: Path, payload: Mapping[str, Any]) -> CandidateInventoryRecord:
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    profile = config.get("profile") if isinstance(config, Mapping) and isinstance(config.get("profile"), Mapping) else {}
    checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), Mapping) else {}
    tensor_path = checkpoint.get("tensor_file") or payload.get("tensor_file")
    metadata_path = checkpoint.get("metadata") or payload.get("metadata")
    tensor = Path(str(tensor_path)) if tensor_path else None
    metadata = Path(str(metadata_path)) if metadata_path else None
    if tensor is not None and not tensor.is_absolute() and not tensor.exists():
        tensor = path.parent / tensor
    if metadata is not None and not metadata.is_absolute() and not metadata.exists():
        metadata = path.parent / metadata
    candidate_id = str(payload.get("candidate_id") or payload.get("config_id") or path.parent.parent.name)
    source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    fit_train = dict(inputs.get("fit_train", {})) if isinstance(inputs.get("fit_train"), Mapping) else {}
    fit_dev = dict(inputs.get("fit_dev", {})) if isinstance(inputs.get("fit_dev"), Mapping) else {}
    if not fit_train or not fit_dev:
        inferred_train, inferred_dev = _find_capture_manifests(path)
        fit_train = fit_train or inferred_train
        fit_dev = fit_dev or inferred_dev
    source_revision = str(source.get("revision") or profile.get("revision") or fit_train.get("source_revision") or fit_dev.get("source_revision") or "") or None
    checkpoint_hash = str(payload.get("checkpoint_sha256") or checkpoint.get("tensor_sha256") or "") or (_sha256(tensor) if tensor and tensor.exists() else None)
    raw_inputs = [str(item) for item in (tensor, metadata) if item is not None and item.exists()]
    raw_inputs.extend(str(item["path"]) for item in (fit_train, fit_dev) if isinstance(item, Mapping) and item.get("path") and Path(str(item["path"])).exists())
    receipts = [str(path)]
    if metadata is not None:
        receipts.append(str(metadata))
    checkpoint_exists = tensor is not None and tensor.exists()
    data_exists = bool(fit_train.get("path") and fit_dev.get("path") and Path(str(fit_train["path"])).exists() and Path(str(fit_dev["path"])).exists())
    source_ok = source_revision in {None, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
    reload_ok = payload.get("checkpoint_reload", payload.get("reload_verified", checkpoint.get("strict_reload", {}).get("passed", True))) is not False
    if not checkpoint_exists:
        eligibility, status, reason = "not_replayable", "NOT_REPLAYABLE_MISSING_CHECKPOINT", "checkpoint tensor is missing"
    elif not data_exists:
        eligibility, status, reason = "not_replayable", "NOT_REPLAYABLE_MISSING_DATA", "paired FIT-TRAIN/FIT-DEV manifests are missing"
    elif not source_ok:
        eligibility, status, reason = "not_replayable", "NOT_REPLAYABLE_IDENTITY_MISMATCH", "source revision differs from the pinned Qwen source contract"
    elif not reload_ok:
        eligibility, status, reason = "not_replayable", "FAILED_VALIDATION", "strict checkpoint reload did not pass"
    else:
        eligibility, status, reason = "eligible", "PARTIAL_METRICS_ONLY", "replayable inputs found; V2 recomputation not yet emitted"
    layer_match = re.search(r"layer[-_](\d+)", str(tensor or metadata or path), re.IGNORECASE)
    layer = int(layer_match.group(1)) if layer_match else 0
    return CandidateInventoryRecord(
        candidate_id=candidate_id,
        design_id=_infer_design(path, payload),
        topology=str(profile.get("name") or profile.get("topology") or "") or None,
        routing_mode=str(profile.get("routing_mode") or payload.get("routing_mode") or "") or None,
        seed=int(payload.get("seed")) if payload.get("seed") is not None else int(config.get("seed")) if isinstance(config, Mapping) and config.get("seed") is not None else None,
        stage=_infer_stage(payload, path),
        checkpoint=str(tensor or metadata or "") or None,
        checkpoint_hash=checkpoint_hash,
        source_revision=source_revision,
        layer=layer,
        fit_train_data_identity=fit_train,
        fit_dev_data_identity=fit_dev,
        existing_metric_policy_version=str(payload.get("metric_policy_version") or payload.get("policy_version") or "") or None,
        raw_replayable_inputs=sorted(set(raw_inputs)),
        receipt_locations=sorted(set(receipts)),
        rerun_eligibility=eligibility,
        rerun_status=status,
        reason=reason,
        source_receipt_lineage=[str(item) for item in (payload.get("source_receipt_lineage", []) if isinstance(payload.get("source_receipt_lineage"), list) else [])],
        artifact_kind=str(payload.get("artifact_type") or payload.get("receipt_type") or "v23-v24-seed-result"),
    )


def _record_from_checkpoint_metadata(path: Path, payload: Mapping[str, Any]) -> CandidateInventoryRecord:
    """Adapt legacy V2.3 layer metadata into the common inventory contract."""

    seed_match = re.search(r"seed[-_](\d+)", str(path), re.IGNORECASE)
    candidate = path.parent.parent.name if path.parent.parent != path.parent else path.stem
    synthetic = {
        "artifact_type": "dense2moe-v2.3-checkpoint-metadata",
        "candidate_id": candidate,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "checkpoint": {
            "metadata": str(path),
            "tensor_file": payload.get("tensor_file"),
            "tensor_sha256": payload.get("tensor_sha256"),
            "strict_reload": {"passed": payload.get("status") not in {"CHECKPOINT_INVALID", "FAILED"}},
        },
        "config": {"profile": {"name": payload.get("profile"), "routing_mode": payload.get("routing_mode"), "revision": payload.get("source_revision")}},
        "source": {"revision": payload.get("source_revision")},
    }
    return _record_from_seed(path, synthetic)


def discover_candidate_inventory(roots: Sequence[str | Path] | None = None) -> list[CandidateInventoryRecord]:
    """Discover every seed/checkpoint receipt under the supplied run roots."""

    roots = list(roots or (Path(".nsp/artifacts/runs"), Path("runs")))
    records: list[CandidateInventoryRecord] = []
    seen: set[tuple[str, str | None, int | None, str | None]] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("seed-result.json"):
            payload = _read_json(path)
            if payload is None:
                continue
            record = _record_from_seed(path, payload)
            key = (record.candidate_id, record.checkpoint_hash, record.seed, record.checkpoint)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
        # V2.3 pilot runs predate seed-result.json and store the same
        # checkpoint identity in layer-0000.json metadata.  Include those
        # failed and near-miss artifacts rather than silently dropping them.
        for path in root.rglob("layer-0000.json"):
            payload = _read_json(path)
            if payload is None or "tensor_file" not in payload or "profile" not in payload:
                continue
            record = _record_from_checkpoint_metadata(path, payload)
            key = (record.candidate_id, record.checkpoint_hash, record.seed, record.checkpoint)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    records.sort(key=lambda item: (item.candidate_id, item.seed if item.seed is not None else -1, item.checkpoint_hash or "", item.receipt_locations[0] if item.receipt_locations else ""))
    return records


def inventory_payload(records: Sequence[CandidateInventoryRecord], *, source_revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", code_science_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    statuses = [record.rerun_status for record in records]
    # A terminal status alone does not prove that a replayable candidate was
    # actually recomputed.  Cohort completeness is reserved for candidates
    # that are explicitly non-replayable or have a V2 receipt-producing rerun.
    replayable_unresolved = [
        record.candidate_id
        for record in records
        if record.rerun_eligibility == "eligible" and record.rerun_status != "RERUN_COMPLETE"
    ]
    non_replayable = {
        "not_replayable",
        "not_applicable",
        "blocked_runtime_identity",
    }
    cohort_complete = bool(records) and all(
        record.rerun_status == "RERUN_COMPLETE"
        or record.rerun_eligibility in non_replayable
        for record in records
    )
    return {
        "schema_version": 2,
        "receipt_type": "dense2moe-v2-candidate-inventory-v2",
        "source_revision": source_revision,
        "candidate_count": len(records),
        "terminal_statuses": {status: statuses.count(status) for status in TERMINAL_RERUN_STATUSES if status in statuses},
        "cohort_complete": cohort_complete,
        "replayable_unresolved_count": len(replayable_unresolved),
        "replayable_unresolved_candidates": sorted(replayable_unresolved),
        "records": [record.as_dict() for record in records],
        "code_science_identity": dict(code_science_identity or {}),
        "protected_tiers_opened": False,
    }


def write_inventory(records: Sequence[CandidateInventoryRecord], path: str | Path, **kwargs: Any) -> dict[str, Any]:
    target = Path(path)
    payload = inventory_payload(records, **kwargs)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = _read_json(target)
        if existing == payload:
            return existing or payload
        raise ValueError(f"refusing to overwrite existing inventory: {target}")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def terminal_status_for_record(
    record: CandidateInventoryRecord,
    *,
    structural_metrics_available: bool = False,
    valid_v2_receipt: bool = False,
    resource_blocked: bool = False,
) -> str:
    if resource_blocked and record.rerun_eligibility == "eligible":
        return "BLOCKED_RESOURCE_LIMIT"
    if record.rerun_eligibility != "eligible":
        return record.rerun_status
    # Metrics in memory or a legacy receipt are not enough to claim a
    # completed rerun.  Completion requires the immutable V2 structural
    # receipt, whose policy hash/schema were validated by the writer.
    if structural_metrics_available and valid_v2_receipt:
        return "RERUN_COMPLETE"
    return "PARTIAL_METRICS_ONLY"


__all__ = [
    "CandidateInventoryRecord",
    "TERMINAL_RERUN_STATUSES",
    "discover_candidate_inventory",
    "inventory_payload",
    "terminal_status_for_record",
    "write_inventory",
]
