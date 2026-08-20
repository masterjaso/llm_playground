"""Bounded layer-0 capture for explicitly named frozen evaluation tiers.

The development activation plan is intentionally not reused for promotion:
it contains only FIT-TRAIN/FIT-DEV.  This module derives one immutable plan
from the content-addressed Corpus V2.2 freeze artifacts, then reuses the
native streaming teacher.  It never opens or writes the promotion
contamination ledger and never invokes candidate fitting.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..hardware import load_runtime_lock
from ..provenance import current_git_commit
from .real_method_proof import QWEN_SOURCE_REVISION, RealCaptureBlocked
from .streaming_teacher import stream_teacher_split

PROMOTION_EVALUATION_TIERS = ("GATE-A", "SHADOW-B", "SHADOW-C")
DEVELOPMENT_TIERS = ("FIT-TRAIN", "FIT-DEV")
PROMOTION_MANIFEST_NAME = "layer-0000.json"
PROMOTION_SCHEMA_VERSION = 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(item) for item in ids).encode("utf-8")).hexdigest()


def _frozen_normalized_content_hash(text: str) -> str:
    """Hash the Corpus V2.2 freeze convention without changing its text.

    The frozen producer predates the tokenizer resolver's NFC normalization.
    Promotion validates that legacy digest against the immutable row, then
    carries it under an explicit compatibility name in the derived plan.
    """

    normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealCaptureBlocked(code, details={"path": str(path), "error": str(exc)}) from exc
    if not isinstance(value, dict):
        raise RealCaptureBlocked(code, details={"path": str(path), "error": "expected JSON object"})
    return value


def _resolve_manifest_path(manifest_path: Path, raw: str) -> Path:
    candidate = Path(raw.replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    local = manifest_path.parent / candidate
    if local.exists():
        return local
    return manifest_path.parent.parent / candidate


def _load_frozen_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"line {line_number} is not an object")
            rows.append(dict(value))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealCaptureBlocked("FROZEN_CORPUS_MANIFEST_INVALID", details={"path": str(manifest_path), "error": str(exc)}) from exc
    if not rows:
        raise RealCaptureBlocked("FROZEN_CORPUS_MANIFEST_EMPTY", details={"path": str(manifest_path)})
    return rows


def _verify_receipt_artifacts(root: Path, receipt: Mapping[str, Any]) -> dict[str, str]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RealCaptureBlocked("FROZEN_CORPUS_ARTIFACTS_MISSING")
    hashes: dict[str, str] = {}
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping) or not raw.get("path") or not raw.get("sha256"):
            raise RealCaptureBlocked("FROZEN_CORPUS_ARTIFACT_REFERENCE_INVALID", details={"artifact": str(name)})
        path = root / str(raw["path"])
        if not path.is_file():
            raise RealCaptureBlocked("FROZEN_CORPUS_ARTIFACT_MISSING", details={"artifact": str(name), "path": str(path)})
        actual = _sha256_file(path)
        if actual != str(raw["sha256"]):
            raise RealCaptureBlocked(
                "FROZEN_CORPUS_ARTIFACT_HASH_MISMATCH",
                details={"artifact": str(name), "path": str(path), "expected": str(raw["sha256"]), "actual": actual},
            )
        hashes[str(name)] = actual
    return hashes


def _validate_contamination_ledger(path: Path) -> tuple[dict[str, Any], str]:
    ledger = _read_json(path, "CONTAMINATION_LEDGER_INVALID")
    if ledger.get("ledger_type") != "dense2moe-evaluation-contamination":
        raise RealCaptureBlocked("CONTAMINATION_LEDGER_TYPE_INVALID", details={"path": str(path)})
    opened = sorted(str(item) for item in dict(ledger.get("tiers", {})))
    declared = sorted(str(item) for item in ledger.get("opened_tiers", [])) if isinstance(ledger.get("opened_tiers"), list) else []
    if ledger.get("status") != "SEALED" or opened or declared:
        raise RealCaptureBlocked(
            "CONTAMINATION_LEDGER_NOT_SEALED",
            details={"path": str(path), "status": ledger.get("status"), "opened_tiers": opened or declared},
        )
    return ledger, _sha256_file(path)


def _runtime_identity(path: str | Path) -> dict[str, Any]:
    lock_path = Path(path)
    loaded = load_runtime_lock(lock_path)
    if loaded.get("status") != "LOCKED" or not loaded.get("ok"):
        raise RealCaptureBlocked("RUNTIME_LOCK_INVALID", details={"path": str(lock_path), "error": loaded.get("error")})
    payload = loaded.get("payload")
    if not isinstance(payload, Mapping) or payload.get("status") != "APPROVED":
        raise RealCaptureBlocked("RUNTIME_LOCK_NOT_APPROVED", details={"path": str(lock_path)})
    lock_sha = str(payload.get("lock_sha256", ""))
    if not lock_sha:
        raise RealCaptureBlocked("RUNTIME_LOCK_IDENTITY_MISSING", details={"path": str(lock_path)})
    return {"path": str(lock_path), "sha256": _sha256_file(lock_path), "lock_sha256": lock_sha}


def _validate_frozen_plan_inputs(root: Path, tier: str) -> dict[str, Any]:
    if tier not in PROMOTION_EVALUATION_TIERS:
        reason = "PROMOTION_TIER_DEVELOPMENT_UNAUTHORIZED" if tier in DEVELOPMENT_TIERS else "PROMOTION_TIER_UNKNOWN"
        raise RealCaptureBlocked(reason, details={"tier": tier, "allowed_tiers": list(PROMOTION_EVALUATION_TIERS)})
    receipt_path = root / "corpus-v2.2-receipt.json"
    manifest_path = root / "corpus-v2.2.jsonl"
    splits_path = root / "corpus-v2.2-splits.json"
    overlap_path = root / "corpus-v2.2-overlap-audit.json"
    tier_ledger_path = root / "corpus-v2.2-tier-ledger.json"
    activation_plan_path = root / "corpus-v2.2-activation-plan.json"
    for path in (receipt_path, manifest_path, splits_path, overlap_path, tier_ledger_path, activation_plan_path):
        if not path.is_file():
            raise RealCaptureBlocked("FROZEN_CORPUS_ARTIFACT_MISSING", details={"path": str(path)})
    receipt = _read_json(receipt_path, "FROZEN_CORPUS_RECEIPT_INVALID")
    if receipt.get("status") != "CORPUS_V22_FROZEN" or receipt.get("immutable") is not True:
        raise RealCaptureBlocked("FROZEN_CORPUS_RECEIPT_NOT_FROZEN", details={"path": str(receipt_path), "status": receipt.get("status")})
    if receipt.get("component") != "development-internal" or tier not in {str(item) for item in receipt.get("required_tiers", [])}:
        raise RealCaptureBlocked("FROZEN_CORPUS_TIER_NOT_ATTACHED", details={"tier": tier, "component": receipt.get("component")})
    _verify_receipt_artifacts(root, receipt)
    manifest_hash = _sha256_file(manifest_path)
    manifest_ref = receipt.get("manifest")
    if not isinstance(manifest_ref, Mapping) or str(manifest_ref.get("sha256", "")) != manifest_hash:
        raise RealCaptureBlocked("FROZEN_CORPUS_MANIFEST_HASH_MISMATCH", details={"expected": manifest_ref.get("sha256") if isinstance(manifest_ref, Mapping) else None, "actual": manifest_hash})
    splits = _read_json(splits_path, "FROZEN_SPLIT_LEDGER_INVALID")
    records_by_tier = splits.get("records")
    if not isinstance(records_by_tier, Mapping):
        raise RealCaptureBlocked("FROZEN_SPLIT_LEDGER_INVALID", details={"path": str(splits_path)})
    selected_ids = [str(item) for item in records_by_tier.get(tier, [])]
    train_ids = {str(item) for item in records_by_tier.get("FIT-TRAIN", [])}
    dev_ids = {str(item) for item in records_by_tier.get("FIT-DEV", [])}
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise RealCaptureBlocked("FROZEN_TIER_ROWS_INVALID", details={"tier": tier, "count": len(selected_ids)})
    overlap = sorted(set(selected_ids) & (train_ids | dev_ids))
    if overlap:
        raise RealCaptureBlocked("FROZEN_EVALUATION_DEVELOPMENT_OVERLAP", details={"tier": tier, "row_ids": overlap[:20]})
    rows = _load_frozen_rows(manifest_path)
    rows_by_id = {str(row.get("id", "")): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise RealCaptureBlocked("FROZEN_CORPUS_ROW_IDS_INVALID")
    selected_rows: list[dict[str, Any]] = []
    for row_id in selected_ids:
        row = rows_by_id.get(row_id)
        if row is None or str(row.get("tier")) != tier or str(row.get("split")) != tier:
            raise RealCaptureBlocked("FROZEN_TIER_ROW_MISMATCH", details={"tier": tier, "row_id": row_id})
        expected_content = str(row.get("content_sha256", ""))
        expected_normalized = str(row.get("normalized_content_sha256", ""))
        text = str(row.get("text", row.get("content", "")))
        if not expected_content or not expected_normalized or not str(row.get("source_record_id", "")) or not str(row.get("source_revision", "")):
            raise RealCaptureBlocked("FROZEN_TIER_PROVENANCE_MISSING", details={"tier": tier, "row_id": row_id})
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_content or _frozen_normalized_content_hash(text) != expected_normalized:
            raise RealCaptureBlocked("FROZEN_TIER_PROVENANCE_HASH_MISMATCH", details={"tier": tier, "row_id": row_id})
        selected_rows.append(row)
    overlap_audit = _read_json(overlap_path, "FROZEN_OVERLAP_AUDIT_INVALID")
    if overlap_audit.get("status") != "PASS" or overlap_audit.get("zero_forbidden_overlap") is not True:
        raise RealCaptureBlocked("FROZEN_OVERLAP_AUDIT_FAILED", details={"path": str(overlap_path)})
    if not {"FIT-TRAIN", "FIT-DEV", tier}.issubset({str(item) for item in overlap_audit.get("checked_tiers", [])}):
        raise RealCaptureBlocked("FROZEN_OVERLAP_AUDIT_INCOMPLETE", details={"tier": tier})
    tier_ledger = _read_json(tier_ledger_path, "FROZEN_TIER_LEDGER_INVALID")
    if tier_ledger.get("status") != "SEALED" or tier_ledger.get("one_way_opening") is not True:
        raise RealCaptureBlocked("FROZEN_TIER_LEDGER_NOT_SEALED", details={"path": str(tier_ledger_path)})
    if str(tier_ledger.get("manifest_sha256")) != manifest_hash or str(tier_ledger.get("method_version")) != str(receipt.get("method_version")):
        raise RealCaptureBlocked("FROZEN_TIER_LEDGER_PROVENANCE_MISMATCH")
    tier_entry = tier_ledger.get("tiers", {}).get(tier) if isinstance(tier_ledger.get("tiers"), Mapping) else None
    if not isinstance(tier_entry, Mapping) or tier_entry.get("status") != "FROZEN" or tier_entry.get("opened") is not False or tier_entry.get("optimizer_updates") is not False:
        raise RealCaptureBlocked("FROZEN_TIER_NOT_CAPTURE_ELIGIBLE", details={"tier": tier, "entry": tier_entry})
    dataset_hash = hashlib.sha256("".join(selected_ids).encode("utf-8")).hexdigest()
    if dataset_hash != str(tier_entry.get("dataset_sha256")):
        raise RealCaptureBlocked("FROZEN_TIER_DATASET_HASH_MISMATCH", details={"tier": tier, "expected": tier_entry.get("dataset_sha256"), "actual": dataset_hash})
    activation_plan = _read_json(activation_plan_path, "FROZEN_ACTIVATION_PLAN_INVALID")
    eligible_splits = {str(item) for item in activation_plan.get("eligible_splits", [])}
    if tier in eligible_splits or not eligible_splits.issubset(set(DEVELOPMENT_TIERS)):
        raise RealCaptureBlocked("FROZEN_ACTIVATION_PLAN_SCOPE_INVALID", details={"eligible_splits": sorted(eligible_splits)})
    return {
        "tier": tier,
        "rows": selected_rows,
        "row_ids": selected_ids,
        "row_ids_sha256": _ids_hash(selected_ids),
        "dataset_hash": dataset_hash,
        "record_count": len(selected_rows),
        "declared_tokens": sum(int(row.get("token_count", 0) or 0) for row in selected_rows),
        "corpus_root": str(root.resolve()),
        "corpus_manifest_path": str(manifest_path.resolve()),
        "corpus_manifest_sha256": manifest_hash,
        "corpus_receipt_path": str(receipt_path.resolve()),
        "corpus_receipt_sha256": _sha256_file(receipt_path),
        "tier_ledger_path": str(tier_ledger_path.resolve()),
        "tier_ledger_sha256": _sha256_file(tier_ledger_path),
        "splits_path": str(splits_path.resolve()),
        "splits_sha256": _sha256_file(splits_path),
        "overlap_audit_path": str(overlap_path.resolve()),
        "overlap_audit_sha256": _sha256_file(overlap_path),
        "activation_plan_path": str(activation_plan_path.resolve()),
        "activation_plan_sha256": _sha256_file(activation_plan_path),
        "method_version": str(receipt.get("method_version", "")),
        "threshold_fingerprint": str(receipt.get("threshold_fingerprint", "")),
        "tier_ledger_entry": dict(tier_entry),
        "activation_plan_eligible_splits": sorted(eligible_splits),
    }


def build_frozen_evaluation_plan(corpus_root: str | Path, tier: str, *, source_revision: str = QWEN_SOURCE_REVISION, sequence_length: int = 2048) -> dict[str, Any]:
    """Return a tokenizer-resolvable plan containing only one frozen tier."""

    if source_revision != QWEN_SOURCE_REVISION:
        raise RealCaptureBlocked("SOURCE_REVISION_MISMATCH", details={"expected": QWEN_SOURCE_REVISION, "actual": source_revision})
    if sequence_length <= 0:
        raise RealCaptureBlocked("SEQUENCE_LENGTH_INVALID", details={"sequence_length": sequence_length})
    root = Path(corpus_root)
    identity = _validate_frozen_plan_inputs(root, tier)
    rows: list[dict[str, Any]] = []
    for raw in identity["rows"]:
        row = dict(raw)
        # The frozen row itself remains the source of identity.  The derived
        # plan embeds its hash-checked text so the streaming resolver cannot
        # accidentally use a mutable acquisition-side source index.  The
        # declared V2.2 token budget is an immutable prefix/suffix cap; the
        # pinned Transformers tokenizer may expose one or two extra wrapper
        # tokens compared with the acquisition tokenizer.
        for key in ("source_file", "source_file_sha256", "source_record_index"):
            row.pop(key, None)
        if row.get("normalized_content_sha256"):
            row["frozen_normalized_content_sha256"] = row.pop("normalized_content_sha256")
        row["split"] = tier
        row["sample_tokens"] = int(row.get("token_count", 0) or 0)
        rows.append(row)
    plan = {
        "status": "FROZEN_EVALUATION_CAPTURE_READY",
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "capture_kind": "promotion-layer0-frozen-evaluation-tier",
        "source": {"path": identity["corpus_manifest_path"]},
        "resolvability": {"base_dir": identity["corpus_root"]},
        "source_revision": source_revision,
        "tokenizer_revision": source_revision,
        "sequence_length": sequence_length,
        "dataset_hash": identity["dataset_hash"],
        "tier": tier,
        tier: rows,
        "row_ids": identity["row_ids"],
        "row_ids_sha256": identity["row_ids_sha256"],
        "declared_tokens": identity["declared_tokens"],
        "record_count": identity["record_count"],
        "method_version": identity["method_version"],
        "threshold_fingerprint": identity["threshold_fingerprint"],
        "corpus_receipt": {"path": identity["corpus_receipt_path"], "sha256": identity["corpus_receipt_sha256"]},
        "tier_ledger": {"path": identity["tier_ledger_path"], "sha256": identity["tier_ledger_sha256"]},
        "splits": {"path": identity["splits_path"], "sha256": identity["splits_sha256"]},
        "overlap_audit": {"path": identity["overlap_audit_path"], "sha256": identity["overlap_audit_sha256"]},
        "activation_plan": {"path": identity["activation_plan_path"], "sha256": identity["activation_plan_sha256"], "eligible_splits": identity["activation_plan_eligible_splits"]},
        "evaluation_tiers_excluded_from_fitting": list(PROMOTION_EVALUATION_TIERS),
    }
    return {"plan": plan, "identity": identity, "plan_sha256": _canonical_hash(plan)}


def _validate_activation_manifest(path: Path, identity: Mapping[str, Any], *, expected_split: str) -> dict[str, Any]:
    manifest = _read_json(path, "PROMOTION_CAPTURE_MANIFEST_INVALID")
    if manifest.get("status") != "CAPTURE_COMPLETE" or int(manifest.get("layer", -1)) != 0 or manifest.get("split") != expected_split:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_MANIFEST_IDENTITY_MISMATCH", details={"path": str(path)})
    if manifest.get("dataset_hash") != identity["dataset_hash"] or not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_DATASET_HASH_MISMATCH", details={"path": str(path)})
    expected_ids = {str(item) for item in identity["row_ids"]}
    expected_lengths = {str(row["id"]): int(row.get("token_count", 0) or 0) for row in identity["rows"]}
    seen_ids: set[str] = set()
    seen_chunks: set[tuple[str, int]] = set()
    observed_lengths: dict[str, int] = {}
    total = 0
    for shard in manifest["shards"]:
        if not isinstance(shard, Mapping) or not shard.get("path") or not shard.get("sha256"):
            raise RealCaptureBlocked("PROMOTION_CAPTURE_SHARD_METADATA_INVALID", details={"path": str(path)})
        shard_path = _resolve_manifest_path(path, str(shard["path"]))
        if not shard_path.is_file() or _sha256_file(shard_path) != str(shard["sha256"]):
            raise RealCaptureBlocked("PROMOTION_CAPTURE_SHARD_HASH_MISMATCH", details={"path": str(shard_path)})
        if str(shard.get("input_tensor", "")) != "ffn_input" or str(shard.get("target_tensor", "")) != "dense_ffn_target":
            raise RealCaptureBlocked("PROMOTION_CAPTURE_TENSOR_IDENTITY_INVALID", details={"path": str(shard_path)})
        records = shard.get("records")
        if not isinstance(records, list):
            raise RealCaptureBlocked("PROMOTION_CAPTURE_ROW_IDS_MISSING", details={"path": str(shard_path)})
        if int(shard.get("count", -1)) != len(records):
            # Streaming manifests count tokens, while ``records`` count the
            # source chunks that contributed those tokens.
            if not records or not all("length" in record for record in records if isinstance(record, Mapping)):
                raise RealCaptureBlocked("PROMOTION_CAPTURE_COUNT_INVALID", details={"path": str(shard_path)})
            if sum(int(record.get("length", 0)) for record in records if isinstance(record, Mapping)) != int(shard.get("count", -1)):
                raise RealCaptureBlocked("PROMOTION_CAPTURE_TOKEN_COUNT_INVALID", details={"path": str(shard_path)})
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping) or str(record.get("split")) != expected_split:
                raise RealCaptureBlocked("PROMOTION_CAPTURE_SPLIT_MISMATCH", details={"path": str(shard_path)})
            row_id = str(record.get("example_id", ""))
            if row_id not in expected_ids:
                raise RealCaptureBlocked("PROMOTION_CAPTURE_UNAUTHORIZED_ROW", details={"row_id": row_id, "tier": expected_split})
            chunk_index = int(record.get("chunk_index", record_index))
            chunk_key = (row_id, chunk_index)
            if chunk_key in seen_chunks:
                raise RealCaptureBlocked("PROMOTION_CAPTURE_DUPLICATE_ROW", details={"row_id": row_id, "chunk_index": chunk_index, "tier": expected_split})
            seen_chunks.add(chunk_key)
            seen_ids.add(row_id)
            if "length" in record:
                length = int(record["length"])
                if length <= 0:
                    raise RealCaptureBlocked("PROMOTION_CAPTURE_TOKEN_COUNT_INVALID", details={"row_id": row_id, "tier": expected_split})
                observed_lengths[row_id] = observed_lengths.get(row_id, 0) + length
        total += int(shard.get("count", 0) or 0)
    if observed_lengths and observed_lengths != expected_lengths:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_TOKEN_COUNT_INVALID", details={"tier": expected_split})
    if total != int(manifest.get("count", -1)) or seen_ids != expected_ids:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_COUNT_INVALID", details={"path": str(path), "count": manifest.get("count"), "shard_count": total})
    return manifest


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("promotion_provenance") if isinstance(manifest.get("promotion_provenance"), Mapping) else {}
    runtime = provenance.get("runtime_lock") if isinstance(provenance.get("runtime_lock"), Mapping) else {}
    return {
        "tier": manifest.get("tier", manifest.get("split")),
        "dataset_hash": manifest.get("dataset_hash"),
        "corpus_receipt_sha256": provenance.get("corpus_receipt_sha256"),
        "tier_ledger_sha256": provenance.get("tier_ledger_sha256"),
        "row_ids_sha256": provenance.get("row_ids_sha256"),
        "source_revision": provenance.get("source_revision", manifest.get("source_revision")),
        "runtime_lock_sha256": runtime.get("sha256"),
        "code_commit": provenance.get("code_commit", manifest.get("code_commit")),
        "method_version": provenance.get("method_version"),
    }


def _assert_output_scope(output_root: Path) -> None:
    if not output_root.exists():
        return
    allowed = {PROMOTION_MANIFEST_NAME, "layer-0000"}
    unexpected = sorted(path.name for path in output_root.iterdir() if path.name not in allowed)
    if unexpected:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_OUTPUT_SCOPE_INVALID", details={"path": str(output_root), "unexpected": unexpected})
    shard_root = output_root / "layer-0000"
    if shard_root.is_dir():
        unexpected_shards = sorted(path.name for path in shard_root.iterdir() if not path.is_file() or path.suffix != ".safetensors")
        if unexpected_shards:
            raise RealCaptureBlocked("PROMOTION_CAPTURE_OUTPUT_SCOPE_INVALID", details={"path": str(shard_root), "unexpected": unexpected_shards})


def _copy_atomic(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise RealCaptureBlocked("PROMOTION_CAPTURE_PARTIAL_HASH_MISMATCH", details={"path": str(destination)})
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _published_result(output_root: Path, identity: Mapping[str, Any], *, expected_code_commit: str) -> dict[str, Any] | None:
    manifest_path = output_root / PROMOTION_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    manifest = _validate_activation_manifest(manifest_path, identity, expected_split=str(identity["tier"]))
    actual = _manifest_identity(manifest)
    expected = {
        "tier": identity["tier"],
        "dataset_hash": identity["dataset_hash"],
        "corpus_receipt_sha256": identity["corpus_receipt_sha256"],
        "tier_ledger_sha256": identity["tier_ledger_sha256"],
        "row_ids_sha256": identity["row_ids_sha256"],
        "source_revision": identity["source_revision"],
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "code_commit": expected_code_commit,
        "method_version": identity["method_version"],
    }
    if actual != expected:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_PROVENANCE_MISMATCH", details={"path": str(manifest_path), "expected": expected, "actual": actual})
    provenance = manifest.get("promotion_provenance")
    actual_ids = list(provenance.get("row_ids", [])) if isinstance(provenance, Mapping) and isinstance(provenance.get("row_ids"), list) else []
    if actual_ids != list(identity["row_ids"]):
        raise RealCaptureBlocked("PROMOTION_CAPTURE_ROW_ORDER_MISMATCH", details={"path": str(manifest_path)})
    return {"status": "PROMOTION_CAPTURE_RESUMED", "manifest": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path), "count": int(manifest["count"]), "dataset_hash": identity["dataset_hash"], "tier": identity["tier"]}


def capture_frozen_evaluation_tier(
    corpus_root: str | Path,
    source_snapshot: str | Path,
    output_root: str | Path,
    runtime_lock: str | Path,
    *,
    tier: str,
    source_revision: str = QWEN_SOURCE_REVISION,
    device: str = "cuda:1",
    compute_dtype: str = "bfloat16",
    shard_tokens: int = 2048,
    sequence_length: int = 2048,
    attention_implementation: str = "sdpa",
    staging_root: str | Path | None = None,
    contamination_ledger: str | Path | None = None,
) -> dict[str, Any]:
    """Capture one frozen evaluation tier without opening promotion state."""

    if os.name != "nt" or platform.system() != "Windows":
        raise RealCaptureBlocked("NATIVE_WINDOWS_REQUIRED")
    if source_revision != QWEN_SOURCE_REVISION:
        raise RealCaptureBlocked("SOURCE_REVISION_MISMATCH", details={"expected": QWEN_SOURCE_REVISION, "actual": source_revision})
    if not str(device).startswith("cuda"):
        raise RealCaptureBlocked("PROMOTION_CAPTURE_CUDA_REQUIRED", details={"device": device})
    source = Path(source_snapshot)
    if not source.is_dir() or not (source / "config.json").is_file() or not (source / "model.safetensors.index.json").is_file():
        raise RealCaptureBlocked("SOURCE_SNAPSHOT_INVALID", details={"path": str(source)})
    identity_bundle = build_frozen_evaluation_plan(corpus_root, tier, source_revision=source_revision, sequence_length=sequence_length)
    identity = dict(identity_bundle["identity"])
    identity["source_revision"] = source_revision
    identity["runtime_lock"] = _runtime_identity(runtime_lock)
    identity["runtime_lock_sha256"] = identity["runtime_lock"]["sha256"]
    identity["source_snapshot"] = str(source.resolve())
    identity["source_config_sha256"] = _sha256_file(source / "config.json")
    identity["source_index_sha256"] = _sha256_file(source / "model.safetensors.index.json")
    identity["code_commit"] = current_git_commit()
    ledger_path = Path(contamination_ledger) if contamination_ledger is not None else Path(output_root).parent / "contamination-ledger.json"
    _ledger, ledger_sha256 = _validate_contamination_ledger(ledger_path)
    identity["contamination_ledger_path"] = str(ledger_path.resolve())
    identity["contamination_ledger_sha256"] = ledger_sha256
    output = Path(output_root)
    _assert_output_scope(output)
    resumed = _published_result(output, identity, expected_code_commit=str(identity["code_commit"])) if (output / PROMOTION_MANIFEST_NAME).exists() else None
    if resumed is not None:
        return resumed | {"row_count": identity["record_count"], "declared_tokens": identity["declared_tokens"], "runtime_lock_sha256": identity["runtime_lock_sha256"], "corpus_receipt_sha256": identity["corpus_receipt_sha256"], "tier_ledger_sha256": identity["tier_ledger_sha256"]}
    stage = Path(staging_root) if staging_root is not None else output.parent.parent.parent / "capture-work" / tier
    stage.mkdir(parents=True, exist_ok=True)
    stage_identity_path = stage / "capture-identity.json"
    stage_identity = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "tier": tier,
        "dataset_hash": identity["dataset_hash"],
        "row_ids_sha256": identity["row_ids_sha256"],
        "corpus_receipt_sha256": identity["corpus_receipt_sha256"],
        "tier_ledger_sha256": identity["tier_ledger_sha256"],
        "source_revision": source_revision,
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "code_commit": identity["code_commit"],
        "method_version": identity["method_version"],
        "plan_sha256": identity_bundle["plan_sha256"],
    }
    if stage_identity_path.exists():
        existing_identity = _read_json(stage_identity_path, "PROMOTION_CAPTURE_STAGING_INVALID")
        if existing_identity != stage_identity:
            raise RealCaptureBlocked("PROMOTION_CAPTURE_STAGING_PROVENANCE_MISMATCH", details={"path": str(stage_identity_path)})
    elif any(stage.iterdir()):
        raise RealCaptureBlocked("PROMOTION_CAPTURE_STAGING_UNIDENTIFIED", details={"path": str(stage)})
    else:
        from ..data import write_immutable_json

        write_immutable_json(stage_identity_path, stage_identity)
    plan_path = stage / "frozen-evaluation-data-plan.json"
    from ..data import write_immutable_json

    write_immutable_json(plan_path, identity_bundle["plan"])
    stream_teacher_split(
        source,
        plan_path,
        stage,
        split=tier,
        layers=(0,),
        device=device,
        compute_dtype=compute_dtype,
        shard_tokens=shard_tokens,
        attention_implementation=attention_implementation,
    )
    stage_manifest_path = stage / "capture" / f"layer-0000-{tier}.json"
    stage_manifest = _validate_activation_manifest(stage_manifest_path, identity, expected_split=tier)
    provenance = {
        "capture_mode": "promotion",
        "tier": tier,
        "method_version": identity["method_version"],
        "threshold_fingerprint": identity["threshold_fingerprint"],
        "row_ids": identity["row_ids"],
        "row_ids_sha256": identity["row_ids_sha256"],
        "declared_tokens": identity["declared_tokens"],
        "corpus_receipt_path": identity["corpus_receipt_path"],
        "corpus_receipt_sha256": identity["corpus_receipt_sha256"],
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "tier_ledger_path": identity["tier_ledger_path"],
        "tier_ledger_sha256": identity["tier_ledger_sha256"],
        "splits_path": identity["splits_path"],
        "splits_sha256": identity["splits_sha256"],
        "overlap_audit_path": identity["overlap_audit_path"],
        "overlap_audit_sha256": identity["overlap_audit_sha256"],
        "activation_plan_path": identity["activation_plan_path"],
        "activation_plan_sha256": identity["activation_plan_sha256"],
        "activation_plan_eligible_splits": identity["activation_plan_eligible_splits"],
        "source_revision": source_revision,
        "source_snapshot": identity["source_snapshot"],
        "source_config_sha256": identity["source_config_sha256"],
        "source_index_sha256": identity["source_index_sha256"],
        "runtime_lock": identity["runtime_lock"],
        "code_commit": identity["code_commit"],
        "contamination_ledger": {"path": identity["contamination_ledger_path"], "sha256": identity["contamination_ledger_sha256"], "status": "SEALED", "opened_tiers": []},
    }
    published = dict(stage_manifest)
    published["schema_version"] = PROMOTION_SCHEMA_VERSION
    published["manifest_type"] = "dense2moe-promotion-layer0-activation"
    published["tier"] = tier
    published["promotion_provenance"] = provenance
    published["source_revision"] = source_revision
    published["source_snapshot"] = identity["source_snapshot"]
    published["code_commit"] = identity["code_commit"]
    published_shards: list[dict[str, Any]] = []
    target_names: set[str] = set()
    for item in stage_manifest["shards"]:
        source_path = _resolve_manifest_path(stage_manifest_path, str(item["path"]))
        target_name = source_path.name
        if target_name in target_names:
            raise RealCaptureBlocked("PROMOTION_CAPTURE_OUTPUT_PATH_INVALID", details={"path": str(source_path), "reason": "duplicate shard name"})
        target_names.add(target_name)
        published_shards.append({**dict(item), "path": f"layer-0000/{target_name}"})
    published["shards"] = published_shards
    output.mkdir(parents=True, exist_ok=True)
    for item, stage_item in zip(published["shards"], stage_manifest["shards"], strict=True):
        relative = Path(str(item["path"]).replace("/", os.sep))
        if not relative.parts or relative.parts[0] != "layer-0000" or relative.is_absolute() or ".." in relative.parts:
            raise RealCaptureBlocked("PROMOTION_CAPTURE_OUTPUT_PATH_INVALID", details={"path": str(item["path"])})
        source_path = _resolve_manifest_path(stage_manifest_path, str(stage_item["path"]))
        _copy_atomic(source_path, output / relative, str(item["sha256"]))
    manifest_path = output / PROMOTION_MANIFEST_NAME
    manifest_sha256 = write_immutable_json(manifest_path, published)
    final = _published_result(output, identity, expected_code_commit=str(identity["code_commit"]))
    if final is None:
        raise RealCaptureBlocked("PROMOTION_CAPTURE_PUBLISH_FAILED", details={"path": str(manifest_path)})
    return final | {
        "status": "PROMOTION_CAPTURE_COMPLETE",
        "manifest_sha256": manifest_sha256,
        "row_count": identity["record_count"],
        "declared_tokens": identity["declared_tokens"],
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "corpus_receipt_sha256": identity["corpus_receipt_sha256"],
        "tier_ledger_sha256": identity["tier_ledger_sha256"],
        "contamination_ledger_sha256": identity["contamination_ledger_sha256"],
    }


__all__ = [
    "DEVELOPMENT_TIERS",
    "PROMOTION_EVALUATION_TIERS",
    "PROMOTION_MANIFEST_NAME",
    "build_frozen_evaluation_plan",
    "capture_frozen_evaluation_tier",
]
