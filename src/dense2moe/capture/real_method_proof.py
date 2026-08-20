"""Contracts for provenance-safe real Qwen layer-0 method-proof captures.

The existing streaming teacher writes bounded activation shards.  This module
adds the scientific receipt around those shards and a lazy iterator used by
the capture-backed Phase 01 runner.  It deliberately does not load the whole
capture into memory and it never treats the legacy ``mlp_input``-only or
synthetic-smoke manifests as real evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..hardware import check_runtime_lock, collect_environment, load_runtime_lock
from ..provenance import current_git_commit

REAL_CAPTURE_SCHEMA_VERSION = 1
REAL_CAPTURE_RECEIPT_TYPE = "dense2moe-real-qwen-layer-capture"
REAL_CAPTURE_STATUS = "REAL_QWEN_LAYER_CAPTURE_READY"
REAL_CAPTURE_EVIDENCE_CLASS = "real-qwen-layer-capture"
QWEN_SOURCE_MODEL = "Qwen/Qwen3.8-27B"
QWEN_SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN_SOURCE_MODEL_TYPE = "qwen3_5_text"
QWEN_NUM_HIDDEN_LAYERS = 64
QWEN_HIDDEN_SIZE = 5120
QWEN_DENSE_INTERMEDIATE_SIZE = 17_408
METHOD_PROOF_MIN_TOKENS = 32_768


class RealCaptureBlocked(ValueError):
    """Raised when a capture cannot be accepted as real method-proof data."""

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.reason = str(reason)
        self.details = dict(details or {})
        super().__init__(self.reason)

    def as_dict(self) -> dict[str, Any]:
        return {"status": "BLOCKED", "blocker_code": self.reason, "details": self.details}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_dtype_name(value: Any) -> str:
    raw = str(value).replace("torch.", "").casefold()
    aliases = {
        "f32": "float32",
        "f16": "float16",
        "bf16": "bfloat16",
        "f64": "float64",
        "i64": "int64",
        "i32": "int32",
        "i16": "int16",
        "i8": "int8",
        "u8": "uint8",
    }
    return aliases.get(raw, raw)


def _resolve_artifact(receipt_path: Path, raw: str | Path) -> Path:
    raw_text = str(raw)
    # Receipts are often produced on Windows and inspected from WSL during
    # review.  Normalize separators while retaining native absolute paths.
    normalized = raw_text.replace("\\", "/") if os.sep != "\\" else raw_text
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    # ``Path`` on POSIX does not understand a Windows drive-qualified path.
    # Keep the path unresolved here; a sibling/cwd locator may still be valid,
    # but never silently reinterpret a drive path as a relative artifact.
    if re.match(r"^[A-Za-z]:/", normalized):
        candidate = Path(normalized)
    options = (receipt_path.parent / candidate, Path.cwd() / candidate)
    for option in options:
        if option.is_file():
            return option
    raise RealCaptureBlocked("CAPTURE_ARTIFACT_MISSING", details={"path": str(raw)})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealCaptureBlocked("CAPTURE_RECEIPT_INVALID", details={"path": str(path), "error": str(exc)}) from exc
    if not isinstance(payload, dict):
        raise RealCaptureBlocked("CAPTURE_RECEIPT_INVALID", details={"path": str(path), "error": "expected object"})
    return payload


def _record_ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(item) for item in ids).encode("utf-8")).hexdigest()


def _has_provenance(row: Mapping[str, Any]) -> bool:
    return all(
        str(row.get(key, "")).strip().casefold() not in {"", "unknown", "none", "null"}
        for key in ("source_record_id", "source_family", "source_name", "source_revision")
    )


def validate_method_proof_receipt(path: str | Path) -> dict[str, Any]:
    """Validate the frozen FIT-TRAIN receipt used to select capture records."""

    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_MISSING", details={"path": str(receipt_path)})
    payload = _read_json(receipt_path)
    if payload.get("receipt_type") != "dense2moe-method-proof-data":
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_TYPE_INVALID")
    recorded = str(payload.get("receipt_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not recorded or recorded != canonical_sha256(unsigned):
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_HASH_MISMATCH")
    if payload.get("status") != "METHOD_PROOF_READY":
        raise RealCaptureBlocked("METHOD_PROOF_NOT_READY", details={"status": payload.get("status")})
    policy = payload.get("method_proof_policy")
    if not isinstance(policy, Mapping) or policy.get("eligible_split") != "FIT-TRAIN":
        raise RealCaptureBlocked("METHOD_PROOF_NOT_FIT_TRAIN")
    minimum = int(policy.get("minimum_tokens", 0) or 0)
    if minimum < METHOD_PROOF_MIN_TOKENS:
        raise RealCaptureBlocked("METHOD_PROOF_TOKEN_POLICY_INVALID", details={"minimum_tokens": minimum})
    source_manifest = policy.get("source_manifest")
    source_hash = str(policy.get("source_manifest_sha256", ""))
    if not source_manifest or not source_hash:
        raise RealCaptureBlocked("METHOD_PROOF_SOURCE_MANIFEST_UNHASHED")
    source_path = _resolve_artifact(receipt_path, source_manifest)
    if sha256_file(source_path) != source_hash:
        raise RealCaptureBlocked("METHOD_PROOF_SOURCE_MANIFEST_HASH_MISMATCH")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping) or not manifest.get("path") or not manifest.get("sha256"):
        raise RealCaptureBlocked("METHOD_PROOF_MANIFEST_UNHASHED")
    manifest_path = _resolve_artifact(receipt_path, str(manifest["path"]))
    if sha256_file(manifest_path) != str(manifest["sha256"]):
        raise RealCaptureBlocked("METHOD_PROOF_MANIFEST_HASH_MISMATCH")
    rows: list[dict[str, Any]] = []
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"line {line_number} is not an object")
                rows.append(value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealCaptureBlocked("METHOD_PROOF_MANIFEST_INVALID", details={"error": str(exc)}) from exc
    if not rows:
        raise RealCaptureBlocked("METHOD_PROOF_MANIFEST_EMPTY")
    if any(str(row.get("split", "")) != "FIT-TRAIN" for row in rows):
        raise RealCaptureBlocked("METHOD_PROOF_NON_FIT_ROW")
    if any(
        bool(row.get(name))
        for row in rows
        for name in (
            "benchmark_quarantine",
            "benchmark_quarantine_reason",
            "benchmark_membership",
            "benchmark_denylist",
            "benchmark_context",
            "benchmark_original_split",
        )
    ):
        raise RealCaptureBlocked("METHOD_PROOF_BENCHMARK_CONTAMINATION")
    if any(not _has_provenance(row) for row in rows):
        raise RealCaptureBlocked("METHOD_PROOF_PROVENANCE_MISSING")
    if any(int(row.get("token_count", 0) or 0) <= 0 for row in rows):
        raise RealCaptureBlocked("METHOD_PROOF_TOKEN_COUNT_INVALID")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise RealCaptureBlocked("METHOD_PROOF_IDS_INVALID")
    selected_tokens = sum(int(row.get("token_count", 0) or 0) for row in rows)
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise RealCaptureBlocked("METHOD_PROOF_SELECTION_MISSING")
    if int(selection.get("selected_rows", -1)) != len(rows) or int(selection.get("selected_tokens", -1)) != selected_tokens:
        raise RealCaptureBlocked("METHOD_PROOF_SELECTION_MISMATCH")
    selected_ids = selection.get("selected_ids")
    if selected_ids is not None and [str(item) for item in selected_ids] != ids:
        raise RealCaptureBlocked("METHOD_PROOF_SELECTED_IDS_MISMATCH")
    if selected_tokens < minimum:
        raise RealCaptureBlocked("METHOD_PROOF_TOKEN_THRESHOLD_NOT_MET")
    overlap_audit = payload.get("split_overlap_audit")
    if not isinstance(overlap_audit, Mapping) or str(overlap_audit.get("status", "")).upper() != "PASS":
        raise RealCaptureBlocked("METHOD_PROOF_OVERLAP_PROOF_MISSING")
    excluded_splits = {str(value) for value in policy.get("excluded_splits", [])}
    if not excluded_splits.intersection({"FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C", "PRESERVATION-CANARY"}):
        raise RealCaptureBlocked("METHOD_PROOF_EXCLUSION_POLICY_INVALID")
    declared_buckets = {str(value).casefold() for value in policy.get("diversity_buckets", [])}
    if not {"code", "technical"}.issubset(declared_buckets):
        raise RealCaptureBlocked("METHOD_PROOF_DIVERSITY_POLICY_INVALID")
    bucket_tokens = {"code": 0, "technical": 0}
    for row in rows:
        domain = str(row.get("domain", "")).casefold().replace("_", "-")
        bucket = (
            "code"
            if domain == "code" or domain.startswith("code/") or domain.endswith("/code")
            else "technical"
            if "agentic" in domain or "software-engineering" in domain or domain == "structured"
            else "other"
        )
        if bucket in bucket_tokens:
            bucket_tokens[bucket] += int(row.get("token_count", 0) or 0)
    bucket_target = max(1, int(minimum * 0.25))
    if any(bucket_tokens[bucket] < bucket_target for bucket in ("code", "technical")):
        raise RealCaptureBlocked(
            "METHOD_PROOF_DIVERSITY_BLOCKED",
            details={"bucket_tokens": bucket_tokens, "bucket_target": bucket_target},
        )
    return {
        **payload,
        "_path": str(receipt_path),
        "_manifest_path": str(manifest_path),
        "_rows": rows,
        "_selected_record_ids": ids,
        "_selected_record_ids_sha256": _record_ids_hash(ids),
        "_selected_tokens": selected_tokens,
    }


def _source_value(payload: Mapping[str, Any], name: str) -> Any:
    source = payload.get("source")
    if isinstance(source, Mapping) and name in source:
        return source[name]
    aliases = {
        "model": ("source_model", "model"),
        "revision": ("source_revision", "revision"),
        "model_type": ("source_model_type", "model_type"),
        "num_hidden_layers": ("num_hidden_layers",),
        "hidden_size": ("hidden_size",),
        "dense_intermediate_size": ("dense_intermediate_size",),
    }
    for key in aliases.get(name, (name,)):
        if key in payload:
            return payload[key]
    return None


def _inspect_npy_pair(input_path: Path, target_path: Path) -> tuple[list[int], str, str]:
    """Inspect paired NumPy shards through mmap-only headers."""

    try:
        import numpy as np  # type: ignore

        inputs = np.load(input_path, mmap_mode="r", allow_pickle=False)
        targets = np.load(target_path, mmap_mode="r", allow_pickle=False)
    except (ImportError, OSError, ValueError) as exc:
        raise RealCaptureBlocked("CAPTURE_SHARD_INVALID", details={"path": str(input_path), "error": str(exc)}) from exc
    if inputs.ndim != 2 or targets.ndim != 2:
        raise RealCaptureBlocked(
            "CAPTURE_SHARD_RANK_INVALID",
            details={"path": str(input_path), "input_shape": list(inputs.shape), "target_shape": list(targets.shape)},
        )
    if tuple(inputs.shape) != tuple(targets.shape):
        raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(input_path)})
    return list(inputs.shape), _normalise_dtype_name(inputs.dtype), _normalise_dtype_name(targets.dtype)


def _inspect_shard(path: Path, *, input_name: str, target_name: str) -> tuple[list[int], str, str, str]:
    """Inspect a safetensors shard without eagerly materialising it."""

    if path.suffix.lower() == ".npy":
        raise RealCaptureBlocked(
            "CAPTURE_NPY_TARGET_PATH_MISSING",
            details={"path": str(path), "error": "paired NumPy captures must declare target_path"},
        )
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        raise RealCaptureBlocked("CAPTURE_RUNTIME_UNAVAILABLE", details={"error": "safetensors is required"}) from exc
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = set(handle.keys())
            if input_name not in keys or target_name not in keys:
                raise RealCaptureBlocked(
                    "CAPTURE_SHARD_TENSORS_MISSING",
                    details={"path": str(path), "keys": sorted(keys)},
                )
            input_shape = list(handle.get_slice(input_name).get_shape())
            target_shape = list(handle.get_slice(target_name).get_shape())
            if input_shape != target_shape:
                raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(path)})
            dtype = _normalise_dtype_name(handle.get_slice(input_name).get_dtype())
    except (RuntimeError, TypeError, ValueError) as np_exc:
        # NumPy cannot represent every BF16 safetensors dtype.  The PT
        # framework still exposes shape/dtype lazily without materialising the
        # shard, so use it as the inspection fallback.
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if input_name not in keys or target_name not in keys:
                    raise RealCaptureBlocked(
                        "CAPTURE_SHARD_TENSORS_MISSING",
                        details={"path": str(path), "keys": sorted(keys)},
                    )
                input_shape = list(handle.get_slice(input_name).get_shape())
                target_shape = list(handle.get_slice(target_name).get_shape())
                if input_shape != target_shape:
                    raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(path)})
                dtype = _normalise_dtype_name(handle.get_slice(input_name).get_dtype())
        except RealCaptureBlocked:
            raise
        except (OSError, RuntimeError, TypeError, KeyError) as pt_exc:
            raise RealCaptureBlocked("CAPTURE_SHARD_INVALID", details={"path": str(path), "error": str(pt_exc)}) from np_exc
    except RealCaptureBlocked:
        raise
    except (OSError, KeyError) as exc:
        raise RealCaptureBlocked("CAPTURE_SHARD_INVALID", details={"path": str(path), "error": str(exc)}) from exc
    return input_shape, dtype, "safetensors", input_name


def _verify_runtime_lock(
    receipt_path: Path,
    payload: Mapping[str, Any],
    runtime_lock_path: str | Path | None,
    *,
    enforce_current_runtime: bool = False,
) -> dict[str, Any]:
    identity = payload.get("runtime_lock")
    if not isinstance(identity, Mapping):
        raise RealCaptureBlocked("RUNTIME_LOCK_MISSING")
    raw_path = runtime_lock_path or identity.get("path")
    if not raw_path:
        raise RealCaptureBlocked("RUNTIME_LOCK_MISSING")
    lock_path = _resolve_artifact(receipt_path, str(raw_path))
    loaded = load_runtime_lock(lock_path)
    if loaded.get("status") != "LOCKED" or not loaded.get("ok"):
        raise RealCaptureBlocked("RUNTIME_LOCK_INVALID", details={"status": loaded.get("status"), "error": loaded.get("error")})
    expected_file_hash = str(identity.get("sha256", ""))
    if not expected_file_hash:
        raise RealCaptureBlocked("RUNTIME_LOCK_HASH_MISSING")
    if sha256_file(lock_path) != expected_file_hash:
        raise RealCaptureBlocked("RUNTIME_LOCK_HASH_MISMATCH")
    expected_identity = str(identity.get("lock_sha256", ""))
    if not expected_identity:
        raise RealCaptureBlocked("RUNTIME_LOCK_IDENTITY_MISSING")
    actual_identity = str(loaded.get("payload", {}).get("lock_sha256", ""))
    if expected_identity != actual_identity:
        raise RealCaptureBlocked("RUNTIME_LOCK_IDENTITY_MISMATCH")
    drift: dict[str, Any] | None = None
    if enforce_current_runtime:
        try:
            environment = collect_environment(repo_root=Path.cwd())
            checked = check_runtime_lock(environment, lock_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RealCaptureBlocked("WINDOWS_RUNTIME_DRIFT", details={"error": str(exc)}) from exc
        if checked.get("status") != "LOCKED" or not checked.get("ok"):
            raise RealCaptureBlocked(
                "WINDOWS_RUNTIME_DRIFT",
                details={"drift": checked.get("drift", []), "status": checked.get("status")},
            )
        drift = {"status": checked.get("status"), "drift": checked.get("drift", [])}
    return {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "lock_sha256": actual_identity,
        "payload": loaded["payload"],
        "current_runtime": drift,
    }


def _verify_source_snapshot(receipt_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise RealCaptureBlocked("CAPTURE_SOURCE_METADATA_MISSING")
    config_raw = source.get("config_path")
    config_hash = str(source.get("config_sha256", ""))
    index_raw = source.get("source_index_path")
    index_hash = str(source.get("source_index_sha256", ""))
    if not config_raw or not config_hash or not index_raw or not index_hash:
        raise RealCaptureBlocked("CAPTURE_SOURCE_HASHES_MISSING")
    config_path = _resolve_artifact(receipt_path, str(config_raw))
    index_path = _resolve_artifact(receipt_path, str(index_raw))
    if sha256_file(config_path) != config_hash:
        raise RealCaptureBlocked("CAPTURE_SOURCE_CONFIG_HASH_MISMATCH")
    if sha256_file(index_path) != index_hash:
        raise RealCaptureBlocked("CAPTURE_SOURCE_INDEX_HASH_MISMATCH")
    config = _read_json(config_path)
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    expected = {
        "model_type": QWEN_SOURCE_MODEL_TYPE,
        "num_hidden_layers": QWEN_NUM_HIDDEN_LAYERS,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "dense_intermediate_size": QWEN_DENSE_INTERMEDIATE_SIZE,
    }
    observed = {
        "model_type": text_config.get("model_type"),
        "num_hidden_layers": int(text_config.get("num_hidden_layers", -1)),
        "hidden_size": int(text_config.get("hidden_size", -1)),
        "dense_intermediate_size": int(text_config.get("intermediate_size", text_config.get("dense_intermediate_size", -1))),
    }
    mismatches = {key: {"expected": value, "actual": observed[key]} for key, value in expected.items() if observed[key] != value}
    if mismatches:
        raise RealCaptureBlocked("CAPTURE_SOURCE_GEOMETRY_MISMATCH", details=mismatches)
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or not str(tokenizer.get("revision", "")) or not str(tokenizer.get("hash", "")):
        raise RealCaptureBlocked("CAPTURE_TOKENIZER_IDENTITY_MISSING")
    tokenizer_files = tokenizer.get("files")
    if tokenizer_files is not None and not isinstance(tokenizer_files, Mapping):
        raise RealCaptureBlocked("CAPTURE_TOKENIZER_IDENTITY_INVALID")
    if isinstance(tokenizer_files, Mapping):
        # A tokenizer hash is only meaningful when every declared local file is
        # still present and byte-identical.  This check is intentionally
        # optional for legacy receipts that carry only an immutable aggregate
        # hash, but new captures always include the file inventory.
        for raw_name, raw_hash in tokenizer_files.items():
            tokenizer_path = _resolve_artifact(receipt_path, str(raw_name))
            if sha256_file(tokenizer_path) != str(raw_hash):
                raise RealCaptureBlocked(
                    "CAPTURE_TOKENIZER_HASH_MISMATCH",
                    details={"path": str(tokenizer_path)},
                )
    return {
        "config_path": str(config_path),
        "source_index_path": str(index_path),
        "config_sha256": config_hash,
        "source_index_sha256": index_hash,
        "tokenizer": dict(tokenizer),
    }


def validate_capture_receipt(
    path: str | Path,
    *,
    method_proof_receipt: str | Path | None = None,
    runtime_lock_path: str | Path | None = None,
    require_native_windows: bool = False,
    enforce_runtime_drift: bool | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate a real Qwen layer-0 capture and every referenced shard."""

    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise RealCaptureBlocked("CAPTURE_RECEIPT_MISSING", details={"path": str(receipt_path)})
    payload = _read_json(receipt_path)
    if payload.get("receipt_type") != REAL_CAPTURE_RECEIPT_TYPE:
        raise RealCaptureBlocked("CAPTURE_EVIDENCE_CLASS_INVALID", details={"receipt_type": payload.get("receipt_type")})
    if int(payload.get("schema_version", 0)) != REAL_CAPTURE_SCHEMA_VERSION:
        raise RealCaptureBlocked("CAPTURE_SCHEMA_UNSUPPORTED")
    if payload.get("evidence_class") != REAL_CAPTURE_EVIDENCE_CLASS:
        raise RealCaptureBlocked("CAPTURE_EVIDENCE_CLASS_INVALID")
    if payload.get("status") != REAL_CAPTURE_STATUS:
        raise RealCaptureBlocked("CAPTURE_NOT_READY", details={"status": payload.get("status")})
    if payload.get("scientific_promotion_eligible") is not True:
        raise RealCaptureBlocked("CAPTURE_NOT_SCIENTIFIC_ELIGIBLE")
    if payload.get("production_promotion_eligible") is True:
        raise RealCaptureBlocked("CAPTURE_PRODUCTION_ELIGIBILITY_INVALID")
    recorded = str(payload.get("receipt_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not recorded or recorded != canonical_sha256(unsigned):
        raise RealCaptureBlocked("CAPTURE_RECEIPT_HASH_MISMATCH")
    source_expected = {
        "model": QWEN_SOURCE_MODEL,
        "revision": QWEN_SOURCE_REVISION,
        "model_type": QWEN_SOURCE_MODEL_TYPE,
        "num_hidden_layers": QWEN_NUM_HIDDEN_LAYERS,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "dense_intermediate_size": QWEN_DENSE_INTERMEDIATE_SIZE,
    }
    mismatches = {
        key: {"expected": expected, "actual": _source_value(payload, key)}
        for key, expected in source_expected.items()
        if _source_value(payload, key) != expected
    }
    if mismatches:
        raise RealCaptureBlocked("CAPTURE_SOURCE_IDENTITY_MISMATCH", details=mismatches)
    if int(payload.get("layer", -1)) != 0:
        raise RealCaptureBlocked("CAPTURE_LAYER_NOT_ZERO")
    if payload.get("native_windows") is not True:
        raise RealCaptureBlocked("CAPTURE_NATIVE_WINDOWS_PROOF_MISSING")
    if require_native_windows and (os.name != "nt" or platform.system() != "Windows"):
        raise RealCaptureBlocked("NATIVE_WINDOWS_REQUIRED")
    if payload.get("benchmark_material") is True or payload.get("evaluation_contamination") is True:
        raise RealCaptureBlocked("CAPTURE_EVALUATION_CONTAMINATION")
    source_snapshot = _verify_source_snapshot(receipt_path, payload)
    if payload.get("split") != "FIT-TRAIN":
        raise RealCaptureBlocked("CAPTURE_SPLIT_INVALID")
    excluded = payload.get("excluded_evaluation_identities")
    if not isinstance(excluded, Mapping) or str(excluded.get("status", "")).upper() != "PASS":
        raise RealCaptureBlocked("CAPTURE_EVALUATION_EXCLUSION_PROOF_MISSING")
    linked_method = payload.get("method_proof_receipt")
    method_path = method_proof_receipt
    if method_path is None and isinstance(linked_method, Mapping):
        method_path = linked_method.get("path")
    if not method_path:
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_MISSING")
    method = validate_method_proof_receipt(method_path)
    method_identity = payload.get("method_proof_receipt")
    if not isinstance(method_identity, Mapping):
        raise RealCaptureBlocked("METHOD_PROOF_LINK_MISSING")
    method_file_hash = sha256_file(method["_path"])
    if str(method_identity.get("sha256", "")) != method_file_hash:
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_HASH_MISMATCH")
    if str(method_identity.get("selected_record_ids_sha256", "")) != method["_selected_record_ids_sha256"]:
        raise RealCaptureBlocked("METHOD_PROOF_SELECTED_IDS_HASH_MISMATCH")
    if int(method_identity.get("selected_tokens", -1)) != method["_selected_tokens"]:
        raise RealCaptureBlocked("METHOD_PROOF_SELECTED_TOKEN_MISMATCH")
    lock = _verify_runtime_lock(
        receipt_path,
        payload,
        runtime_lock_path,
        enforce_current_runtime=require_native_windows if enforce_runtime_drift is None else bool(enforce_runtime_drift),
    )
    input_spec = payload.get("input_activation")
    target_spec = payload.get("dense_ffn_target")
    if not isinstance(input_spec, Mapping) or not isinstance(target_spec, Mapping):
        raise RealCaptureBlocked("CAPTURE_TENSOR_CONTRACT_MISSING")
    expected_shape = [int(payload.get("row_count", -1)), QWEN_HIDDEN_SIZE]
    if list(input_spec.get("shape", [])) != expected_shape or list(target_spec.get("shape", [])) != expected_shape:
        raise RealCaptureBlocked("CAPTURE_TENSOR_SHAPE_INVALID", details={"expected": expected_shape})
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RealCaptureBlocked("CAPTURE_SHARDS_MISSING")
    total = 0
    inspected: list[dict[str, Any]] = []
    for item in shards:
        if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
            raise RealCaptureBlocked("CAPTURE_SHARD_METADATA_INVALID")
        shard_path = _resolve_artifact(receipt_path, str(item["path"]))
        actual_hash = sha256_file(shard_path)
        if actual_hash != str(item["sha256"]):
            raise RealCaptureBlocked("CAPTURE_SHARD_HASH_MISMATCH", details={"path": str(shard_path)})
        input_name = str(item.get("input_tensor", "ffn_input"))
        target_name = str(item.get("target_tensor", "dense_ffn_target"))
        target_raw = item.get("target_path")
        target_path: Path | None = None
        target_dtype: str | None = None
        if shard_path.suffix.lower() == ".npy":
            if not target_raw or not item.get("target_sha256"):
                raise RealCaptureBlocked(
                    "CAPTURE_NPY_TARGET_PATH_MISSING",
                    details={"path": str(shard_path)},
                )
            target_path = _resolve_artifact(receipt_path, str(target_raw))
            target_hash = sha256_file(target_path)
            if target_hash != str(item["target_sha256"]):
                raise RealCaptureBlocked("CAPTURE_TARGET_SHARD_HASH_MISMATCH", details={"path": str(target_path)})
            shape, dtype, target_dtype = _inspect_npy_pair(shard_path, target_path)
            fmt = "npy-pair"
        else:
            shape, dtype, fmt, _ = _inspect_shard(shard_path, input_name=input_name, target_name=target_name)
        if len(shape) != 2 or shape[1] != QWEN_HIDDEN_SIZE or int(item.get("count", -1)) != shape[0]:
            raise RealCaptureBlocked("CAPTURE_SHARD_SHAPE_INVALID", details={"path": str(shard_path), "shape": shape})
        declared_dtype = str(item.get("dtype", ""))
        if declared_dtype and declared_dtype not in {dtype, dtype.replace("torch.", "")}:  # pragma: no branch - metadata check
            raise RealCaptureBlocked(
                "CAPTURE_SHARD_DTYPE_MISMATCH",
                details={"path": str(shard_path), "declared": declared_dtype, "actual": dtype},
            )
        total += shape[0]
        inspected.append(
            {
                **dict(item),
                "path": str(shard_path),
                "target_path": str(target_path) if target_path is not None else item.get("target_path"),
                "shape": shape,
                "dtype": dtype,
                "target_dtype": target_dtype,
                "format": fmt,
            }
        )
    if total <= 0 or total != int(payload.get("row_count", -1)):
        raise RealCaptureBlocked("CAPTURE_ROW_COUNT_MISMATCH", details={"expected": payload.get("row_count"), "actual": total})
    if int(payload.get("token_count", -1)) != total:
        raise RealCaptureBlocked("CAPTURE_TOKEN_ROW_MISMATCH")
    if max_tokens is not None and (max_tokens <= 0 or max_tokens > total):
        raise RealCaptureBlocked("CAPTURE_REQUESTED_TOKEN_COUNT_UNAVAILABLE", details={"requested": max_tokens, "available": total})
    return {
        **payload,
        "_path": str(receipt_path),
        "_method_proof": method,
        "_runtime_lock": lock,
        "_source_snapshot": source_snapshot,
        "_shards": inspected,
        "_row_count": total,
    }


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy on the supported Windows stack does not expose a bfloat16
        # dtype.  The PT safetensors fallback is still authoritative; convert
        # only this already-bounded batch slice to float32 at the adapter
        # boundary so the scientific runner can consume it without eagerly
        # materialising the capture.
        if "bfloat16" in str(getattr(value, "dtype", "")):
            value = value.float()
        return value.numpy()
    return value


def iter_capture_batches(
    receipt: Mapping[str, Any] | str | Path,
    *,
    max_tokens: int | None = None,
    batch_rows: int = 256,
) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Yield bounded ``(X, Y, metadata)`` batches from validated shards."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    payload = validate_capture_receipt(receipt, max_tokens=max_tokens) if isinstance(receipt, (str, Path)) else dict(receipt)
    remaining = int(max_tokens if max_tokens is not None else payload["_row_count"])
    for shard in payload.get("_shards", payload.get("shards", [])):
        if remaining <= 0:
            break
        path = Path(str(shard["path"]))
        input_name = str(shard.get("input_tensor", "ffn_input"))
        target_name = str(shard.get("target_tensor", "dense_ffn_target"))
        count = min(int(shard["count"]), remaining)
        if path.suffix.lower() == ".npy":
            target_raw = shard.get("target_path")
            if not target_raw:
                raise RealCaptureBlocked("CAPTURE_NPY_TARGET_PATH_MISSING", details={"path": str(path)})
            target_path = Path(str(target_raw))
            try:
                import numpy as np  # type: ignore

                inputs = np.load(path, mmap_mode="r", allow_pickle=False)
                targets = np.load(target_path, mmap_mode="r", allow_pickle=False)
            except (ImportError, OSError, ValueError) as exc:
                raise RealCaptureBlocked("CAPTURE_SHARD_INVALID", details={"path": str(path), "error": str(exc)}) from exc
            if tuple(inputs.shape) != tuple(targets.shape):
                raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(path)})
            for start in range(0, count, batch_rows):
                stop = min(start + batch_rows, count)
                x = inputs[start:stop]
                y = targets[start:stop]
                yield x, y, {
                    "shard_path": str(path),
                    "target_shard_path": str(target_path),
                    "row_start": start,
                    "row_stop": stop,
                    "sample_count": stop - start,
                    "token_count": stop - start,
                }
            remaining -= count
            continue
        try:
            from safetensors import safe_open  # type: ignore
        except ImportError as exc:
            raise RealCaptureBlocked("CAPTURE_RUNTIME_UNAVAILABLE", details={"error": "safetensors is required"}) from exc
        try:
            # Opening the NumPy backend can succeed for BF16 shards, but the
            # first slice then raises because this NumPy build has no BF16
            # dtype.  Inspect the declared dtype before slicing so the
            # fallback is selected deterministically and no rows are lost.
            with safe_open(str(path), framework="np") as handle:
                input_slice = handle.get_slice(input_name)
                if "bf16" in str(input_slice.get_dtype()).casefold() or "bfloat16" in str(input_slice.get_dtype()).casefold():
                    raise TypeError("NumPy backend cannot slice bfloat16")
                for start in range(0, count, batch_rows):
                    stop = min(start + batch_rows, count)
                    x = _as_numpy(input_slice[start:stop])
                    y = _as_numpy(handle.get_slice(target_name)[start:stop])
                    if getattr(x, "shape", None) != getattr(y, "shape", None):
                        raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(path)})
                    yield x, y, {
                        "shard_path": str(path),
                        "row_start": start,
                        "row_stop": stop,
                        "sample_count": stop - start,
                        "token_count": stop - start,
                    }
        except (RuntimeError, TypeError, ValueError):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                input_slice = handle.get_slice(input_name)
                target_slice = handle.get_slice(target_name)
                for start in range(0, count, batch_rows):
                    stop = min(start + batch_rows, count)
                    x = _as_numpy(input_slice[start:stop])
                    y = _as_numpy(target_slice[start:stop])
                    if getattr(x, "shape", None) != getattr(y, "shape", None):
                        raise RealCaptureBlocked("CAPTURE_INPUT_TARGET_SHAPE_MISMATCH", details={"path": str(path)})
                    yield x, y, {
                        "shard_path": str(path),
                        "row_start": start,
                        "row_stop": stop,
                        "sample_count": stop - start,
                        "token_count": stop - start,
                    }
        remaining -= count


def build_real_capture_receipt(
    streaming_manifest: str | Path,
    *,
    method_proof_receipt: str | Path,
    runtime_lock: str | Path,
    source_snapshot: str | Path,
    source_revision: str = QWEN_SOURCE_REVISION,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build a method-proof receipt from the existing layer-major stream."""

    if os.name != "nt" or platform.system() != "Windows":
        raise RealCaptureBlocked("NATIVE_WINDOWS_REQUIRED")
    stream_path = Path(streaming_manifest)
    stream = _read_json(stream_path)
    if stream.get("status") != "CAPTURE_COMPLETE" or int(stream.get("layer", -1)) != 0:
        raise RealCaptureBlocked("STREAMING_LAYER0_CAPTURE_NOT_READY")
    if stream.get("capture_kind") != "real_qwen_layer0_ffn_input_target":
        raise RealCaptureBlocked("STREAMING_CAPTURE_NOT_REAL_FFN_TARGET")
    if stream.get("split") != "FIT-TRAIN" or stream.get("evidence_class") != REAL_CAPTURE_EVIDENCE_CLASS:
        raise RealCaptureBlocked("STREAMING_CAPTURE_SPLIT_OR_EVIDENCE_INVALID")
    method = validate_method_proof_receipt(method_proof_receipt)
    source_root = Path(source_snapshot)
    config_path = source_root / "config.json"
    config = _read_json(config_path)
    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    source_index = source_root / "model.safetensors.index.json"
    if not source_index.is_file():
        raise RealCaptureBlocked("SOURCE_INDEX_MISSING")
    try:
        from .teacher import snapshot_tokenizer_hashes

        tokenizer_inventory = snapshot_tokenizer_hashes(source_root)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RealCaptureBlocked("CAPTURE_TOKENIZER_IDENTITY_MISSING", details={"error": str(exc)}) from exc
    if not tokenizer_inventory:
        raise RealCaptureBlocked("CAPTURE_TOKENIZER_IDENTITY_MISSING")
    tokenizer_files = {str(source_root / name): digest for name, digest in tokenizer_inventory.items()}
    lock = load_runtime_lock(runtime_lock)
    if lock.get("status") != "LOCKED" or not lock.get("ok"):
        raise RealCaptureBlocked("RUNTIME_LOCK_INVALID", details={"status": lock.get("status"), "error": lock.get("error")})
    source_model_type = text_config.get("model_type", QWEN_SOURCE_MODEL_TYPE)
    if source_model_type != QWEN_SOURCE_MODEL_TYPE:
        raise RealCaptureBlocked("SOURCE_MODEL_TYPE_MISMATCH")
    shard_rows = []
    for item in stream.get("shards", []):
        shard_rows.append(
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "count": int(item["count"]),
                "shape": list(item["shape"]),
                "dtype": str(item.get("dtype", "bfloat16")),
                "format": "safetensors",
                "input_tensor": "ffn_input",
                "target_tensor": "dense_ffn_target",
                "records": list(item.get("records", [])),
            }
        )
    row_count = sum(int(item["count"]) for item in shard_rows)
    receipt: dict[str, Any] = {
        "schema_version": REAL_CAPTURE_SCHEMA_VERSION,
        "receipt_type": REAL_CAPTURE_RECEIPT_TYPE,
        "status": REAL_CAPTURE_STATUS,
        "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
        "scientific_promotion_eligible": True,
        "production_promotion_eligible": False,
        "source": {
            "model": QWEN_SOURCE_MODEL,
            "revision": source_revision,
            "model_type": source_model_type,
            "num_hidden_layers": int(text_config.get("num_hidden_layers", -1)),
            "hidden_size": int(text_config.get("hidden_size", -1)),
            "dense_intermediate_size": int(text_config.get("intermediate_size", text_config.get("dense_intermediate_size", -1))),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_index_path": str(source_index),
            "source_index_sha256": sha256_file(source_index),
        },
        "tokenizer": {
            "revision": source_revision,
            "hash": canonical_sha256(tokenizer_files),
            "files": tokenizer_files,
        },
        "layer": 0,
        "split": "FIT-TRAIN",
        "input_activation": {"name": "X", "tensor": "ffn_input", "shape": [row_count, QWEN_HIDDEN_SIZE]},
        "dense_ffn_target": {"name": "Y", "tensor": "dense_ffn_target", "shape": [row_count, QWEN_HIDDEN_SIZE]},
        "row_count": row_count,
        "token_count": row_count,
        "capture_dtype": str(stream.get("dtype", "bfloat16")),
        "capture_shard_format": "safetensors",
        "shards": shard_rows,
        "method_proof_receipt": {
            "path": str(method_proof_receipt),
            "sha256": sha256_file(method["_path"]),
            "selected_record_ids_sha256": method["_selected_record_ids_sha256"],
            "selected_tokens": method["_selected_tokens"],
        },
        "selected_record_ids_sha256": method["_selected_record_ids_sha256"],
        "selected_tokens": method["_selected_tokens"],
        "excluded_evaluation_identities": {
            "status": "PASS",
            "splits": list(method.get("method_proof_policy", {}).get("excluded_splits", [])),
        },
        "benchmark_material": False,
        "evaluation_contamination": False,
        "native_windows": True,
        "runtime_lock": {
            "path": str(runtime_lock),
            "sha256": sha256_file(runtime_lock),
            "lock_sha256": str(lock.get("payload", {}).get("lock_sha256", "")),
        },
        "source_snapshot": str(source_root),
        "code_commit": code_commit or current_git_commit(),
        "created_at": stream.get("created_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_path": str(stream_path),
    }
    if receipt["source"]["revision"] != QWEN_SOURCE_REVISION:
        raise RealCaptureBlocked("SOURCE_REVISION_MISMATCH")
    if receipt["source"]["num_hidden_layers"] != QWEN_NUM_HIDDEN_LAYERS or receipt["source"]["hidden_size"] != QWEN_HIDDEN_SIZE or receipt["source"]["dense_intermediate_size"] != QWEN_DENSE_INTERMEDIATE_SIZE:
        raise RealCaptureBlocked("SOURCE_GEOMETRY_MISMATCH")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_real_capture_receipt(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a content-addressed receipt atomically."""

    value = dict(payload)
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    value["receipt_sha256"] = canonical_sha256(unsigned)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


__all__ = [
    "METHOD_PROOF_MIN_TOKENS",
    "QWEN_DENSE_INTERMEDIATE_SIZE",
    "QWEN_HIDDEN_SIZE",
    "QWEN_NUM_HIDDEN_LAYERS",
    "QWEN_SOURCE_MODEL",
    "QWEN_SOURCE_MODEL_TYPE",
    "QWEN_SOURCE_REVISION",
    "REAL_CAPTURE_EVIDENCE_CLASS",
    "REAL_CAPTURE_RECEIPT_TYPE",
    "REAL_CAPTURE_SCHEMA_VERSION",
    "REAL_CAPTURE_STATUS",
    "RealCaptureBlocked",
    "build_real_capture_receipt",
    "canonical_sha256",
    "iter_capture_batches",
    "sha256_file",
    "validate_capture_receipt",
    "validate_method_proof_receipt",
    "write_real_capture_receipt",
]
