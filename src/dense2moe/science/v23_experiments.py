"""Receipt-backed experiment helpers for the Dense2MoE V2.3 search.

The V2.3 search deliberately separates *measurement* from the CUDA runners.
This module owns the small, deterministic boundary used by those runners:

* validate a paired FIT-TRAIN/FIT-DEV activation capture;
* turn one record for each registered design into an oracle-ceiling receipt;
* enforce one token/compute budget for equal-budget development; and
* compute a metric-oriented Pareto frontier without opening a promotion tier.

No tensor is loaded here.  The capture runners publish immutable manifests and
the experiment runners pass their measured metrics to these helpers.  Missing
measurement is represented as an explicit failed/blocked design rather than a
synthetic success.  Promotion manifests and tiers are rejected at the input
boundary so a development receipt cannot accidentally become an evaluation
receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .v23_designs import (
    MIN_ACTIVE_FFN_REDUCTION,
    V23Design,
    design_from_mapping,
    validate_design_registry,
)

V23_EXPERIMENT_VERSION = "dense2moe-v2.3-experiments"
V23_EXPERIMENT_SCHEMA_VERSION = 1
V23_METHOD_VERSION = "moe-v23-m01"
V23_DESIGN_IDS = ("A", "B", "C", "D", "E")
V23_DEVELOPMENT_SPLITS = ("FIT-TRAIN", "FIT-DEV")
V23_PROMOTION_TIERS = frozenset(
    {"GATE-A", "SHADOW-B", "SHADOW-C", "G1", "G2", "PRESERVATION-CANARY"}
)
DEFAULT_PILOT_SEEDS = (17, 29, 41)
_PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)

ReceiptInput: TypeAlias = str | os.PathLike[str] | Mapping[str, Any]
DesignRegistryInput: TypeAlias = (
    Mapping[str, Any] | Iterable[V23Design | Mapping[str, Any]]
)


class V23ExperimentBlocked(ValueError):
    """A fail-closed input or receipt validation error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "BLOCKED",
            "blocker_code": self.code,
            "message": self.message,
            **self.details,
        }


@dataclass(frozen=True)
class PairedActivationManifests:
    """Validated development-only activation manifest pair."""

    train: Mapping[str, Any]
    dev: Mapping[str, Any]
    train_path: str | None = None
    dev_path: str | None = None
    train_sha256: str | None = None
    dev_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "FIT-TRAIN": _manifest_summary(
                self.train,
                self.train_path,
                self.train_sha256,
            ),
            "FIT-DEV": _manifest_summary(self.dev, self.dev_path, self.dev_sha256),
        }


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def canonical_json(value: Any) -> str:
    """Return stable JSON used for all V2.3 experiment receipts."""

    return _canonical_bytes(value).decode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(value: ReceiptInput, *, label: str) -> tuple[dict[str, Any], str | None, str | None]:
    if isinstance(value, Mapping):
        return dict(value), None, None
    path = Path(value)
    if not path.is_file():
        raise V23ExperimentBlocked(
            "ACTIVATION_MANIFEST_MISSING",
            f"{label} does not exist: {path}",
            path=str(path),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V23ExperimentBlocked(
            "ACTIVATION_MANIFEST_INVALID",
            f"{label} is not readable JSON: {path}",
            path=str(path),
        ) from exc
    if not isinstance(payload, Mapping):
        raise V23ExperimentBlocked(
            "ACTIVATION_MANIFEST_INVALID",
            f"{label} must contain a JSON object: {path}",
            path=str(path),
        )
    return dict(payload), str(path.resolve()), _sha256_file(path)


def _normalise_split(value: Any) -> str:
    aliases = {"train": "FIT-TRAIN", "fit-train": "FIT-TRAIN", "dev": "FIT-DEV", "holdout": "FIT-DEV", "fit-dev": "FIT-DEV"}
    return aliases.get(str(value).strip().lower(), str(value).strip().upper())


def _promotion_values(value: Any, *, key: str = "") -> list[str]:
    """Find forbidden promotion state in an arbitrary receipt mapping."""

    findings: list[str] = []
    key_lower = str(key).strip().lower()
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            findings.extend(_promotion_values(child_value, key=str(child_key)))
        return findings
    if isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            findings.extend(_promotion_values(child, key=key))
        return findings
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in V23_PROMOTION_TIERS and (
            key_lower in {"tier", "split", "evaluation_tier", "promotion_tier", "name"}
            or "tier" in key_lower
        ):
            findings.append(normalized)
    if key_lower in {
        "opened_evaluation_tiers",
        "promotion_tiers",
        "opened_promotion_tiers",
    } and value and value not in (False, None, [], (), {}, ""):
        findings.append(f"{key_lower}:present")
    if key_lower in {"promotion_mode", "promotion_open", "promotion_eligible"} and value is True:
        findings.append(f"{key_lower}:true")
    return findings


def _guard_development_only(*values: Any) -> None:
    findings: list[str] = []
    for value in values:
        findings.extend(_promotion_values(value))
    if findings:
        unique = sorted(set(findings))
        raise V23ExperimentBlocked(
            "PROMOTION_TIER_INPUT_FORBIDDEN",
            "V2.3 development experiments cannot consume promotion-tier state",
            findings=unique,
        )


def _resolve_artifact_path(raw: Any, *, manifest_path: str | None) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if manifest_path:
        local = Path(manifest_path).parent / path
        if local.exists():
            return local
    return Path.cwd() / path


def _validate_artifact_descriptors(
    payload: Mapping[str, Any],
    *,
    label: str,
    manifest_path: str | None,
) -> tuple[int, bool, bool]:
    """Validate capture shard descriptors and return count/input/target flags."""

    raw_shards = payload.get("shards")
    if raw_shards is None:
        raw_shards = []
    if not isinstance(raw_shards, list):
        raise V23ExperimentBlocked(
            "ACTIVATION_MANIFEST_INVALID",
            f"{label}.shards must be a list",
            split=label,
        )
    total = 0
    saw_input = False
    saw_target = False
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_INVALID",
                f"{label} shard {index} must be an object",
            )
        count = raw_shard.get("count")
        if count is None or int(count) <= 0:
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_INVALID",
                f"{label} shard {index} has no positive count",
            )
        total += int(count)
        input_name = raw_shard.get("input_tensor", raw_shard.get("tensor"))
        target_name = raw_shard.get("target_tensor")
        saw_input = saw_input or isinstance(input_name, str) and bool(input_name.strip())
        saw_target = saw_target or isinstance(target_name, str) and bool(target_name.strip())
        raw_path = raw_shard.get("path")
        if raw_path is None:
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_INVALID",
                f"{label} shard {index} has no artifact path",
            )
        artifact = _resolve_artifact_path(raw_path, manifest_path=manifest_path)
        if not artifact.is_file():
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_MISSING",
                f"{label} shard artifact is missing: {artifact}",
                path=str(artifact),
            )
        expected_hash = raw_shard.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_HASH_MISSING",
                f"{label} shard {index} has no sha256",
            )
        actual_hash = _sha256_file(artifact)
        if actual_hash != expected_hash:
            raise V23ExperimentBlocked(
                "ACTIVATION_SHARD_HASH_MISMATCH",
                f"{label} shard hash mismatch: {artifact}",
                expected=expected_hash,
                actual=actual_hash,
            )
    declared = payload.get("count", payload.get("token_count", payload.get("total_count")))
    if declared is None or int(declared) <= 0:
        raise V23ExperimentBlocked(
            "ACTIVATION_DATA_MISSING",
            f"{label} has no positive captured activation count",
        )
    if raw_shards and total != int(declared):
        raise V23ExperimentBlocked(
            "ACTIVATION_COUNT_MISMATCH",
            f"{label} shard counts do not equal manifest count",
            declared=int(declared),
            shards=total,
        )
    top_input = payload.get("input_tensor", payload.get("tensor"))
    top_target = payload.get("target_tensor")
    saw_input = saw_input or isinstance(top_input, str) and bool(top_input.strip())
    saw_target = saw_target or isinstance(top_target, str) and bool(top_target.strip())
    if not saw_input:
        raise V23ExperimentBlocked(
            "ACTIVATION_INPUT_MISSING",
            f"{label} does not declare an FFN input tensor",
        )
    if not saw_target:
        raise V23ExperimentBlocked(
            "DENSE_TARGET_MISSING",
            f"{label} does not declare a dense FFN target tensor",
        )
    return int(declared), saw_input, saw_target


def _validate_one_manifest(
    value: ReceiptInput,
    *,
    expected_split: str,
    label: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    payload, path, digest = _read_json(value, label=label)
    _guard_development_only(payload)
    if path and any(
        part.lower() in {"promotion", "gate-a", "shadow-b", "shadow-c", "g1", "g2"}
        for part in Path(path).parts
    ):
        raise V23ExperimentBlocked(
            "PROMOTION_TIER_INPUT_FORBIDDEN",
            f"{label} is located in a promotion-tier path",
            path=path,
        )
    status = str(payload.get("status", "")).strip().upper()
    if status not in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED", "V23_CAPTURE_COMPLETE"}:
        raise V23ExperimentBlocked(
            "ACTIVATION_CAPTURE_INCOMPLETE",
            f"{label} is not a complete immutable capture",
            status=status or None,
        )
    split = _normalise_split(payload.get("split", ""))
    if split != expected_split:
        raise V23ExperimentBlocked(
            "ACTIVATION_SPLIT_MISMATCH",
            f"{label} must be {expected_split}, got {split or '<missing>'}",
        )
    dataset_hash = str(payload.get("dataset_hash", "")).strip()
    if not dataset_hash:
        raise V23ExperimentBlocked(
            "ACTIVATION_DATASET_HASH_MISSING",
            f"{label} has no dataset hash",
        )
    source_revision = str(payload.get("source_revision", "")).strip()
    if not source_revision:
        raise V23ExperimentBlocked(
            "ACTIVATION_SOURCE_REVISION_MISSING",
            f"{label} has no pinned teacher source revision",
        )
    if not _PINNED_REVISION_RE.fullmatch(source_revision):
        raise V23ExperimentBlocked(
            "ACTIVATION_SOURCE_REVISION_UNPINNED",
            f"{label} source revision is not an immutable commit SHA",
            source_revision=source_revision,
        )
    _validate_artifact_descriptors(payload, label=expected_split, manifest_path=path)
    return payload, path, digest


def validate_paired_activation_manifests(
    train_manifest: ReceiptInput,
    dev_manifest: ReceiptInput,
    *,
    run_metadata: ReceiptInput | None = None,
) -> PairedActivationManifests:
    """Validate complete, paired, development-only FIT-TRAIN/FIT-DEV captures."""

    train, train_path, train_hash = _validate_one_manifest(
        train_manifest,
        expected_split="FIT-TRAIN",
        label="FIT-TRAIN activation manifest",
    )
    dev, dev_path, dev_hash = _validate_one_manifest(
        dev_manifest,
        expected_split="FIT-DEV",
        label="FIT-DEV activation manifest",
    )
    if run_metadata is not None:
        metadata, metadata_path, _ = _read_json(run_metadata, label="V2.3 run metadata")
        _guard_development_only(metadata)
        if metadata_path and any(part.lower() in {"promotion", "gate-a", "shadow-b"} for part in Path(metadata_path).parts):
            raise V23ExperimentBlocked(
                "PROMOTION_TIER_INPUT_FORBIDDEN",
                "run metadata is located in a promotion-tier path",
                path=metadata_path,
            )
    for key in ("dataset_hash", "source_revision", "layer"):
        left = train.get(key)
        right = dev.get(key)
        if left is not None and right is not None and str(left) != str(right):
            raise V23ExperimentBlocked(
                "ACTIVATION_PAIR_MISMATCH",
                f"FIT-TRAIN and FIT-DEV disagree on {key}",
                field=key,
                train=left,
                dev=right,
            )
    train_meta = train.get("metadata")
    dev_meta = dev.get("metadata")
    if isinstance(train_meta, Mapping) and isinstance(dev_meta, Mapping):
        for key in ("tokenizer_hash", "source_snapshot", "capture_contract_hash"):
            left = train_meta.get(key)
            right = dev_meta.get(key)
            if left is not None and right is not None and str(left) != str(right):
                raise V23ExperimentBlocked(
                    "ACTIVATION_PAIR_MISMATCH",
                    f"FIT-TRAIN and FIT-DEV metadata disagree on {key}",
                    field=key,
                    train=left,
                    dev=right,
                )
    return PairedActivationManifests(
        train=train,
        dev=dev,
        train_path=train_path,
        dev_path=dev_path,
        train_sha256=train_hash,
        dev_sha256=dev_hash,
    )


def load_paired_activation_manifests(
    train_manifest: ReceiptInput,
    dev_manifest: ReceiptInput,
    *,
    run_metadata: ReceiptInput | None = None,
) -> PairedActivationManifests:
    """Readable alias for :func:`validate_paired_activation_manifests`."""

    return validate_paired_activation_manifests(
        train_manifest,
        dev_manifest,
        run_metadata=run_metadata,
    )


def _manifest_summary(
    payload: Mapping[str, Any], path: str | None, digest: str | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": str(payload.get("status", "")),
        "split": _normalise_split(payload.get("split", "")),
        "count": int(payload.get("count", payload.get("token_count", payload.get("total_count", 0)))),
        "dataset_hash": str(payload.get("dataset_hash", "")),
        "source_revision": str(payload.get("source_revision", "")),
        "layer": payload.get("layer"),
    }
    if path is not None:
        result["path"] = path
    if digest is not None:
        result["sha256"] = digest
    for key in ("tokenizer_hash", "input_tensor", "target_tensor", "capture_kind", "evidence_class"):
        if key in payload:
            result[key] = payload[key]
    return result


def _normalise_registry(registry: DesignRegistryInput | None) -> tuple[V23Design, ...]:
    if registry is None:
        return validate_design_registry()
    if isinstance(registry, Mapping):
        values: Any = registry.get("designs", registry)
        if isinstance(values, Mapping):
            values = list(values.values())
    else:
        values = list(registry)
    designs: list[V23Design] = []
    for value in values:
        if isinstance(value, V23Design):
            design = value
        elif isinstance(value, Mapping):
            design = design_from_mapping(value)
        else:
            raise V23ExperimentBlocked(
                "DESIGN_REGISTRY_INVALID",
                "design registry entries must be V23Design objects or mappings",
            )
        design.validate(minimum_reduction=MIN_ACTIVE_FFN_REDUCTION)
        designs.append(design)
    designs.sort(key=lambda item: item.design_id)
    if tuple(item.design_id for item in designs) != V23_DESIGN_IDS:
        raise V23ExperimentBlocked(
            "DESIGN_REGISTRY_INCOMPLETE",
            "V2.3 experiment receipts require exactly designs A, B, C, D, and E",
            design_ids=[item.design_id for item in designs],
        )
    return tuple(designs)


def _design_registry_digest(designs: Sequence[V23Design]) -> str:
    return _sha256_bytes(_canonical_bytes([design.as_dict() for design in designs]))


def _base_receipt(
    receipt_type: str,
    pair: PairedActivationManifests,
    designs: Sequence[V23Design],
) -> dict[str, Any]:
    return {
        "receipt_type": receipt_type,
        "version": V23_EXPERIMENT_VERSION,
        "schema_version": V23_EXPERIMENT_SCHEMA_VERSION,
        "method_version": V23_METHOD_VERSION,
        "fit_only": True,
        "opened_evaluation_tiers": [],
        "promotion_tiers_opened": [],
        "activation_pair": pair.as_dict(),
        "design_registry_sha256": _design_registry_digest(designs),
        "design_ids": list(V23_DESIGN_IDS),
        "minimum_active_ffn_reduction": MIN_ACTIVE_FFN_REDUCTION,
    }


def _as_failure_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    return [str(value)]


def _assert_finite(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise V23ExperimentBlocked(
            "NONFINITE_EXPERIMENT_METRIC",
            f"non-finite metric at {path}",
            metric_path=path,
        )


def _attempt_map(attempts: Any) -> dict[str, Mapping[str, Any]]:
    if attempts is None:
        return {}
    if isinstance(attempts, Mapping):
        if isinstance(attempts.get("designs"), list):
            attempts = attempts["designs"]
        else:
            result: dict[str, Mapping[str, Any]] = {}
            for key, value in attempts.items():
                if not isinstance(value, Mapping):
                    raise V23ExperimentBlocked(
                        "DESIGN_ATTEMPT_INVALID",
                        f"attempt for design {key!r} must be an object",
                    )
                result[str(key).strip().upper()] = dict(value)
            return result
    if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes, bytearray)):
        result = {}
        for value in attempts:
            if not isinstance(value, Mapping) or not str(value.get("design_id", "")).strip():
                raise V23ExperimentBlocked(
                    "DESIGN_ATTEMPT_INVALID",
                    "list attempts require a design_id on every entry",
                )
            result[str(value["design_id"]).strip().upper()] = dict(value)
        return result
    raise V23ExperimentBlocked(
        "DESIGN_ATTEMPT_INVALID",
        "design attempts must be a mapping or sequence of mappings",
    )


def _normalise_oracle_attempt(
    design: V23Design,
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "design_id": design.design_id,
        "name": design.name,
        "topology_id": design.topology_id,
        "active_ffn_reduction": design.active_ffn_reduction,
        "attempted": raw is not None,
    }
    if raw is None:
        record.update(
            {
                "status": "BLOCKED",
                "oracle_eligible": False,
                "failure_reasons": ["ORACLE_ATTEMPT_MISSING"],
            }
        )
        return record
    _guard_development_only(raw)
    _assert_finite(raw, path=f"designs.{design.design_id}")
    status = str(raw.get("status", "ATTEMPTED")).strip().upper()
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise V23ExperimentBlocked(
            "DESIGN_ATTEMPT_INVALID",
            f"oracle metrics for design {design.design_id} must be an object",
        )
    metrics = dict(metrics)
    _assert_finite(metrics, path=f"designs.{design.design_id}.metrics")
    reasons = _as_failure_reasons(raw.get("failure_reasons", raw.get("failure_reason")))
    gate = raw.get("oracle_gate_pass", raw.get("oracle_eligible"))
    if gate is None and isinstance(raw.get("oracle_gate"), Mapping):
        gate = raw["oracle_gate"].get("pass", raw["oracle_gate"].get("status") == "PASS")
    if gate is None:
        gate = status in {"PASS", "ORACLE_PASS", "VIABLE", "ORACLE_ELIGIBLE"}
        if status in {"ATTEMPTED", "MEASURED"}:
            reasons.append("ORACLE_GATE_RESULT_MISSING")
    gate = bool(gate)
    if design.active_ffn_reduction < MIN_ACTIVE_FFN_REDUCTION:
        gate = False
        reasons.append("ACTIVE_FFN_REDUCTION_BELOW_THRESHOLD")
    if not gate and not reasons:
        reasons.append("ORACLE_GATE_FAILED")
    if gate and not metrics:
        gate = False
        reasons.append("ORACLE_METRICS_MISSING")
    if gate:
        canonical_status = "ORACLE_PASS"
    elif status in {"BLOCKED", "MISSING", "NOT_ATTEMPTED"}:
        canonical_status = "BLOCKED"
    else:
        canonical_status = "ORACLE_FAILED"
    record.update(
        {
            "status": canonical_status,
            "oracle_eligible": gate,
            "metrics": metrics,
            "failure_reasons": sorted(set(reasons)),
        }
    )
    for key in ("checkpoint_sha256", "configuration", "attempt_id", "notes"):
        if key in raw:
            record[key] = raw[key]
    return record


def _publish_receipt(payload: Mapping[str, Any], output: str | os.PathLike[str] | None) -> dict[str, Any]:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    body["receipt_sha256"] = _sha256_bytes(_canonical_bytes(body))
    result = json.loads(_canonical_bytes(body).decode("utf-8"))
    if output is None:
        return result
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(result) + b"\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise V23ExperimentBlocked(
                "RECEIPT_IMMUTABLE_MISMATCH",
                f"refusing to overwrite invalid existing receipt: {path}",
            ) from exc
        if existing != result:
            raise V23ExperimentBlocked(
                "RECEIPT_IMMUTABLE_MISMATCH",
                f"refusing to overwrite a different immutable receipt: {path}",
            )
    else:
        fd, temporary = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            temporary_path.write_bytes(encoded)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    result["receipt_path"] = str(path)
    return result


def build_oracle_ceiling_receipt(
    train_manifest: ReceiptInput,
    dev_manifest: ReceiptInput,
    *,
    attempts: Any = None,
    design_registry: DesignRegistryInput | None = None,
    run_metadata: ReceiptInput | None = None,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic selector-independent oracle-ceiling receipt."""

    pair = validate_paired_activation_manifests(
        train_manifest,
        dev_manifest,
        run_metadata=run_metadata,
    )
    designs = _normalise_registry(design_registry)
    supplied = _attempt_map(attempts)
    unknown = sorted(set(supplied) - set(V23_DESIGN_IDS))
    if unknown:
        raise V23ExperimentBlocked(
            "DESIGN_ATTEMPT_UNKNOWN",
            "oracle attempts contain unknown design IDs",
            design_ids=unknown,
        )
    records = [_normalise_oracle_attempt(design, supplied.get(design.design_id)) for design in designs]
    all_attempted = all(bool(record.get("attempted")) for record in records)
    receipt = _base_receipt("dense2moe-v2.3-oracle-ceiling", pair, designs)
    receipt.update(
        {
            "status": "ORACLE_CEILINGS_COMPLETE" if all_attempted else "ORACLE_CEILING_BLOCKED",
            "all_five_designs_recorded": True,
            "all_five_designs_attempted": all_attempted,
            "attempted_design_count": sum(bool(record.get("attempted")) for record in records),
            "designs": records,
            "failed_design_ids": [
                record["design_id"] for record in records if not record.get("oracle_eligible", False)
            ],
            "failure_reasons": {
                record["design_id"]: list(record.get("failure_reasons", []))
                for record in records
                if record.get("failure_reasons")
            },
        }
    )
    return _publish_receipt(receipt, output)


oracle_ceiling_receipt = build_oracle_ceiling_receipt


def _load_receipt(value: ReceiptInput, *, label: str) -> dict[str, Any]:
    payload, _, _ = _read_json(value, label=label)
    _guard_development_only(payload)
    if not str(payload.get("receipt_type", "")).startswith("dense2moe-v2.3-"):
        raise V23ExperimentBlocked(
            "EXPERIMENT_RECEIPT_INVALID",
            f"{label} is not a V2.3 experiment receipt",
        )
    if payload.get("design_ids") != list(V23_DESIGN_IDS):
        raise V23ExperimentBlocked(
            "EXPERIMENT_RECEIPT_INCOMPLETE",
            f"{label} does not contain all five design IDs",
        )
    return payload


def _result_map(results: Any) -> dict[str, Mapping[str, Any]]:
    if results is None:
        return {}
    return _attempt_map(results)


def _seed_map(value: Any) -> dict[int, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        result: dict[int, Mapping[str, Any]] = {}
        for key, item in value.items():
            if not isinstance(item, Mapping):
                raise V23ExperimentBlocked("SEED_RESULT_INVALID", "every seed result must be an object")
            result[int(key)] = dict(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {}
        for item in value:
            if not isinstance(item, Mapping) or item.get("seed") is None:
                raise V23ExperimentBlocked("SEED_RESULT_INVALID", "seed result list entries require seed")
            result[int(item["seed"])] = dict(item)
        return result
    return {}


def _normalise_equal_budget_result(
    design: Mapping[str, Any],
    raw: Mapping[str, Any] | None,
    *,
    expected_tokens: int | None,
    expected_flops: int | None,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    design_id = str(design["design_id"])
    record: dict[str, Any] = {
        "design_id": design_id,
        "topology_id": design.get("topology_id"),
        "active_ffn_reduction": float(design.get("active_ffn_reduction", 0.0)),
        "attempted": raw is not None,
    }
    if raw is None:
        record.update({"status": "BLOCKED", "eligible_for_frontier": False, "failure_reasons": ["EQUAL_BUDGET_RESULT_MISSING"]})
        return record
    _guard_development_only(raw)
    _assert_finite(raw, path=f"equal_budget.{design_id}")
    reasons = _as_failure_reasons(raw.get("failure_reasons", raw.get("failure_reason")))
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise V23ExperimentBlocked("EQUAL_BUDGET_RESULT_INVALID", f"metrics for design {design_id} must be an object")
    metrics = dict(metrics)
    _assert_finite(metrics, path=f"equal_budget.{design_id}.metrics")
    tokens_raw = raw.get("tokens", raw.get("budget_tokens"))
    flops_raw = raw.get("flops", raw.get("budget_flops"))
    if tokens_raw is None:
        reasons.append("EQUAL_BUDGET_TOKENS_MISSING")
    else:
        tokens = int(tokens_raw)
        record["tokens"] = tokens
        if tokens <= 0:
            reasons.append("EQUAL_BUDGET_TOKENS_INVALID")
        if expected_tokens is not None and tokens != expected_tokens:
            reasons.append("EQUAL_BUDGET_TOKENS_MISMATCH")
    if expected_flops is not None:
        if flops_raw is None:
            reasons.append("EQUAL_BUDGET_FLOPS_MISSING")
        else:
            flops = int(flops_raw)
            record["flops"] = flops
            if flops <= 0 or flops != expected_flops:
                reasons.append("EQUAL_BUDGET_FLOPS_MISMATCH")
    elif flops_raw is not None:
        record["flops"] = int(flops_raw)
    seed_results = _seed_map(raw.get("seeds", raw.get("seed_results")))
    missing_seeds = sorted(set(expected_seeds) - set(seed_results))
    extra_seeds = sorted(set(seed_results) - set(expected_seeds))
    if missing_seeds:
        reasons.append("PILOT_SEEDS_MISSING:" + ",".join(str(item) for item in missing_seeds))
    if extra_seeds:
        reasons.append("PILOT_SEEDS_UNEXPECTED:" + ",".join(str(item) for item in extra_seeds))
    for seed, seed_result in seed_results.items():
        _assert_finite(seed_result, path=f"equal_budget.{design_id}.seeds.{seed}")
        if seed_result.get("finite", seed_result.get("finite_gradients")) is False:
            reasons.append(f"NONFINITE_GRADIENTS_SEED_{seed}")
        if seed_result.get("checkpoint_reload") is False or seed_result.get("reload_verified") is False:
            reasons.append(f"CHECKPOINT_RELOAD_FAILED_SEED_{seed}")
        dead = seed_result.get("dead_experts")
        if dead is not None and int(dead) != 0:
            reasons.append(f"DEAD_EXPERTS_SEED_{seed}")
    if not metrics:
        reasons.append("EQUAL_BUDGET_METRICS_MISSING")
    if raw.get("checkpoint_reload") is False or raw.get("reload_verified") is False:
        reasons.append("CHECKPOINT_RELOAD_FAILED")
    if raw.get("dead_experts") is not None and int(raw["dead_experts"]) != 0:
        reasons.append("DEAD_EXPERTS_PRESENT")
    if raw.get("status") in {"FAILED", "BLOCKED", "REJECTED"} and not reasons:
        reasons.append("DESIGN_REPORTED_FAILURE")
    eligible = not reasons and bool(seed_results)
    record.update(
        {
            "status": "DEV_PASS" if eligible else "DEV_FAILED",
            "eligible_for_frontier": eligible,
            "metrics": metrics,
            "seeds": {str(seed): seed_results[seed] for seed in sorted(seed_results)},
            "failure_reasons": sorted(set(reasons)),
        }
    )
    for key in ("checkpoint_sha256", "candidate_id", "throughput_tokens_per_second", "peak_memory_bytes"):
        if key in raw:
            record[key] = raw[key]
    return record


def build_equal_budget_receipt(
    oracle_receipt: ReceiptInput,
    train_manifest: ReceiptInput,
    dev_manifest: ReceiptInput,
    *,
    results: Any = None,
    token_budget: int | None = None,
    flops_budget: int | None = None,
    seeds: Sequence[int] = DEFAULT_PILOT_SEEDS,
    run_metadata: ReceiptInput | None = None,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build an equal-budget development receipt from measured design results."""

    if token_budget is None or int(token_budget) <= 0:
        raise V23ExperimentBlocked(
            "EQUAL_BUDGET_TOKENS_MISSING",
            "an explicit positive token budget is required for equal-budget comparison",
        )
    if flops_budget is None or int(flops_budget) <= 0:
        raise V23ExperimentBlocked(
            "EQUAL_BUDGET_FLOPS_MISSING",
            "an explicit positive active-FLOPs budget is required for equal-budget comparison",
        )
    token_budget = int(token_budget)
    flops_budget = int(flops_budget)
    oracle = _load_receipt(oracle_receipt, label="oracle ceiling receipt")
    pair = validate_paired_activation_manifests(
        train_manifest,
        dev_manifest,
        run_metadata=run_metadata,
    )
    if oracle.get("activation_pair") != pair.as_dict():
        raise V23ExperimentBlocked(
            "ACTIVATION_PAIR_MISMATCH",
            "equal-budget manifests do not match the oracle-ceiling receipt",
        )
    expected_seeds = tuple(sorted({int(seed) for seed in seeds}))
    if not expected_seeds:
        raise V23ExperimentBlocked("PILOT_SEEDS_MISSING", "at least one deterministic seed is required")
    supplied = _result_map(results)
    unknown = sorted(set(supplied) - set(V23_DESIGN_IDS))
    if unknown:
        raise V23ExperimentBlocked("DESIGN_RESULT_UNKNOWN", "equal-budget results contain unknown designs", design_ids=unknown)
    oracle_designs = {str(item["design_id"]): item for item in oracle.get("designs", [])}
    records: list[dict[str, Any]] = []
    for design_id in V23_DESIGN_IDS:
        oracle_design = oracle_designs.get(design_id)
        if not isinstance(oracle_design, Mapping):
            raise V23ExperimentBlocked("ORACLE_RECEIPT_INCOMPLETE", f"oracle receipt is missing design {design_id}")
        if not oracle_design.get("oracle_eligible", False):
            records.append(
                {
                    "design_id": design_id,
                    "topology_id": oracle_design.get("topology_id"),
                    "active_ffn_reduction": oracle_design.get("active_ffn_reduction"),
                    "attempted": False,
                    "status": "PRUNED",
                    "eligible_for_frontier": False,
                    "failure_reasons": sorted(set(_as_failure_reasons(oracle_design.get("failure_reasons")) + ["ORACLE_GATE_FAILED"])),
                }
            )
        else:
            records.append(
                _normalise_equal_budget_result(
                    oracle_design,
                    supplied.get(design_id),
                    expected_tokens=token_budget,
                    expected_flops=flops_budget,
                    expected_seeds=expected_seeds,
                )
            )
    all_viable_measured = all(
        record.get("status") in {"DEV_PASS", "DEV_FAILED", "PRUNED"} for record in records
    )
    blocked = any(record.get("status") == "BLOCKED" for record in records)
    receipt = _base_receipt("dense2moe-v2.3-equal-budget", pair, _normalise_registry(None))
    receipt["design_registry_sha256"] = oracle.get("design_registry_sha256")
    receipt.update(
        {
            "status": "EQUAL_BUDGET_COMPLETE" if all_viable_measured and not blocked else "EQUAL_BUDGET_BLOCKED",
            "token_budget": token_budget,
            "flops_budget": flops_budget,
            "seeds": list(expected_seeds),
            "all_five_designs_recorded": True,
            "designs": records,
            "failed_design_ids": [record["design_id"] for record in records if record.get("failure_reasons")],
            "failure_reasons": {
                record["design_id"]: list(record.get("failure_reasons", []))
                for record in records
                if record.get("failure_reasons")
            },
        }
    )
    return _publish_receipt(receipt, output)


equal_budget_receipt = build_equal_budget_receipt


_LOWER_IS_BETTER = ("nmse", "oracle_regret", "load_cv", "median_norm_ratio_error", "p95_relative_norm_error", "peak_memory_bytes")
_HIGHER_IS_BETTER = ("cosine", "throughput_tokens_per_second", "active_ffn_reduction")


def _candidate_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    raw = record.get("metrics", {})
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float] = {}
    for key in (*_LOWER_IS_BETTER, *_HIGHER_IS_BETTER):
        value = raw.get(key, record.get(key))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            result[key] = float(value)
    reduction = record.get("active_ffn_reduction")
    if "active_ffn_reduction" not in result and isinstance(reduction, (int, float)):
        result["active_ffn_reduction"] = float(reduction)
    return result


def _dominates(left: Mapping[str, float], right: Mapping[str, float], dimensions: Sequence[str]) -> bool:
    no_worse = True
    strictly_better = False
    for dimension in dimensions:
        a = left[dimension]
        b = right[dimension]
        if dimension in _LOWER_IS_BETTER:
            if a > b:
                no_worse = False
            if a < b:
                strictly_better = True
        else:
            if a < b:
                no_worse = False
            if a > b:
                strictly_better = True
    return no_worse and strictly_better


def build_pareto_frontier_receipt(
    equal_budget_receipt: ReceiptInput,
    *,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compute a deterministic frontier from a complete equal-budget receipt."""

    equal = _load_receipt(equal_budget_receipt, label="equal-budget receipt")
    if equal.get("receipt_type") != "dense2moe-v2.3-equal-budget":
        raise V23ExperimentBlocked("EXPERIMENT_RECEIPT_INVALID", "Pareto computation requires an equal-budget receipt")
    if equal.get("status") != "EQUAL_BUDGET_COMPLETE":
        raise V23ExperimentBlocked("EQUAL_BUDGET_RECEIPT_BLOCKED", "cannot compute a frontier from blocked equal-budget data")
    records = equal.get("designs")
    if not isinstance(records, list) or len(records) != len(V23_DESIGN_IDS):
        raise V23ExperimentBlocked("EXPERIMENT_RECEIPT_INCOMPLETE", "equal-budget receipt must record all five designs")
    candidates: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise V23ExperimentBlocked("EXPERIMENT_RECEIPT_INVALID", "equal-budget design record must be an object")
        if not record.get("eligible_for_frontier"):
            continue
        metrics = _candidate_metrics(record)
        if "active_ffn_reduction" not in metrics or metrics["active_ffn_reduction"] < MIN_ACTIVE_FFN_REDUCTION:
            continue
        if not metrics:
            raise V23ExperimentBlocked("PARETO_METRICS_MISSING", f"metrics missing for {record.get('design_id')}")
        candidates.append({**dict(record), "metrics": metrics})
    dimensions = [
        dimension
        for dimension in (*_LOWER_IS_BETTER, *_HIGHER_IS_BETTER)
        if candidates and all(dimension in candidate["metrics"] for candidate in candidates)
    ]
    if not dimensions and candidates:
        raise V23ExperimentBlocked(
            "PARETO_METRICS_INCOMPLETE",
            "eligible designs do not share a comparable metric set",
        )
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = any(
            other["design_id"] != candidate["design_id"]
            and _dominates(other["metrics"], candidate["metrics"], dimensions)
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: (str(item["design_id"]), str(item.get("candidate_id", ""))))
    receipt = {
        "receipt_type": "dense2moe-v2.3-pareto-frontier",
        "version": V23_EXPERIMENT_VERSION,
        "schema_version": V23_EXPERIMENT_SCHEMA_VERSION,
        "method_version": V23_METHOD_VERSION,
        "fit_only": True,
        "opened_evaluation_tiers": [],
        "promotion_tiers_opened": [],
        "equal_budget_receipt_sha256": equal.get("receipt_sha256"),
        "design_registry_sha256": equal.get("design_registry_sha256"),
        "activation_pair": equal.get("activation_pair"),
        "design_ids": list(V23_DESIGN_IDS),
        "all_five_designs_recorded": True,
        "status": "PARETO_FRONTIER_COMPLETE" if frontier else "NO_ELIGIBLE_DESIGNS",
        "dimensions": dimensions,
        "designs": [dict(record) for record in records],
        "frontier": frontier,
        "frontier_design_ids": [str(item["design_id"]) for item in frontier],
        "negative_evidence": {
            str(record["design_id"]): list(record.get("failure_reasons", []))
            for record in records
            if record.get("failure_reasons")
        },
    }
    return _publish_receipt(receipt, output)


pareto_frontier_receipt = build_pareto_frontier_receipt

# Report-oriented names keep the runner call sites readable while preserving
# one implementation and one receipt schema for each experiment stage.
build_oracle_ceiling_report = build_oracle_ceiling_receipt
build_equal_budget_report = build_equal_budget_receipt
build_pareto_frontier_report = build_pareto_frontier_receipt
validate_v23_experiment_inputs = validate_paired_activation_manifests


__all__ = [
    "DEFAULT_PILOT_SEEDS",
    "V23_DESIGN_IDS",
    "V23_DEVELOPMENT_SPLITS",
    "V23_EXPERIMENT_SCHEMA_VERSION",
    "V23_EXPERIMENT_VERSION",
    "V23_METHOD_VERSION",
    "V23_PROMOTION_TIERS",
    "PairedActivationManifests",
    "V23ExperimentBlocked",
    "build_equal_budget_receipt",
    "build_equal_budget_report",
    "build_oracle_ceiling_receipt",
    "build_oracle_ceiling_report",
    "build_pareto_frontier_receipt",
    "build_pareto_frontier_report",
    "canonical_json",
    "equal_budget_receipt",
    "load_paired_activation_manifests",
    "oracle_ceiling_receipt",
    "pareto_frontier_receipt",
    "validate_paired_activation_manifests",
    "validate_v23_experiment_inputs",
]
