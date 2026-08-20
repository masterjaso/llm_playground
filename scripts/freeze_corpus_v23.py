#!/usr/bin/env python3
"""Freeze the generalized, grouped-disjoint V2.3 development corpus.

Only FIT-TRAIN and FIT-DEV are accepted.  The command performs provenance,
tokenizer, tier-group, near-duplicate, and historical V2.2 overlap audits
before writing any immutable artifact.  It does not download data and it
never consumes evaluation, retired, promotion, or V2.2 rows as a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    build_balanced_activation_plan,
    sha256_file,
    verify_immutable_artifacts,
    write_immutable_json,
    write_immutable_text,
)
from scripts.acquire_corpus_v23_public import (
    DEFAULT_OUTPUT,
    DEVELOPMENT_TIERS,
    V23_METHOD_VERSION,
    _canonical_id,
    audit_v22_overlap,
    audit_v23_tier_disjointness,
    discover_v22_ledgers,
    load_v22_rows,
    normalized_text_hash,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHOD_VERSION = V23_METHOD_VERSION
DEFAULT_PLANNED_TOKENS = 131_072


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}:{index + 1} is not a JSON object")
            rows.append(dict(value))
    return rows


def _stable_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_tokenizer(path: Path | None) -> tuple[Any | None, dict[str, Any]]:
    if path is None:
        return None, {"path": None, "sha256": None, "revision": "declared-input-token-counts"}
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tokenizers is required when --tokenizer-path is supplied") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer, {"path": str(path), "sha256": sha256_file(path), "revision": f"sha256:{sha256_file(path)}"}


def _source_file_hash(row: Mapping[str, Any]) -> str:
    for key in ("source_file_sha256", "download_sha256", "raw_sha256", "file_sha256"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _normalise_row(raw: Mapping[str, Any], *, tokenizer: Any | None, source_path: Path, root: Path) -> dict[str, Any]:
    row = dict(raw)
    text = str(row.get("text", row.get("content", "")))
    if not text.strip():
        raise ValueError("V2.3 records must contain non-empty text")
    tier = str(row.get("tier", row.get("split", ""))).strip()
    if tier not in DEVELOPMENT_TIERS:
        raise ValueError(f"V2.3 accepts only FIT-TRAIN/FIT-DEV, found {tier!r}")
    source_name = str(row.get("source_name", row.get("source", ""))).strip()
    source_url = str(row.get("source_url", row.get("url", ""))).strip()
    source_revision = str(row.get("source_revision", row.get("revision", ""))).strip()
    source_license = str(row.get("source_license", row.get("license", ""))).strip()
    source_record_id = str(row.get("source_record_id", row.get("record_id", ""))).strip()
    acquisition_time = str(row.get("acquisition_time_utc", row.get("acquisition_time", ""))).strip()
    source_file = str(row.get("source_file", "")).strip()
    source_hash = _source_file_hash(row)
    missing = [
        name
        for name, value in (
            ("source_name", source_name),
            ("source_url", source_url),
            ("source_revision", source_revision),
            ("source_license", source_license),
            ("source_record_id", source_record_id),
            ("acquisition_time_utc", acquisition_time),
            ("source_file", source_file),
            ("source_file_sha256/download_sha256", source_hash),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"{source_record_id or '<unknown>'} is missing provenance: {missing}")
    if not source_url.startswith(("https://", "http://")):
        raise ValueError(f"{source_record_id} has a non-public source URL")
    if any(term in source_license.casefold() for term in ("unknown", "proprietary", "restricted", "non-commercial")):
        raise ValueError(f"{source_record_id} has an ambiguous/non-permissive license")
    if len(source_hash) != 64 or any(char not in "0123456789abcdefABCDEF" for char in source_hash):
        raise ValueError(f"{source_record_id} has an invalid source-file SHA-256")
    try:
        from datetime import datetime

        datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{source_record_id} has an invalid acquisition timestamp") from exc

    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized_hash = normalized_text_hash(text)
    if row.get("content_sha256") and str(row["content_sha256"]) != raw_hash:
        raise ValueError(f"{source_record_id} content_sha256 does not match text")
    if row.get("normalized_content_sha256") and str(row["normalized_content_sha256"]) != normalized_hash:
        raise ValueError(f"{source_record_id} normalized_content_sha256 does not match text")
    canonical = dict(row)
    canonical.update(
        {
            "source_name": source_name,
            "source_revision": source_revision,
            "source_record_id": source_record_id,
            "content_sha256": raw_hash,
        }
    )
    expected_id = _canonical_id(canonical)
    existing_id = str(row.get("id", row.get("stable_id", ""))).strip()
    if existing_id and existing_id != expected_id:
        raise ValueError(f"{source_record_id} has a non-deterministic id (expected {expected_id})")
    row.update(
        {
            "id": expected_id,
            "text": text,
            "content_sha256": raw_hash,
            "normalized_content_sha256": normalized_hash,
            "tier": tier,
            "split": tier,
            "source_name": source_name,
            "source_url": source_url,
            "source_revision": source_revision,
            "source_license": source_license,
            "source_record_id": source_record_id,
            "source_file_sha256": source_hash,
            "acquisition_time_utc": acquisition_time,
            "source_file": source_file,
            "source_path_from_freeze": _stable_path(source_path, root=root),
            "v23_method_version": str(row.get("v23_method_version", DEFAULT_METHOD_VERSION)),
            "v23_parent": "none",
        }
    )
    if tokenizer is not None:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(getattr(encoded, "ids", encoded))
        if row.get("token_count") is not None and int(row["token_count"]) != token_count:
            raise ValueError(f"{source_record_id} token_count does not match the pinned tokenizer")
        row["token_count"] = token_count
    else:
        token_count = int(row.get("token_count", 0) or 0)
        if token_count <= 0:
            raise ValueError(f"{source_record_id} needs token_count or --tokenizer-path")
        row["token_count"] = token_count
    if row["token_count"] <= 0:
        raise ValueError(f"{source_record_id} has no usable tokens")
    return row


def _source_receipt_from_rows(rows: Sequence[Mapping[str, Any]], *, source_paths: Sequence[Path], root: Path, acquisition_receipt: Path | None, capture_commands: Sequence[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id", row.get("source_name", "")))
        item = grouped.setdefault(
            source_id,
            {
                "source_id": source_id,
                "name": str(row.get("source_name", "")),
                "url": str(row.get("source_url", "")),
                "source_revision": str(row.get("source_revision", "")),
                "source_license": str(row.get("source_license", "")),
                "source_file_sha256": _source_file_hash(row),
                "acquisition_time_utc": str(row.get("acquisition_time_utc", "")),
                "raw_paths": set(),
                "tiers": set(),
                "records": 0,
            },
        )
        item["raw_paths"].add(str(row.get("source_file", "")))
        item["tiers"].add(str(row.get("tier", "")))
        item["records"] += 1
    normalized_sources = []
    for item in sorted(grouped.values(), key=lambda value: value["source_id"]):
        item = dict(item)
        item["raw_paths"] = sorted(item["raw_paths"])
        item["tiers"] = sorted(item["tiers"])
        normalized_sources.append(item)
    inherited: dict[str, Any] | None = None
    if acquisition_receipt is not None and acquisition_receipt.exists():
        inherited = json.loads(acquisition_receipt.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "receipt_type": "dense2moe-v23-source-receipt",
        "status": "SEALED",
        "sources": normalized_sources,
        "prepared_sources": [{"path": _stable_path(path, root=root), "sha256": sha256_file(path), "records": len(_read_jsonl(path))} for path in source_paths],
        "inherited_acquisition_receipt": None if inherited is None else {"path": _stable_path(acquisition_receipt, root=root), "sha256": sha256_file(acquisition_receipt)},
        "exact_capture_commands": list(capture_commands),
        "public_only": True,
        "licenses_required": True,
        "source_revisions_required": True,
        "file_hashes_required": True,
        "acquisition_time_required": True,
        "v22_evaluation_or_promotion_rows_used": False,
    }


def _load_sources(source_paths: Iterable[Path], *, prepared_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = [Path(path) for path in source_paths]
    if prepared_dir is not None:
        paths.extend(sorted(Path(prepared_dir).glob("*.jsonl")))
    selected: list[Path] = []
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                if child.resolve() not in seen_paths:
                    selected.append(child.resolve())
                    rows.extend(_read_jsonl(child))
                    seen_paths.add(child.resolve())
        elif path.exists():
            selected.append(path)
            rows.extend(_read_jsonl(path))
    if not rows:
        raise ValueError("no prepared V2.3 JSONL sources were supplied")
    return rows, selected


def _domain_fractions(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    domains = sorted({str(row.get("domain", "general")).strip() or "general" for row in rows if str(row.get("tier", "")) == "FIT-TRAIN"})
    if not domains:
        return {"general": 1.0}
    fraction = 1.0 / len(domains)
    result = {domain: fraction for domain in domains}
    result[domains[0]] += 1.0 - sum(result.values())
    return result


def freeze_v23(
    *,
    sources: Iterable[Path],
    output: Path,
    tokenizer_path: Path | None = None,
    v22_ledgers: Iterable[Path] = (),
    planned_tokens: int = DEFAULT_PLANNED_TOKENS,
    method_version: str = DEFAULT_METHOD_VERSION,
    threshold_fingerprint: str = "sealed-v23-grouped-public-v1",
    acquisition_receipt: Path | None = None,
    capture_commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and write the immutable V2.3 corpus artifacts."""

    output = Path(output)
    raw_rows, source_paths = _load_sources(sources)
    tokenizer, tokenizer_receipt = _load_tokenizer(tokenizer_path)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append(_normalise_row(raw, tokenizer=tokenizer, source_path=source_paths[0] if source_paths else output, root=ROOT))
    if not rows:
        raise ValueError("prepared sources contain no records")
    invalid_method = sorted({str(row.get("v23_method_version", method_version)) for row in rows if str(row.get("v23_method_version", method_version)) not in {method_version, V23_METHOD_VERSION}})
    if invalid_method:
        raise ValueError(f"records belong to another method version: {invalid_method}")

    # Exact duplicates inside one tier are harmless serialization duplicates;
    # drop only the later copy.  A cross-tier duplicate remains visible to the
    # audit and is rejected below.
    seen_by_tier: dict[str, set[str]] = {tier: set() for tier in DEVELOPMENT_TIERS}
    deduped: list[dict[str, Any]] = []
    duplicate_rows_removed: list[str] = []
    for row in sorted(rows, key=lambda item: (str(item["tier"]), str(item["id"]))):
        digest = str(row["normalized_content_sha256"])
        if digest in seen_by_tier[row["tier"]]:
            duplicate_rows_removed.append(str(row["id"]))
            continue
        seen_by_tier[row["tier"]].add(digest)
        deduped.append(row)
    rows = deduped
    tier_audit = audit_v23_tier_disjointness(rows)
    historical_paths = [Path(path) for path in v22_ledgers]
    historical_rows, historical_metadata = load_v22_rows(historical_paths)
    overlap_audit = audit_v22_overlap(rows, historical_rows=historical_rows, historical_paths=historical_paths)
    if tier_audit["status"] != "PASS":
        raise ValueError(f"V2.3 FIT-TRAIN/FIT-DEV overlap detected: {tier_audit['group_conflicts'][:3]}")
    if overlap_audit["status"] != "PASS":
        raise ValueError(f"V2.3 overlaps V2.2 identities: {overlap_audit['conflicts'][:3]}")

    tier_counts = Counter(str(row["tier"]) for row in rows)
    missing_tiers = [tier for tier in DEVELOPMENT_TIERS if tier_counts[tier] == 0]
    if missing_tiers:
        raise ValueError(f"both V2.3 development tiers are required: {missing_tiers}")
    domains = _domain_fractions(rows)
    activation_plan = build_balanced_activation_plan(
        rows,
        planned_tokens=int(planned_tokens),
        target_fractions=domains,
        task_token_cap=4_096,
        repository_token_cap=65_536,
        source_family_token_cap=max(int(planned_tokens) // 2, 1),
        trajectory_token_cap=4_096,
        eligible_splits=DEVELOPMENT_TIERS,
        seed=23,
    )
    gate_status = "PASS" if activation_plan["status"] == "READY_FOR_BALANCED_CAPTURE" and not missing_tiers else "BLOCKED"
    rows.sort(key=lambda row: (str(row["tier"]), str(row["id"])))
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    tier_ledger = {
        "schema_version": 1,
        "ledger_type": "dense2moe-v23-tier-opening",
        "status": "SEALED",
        "corpus_version": "v2.3",
        "method_version": method_version,
        "threshold_fingerprint": threshold_fingerprint,
        "manifest_sha256": manifest_hash,
        "tier_order": list(DEVELOPMENT_TIERS),
        "tiers": {
            tier: {
                "records": int(tier_counts[tier]),
                "dataset_sha256": hashlib.sha256("".join(row["id"] for row in rows if row["tier"] == tier).encode()).hexdigest(),
                "status": "FROZEN" if tier_counts[tier] else "UNATTACHED",
                "opened": False,
                "retired": False,
                "optimizer_updates": tier == "FIT-TRAIN",
            }
            for tier in DEVELOPMENT_TIERS
        },
        "evaluation_tiers": [],
        "promotion_tiers": [],
        "one_way_opening": True,
    }
    source_receipt = _source_receipt_from_rows(rows, source_paths=source_paths, root=ROOT, acquisition_receipt=acquisition_receipt, capture_commands=capture_commands)
    source_receipt["historical_v22_ledgers"] = historical_metadata
    source_receipt["historical_overlap_status"] = overlap_audit["status"]
    source_receipt["tokenizer"] = tokenizer_receipt
    source_receipt["records"] = len(rows)
    artifacts_payload: dict[str, Any] = {
        "v23-tier-ledger.json": tier_ledger,
        "v23-overlap-audit.json": {"tier_disjointness": tier_audit, "v22_overlap": overlap_audit, "historical_v22_ledgers": historical_metadata},
        "v23-source-receipt.json": source_receipt,
        "v23-activation-plan.json": activation_plan,
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {
        "v23-corpus.jsonl": {"path": "v23-corpus.jsonl", "sha256": write_immutable_text(output / "v23-corpus.jsonl", manifest_text), "format": "jsonl"}
    }
    for name, payload in artifacts_payload.items():
        artifacts[name] = {"path": name, "sha256": write_immutable_json(output / name, payload), "format": "json"}
    freeze_receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-v23-freeze-receipt",
        "status": "CORPUS_V23_FROZEN",
        "phase_gate": gate_status,
        "corpus_version": "v2.3",
        "method_version": method_version,
        "immutable": True,
        "manifest": artifacts["v23-corpus.jsonl"],
        "artifacts": artifacts,
        "tier_order": list(DEVELOPMENT_TIERS),
        "tier_counts": {tier: int(tier_counts[tier]) for tier in DEVELOPMENT_TIERS},
        "token_counts": {tier: sum(int(row["token_count"]) for row in rows if row["tier"] == tier) for tier in DEVELOPMENT_TIERS},
        "duplicate_rows_removed_within_tier": duplicate_rows_removed,
        "tier_disjointness": tier_audit,
        "v22_overlap": overlap_audit,
        "activation_plan": activation_plan,
        "tokenizer": tokenizer_receipt,
        "historical_v22_ledgers": historical_metadata,
        "exact_capture_commands": list(capture_commands),
        "evaluation_tiers_opened": [],
        "promotion_data_used": False,
    }
    receipt_path = output / "v23-freeze-receipt.json"
    artifacts[receipt_path.name] = {"path": receipt_path.name, "sha256": write_immutable_json(receipt_path, freeze_receipt), "format": "json"}
    evidence = verify_immutable_artifacts(output, artifacts)
    if evidence["status"] != "PASS":
        raise ValueError(f"V2.3 immutable artifact verification failed: {evidence['failures']}")
    return {
        "status": freeze_receipt["status"],
        "phase_gate": gate_status,
        "output": str(output),
        "receipt": artifacts[receipt_path.name],
        "artifacts": artifacts,
        "tier_counts": freeze_receipt["tier_counts"],
        "token_counts": freeze_receipt["token_counts"],
        "overlap_status": overlap_audit["status"],
        "activation_plan": {"status": activation_plan["status"], "selected_tokens": activation_plan["selected_tokens"]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=False, default=[])
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--v22-ledger", type=Path, action="append", default=None)
    parser.add_argument("--planned-tokens", type=int, default=DEFAULT_PLANNED_TOKENS)
    parser.add_argument("--method-version", default=DEFAULT_METHOD_VERSION)
    parser.add_argument("--threshold-fingerprint", default="sealed-v23-grouped-public-v1")
    parser.add_argument("--acquisition-receipt", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.source and args.prepared_dir is None:
        args.prepared_dir = args.output / "prepared"
    historical = args.v22_ledger if args.v22_ledger is not None else discover_v22_ledgers(ROOT)
    command = "python " + " ".join(json.dumps(str(item)) for item in ([Path(__file__).resolve(), *(argv or sys.argv[1:])]))
    acquisition_receipt = args.acquisition_receipt
    if acquisition_receipt is None and args.prepared_dir is not None:
        candidate = args.prepared_dir / "acquisition-receipt.json"
        if candidate.exists():
            acquisition_receipt = candidate
    source_paths = list(args.source)
    if args.prepared_dir is not None:
        source_paths.append(args.prepared_dir)
    result = freeze_v23(
        sources=source_paths,
        output=args.output,
        tokenizer_path=args.tokenizer_path,
        v22_ledgers=historical,
        planned_tokens=args.planned_tokens,
        method_version=args.method_version,
        threshold_fingerprint=args.threshold_fingerprint,
        acquisition_receipt=acquisition_receipt,
        capture_commands=(command,),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["phase_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
