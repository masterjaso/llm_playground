#!/usr/bin/env python3
"""Attach one untouched external corpus component to a locked V2.2 freeze.

The parent receipt and every external component are immutable.  This command
never mutates the development/internal manifest; it writes a new aggregate
receipt that references the parent and the newly attached tier.  A method lock
is mandatory, so external data cannot influence recipe selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    V22_EXTERNAL_TIERS,
    V22_TIER_ORDER,
    audit_agent_task_diversity,
    audit_corpus_tier_disjointness,
    is_benchmark_derived,
    sha256_file,
    validate_pinned_source_records,
    verify_immutable_artifacts,
    write_immutable_json,
    write_immutable_text,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}:{index + 1} is not a JSON object")
        rows.append(dict(value))
    return rows


def _resolve_artifact(receipt_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return receipt_path.parent / candidate


def _load_parent(receipt_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("receipt_type") not in {"dense2moe-corpus-v2.2-freeze-receipt", "dense2moe-corpus-v2.2-external-attachment-receipt"}:
        raise ValueError("parent must be a Corpus V2.2 freeze or external attachment receipt")
    if payload.get("phase_gate") != "PASS":
        raise ValueError("parent corpus receipt is not phase-green")
    manifest_ref = payload.get("manifest", {}).get("path")
    manifest_hash = str(payload.get("manifest", {}).get("sha256", ""))
    manifest_path = _resolve_artifact(receipt_path, manifest_ref)
    if not manifest_path.exists() or not manifest_hash or sha256_file(manifest_path) != manifest_hash:
        raise ValueError("parent manifest is missing or hash-invalid")
    rows = _read_jsonl(manifest_path)
    return payload, rows, manifest_hash


def _validate_lock(lock_path: Path, *, method_version: str, threshold_fingerprint: str) -> dict[str, Any]:
    if not lock_path.exists():
        raise ValueError("method lock does not exist")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    if status not in {"METHOD_LOCKED", "METHOD_LOCKS_FROZEN", "FROZEN", "GREEN"}:
        raise ValueError("method lock is not frozen")
    if str(payload.get("method_version", method_version)) != method_version:
        raise ValueError("method lock method_version mismatch")
    if str(payload.get("threshold_fingerprint", threshold_fingerprint)) != threshold_fingerprint:
        raise ValueError("method lock threshold fingerprint mismatch")
    if payload.get("external_tuning_forbidden") is not True:
        raise ValueError("method lock must forbid external tuning")
    return payload


def attach_external(
    *,
    parent_receipt: Path,
    sources: Iterable[Path],
    output: Path,
    tier: str,
    method_lock: Path,
    method_version: str,
    threshold_fingerprint: str = "sealed-qwen38-promotion-v1",
) -> dict[str, Any]:
    if tier not in V22_EXTERNAL_TIERS:
        raise ValueError(f"tier must be one of {V22_EXTERNAL_TIERS}")
    source_paths = [Path(path) for path in sources]
    if not source_paths or any(not path.exists() for path in source_paths):
        raise FileNotFoundError("all external source files must exist")
    _validate_lock(method_lock, method_version=method_version, threshold_fingerprint=threshold_fingerprint)
    parent, parent_rows, parent_manifest_hash = _load_parent(parent_receipt)
    parent_tiers = parent.get("tier_counts", {})
    if int(parent_tiers.get(tier, 0) or 0) > 0:
        raise ValueError(f"external tier {tier} already exists in the parent receipt")
    rows: list[dict[str, Any]] = []
    for source in source_paths:
        for row in _read_jsonl(source):
            item = dict(row)
            if str(item.get("tier", item.get("split", ""))) != tier:
                raise ValueError(f"every attached row must explicitly belong to {tier}")
            item["tier"] = tier
            item["split"] = tier
            rows.append(item)
    if not rows:
        raise ValueError("external sources contain no rows")
    combined = parent_rows + rows
    if any(is_benchmark_derived(row) for row in rows):
        raise ValueError("benchmark-derived external material is not eligible for G1/G2")
    pinned = validate_pinned_source_records(combined)
    if pinned["status"] != "PASS":
        raise ValueError(f"pinned provenance failed: {pinned['failures'][:3]}")
    overlap = audit_corpus_tier_disjointness(combined)
    if overlap["status"] != "PASS":
        raise ValueError("external rows overlap the parent corpus or each other")
    diversity = audit_agent_task_diversity(combined, minimum_tasks=0, eligible_splits=V22_TIER_ORDER)
    tier_ids = [str(row.get("id", "")) for row in rows]
    if not all(tier_ids) or len(set(tier_ids)) != len(tier_ids):
        raise ValueError("external row identities must be present and unique")
    manifest_rows = sorted(combined, key=lambda row: (str(row.get("tier", "")), str(row.get("id", ""))))
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in manifest_rows)
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    tier_counts = Counter(str(row.get("tier", "")) for row in manifest_rows)
    artifacts: dict[str, dict[str, Any]] = {
        "corpus-v2.2.jsonl": {"path": "corpus-v2.2.jsonl", "sha256": write_immutable_text(output / "corpus-v2.2.jsonl", manifest_text), "format": "jsonl"},
        "parent-receipt.json": {"path": "parent-receipt.json", "sha256": write_immutable_json(output / "parent-receipt.json", parent), "format": "json"},
        "external-source-receipt.json": {
            "path": "external-source-receipt.json",
            "sha256": write_immutable_json(
                output / "external-source-receipt.json",
                {"schema_version": 1, "tier": tier, "sources": [{"path": str(path), "sha256": sha256_file(path), "records": len(_read_jsonl(path))} for path in source_paths]},
            ),
            "format": "json",
        },
        "overlap-audit.json": {"path": "overlap-audit.json", "sha256": write_immutable_json(output / "overlap-audit.json", overlap), "format": "json"},
        "agent-diversity.json": {"path": "agent-diversity.json", "sha256": write_immutable_json(output / "agent-diversity.json", diversity), "format": "json"},
    }
    receipt = {
        "schema_version": 2,
        "receipt_type": "dense2moe-corpus-v2.2-external-attachment-receipt",
        "status": "CORPUS_V22_EXTERNAL_ATTACHED",
        "phase_gate": "PASS",
        "immutable": True,
        "component": tier,
        "method_version": method_version,
        "threshold_fingerprint": threshold_fingerprint,
        "parent_receipt": {"path": str(parent_receipt), "sha256": sha256_file(parent_receipt)},
        "parent_manifest_sha256": parent_manifest_hash,
        "manifest": artifacts["corpus-v2.2.jsonl"],
        "artifacts": artifacts,
        "tier_counts": dict(sorted(tier_counts.items())),
        "attached_tier": tier,
        "external_data_untouched": True,
        "external_tuning_forbidden": True,
        "historical_holdout": "CLOSED",
    }
    receipt_ref = {"path": "corpus-v2.2-receipt.json", "sha256": write_immutable_json(output / "corpus-v2.2-receipt.json", receipt), "format": "json"}
    artifacts["corpus-v2.2-receipt.json"] = receipt_ref
    evidence = verify_immutable_artifacts(output, artifacts)
    if evidence["status"] != "PASS":
        raise ValueError(f"external attachment verification failed: {evidence['failures']}")
    return {"status": receipt["status"], "phase_gate": receipt["phase_gate"], "tier": tier, "receipt": receipt_ref, "manifest_sha256": manifest_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=V22_EXTERNAL_TIERS, required=True)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--method-version", required=True)
    parser.add_argument("--threshold-fingerprint", default="sealed-qwen38-promotion-v1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = attach_external(
        parent_receipt=args.parent_receipt,
        sources=args.source,
        output=args.output,
        tier=args.tier,
        method_lock=args.method_lock,
        method_version=args.method_version,
        threshold_fingerprint=args.threshold_fingerprint,
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
