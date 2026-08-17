#!/usr/bin/env python3
"""Freeze the sealed Corpus V2.2 anti-overfitting protocol.

The command consumes already acquired JSONL records.  It never downloads or
silently invents evaluation data.  Every row must carry an explicit tier,
pinned source revision/license, and source-record identity; grouped split and
near-duplicate audits run before any immutable artifact is written.
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
    V22_DEVELOPMENT_TIERS,
    V22_EVALUATION_TIERS,
    V22_EXTERNAL_TIERS,
    V22_TIER_ORDER,
    audit_agent_task_diversity,
    audit_corpus_tier_disjointness,
    build_balanced_activation_plan,
    is_benchmark_derived,
    sha256_file,
    stable_corpus_record_id,
    validate_pinned_source_records,
    verify_immutable_artifacts,
    write_immutable_json,
    write_immutable_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/public_v22"
DEFAULT_METHOD_VERSION = "qwen38-dense2moe-v22-method-1"


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


def _stable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalise_row(raw: Mapping[str, Any], *, source_path: Path, tokenizer: Any | None = None) -> dict[str, Any] | None:
    text = str(raw.get("text", raw.get("content", "")))
    if not text.strip():
        return None
    row = dict(raw)
    tier = str(row.get("tier", row.get("split", ""))).strip()
    if tier in {"train", "FIT", "fit"}:
        tier = "FIT-TRAIN"
    elif tier in {"dev", "validation", "FIT-DEV"}:
        tier = "FIT-DEV"
    row["tier"] = tier
    row["split"] = tier
    row["text"] = text
    row["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    row["normalized_content_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    row["id"] = stable_corpus_record_id(row)
    if row.get("token_count") is None:
        if tokenizer is None:
            raise ValueError(f"{row['id']} has no token_count and no tokenizer was supplied")
        encoded = tokenizer.encode(text, add_special_tokens=False)
        row["token_count"] = len(getattr(encoded, "ids", encoded))
    row["token_count"] = int(row.get("token_count", 0) or 0)
    if row["token_count"] <= 0:
        return None
    row.setdefault("source_file", _stable_path(source_path))
    row.setdefault("source_name", str(row.get("source", source_path.stem)))
    row.setdefault("source_record_id", row.get("record_id", row["id"]))
    row.setdefault("source_family", "unknown")
    row.setdefault("domain", "general")
    row.setdefault("task_family", "general")
    return row


def _load_tokenizer(path: Path | None) -> Any | None:
    if path is None:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tokenizers is required when --tokenizer-path is supplied") from exc
    return Tokenizer.from_file(str(path))


def freeze_v22(
    *,
    sources: Iterable[Path],
    output: Path,
    tokenizer_path: Path | None = None,
    require_agent_tasks: int = 96,
    planned_tokens: int = 750_000,
    method_version: str = DEFAULT_METHOD_VERSION,
    threshold_fingerprint: str = "sealed-qwen38-promotion-v1",
) -> dict[str, Any]:
    source_paths = [Path(path) for path in sources]
    if not source_paths:
        raise ValueError("at least one acquired JSONL source is required")
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    tokenizer = _load_tokenizer(tokenizer_path)
    rows: list[dict[str, Any]] = []
    for source_path in source_paths:
        for raw in _read_jsonl(source_path):
            row = _normalise_row(raw, source_path=source_path, tokenizer=tokenizer)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError("acquired sources contain no usable records")
    invalid_tiers = sorted({str(row.get("tier", "")) for row in rows if str(row.get("tier", "")) not in V22_TIER_ORDER})
    if invalid_tiers:
        raise ValueError(f"every record needs an explicit V2.2 tier; invalid/missing tiers: {invalid_tiers[:5]}")

    # Remove only exact duplicates within one tier.  Keeping a duplicate that
    # crosses tiers is intentional: the leakage audit must see and reject it.
    seen_by_tier: dict[str, set[str]] = {tier: set() for tier in V22_TIER_ORDER}
    deduped: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    for row in rows:
        digest = str(row["normalized_content_sha256"])
        tier = str(row["tier"])
        if digest in seen_by_tier[tier]:
            duplicate_ids.append(str(row["id"]))
            continue
        seen_by_tier[tier].add(digest)
        deduped.append(row)
    rows = sorted(deduped, key=lambda item: (str(item["tier"]), str(item["id"])))
    for index, row in enumerate(rows):
        row["source_record_index"] = index
        row["v22_method_version"] = method_version
        row["v22_parent"] = "none"
        if is_benchmark_derived(row) and row["tier"] != "BENCHMARK-CANARY-EXCLUDED":
            raise ValueError(f"benchmark-derived record {row['id']} must be explicitly quarantined")

    pinned = validate_pinned_source_records(rows)
    if pinned["status"] != "PASS":
        raise ValueError(f"pinned source provenance failed: {pinned['failures'][:3]}")
    overlap = audit_corpus_tier_disjointness(rows)
    if overlap["status"] != "PASS":
        raise ValueError("cross-tier grouping or duplicate leakage detected; inspect the audit before retrying")
    diversity = audit_agent_task_diversity(rows, minimum_tasks=require_agent_tasks, eligible_splits=V22_DEVELOPMENT_TIERS + V22_EVALUATION_TIERS)
    plan = build_balanced_activation_plan(rows, planned_tokens=planned_tokens, eligible_splits=V22_DEVELOPMENT_TIERS)
    tier_counts = Counter(str(row["tier"]) for row in rows)
    missing_external = [tier for tier in V22_EXTERNAL_TIERS if tier_counts[tier] == 0]
    missing_internal = [tier for tier in ("GATE-A", "SHADOW-B", "SHADOW-C") if tier_counts[tier] == 0]
    gate_status = "PASS" if not missing_external and not missing_internal and diversity["status"] == "PASS" and plan["status"] == "READY_FOR_BALANCED_CAPTURE" else "BLOCKED"
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    splits = {
        "schema_version": 2,
        "manifest_type": "dense2moe-corpus-v2.2-splits",
        "status": "FROZEN",
        "tier_order": list(V22_TIER_ORDER),
        "records": {tier: [row["id"] for row in rows if row["tier"] == tier] for tier in V22_TIER_ORDER},
        "grouping": "task/repository/trajectory/document/source-lineage; no row-random split",
    }
    tier_ledger = {
        "schema_version": 1,
        "ledger_type": "dense2moe-corpus-v2.2-tier-opening",
        "status": "SEALED",
        "method_version": method_version,
        "threshold_fingerprint": threshold_fingerprint,
        "manifest_sha256": manifest_hash,
        "tiers": {
            tier: {
                "records": tier_counts[tier],
                "dataset_sha256": hashlib.sha256("".join(row["id"] for row in rows if row["tier"] == tier).encode()).hexdigest(),
                "opened": False,
                "retired": False,
                "optimizer_updates": tier in V22_DEVELOPMENT_TIERS,
            }
            for tier in V22_TIER_ORDER
        },
        "one_way_opening": True,
    }
    source_receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-corpus-v2.2-source-receipt",
        "sources": [{"path": _stable_path(path), "sha256": sha256_file(path), "records": len(_read_jsonl(path))} for path in source_paths],
        "pinned_revisions_required": True,
        "licenses_required": True,
        "visible_transcripts_only": True,
        "private_reasoning_included": False,
    }
    artifacts_payload: dict[str, Any] = {
        "corpus-v2.2-splits.json": splits,
        "corpus-v2.2-tier-ledger.json": tier_ledger,
        "corpus-v2.2-source-receipt.json": source_receipt,
        "corpus-v2.2-overlap-audit.json": overlap,
        "corpus-v2.2-agent-diversity.json": diversity,
        "corpus-v2.2-activation-plan.json": plan,
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {
        "corpus-v2.2.jsonl": {"path": "corpus-v2.2.jsonl", "sha256": write_immutable_text(output / "corpus-v2.2.jsonl", manifest_text), "format": "jsonl"}
    }
    for name, payload in artifacts_payload.items():
        artifacts[name] = {"path": name, "sha256": write_immutable_json(output / name, payload), "format": "json"}
    receipt = {
        "schema_version": 2,
        "receipt_type": "dense2moe-corpus-v2.2-freeze-receipt",
        "status": "CORPUS_V22_FROZEN",
        "phase_gate": gate_status,
        "immutable": True,
        "method_version": method_version,
        "threshold_fingerprint": threshold_fingerprint,
        "manifest": artifacts["corpus-v2.2.jsonl"],
        "artifacts": artifacts,
        "tier_order": list(V22_TIER_ORDER),
        "tier_counts": dict(sorted(tier_counts.items())),
        "duplicate_rows_removed_within_tier": sorted(duplicate_ids),
        "pinned_provenance": pinned,
        "overlap_audit": overlap,
        "agent_task_diversity": diversity,
        "activation_plan": plan,
        "internal_promotion_tiers": ["GATE-A", "SHADOW-B", "SHADOW-C"],
        "external_generalization_tiers": list(V22_EXTERNAL_TIERS),
        "historical_holdout": "CLOSED",
        "promotion_policy": "all untouched corpora and major domain slices must be green; no G1/G2 tuning",
    }
    receipt_path = output / "corpus-v2.2-receipt.json"
    artifacts["corpus-v2.2-receipt.json"] = {"path": receipt_path.name, "sha256": write_immutable_json(receipt_path, receipt), "format": "json"}
    evidence = verify_immutable_artifacts(output, artifacts)
    if evidence["status"] != "PASS":
        raise ValueError(f"V2.2 artifact verification failed: {evidence['failures']}")
    return {"status": receipt["status"], "phase_gate": gate_status, "receipt": artifacts["corpus-v2.2-receipt.json"], "tier_counts": dict(sorted(tier_counts.items())), "activation_plan": {"status": plan["status"], "selected_tokens": plan["selected_tokens"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True, help="acquired JSONL source (repeatable)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--require-agent-tasks", type=int, default=96)
    parser.add_argument("--planned-tokens", type=int, default=750_000)
    parser.add_argument("--method-version", default=DEFAULT_METHOD_VERSION)
    parser.add_argument("--threshold-fingerprint", default="sealed-qwen38-promotion-v1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = freeze_v22(
        sources=args.source,
        output=args.output,
        tokenizer_path=args.tokenizer_path,
        require_agent_tasks=args.require_agent_tasks,
        planned_tokens=args.planned_tokens,
        method_version=args.method_version,
        threshold_fingerprint=args.threshold_fingerprint,
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["phase_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

