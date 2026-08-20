#!/usr/bin/env python3
"""Derive and freeze Corpus V2.1 without mutating the frozen V2 artifacts.

The command is intentionally local and deterministic.  It consumes the V2
manifest plus one or more optional acquisition JSONL files, quarantines every
benchmark-derived row before assigning optimization roles, and emits
content-addressed audit receipts suitable for teacher-capture gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    V21_OPTIMIZATION_SPLITS,
    V21_PROMOTION_SPLITS,
    V21_QUARANTINE_SPLIT,
    V21_SPLITS,
    audit_agent_task_diversity,
    audit_split_disjointness,
    audit_tokenizer_records,
    build_balanced_activation_plan,
    is_benchmark_derived,
    quarantine_benchmark_records,
    sha256_file,
    stable_corpus_record_id,
    verify_immutable_artifacts,
    write_immutable_json,
    write_immutable_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_MANIFEST = ROOT / "data/public_v2/corpus.jsonl"
DEFAULT_V2_SPLITS = ROOT / "data/public_v2/corpus-v2-splits.json"
DEFAULT_SOURCE = ROOT / "data/public_v2/corpus-v2-source.jsonl"
DEFAULT_OUTPUT = ROOT / "data/public_v21"
DEFAULT_TOKENIZER = ROOT / "runs/20260815-030931-windows/source/tokenizer.json"
MANIFEST_NAME = "corpus-v2.1.jsonl"
SPLITS_NAME = "corpus-v2.1-splits.json"
RECEIPT_NAME = "corpus-v2.1-receipt.json"
SOURCE_RECEIPT_NAME = "corpus-v2.1-source-receipt.json"
BENCHMARK_RECEIPT_NAME = "corpus-v2.1-benchmark-exclusion.json"
TOKENIZER_RECEIPT_NAME = "corpus-v2.1-tokenizer-audit.json"
PLAN_NAME = "corpus-v2.1-activation-plan.json"


def _stable_path_label(path: Path) -> str:
    """Represent repository-local inputs without host-specific prefixes."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


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


def _hash_jsonl(rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity_group(row: Mapping[str, Any]) -> str:
    for key in ("split_group", "repository_id", "repository", "repo", "repo_name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value if key == "split_group" else f"repo:{value}"
    for key in ("task_id", "issue_id", "document_id", "source_record_id", "id"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"document:{value}"
    return f"content:{row.get('normalized_content_sha256', row.get('content_sha256', ''))}"


def _is_canary(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("split", "")) == "PRESERVATION-CANARY"
        or str(row.get("domain", "")).casefold() in {"general-preservation", "long-context"}
        or str(row.get("source_family", "")).casefold() in {"preservation-canary", "general-public-domain"}
    )


def _assign_new_roles(rows: list[dict[str, Any]]) -> None:
    """Assign source rows deterministically while keeping groups together."""

    role_for_group: dict[str, str] = {}
    for row in rows:
        split = str(row.get("split", "")).strip()
        group = _identity_group(row)
        if split in V21_SPLITS and not is_benchmark_derived(row):
            role_for_group.setdefault(group, split)
    for row in rows:
        if is_benchmark_derived(row):
            continue
        if _is_canary(row):
            row["split"] = "PRESERVATION-CANARY"
            continue
        group = _identity_group(row)
        existing = role_for_group.get(group)
        if existing:
            row["split"] = existing
            continue
        digest = int(hashlib.sha256(f"v2.1:{group}".encode()).hexdigest()[:8], 16) % 100
        if digest < 56:
            split = "FIT-TRAIN"
        elif digest < 68:
            split = "FIT-DEV"
        elif digest < 81:
            split = "GATE-A"
        elif digest < 91:
            split = "SHADOW-B"
        else:
            split = "SHADOW-C"
        role_for_group[group] = split
        row["split"] = split


def _normalise_row(row: Mapping[str, Any], *, tokenizer: Any) -> dict[str, Any] | None:
    text = str(row.get("text", row.get("content", "")))
    if not text.strip():
        return None
    item = dict(row)
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    item["text"] = text
    item["content_sha256"] = raw_hash
    item["normalized_content_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    item["id"] = stable_corpus_record_id(item)
    if item.get("token_count") is None:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        item["token_count"] = len(getattr(encoded, "ids", encoded))
    item["token_count"] = int(item.get("token_count", 0) or 0)
    if item["token_count"] <= 0:
        return None
    item.setdefault("source_name", str(item.get("source", "unknown")))
    item.setdefault("source_revision", str(item.get("revision", "unknown")))
    item.setdefault("source_license", str(item.get("license", "unknown")))
    item.setdefault("source_family", "unknown")
    item.setdefault("domain", "general")
    item.setdefault("task_family", "general")
    item.setdefault("source_record_id", item.get("record_id", item["id"]))
    return item


def _load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tokenizers is required to freeze Corpus V2.1") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))


def _verify_parent(v2_manifest: Path, v2_splits: Path) -> dict[str, Any]:
    receipt = v2_manifest.with_name("corpus-v2-receipt.json")
    expected_manifest = None
    expected_splits = None
    if receipt.exists():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        expected_manifest = str(payload.get("manifest", {}).get("sha256", "")) or None
        expected_splits = str(payload.get("split_identity", {}).get("sha256", "")) or None
    actual_manifest = sha256_file(v2_manifest)
    actual_splits = sha256_file(v2_splits)
    if expected_manifest and expected_manifest != actual_manifest:
        raise ValueError("frozen V2 manifest hash does not match its receipt")
    if expected_splits and expected_splits != actual_splits:
        raise ValueError("frozen V2 split hash does not match its receipt")
    return {
        "manifest": {"path": _stable_path_label(v2_manifest), "sha256": actual_manifest, "records": sum(1 for _ in v2_manifest.open("rb"))},
        "splits": {"path": _stable_path_label(v2_splits), "sha256": actual_splits},
        "receipt": _stable_path_label(receipt) if receipt.exists() else None,
        "receipt_verified": bool(receipt.exists() and expected_manifest and expected_splits),
    }


def freeze_v21(
    *,
    source: Path,
    v2_manifest: Path,
    v2_splits: Path,
    output: Path,
    tokenizer_path: Path,
    agent_sources: Iterable[Path] = (),
    require_agent_tasks: int = 96,
    planned_tokens: int = 750_000,
) -> dict[str, Any]:
    parent = _verify_parent(v2_manifest, v2_splits)
    tokenizer = _load_tokenizer(tokenizer_path)
    parent_rows = _read_jsonl(v2_manifest)
    parent_ids = {stable_corpus_record_id(item) for item in parent_rows}
    acquisition_paths = [source, *agent_sources]
    acquisition_rows: list[dict[str, Any]] = []
    for path in acquisition_paths:
        if path.exists():
            acquisition_rows.extend(_read_jsonl(path))
    rows: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for raw in [*parent_rows, *acquisition_rows]:
        item = _normalise_row(raw, tokenizer=tokenizer)
        if item is None or item["normalized_content_sha256"] in seen_content:
            continue
        seen_content.add(item["normalized_content_sha256"])
        rows.append(item)
    # Existing V2 role identities win; newly acquired rows are assigned only
    # after all parent rows have established their deterministic group map.
    _assign_new_roles(rows)
    rows, benchmark_audit = quarantine_benchmark_records(rows)
    # Canonical output locators point at the immutable V2.1 JSONL itself.
    rows.sort(key=lambda row: (str(row.get("split", "")), stable_corpus_record_id(row)))
    for index, row in enumerate(rows):
        row["id"] = stable_corpus_record_id(row)
        row["source_file"] = MANIFEST_NAME
        row["source_record_index"] = index
        row["v21_parent_v2"] = row["id"] in parent_ids
    overlap = audit_split_disjointness(rows)
    diversity = audit_agent_task_diversity(rows, minimum_tasks=require_agent_tasks)
    tokenizer_audit = audit_tokenizer_records(rows, tokenizer=tokenizer, tokenizer_path=tokenizer_path, tokenizer_revision="pinned-local-snapshot")
    if tokenizer_audit["status"] != "PASS":
        raise ValueError(f"V2.1 tokenizer audit failed: {tokenizer_audit['mismatches'][:3]}")
    plan = build_balanced_activation_plan(rows, planned_tokens=planned_tokens)
    manifest_text, _manifest_hash = _hash_jsonl(rows)
    splits_payload = {
        "schema_version": 1,
        "manifest_type": "dense2moe-corpus-v2.1-splits",
        "status": "FROZEN",
        "split_order": [*V21_SPLITS, V21_QUARANTINE_SPLIT],
        "records": {
            split: [row["id"] for row in rows if str(row.get("split")) == split]
            for split in [*V21_SPLITS, V21_QUARANTINE_SPLIT]
        },
    }
    source_receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-corpus-v2.1-source-receipt",
        "parent_v2": parent,
        "acquisition_sources": [
            {"path": _stable_path_label(path), "sha256": sha256_file(path), "records": sum(1 for _ in path.open("rb"))}
            for path in acquisition_paths
            if path.exists()
        ],
        "records": len(rows),
        "provenance_retained": True,
    }
    benchmark_receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-corpus-v2.1-benchmark-exclusion",
        "status": benchmark_audit["status"],
        "quarantine_split": V21_QUARANTINE_SPLIT,
        "policy": {
            "excluded_from": [*V21_OPTIMIZATION_SPLITS, *V21_PROMOTION_SPLITS],
            "allowed_for": [V21_QUARANTINE_SPLIT],
            "gradients": False,
            "checkpoint_selection": False,
            "shadow_promotion": False,
        },
        "audit": benchmark_audit,
    }
    artifact_payloads: dict[str, Any] = {
        SPLITS_NAME: splits_payload,
        SOURCE_RECEIPT_NAME: source_receipt,
        BENCHMARK_RECEIPT_NAME: benchmark_receipt,
        TOKENIZER_RECEIPT_NAME: tokenizer_audit,
        PLAN_NAME: plan,
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {
        MANIFEST_NAME: {"path": MANIFEST_NAME, "sha256": write_immutable_text(output / MANIFEST_NAME, manifest_text), "format": "jsonl"}
    }
    for name, payload in artifact_payloads.items():
        artifacts[name] = {"path": name, "sha256": write_immutable_json(output / name, payload), "format": "json"}
    receipt_payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "dense2moe-corpus-v2.1-freeze-receipt",
        "status": "CORPUS_V21_FROZEN",
        "corpus_version": "v2.1",
        "immutable": True,
        "parent_v2": parent,
        "manifest": artifacts[MANIFEST_NAME],
        "artifacts": artifacts,
        "benchmark_exclusion": benchmark_receipt,
        "agent_task_diversity": diversity,
        "overlap_audit": overlap,
        "tokenizer_audit": tokenizer_audit,
        "activation_sampling_plan": plan,
        "phase_gate": {
            "status": "PASS" if diversity["status"] == "PASS" and overlap["status"] == "PASS" and plan["status"] == "READY_FOR_BALANCED_CAPTURE" else "BLOCKED",
            "agent_task_acquisition": diversity["status"],
            "overlap": overlap["status"],
            "tokenizer": tokenizer_audit["status"],
            "balanced_capture": plan["status"],
        },
        "official_holdout": "CLOSED",
    }
    receipt_path = output / RECEIPT_NAME
    receipt_hash = write_immutable_json(receipt_path, receipt_payload)
    evidence = verify_immutable_artifacts(output, artifacts)
    if evidence["status"] != "PASS":
        raise ValueError(f"V2.1 artifact verification failed: {evidence['failures']}")
    return {
        "status": receipt_payload["status"],
        "phase_gate": receipt_payload["phase_gate"],
        "manifest": artifacts[MANIFEST_NAME],
        "receipt": {"path": RECEIPT_NAME, "sha256": receipt_hash},
        "artifacts": artifacts,
        "agent_task_diversity": diversity,
        "overlap_audit": overlap,
        "tokenizer_audit": {"status": tokenizer_audit["status"], "records_checked": tokenizer_audit["records_checked"]},
        "activation_sampling_plan": {"status": plan["status"], "selected_tokens": plan["selected_tokens"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--agent-source", type=Path, action="append", default=[])
    parser.add_argument("--v2-manifest", type=Path, default=DEFAULT_V2_MANIFEST)
    parser.add_argument("--v2-splits", type=Path, default=DEFAULT_V2_SPLITS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-agent-tasks", type=int, default=96)
    parser.add_argument("--planned-tokens", type=int, default=750_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = freeze_v21(
        source=args.source,
        agent_sources=args.agent_source,
        v2_manifest=args.v2_manifest,
        v2_splits=args.v2_splits,
        tokenizer_path=args.tokenizer_path,
        output=args.output,
        require_agent_tasks=args.require_agent_tasks,
        planned_tokens=args.planned_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
