"""Deterministic preparation of the selector-method proof subset.

The method-proof subset is deliberately derived from the immutable Corpus V2.1
manifest.  It is a control-plane artifact: it does not tokenize, download, or
run a model, and it never rewrites the source manifest.  Only FIT-TRAIN rows
with retained provenance are eligible; benchmark and preservation/evaluation
rows are excluded before selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .data import is_benchmark_derived, sha256_file, stable_corpus_record_id

METHOD_PROOF_STATUS = "METHOD_PROOF_READY"
METHOD_PROOF_DATA_BLOCKED = "METHOD_PROOF_DATA_BLOCKED"
# Compatibility alias for callers that used the shorter pre-gate name.
METHOD_PROOF_BLOCKED = METHOD_PROOF_DATA_BLOCKED
METHOD_PROOF_SPLIT = "FIT-TRAIN"
METHOD_PROOF_EVAL_SPLITS = ("FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C", "PRESERVATION-CANARY")
METHOD_PROOF_DEFAULT_MIN_TOKENS = 32_768


class MethodProofBlocked(ValueError):
    """Raised when the immutable corpus cannot satisfy the method-proof gate."""

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        self.details = dict(details or {})
        super().__init__(reason)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "status": METHOD_PROOF_DATA_BLOCKED,
            "reason": self.reason,
            "details": self.details,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _domain_bucket(row: Mapping[str, Any]) -> str:
    domain = str(row.get("domain", "")).casefold().replace("_", "-")
    if domain == "code" or domain.startswith("code/") or domain.endswith("/code"):
        return "code"
    if "agentic" in domain or "software-engineering" in domain or domain == "structured":
        return "technical"
    return "other"


def _has_retained_provenance(row: Mapping[str, Any]) -> bool:
    return all(
        str(row.get(key, "")).strip().casefold() not in {"", "unknown", "none", "null"}
        for key in ("source_record_id", "source_family", "source_name", "source_revision")
    )


def _is_clean_fit_train(row: Mapping[str, Any]) -> bool:
    if str(row.get("split", "")).strip() != METHOD_PROOF_SPLIT:
        return False
    if bool(row.get("benchmark_quarantine")):
        return False
    if row.get("benchmark_quarantine_reason"):
        return False
    if str(row.get("split", "")).strip() == "BENCHMARK-CANARY-EXCLUDED":
        return False
    if is_benchmark_derived(row):
        return False
    # A method-proof row must remain traceable to the immutable source.  The
    # source record, family, revision, and name are all needed to make a
    # later receipt audit meaningful; a text-only fixture is not evidence.
    if not _has_retained_provenance(row):
        return False
    if not str(row.get("text", row.get("content", ""))).strip():
        return False
    try:
        return int(row.get("token_count", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _row_key(row: Mapping[str, Any]) -> str:
    return stable_corpus_record_id(row)


def _split_identity_values(row: Mapping[str, Any]) -> set[str]:
    """Return repository/task/document identities used for split closure."""

    values: set[str] = set()
    for key in ("split_group", "repository_id", "repository", "repo", "repo_name"):
        value = str(row.get(key, "")).strip().casefold()
        if value:
            values.add(f"repository:{value}")
    for key in ("task_id", "issue_id", "instance_id", "task", "issue"):
        value = str(row.get(key, "")).strip().casefold()
        if value:
            values.add(f"task:{value}")
    for key in ("document_id", "source_document_id", "source_record_id"):
        value = str(row.get(key, "")).strip().casefold()
        if value:
            values.add(f"document:{value}")
    return values


def select_method_proof_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_tokens: int = METHOD_PROOF_DEFAULT_MIN_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stable, clean FIT-TRAIN subset and return selection facts."""

    if min_tokens <= 0:
        raise ValueError("min_tokens must be positive")
    all_rows = [dict(row) for row in rows]
    evaluation_identities: set[str] = set()
    evaluation_content_hashes: set[str] = set()
    for raw in all_rows:
        if str(raw.get("split", "")).strip() in METHOD_PROOF_EVAL_SPLITS or str(raw.get("split", "")).strip() == "BENCHMARK-CANARY-EXCLUDED":
            evaluation_identities.update(_split_identity_values(raw))
            content_hash = str(raw.get("normalized_content_sha256", raw.get("content_sha256", ""))).strip()
            if content_hash:
                evaluation_content_hashes.add(content_hash)
    eligible: list[dict[str, Any]] = []
    excluded_benchmark = 0
    excluded_non_fit = 0
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for raw in all_rows:
        if not _is_clean_fit_train(raw):
            if is_benchmark_derived(raw) or bool(raw.get("benchmark_quarantine")) or raw.get("benchmark_quarantine_reason"):
                excluded_benchmark += 1
            else:
                excluded_non_fit += 1
            continue
        row = dict(raw)
        row["id"] = _row_key(row)
        content_hash = str(row.get("normalized_content_sha256", row.get("content_sha256", ""))).strip()
        overlap = _split_identity_values(row) & evaluation_identities
        if overlap or (content_hash and content_hash in evaluation_content_hashes):
            raise MethodProofBlocked(
                "METHOD_PROOF_DATA_OVERLAP_BLOCKED",
                details={"row_id": row["id"], "overlap_identities": sorted(overlap), "content_hash_overlap": bool(content_hash and content_hash in evaluation_content_hashes)},
            )
        if row["id"] in seen_ids or (content_hash and content_hash in seen_content):
            continue
        seen_ids.add(row["id"])
        if content_hash:
            seen_content.add(content_hash)
        row["method_proof_split"] = METHOD_PROOF_SPLIT
        row["method_proof_source_id"] = row["id"]
        eligible.append(row)

    eligible.sort(key=lambda row: (_row_key(row), str(row.get("source_record_id", ""))))
    by_bucket: dict[str, list[dict[str, Any]]] = {"code": [], "technical": [], "other": []}
    for row in eligible:
        by_bucket[_domain_bucket(row)].append(row)
    missing = [bucket for bucket in ("code", "technical") if not by_bucket[bucket]]
    if missing:
        raise MethodProofBlocked(
            "METHOD_PROOF_DIVERSITY_BLOCKED",
            details={"missing_buckets": missing, "eligible_rows": len(eligible)},
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_bucket_tokens: Counter[str] = Counter()
    bucket_targets = {bucket: max(1, int(min_tokens * 0.25)) for bucket in ("code", "technical")}
    for bucket in ("code", "technical"):
        for row in by_bucket[bucket]:
            if selected_bucket_tokens[bucket] >= bucket_targets[bucket]:
                break
            selected.append(row)
            selected_ids.add(row["id"])
            selected_bucket_tokens[bucket] += int(row.get("token_count", 0) or 0)
        if selected_bucket_tokens[bucket] < bucket_targets[bucket]:
            raise MethodProofBlocked(
                "METHOD_PROOF_DIVERSITY_BLOCKED",
                details={
                    "bucket": bucket,
                    "required_bucket_tokens": bucket_targets[bucket],
                    "available_bucket_tokens": selected_bucket_tokens[bucket],
                },
            )
    selected_tokens = sum(int(row.get("token_count", 0) or 0) for row in selected)
    for row in eligible:
        if selected_tokens >= min_tokens:
            break
        if row["id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["id"])
        selected_tokens += int(row.get("token_count", 0) or 0)
    selected.sort(key=lambda row: (_row_key(row), str(row.get("source_record_id", ""))))
    if selected_tokens < min_tokens:
        raise MethodProofBlocked(
            "METHOD_PROOF_TOKEN_BUDGET_BLOCKED",
            details={
                "required_tokens": int(min_tokens),
                "available_tokens": int(selected_tokens),
                "eligible_rows": len(eligible),
            },
        )

    domain_counts = Counter(str(row.get("domain", "unknown")) for row in selected)
    source_families = Counter(str(row.get("source_family", "unknown")) for row in selected)
    return selected, {
        "selected_rows": len(selected),
        "selected_tokens": int(selected_tokens),
        "bucket_token_counts": dict(sorted(selected_bucket_tokens.items())),
        "bucket_token_targets": dict(sorted(bucket_targets.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_family_counts": dict(sorted(source_families.items())),
        "selected_ids": [row["id"] for row in selected],
        "eligible_rows": len(eligible),
        "excluded_benchmark_rows": excluded_benchmark,
        "excluded_non_fit_rows": excluded_non_fit,
        "split": METHOD_PROOF_SPLIT,
        "evaluation_splits_excluded": list(METHOD_PROOF_EVAL_SPLITS),
        "evaluation_overlap_checked": True,
        "content_deduplication": True,
        "provenance_complete": all(_has_retained_provenance(row) for row in selected),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    return rows


def prepare_method_proof_data(
    corpus_manifest: str | Path,
    output: str | Path,
    *,
    min_tokens: int = METHOD_PROOF_DEFAULT_MIN_TOKENS,
) -> dict[str, Any]:
    """Write the method-proof manifest and a receipt without changing V2.1."""

    source_path = Path(corpus_manifest)
    output_dir = Path(output)
    if int(min_tokens) < METHOD_PROOF_DEFAULT_MIN_TOKENS:
        raise MethodProofBlocked(
            "METHOD_PROOF_TOKEN_POLICY_BLOCKED",
            details={"required_minimum_tokens": METHOD_PROOF_DEFAULT_MIN_TOKENS, "requested_minimum_tokens": int(min_tokens)},
        )
    if not source_path.exists():
        raise MethodProofBlocked("METHOD_PROOF_SOURCE_MISSING", details={"path": str(source_path)})
    source_sha256 = sha256_file(source_path)
    selected, facts = select_method_proof_rows(_read_jsonl(source_path), min_tokens=min_tokens)
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in selected)
    manifest_path = output_dir / "manifest.jsonl"
    _write_text_atomic(manifest_path, manifest_text)
    manifest_sha256 = sha256_file(manifest_path)
    receipt_body: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "dense2moe-method-proof-data",
        "status": METHOD_PROOF_STATUS,
        "method_proof_policy": {
            "source_manifest": str(source_path),
            "source_manifest_sha256": source_sha256,
            "eligible_split": METHOD_PROOF_SPLIT,
            "excluded_splits": list(METHOD_PROOF_EVAL_SPLITS) + ["BENCHMARK-CANARY-EXCLUDED"],
            "benchmark_gradients": False,
            "benchmark_checkpoint_selection": False,
            "minimum_tokens": int(min_tokens),
            "diversity_buckets": ["code", "technical"],
        },
        "manifest": {"path": "manifest.jsonl", "sha256": manifest_sha256, "format": "jsonl"},
        "selection": facts,
        "split_overlap_audit": {"status": "PASS", "checked_against": list(METHOD_PROOF_EVAL_SPLITS) + ["BENCHMARK-CANARY-EXCLUDED"]},
        "provenance": {
            "retained_per_row": True,
            "stable_ids": True,
            "content_hash_deduplicated": True,
            "source_manifest_immutable": True,
        },
        "receipt_hash_basis": "canonical JSON excluding receipt_sha256",
    }
    receipt_body["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt_body)).hexdigest()
    receipt_path = output_dir / "receipt.json"
    _write_text_atomic(receipt_path, json.dumps(receipt_body, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return receipt_body | {"output": str(output_dir)}


__all__ = [
    "METHOD_PROOF_BLOCKED",
    "METHOD_PROOF_DATA_BLOCKED",
    "METHOD_PROOF_DEFAULT_MIN_TOKENS",
    "METHOD_PROOF_STATUS",
    "MethodProofBlocked",
    "prepare_method_proof_data",
    "select_method_proof_rows",
]
