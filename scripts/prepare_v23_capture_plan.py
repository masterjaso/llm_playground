#!/usr/bin/env python3
"""Prepare a fresh, hash-bound FIT-TRAIN/FIT-DEV teacher-capture manifest.

The frozen corpus remains the only source of records.  This adapter expands
the balanced activation plan into the exact source-indexed rows required by
the native streaming teacher, caps both splits to deterministic equal-budget
windows, and never reads retired evaluation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import sha256_file, write_immutable_json

METHOD_VERSION = "moe-v23-m01"
DEFAULT_SEQUENCE_LENGTH = 2_048
DEFAULT_TRAIN_TOKENS = 750_000
DEFAULT_DEV_TOKENS = 100_000


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_corpus(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    line_indices: dict[str, int] = {}
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"corpus line {line_index + 1} is not an object")
        row_id = str(value.get("id", ""))
        if not row_id or row_id in line_indices:
            raise ValueError(f"corpus row has a duplicate or missing id at line {line_index + 1}")
        rows.append(value)
        line_indices[row_id] = line_index
    if not rows:
        raise ValueError(f"corpus is empty: {path}")
    return rows, line_indices


def _row_for_manifest(row: Mapping[str, Any], *, line_index: int, sample_tokens: int) -> dict[str, Any]:
    """Copy only provenance needed to reopen and hash-check one row."""

    if sample_tokens <= 0:
        raise ValueError("sample_tokens must be positive")
    required = (
        "id",
        "source_record_id",
        "source_family",
        "source_name",
        "source_revision",
        "source_license",
        "source_id",
        "source_lineage",
        "content_sha256",
        "normalized_content_sha256",
        "token_count",
        "domain",
        "split",
        "task_id",
        "tree_id",
        "trajectory_id",
        "document_id",
        "group_identity",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"frozen row is missing capture provenance: {missing}")
    result = {key: row[key] for key in required}
    result.update(
        {
            "source_record_index": int(line_index),
            "source_file_sha256": str(row.get("source_file_sha256", "")),
            "sample_tokens": int(sample_tokens),
            "capture_split": str(row["split"]),
        }
    )
    return result


def _select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    budget: int,
    selected_ids: set[str] | None,
    line_indices: Mapping[str, int],
) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("capture token budgets must be positive")
    candidates = [
        row
        for row in rows
        if str(row.get("split", row.get("tier", ""))) == split
        and (selected_ids is None or str(row.get("id", "")) in selected_ids)
    ]
    candidates.sort(key=lambda row: str(row.get("id", "")))
    selected: list[dict[str, Any]] = []
    remaining = int(budget)
    for row in candidates:
        if remaining <= 0:
            break
        available = int(row.get("token_count", 0) or 0)
        if available <= 0:
            continue
        amount = min(available, remaining)
        row_id = str(row["id"])
        selected.append(
            _row_for_manifest(row, line_index=int(line_indices[row_id]), sample_tokens=amount)
        )
        remaining -= amount
    if remaining > 0:
        raise ValueError(
            f"insufficient frozen {split} tokens for capture budget: "
            f"requested={budget}, available={budget - remaining}"
        )
    return selected


def prepare_v23_capture_plan(
    *,
    freeze_receipt: str | Path,
    activation_plan: str | Path,
    output: str | Path,
    train_tokens: int = DEFAULT_TRAIN_TOKENS,
    dev_tokens: int = DEFAULT_DEV_TOKENS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    method_version: str = METHOD_VERSION,
) -> dict[str, Any]:
    """Build and immutably publish the V2.3 native capture manifest."""

    freeze_path = Path(freeze_receipt).resolve()
    activation_path = Path(activation_plan).resolve()
    freeze = _read_json(freeze_path)
    activation = _read_json(activation_path)
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("v23-corpus.jsonl"), Mapping):
        raise TypeError("freeze receipt does not expose the immutable v23-corpus.jsonl artifact")
    corpus_raw = str(artifacts["v23-corpus.jsonl"].get("path", ""))
    corpus_path = (freeze_path.parent / corpus_raw).resolve()
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    expected_corpus_hash = str(artifacts["v23-corpus.jsonl"].get("sha256", ""))
    actual_corpus_hash = sha256_file(corpus_path)
    if expected_corpus_hash and expected_corpus_hash != actual_corpus_hash:
        raise ValueError("frozen corpus hash does not match freeze receipt")
    if activation.get("status") != "READY_FOR_BALANCED_CAPTURE":
        raise ValueError(f"activation plan is not ready: {activation.get('status')!r}")
    rows, line_indices = _read_corpus(corpus_path)
    selected_rows = activation.get("selected_rows")
    if not isinstance(selected_rows, list):
        raise TypeError("activation plan has no selected_rows list")
    selected_train_ids = {
        str(item.get("id"))
        for item in selected_rows
        if isinstance(item, Mapping) and str(item.get("split", "")) == "FIT-TRAIN"
    }
    train = _select_rows(
        rows,
        split="FIT-TRAIN",
        budget=int(train_tokens),
        selected_ids=selected_train_ids,
        line_indices=line_indices,
    )
    # FIT-DEV is selected from the frozen tier itself, not from any old
    # evaluation manifest.  Sorting by the stable row ID makes retries byte
    # identical while keeping all designs on the same development slice.
    dev = _select_rows(
        rows,
        split="FIT-DEV",
        budget=int(dev_tokens),
        selected_ids=None,
        line_indices=line_indices,
    )
    # The capture resolver reopens the canonical JSONL source, so its optional
    # per-row source-file hash must bind that JSONL rather than the raw public
    # archive hash carried by the corpus provenance row.
    for capture_row in (*train, *dev):
        capture_row["source_file_sha256"] = actual_corpus_hash
    source = {
        "path": str(corpus_path),
        "sha256": actual_corpus_hash,
        "format": "jsonl",
        "source_kind": "fresh-v23-frozen-corpus",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "V23_CAPTURE_PLAN_READY",
        "method_version": method_version,
        "dataset_hash": _canonical_hash(
            {
                "corpus_sha256": actual_corpus_hash,
                "FIT-TRAIN": [row["id"] for row in train],
                "FIT-DEV": [row["id"] for row in dev],
                "sample_tokens": {
                    "FIT-TRAIN": [row["sample_tokens"] for row in train],
                    "FIT-DEV": [row["sample_tokens"] for row in dev],
                },
            }
        ),
        "sequence_length": int(sequence_length),
        "source": source,
        "resolvability": {"base_dir": str(corpus_path.parent)},
        "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "freeze_receipt": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "activation_plan": {
            "path": str(activation_path),
            "sha256": sha256_file(activation_path),
            "selected_rows": len(selected_rows),
        },
        "FIT-TRAIN": train,
        "FIT-DEV": dev,
        "selected_tokens": {
            "FIT-TRAIN": sum(int(row["sample_tokens"]) for row in train),
            "FIT-DEV": sum(int(row["sample_tokens"]) for row in dev),
        },
        "identities": {
            "FIT-TRAIN": _canonical_hash([row["id"] for row in train]),
            "FIT-DEV": _canonical_hash([row["id"] for row in dev]),
        },
        "evaluation_splits_excluded": [
            "GATE-A",
            "SHADOW-B",
            "SHADOW-C",
            "PRESERVATION-CANARY",
        ],
        "teacher_capture": "NOT_STARTED",
    }
    output_path = Path(output)
    write_immutable_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--activation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--dev-tokens", type=int, default=DEFAULT_DEV_TOKENS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--method-version", default=METHOD_VERSION)
    args = parser.parse_args()
    result = prepare_v23_capture_plan(
        freeze_receipt=args.freeze_receipt,
        activation_plan=args.activation_plan,
        output=args.output,
        train_tokens=args.train_tokens,
        dev_tokens=args.dev_tokens,
        sequence_length=args.sequence_length,
        method_version=args.method_version,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
