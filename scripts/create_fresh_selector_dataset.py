#!/usr/bin/env python3
"""Build a fresh, provenance-complete selector corpus and frozen A/B receipts.

This utility deliberately does not reuse ``data/public_v2/corpus.jsonl`` or
the historical FIT/holdout records.  It fetches a small, diverse set of
permissively usable text sources, records the exact bytes' SHA-256, rejects
normalized-content overlap with the historical corpus, prepares an exact
tokenized calibration manifest with the pinned tokenizer, and freezes
validation-A/B row identities before selector optimization.

The resulting manifest is intended for a layer-0-only streaming capture.  A
caller can pass the generated ``data-plan.json`` to
``d2m streaming-capture --layers 0 --split train``; no higher layer is
implicitly replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import prepare_calibration_manifest
from dense2moe.provenance import current_git_commit
from dense2moe.training import deterministic_shadow_validation_indices


DEFAULT_SOURCES: tuple[dict[str, str], ...] = (
    {
        "name": "Wikitext-2 raw train",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "domain": "general",
        "license": "CC-BY-SA-3.0",
        "rationale": "Diverse encyclopedia prose and markup not present in the historical calibration mixture.",
    },
    {
        "name": "Project Gutenberg Frankenstein",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "domain": "long-context",
        "license": "Public Domain",
        "rationale": "New public-domain long-form prose for distribution-shift coverage.",
    },
    {
        "name": "Project Gutenberg Shakespeare",
        "url": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
        "domain": "general",
        "license": "Public Domain",
        "rationale": "Additional public-domain dramatic prose and dialogue from a distinct book.",
    },
    {
        "name": "CPython asyncio documentation",
        "url": "https://raw.githubusercontent.com/python/cpython/main/Doc/library/asyncio.rst",
        "domain": "code",
        "license": "PSF-2.0",
        "rationale": "Technical/API prose and code examples from a distinct permissive source.",
    },
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_hash(text: str) -> str:
    normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    return _sha256_bytes(normalized.encode("utf-8"))


def _content_hash(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _download(url: str, destination: Path) -> tuple[Path, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-fresh-selector/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    digest = _sha256_bytes(payload)
    if destination.exists() and _sha256_file(destination) != digest:
        raise RuntimeError(f"refusing to overwrite changed source cache: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)
    return destination, digest


def _clean_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if start:
        text = text[start.end() :]
    if end:
        text = text[: end.start()]
    return text.strip()


def _chunks(text: str, *, max_chars: int) -> list[str]:
    """Chunk paragraphs without duplicating or reordering source text."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    result: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            result.append(paragraph)
            continue
        result.extend(paragraph[start : start + max_chars] for start in range(0, len(paragraph), max_chars))
    return result


def _historical_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            candidate = value.get("normalized_content_sha256")
            if isinstance(candidate, str) and candidate:
                hashes.add(candidate)
            elif isinstance(value.get("text"), str):
                hashes.add(_normalized_hash(value["text"]))
    return hashes


def build_corpus(
    *,
    output: Path,
    source_cache: Path,
    historical_corpus: Path | None,
    max_chars: int,
    max_records_per_source: int | None = None,
) -> dict[str, Any]:
    historical = _historical_hashes(historical_corpus)
    records: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, source in enumerate(DEFAULT_SOURCES):
        filename = re.sub(r"[^a-z0-9]+", "-", source["name"].lower()).strip("-") + ".txt"
        local, digest = _download(source["url"], source_cache / filename)
        text = local.read_text(encoding="utf-8", errors="replace")
        if "Gutenberg" in source["name"]:
            text = _clean_gutenberg(text)
        chunks = _chunks(text, max_chars=max_chars)
        if max_records_per_source is not None:
            chunks = chunks[:max_records_per_source]
        source_receipts.append(
            {
                "name": source["name"],
                "url": source["url"],
                "local_path": str(local),
                "download_sha256": digest,
                "source_revision": f"sha256:{digest}",
                "record_count": len(chunks),
            }
        )
        for record_index, chunk in enumerate(chunks):
            content_hash = _content_hash(chunk)
            normalized_hash = _normalized_hash(chunk)
            if not chunk.strip() or normalized_hash in historical or normalized_hash in seen:
                continue
            seen.add(normalized_hash)
            stable_id = hashlib.sha256(
                f"fresh:{source_index}:{record_index}:{content_hash}".encode("utf-8")
            ).hexdigest()[:32]
            records.append(
                {
                    "id": stable_id,
                    "source_record_id": f"{source_index}:{record_index}",
                    "source_record_index": len(records),
                    "source_name": source["name"],
                    "source_revision": f"sha256:{digest}",
                    "source_license": source["license"],
                    "source_url": source["url"],
                    "download_sha256": digest,
                    "domain": source["domain"],
                    "rationale": source["rationale"],
                    "selection_rationale": source["rationale"],
                    "source_file": output.name,
                    "text": chunk,
                    "content_sha256": content_hash,
                    "normalized_content_sha256": normalized_hash,
                    "add_special_tokens": True,
                }
            )
    if not records:
        raise RuntimeError("fresh corpus is empty after historical-overlap rejection")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            record["source_record_index"] = index
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "status": "FRESH_CORPUS_MATERIALIZED",
        "path": str(output),
        "records": len(records),
        "domains": sorted({str(item["domain"]) for item in records}),
        "source_receipts": source_receipts,
        "historical_overlap_rejected": True,
        "historical_hash_count": len(historical),
        "corpus_sha256": _sha256_file(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--historical-corpus", type=Path, default=Path("data/public_v2/corpus.jsonl"))
    parser.add_argument("--train-tokens", type=int, default=350_000)
    parser.add_argument("--holdout-tokens", type=int, default=16_384)
    parser.add_argument("--validation-a-count", type=int, default=16_384)
    parser.add_argument("--validation-b-count", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--holdout-seed", type=int, default=20260817)
    parser.add_argument("--shadow-seed", type=int, default=20260818)
    parser.add_argument("--max-chars", type=int, default=16_000)
    parser.add_argument("--max-records-per-source", type=int, default=800)
    args = parser.parse_args()
    if args.validation_a_count <= 0 or args.validation_b_count <= 0:
        raise ValueError("validation A/B counts must be positive")
    root = args.run_dir
    corpus_path = root / "source" / "fresh-selector-corpus.jsonl"
    cache_dir = root / "source" / "fresh-text-sources"
    corpus = build_corpus(
        output=corpus_path,
        source_cache=cache_dir,
        historical_corpus=args.historical_corpus,
        max_chars=args.max_chars,
        max_records_per_source=args.max_records_per_source,
    )
    data_plan = root / "capture" / "fresh-selector-data-plan.json"
    receipt = root / "capture" / "fresh-selector-corpus-receipt.json"
    manifest = prepare_calibration_manifest(
        corpus_path,
        data_plan,
        train_tokens=args.train_tokens,
        holdout_tokens=args.holdout_tokens,
        seed=args.seed,
        holdout_seed=args.holdout_seed,
        sequence_length=2048,
        tokenizer_revision=args.source_revision,
        source_snapshot=args.source_snapshot,
        add_special_tokens=True,
        receipt_output=receipt,
        required_domains=("code", "general", "long-context"),
    )
    train_count = int(manifest["train_tokens"])
    validation_a, validation_a_hash = deterministic_shadow_validation_indices(
        train_count,
        excluded_indices=(),
        shadow_count=args.validation_a_count,
        seed=args.shadow_seed,
    )
    validation_b, validation_b_hash = deterministic_shadow_validation_indices(
        train_count,
        excluded_indices=validation_a,
        shadow_count=args.validation_b_count,
        seed=args.shadow_seed + 1,
    )
    shadow = {
        "schema_version": 2,
        "status": "FRESH_SELECTOR_AB_FROZEN",
        "classification": "NEW_LAYER0_CAPTURE_VALIDATION_A_SELECTION_B_CONFIRMATION_ONLY",
        "dataset_manifest": str(data_plan),
        "dataset_hash": manifest["dataset_hash"],
        "train_token_count": train_count,
        "validation_a": {
            "count": len(validation_a),
            "indices": list(validation_a),
            "indices_hash": validation_a_hash,
            "checkpoint_selection": True,
            "gradient_updates": False,
        },
        "validation_b": {
            "count": len(validation_b),
            "indices": list(validation_b),
            "indices_hash": validation_b_hash,
            "checkpoint_selection": False,
            "gradient_updates": False,
            "confirmation_only": True,
        },
        "disjoint": True,
        "selection_union": {"enabled": False},
        "historical_holdout_opened": False,
        "source_revision": args.source_revision,
        "code_commit": current_git_commit(),
        "corpus": corpus,
    }
    shadow_path = root / "capture" / "fresh-selector-validation-ab.json"
    shadow_path.parent.mkdir(parents=True, exist_ok=True)
    shadow_path.write_text(json.dumps(shadow, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "FRESH_SELECTOR_DATASET_READY",
                "data_plan": str(data_plan),
                "validation_ab": str(shadow_path),
                "train_tokens": train_count,
                "validation_a_count": len(validation_a),
                "validation_b_count": len(validation_b),
                "dataset_hash": manifest["dataset_hash"],
                "code_commit": current_git_commit(),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
