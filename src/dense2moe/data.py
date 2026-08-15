"""Reproducible calibration/holdout manifest construction."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _example_id(text: str, index: int) -> str:
    return hashlib.sha256(f"{index}:{text}".encode()).hexdigest()[:24]


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    records.append(dict(value))
        return records
    if path.suffix.lower() == ".tsv":
        lines = path.read_text(encoding="utf-8").splitlines()
        headers = lines[0].split("\t") if lines else []
        return [dict(zip(headers, row.split("\t"))) for row in lines[1:]]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("examples", value.get("records", []))
    if not isinstance(value, list):
        raise TypeError("corpus manifest must contain a list of examples")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def prepare_calibration_manifest(
    corpus_manifest: str | Path,
    output: str | Path,
    *,
    train_tokens: int = 131_072,
    holdout_tokens: int = 16_384,
    seed: int = 17,
    holdout_seed: int = 29,
    sequence_length: int = 2048,
    tokenizer_revision: str = "declared-by-input",
) -> dict[str, Any]:
    """Create a deterministic, non-overlapping split manifest from local data."""

    if train_tokens <= 0 or holdout_tokens <= 0 or sequence_length <= 0:
        raise ValueError("token and sequence-length targets must be positive")
    source = Path(corpus_manifest)
    records = _read_records(source)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        text = str(item.get("text", item.get("content", "")))
        if not text.strip():
            continue
        token_count = int(item.get("token_count", item.get("tokens", len(text.split()))))
        if token_count <= 0:
            continue
        normalized.append(
            {
                "id": str(item.get("id", _example_id(text, index))),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_count": token_count,
                "domain": str(item.get("domain", "general")),
                "sequence_length": min(sequence_length, int(item.get("sequence_length", sequence_length))),
                "source_id": str(item.get("source_id", source.name)),
            }
        )
    if not normalized:
        raise ValueError("corpus contains no non-empty examples")
    by_id = {item["id"]: item for item in normalized}
    ids = sorted(by_id)
    train_order = ids[:]
    holdout_order = ids[:]
    random.Random(seed).shuffle(train_order)
    random.Random(holdout_seed).shuffle(holdout_order)
    train: list[dict[str, Any]] = []
    used: set[str] = set()
    total = 0
    for identifier in train_order:
        if total >= train_tokens:
            break
        item = by_id[identifier]
        train.append(item)
        used.add(identifier)
        total += int(item["token_count"])
    holdout: list[dict[str, Any]] = []
    total_holdout = 0
    for identifier in holdout_order:
        if identifier in used or total_holdout >= holdout_tokens:
            continue
        item = by_id[identifier]
        holdout.append(item)
        total_holdout += int(item["token_count"])
    if total < train_tokens or total_holdout < holdout_tokens:
        raise ValueError(f"corpus cannot satisfy disjoint token targets: train={total}, holdout={total_holdout}")
    payload = {
        "schema_version": 1,
        "status": "CALIBRATION_READY",
        "source": {"path": str(source), "sha256": sha256_file(source), "format": source.suffix.lower().lstrip(".")},
        "tokenizer_revision": tokenizer_revision,
        "tokenization_method": "declared_token_count_or_whitespace_fallback",
        "sequence_length": sequence_length,
        "deduplication": "stable-example-id",
        "seed": seed,
        "holdout_seed": holdout_seed,
        "train": train,
        "holdout": holdout,
        "train_tokens": total,
        "holdout_tokens": total_holdout,
        "train_ids_sha256": hashlib.sha256("\n".join(item["id"] for item in train).encode()).hexdigest(),
        "holdout_ids_sha256": hashlib.sha256("\n".join(item["id"] for item in holdout).encode()).hexdigest(),
        "domains": sorted({item["domain"] for item in normalized}),
    }
    payload["dataset_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["prepare_calibration_manifest", "sha256_file"]
