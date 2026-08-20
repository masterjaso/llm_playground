#!/usr/bin/env python3
"""Acquire and prepare a fresh, public-only V2.3 development corpus.

The V2.3 acquisition boundary is intentionally independent from the V2.2
corpus.  Sources are downloaded into content-addressed files, then split by
whole source/group identities into FIT-TRAIN and FIT-DEV.  This module does
not open, copy, or relabel any evaluation tier.  The freeze script remains
the authority which writes the immutable V2.3 manifest and receipts.

The public source list is deliberately small enough to run in a fresh
checkout.  Callers can provide a JSON source specification or pass source
records directly to :func:`build_corpus` in tests and offline workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import (
    _minhash_signature,
    _shingle_jaccard,
    sha256_file,
    write_immutable_json,
    write_immutable_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "d2m-qwen38-moe-v23-20260818t190044z-fc0b2647"
DEFAULT_OUTPUT = ROOT / ".nsp" / "artifacts" / "runs" / DEFAULT_RUN_ID / "data"
DEVELOPMENT_TIERS = ("FIT-TRAIN", "FIT-DEV")
V23_METHOD_VERSION = "moe-v23-m01"

# These are distinct from every source used by the V2.2 acquisition path.
# Revisions are content-pinned after download (sha256:<file digest>), so a
# moving public branch cannot silently change the run identity.
DEFAULT_SOURCES: tuple[dict[str, str], ...] = (
    {
        "source_id": "wikitext2-pytorch-train",
        "name": "PyTorch examples WikiText-2 raw train",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "license": "CC-BY-SA-3.0",
        "domain": "general",
        "tier": "FIT-TRAIN",
        "rationale": "Public WikiText prose from a source family absent from V2.2.",
    },
    {
        "source_id": "gutenberg-frankenstein-84",
        "name": "Project Gutenberg Frankenstein",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "license": "Public Domain",
        "domain": "long-context",
        "revision": "ebook-84",
        "tier": "FIT-TRAIN",
        "rationale": "A distinct public-domain long-form book for distribution-shift coverage.",
    },
    {
        "source_id": "gutenberg-shakespeare-100",
        "name": "Project Gutenberg Shakespeare",
        "url": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
        "license": "Public Domain",
        "domain": "general",
        "revision": "ebook-100",
        "tier": "FIT-TRAIN",
        "rationale": "A second public-domain book with dialogue and dramatic prose.",
    },
    {
        "source_id": "cpython-asyncio-doc",
        "name": "CPython asyncio documentation",
        "url": "https://raw.githubusercontent.com/python/cpython/main/Doc/library/asyncio.rst",
        "license": "PSF-2.0",
        "domain": "code",
        "tier": "FIT-DEV",
        "rationale": "Permissively licensed API documentation and code examples.",
    },
    {
        "source_id": "rust-book-getting-started",
        "name": "The Rust Programming Language getting started chapter",
        "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch01-00-getting-started.md",
        "license": "MIT OR Apache-2.0",
        "domain": "code",
        "tier": "FIT-DEV",
        "rationale": "A distinct permissively licensed programming-language source.",
    },
    {
        "source_id": "rust-book-common-concepts",
        "name": "The Rust Programming Language common concepts chapter",
        "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch03-00-common-programming-concepts.md",
        "license": "MIT OR Apache-2.0",
        "domain": "code",
        "tier": "FIT-DEV",
        "rationale": "Additional independent code-language documentation from a pinned path.",
    },
    {
        "source_id": "gutenberg-alice-11",
        "name": "Project Gutenberg Alice's Adventures in Wonderland",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "license": "Public Domain",
        "domain": "general",
        "revision": "ebook-11",
        "tier": "FIT-DEV",
        "rationale": "A distinct public-domain short novel for an independently grouped development tier.",
    },
    {
        "source_id": "gutenberg-tom-sawyer-74",
        "name": "Project Gutenberg The Adventures of Tom Sawyer",
        "url": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
        "license": "Public Domain",
        "domain": "general",
        "revision": "ebook-74",
        "tier": "FIT-DEV",
        "rationale": "A distinct public-domain novel for general-language coverage.",
    },
    {
        "source_id": "gutenberg-sherlock-1661",
        "name": "Project Gutenberg The Adventures of Sherlock Holmes",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "license": "Public Domain",
        "domain": "general",
        "revision": "ebook-1661",
        "tier": "FIT-DEV",
        "rationale": "A distinct public-domain dialogue-heavy source for development coverage.",
    },
    {
        "source_id": "gutenberg-moby-dick-2701",
        "name": "Project Gutenberg Moby-Dick",
        "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "license": "Public Domain",
        "domain": "long-context",
        "revision": "ebook-2701",
        "tier": "FIT-TRAIN",
        "rationale": "A distinct public-domain long-form source to ensure the common train budget is well populated.",
    },
    {
        "source_id": "gutenberg-tale-two-cities-98",
        "name": "Project Gutenberg A Tale of Two Cities",
        "url": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
        "license": "Public Domain",
        "domain": "long-context",
        "revision": "ebook-98",
        "tier": "FIT-TRAIN",
        "rationale": "A distinct public-domain novel adding independent train-side prose coverage.",
    },
)

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "mit",
    "mit or apache-2.0",
    "mit/apache-2.0",
    "psf-2.0",
    "public domain",
    "unlicense",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalise_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def normalized_text_hash(text: str) -> str:
    return _sha256_bytes(_normalise_text(text).encode("utf-8"))


def content_hash(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "source"


def _as_source(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = dict(vars(value))
    source_id = str(result.get("source_id", result.get("id", ""))).strip()
    if not source_id:
        source_id = _slug(str(result.get("name", result.get("url", ""))))
    result["source_id"] = source_id
    result["name"] = str(result.get("name", source_id)).strip()
    result["url"] = str(result.get("url", "")).strip()
    result["license"] = str(result.get("license", result.get("source_license", ""))).strip()
    result["domain"] = str(result.get("domain", "general")).strip() or "general"
    result["rationale"] = str(result.get("rationale", result.get("selection_rationale", "Fresh public source selected for V2.3."))).strip()
    return result


def _validate_source(source: Mapping[str, Any]) -> None:
    license_name = str(source.get("license", "")).strip()
    if license_name.casefold() not in PERMISSIVE_LICENSES:
        raise ValueError(f"source {source.get('source_id', '')!r} has an unapproved or missing license: {license_name!r}")
    url = str(source.get("url", "")).strip()
    local_path = source.get("local_path")
    if local_path:
        return
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"source {source.get('source_id', '')!r} must use an HTTPS/HTTP public URL")
    host = parsed.hostname.casefold() if parsed.hostname else ""
    if parsed.username or parsed.password or host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".internal"):
        raise ValueError(f"source {source.get('source_id', '')!r} uses a private or credentialed URL")


def _read_source_bytes(source: Mapping[str, Any], *, timeout: int = 120, opener: Callable[..., Any] | None = None) -> bytes:
    local_path = source.get("local_path")
    if local_path:
        return Path(str(local_path)).read_bytes()
    url = str(source["url"])
    request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-v23-public-acquisition/1.0"})
    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=timeout) as response:
        return response.read()


def download_source(
    source: Mapping[str, Any],
    *,
    raw_dir: Path,
    timeout: int = 120,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Download one public source to a content-addressed immutable raw file."""

    source = _as_source(source)
    _validate_source(source)
    payload = _read_source_bytes(source, timeout=timeout, opener=opener)
    digest = _sha256_bytes(payload)
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(str(source['source_id']))}-{digest[:16]}.txt"
    destination = raw_dir / filename
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError(f"content-addressed raw source mismatch: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)
    revision = str(source.get("revision", "")).strip() or f"sha256:{digest}"
    return {
        "source_id": str(source["source_id"]),
        "name": str(source["name"]),
        "url": str(source["url"]),
        "source_url": str(source["url"]),
        "source_revision": revision,
        "source_license": str(source["license"]),
        "domain": str(source["domain"]),
        "rationale": str(source["rationale"]),
        "raw_path": destination,
        "raw_sha256": digest,
        "download_sha256": digest,
        "source_file_sha256": digest,
        "bytes": len(payload),
    }


def _chunks(text: str, *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        for start in range(0, len(paragraph), max_chars):
            piece = paragraph[start : start + max_chars].strip()
            if piece:
                chunks.append(piece)
    return chunks


def _prepare_source_text(source: Mapping[str, Any], text: str) -> str:
    """Remove publisher boilerplate before making identity-bearing records.

    Project Gutenberg distributes a standard licence/header/footer around the
    public-domain work.  Those paragraphs are identical across otherwise
    independent books and would make the grouped FIT-TRAIN/FIT-DEV overlap
    gate fail closed.  Raw files remain untouched and are retained for the
    provenance receipt; only the prepared corpus payload is trimmed.
    """

    source_id = str(source.get("source_id", "")).casefold()
    url = str(source.get("url", "")).casefold()
    if not source_id.startswith("gutenberg-") and "gutenberg.org" not in url:
        return text
    start = re.search(r"^\s*\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK[^\n]*\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if start:
        text = text[start.end() :]
    end = re.search(r"^\s*\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK[^\n]*\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if end:
        text = text[: end.start()]
    return text.strip()


def _assign_source_tiers(sources: Sequence[Mapping[str, Any]], *, seed: int = 23) -> dict[str, str]:
    """Assign whole sources to development tiers, never individual rows."""

    if len(sources) < 2:
        raise ValueError("at least two independent public sources are required for FIT-TRAIN/FIT-DEV")
    normalized = [_as_source(item) for item in sources]
    ids = [str(item["source_id"]) for item in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("source_id values must be unique")
    explicit = {source_id: str(item.get("tier", item.get("split", ""))).strip() for source_id, item in zip(ids, normalized)}
    if any(value and value not in DEVELOPMENT_TIERS for value in explicit.values()):
        raise ValueError("source tiers may only be FIT-TRAIN or FIT-DEV")
    if any(explicit.values()):
        if any(not value for value in explicit.values()):
            raise ValueError("either all source tiers must be explicit or none may be explicit")
        if set(explicit.values()) != set(DEVELOPMENT_TIERS):
            raise ValueError("explicit source tiers must include both FIT-TRAIN and FIT-DEV")
        return explicit
    result: dict[str, str] = {}
    for source_id in sorted(ids):
        bucket = int(hashlib.sha256(f"v23-source:{seed}:{source_id}".encode()).hexdigest()[:8], 16) % 100
        result[source_id] = "FIT-DEV" if bucket < 35 else "FIT-TRAIN"
    # Hash assignment is deterministic but a small source list can land in
    # one bucket.  Move only the lexicographically first source when needed.
    if len(set(result.values())) == 1:
        ordered = sorted(ids)
        moved = ordered[0]
        result[moved] = "FIT-DEV" if result[moved] == "FIT-TRAIN" else "FIT-TRAIN"
    return result


def _canonical_id(row: Mapping[str, Any]) -> str:
    seed = "|".join(
        (
            str(row.get("source_name", "")),
            str(row.get("source_revision", "")),
            str(row.get("source_record_id", "")),
            str(row.get("content_sha256", "")),
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    yield dict(value)
        return
    if path.suffix.casefold() != ".json":
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    def walk(value: Any, *, tier: str = "") -> Iterable[dict[str, Any]]:
        if isinstance(value, Mapping):
            if any(key in value for key in ("text", "content", "source_record_id", "normalized_content_sha256")):
                row = dict(value)
                if tier and not row.get("tier"):
                    row["tier"] = tier
                yield row
                return
            for key, child in value.items():
                next_tier = tier
                if str(key) in DEVELOPMENT_TIERS or str(key).startswith(("GATE", "SHADOW", "G1", "G2")):
                    next_tier = str(key)
                yield from walk(child, tier=next_tier)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, Mapping):
                    yield from walk(child, tier=tier)
                elif isinstance(child, str) and tier:
                    yield {"id": child, "stable_id": child, "tier": tier}

    yield from walk(payload)


def discover_v22_ledgers(root: Path = ROOT) -> list[Path]:
    """Find every existing V2.2 manifest/ledger without using V2.2 data as input."""

    candidates: set[Path] = set()
    for base in (root / "data", root / ".nsp" / "artifacts" / "runs"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.casefold()
            normalized = path.as_posix().casefold()
            is_v22_artifact = "v2.2" in name
            is_v22_prepared = "/inputs/prepared/" in normalized or "/inputs/prepared-final/" in normalized
            is_historical_public = any(
                marker in normalized
                for marker in ("/data/public_v2/", "/data/public_v21/")
            ) and any(
                marker in name
                for marker in ("corpus", "split", "receipt", "quarantine", "benchmark")
            )
            if (is_v22_artifact or is_v22_prepared or is_historical_public) and path.suffix.casefold() in {".json", ".jsonl", ".ndjson"}:
                candidates.add(path.resolve())
    return sorted(candidates)


def _identity_values(row: Mapping[str, Any], dimension: str) -> set[str]:
    def values(*keys: str) -> set[str]:
        result: set[str] = set()
        for key in keys:
            value = row.get(key)
            if isinstance(value, (list, tuple, set)):
                result.update(str(item).strip().casefold() for item in value if str(item).strip())
            elif value is not None and str(value).strip():
                result.add(str(value).strip().casefold())
        return result

    if dimension == "source":
        raw = values("source_id", "source_name", "source", "source_url", "source_artifact_url")
        return {f"source:{item}" for item in raw}
    if dimension == "task":
        return {f"task:{item}" for item in values("task_id", "task", "issue_id", "instance_id")}
    if dimension == "tree":
        return {f"tree:{item}" for item in values("tree_id", "message_tree_id", "conversation_id", "thread_id")}
    if dimension == "trajectory":
        return {f"trajectory:{item}" for item in values("trajectory_id", "trajectory", "trace_id")}
    if dimension == "document":
        return {f"document:{item}" for item in values("document_id", "source_document_id", "document")}
    if dimension == "repository_path_revision":
        explicit = values("repository_path_revision")
        if explicit:
            return {f"repository_path_revision:{item}" for item in explicit}
        repositories = values("repository_id", "repository", "repo", "repo_name")
        paths = values("repo_path", "repository_path", "path", "source_file")
        revisions = values("repo_commit", "repository_revision", "source_revision", "revision")
        result = {f"repository:{repo}|path:{path}|revision:{revision}" for repo in repositories or {""} for path in paths or {""} for revision in revisions or {""}}
        return {item for item in result if item != "repository:|path:|revision:"}
    if dimension == "normalized_text":
        candidate = str(row.get("normalized_content_sha256", "")).strip().casefold()
        if not candidate and str(row.get("text", row.get("content", ""))).strip():
            candidate = normalized_text_hash(str(row.get("text", row.get("content", ""))))
        return {f"normalized:{candidate}"} if candidate else set()
    if dimension == "group":
        return {f"group:{item}" for item in values("group_identity", "split_group", "source_lineage", "source_family_group", "source_family_id")}
    if dimension == "source_record":
        return {f"record:{item}" for item in values("source_record_id", "record_id", "id", "stable_id")}
    raise ValueError(f"unknown identity dimension: {dimension}")


IDENTITY_DIMENSIONS = ("source", "task", "tree", "trajectory", "document", "repository_path_revision", "normalized_text", "group", "source_record")


def load_v22_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read identity ledgers/manifests, preferring content-bearing rows.

    Activation plans and split ledgers often appear before the frozen JSONL
    manifest in a directory walk and contain only row IDs.  A first-row-wins
    dedupe would therefore erase text, hashes, and task/document identities
    from the historical overlap audit.  Keep the best representation for each
    stable ID and retain the source-file metadata for every ledger read.
    """

    metadata: list[dict[str, Any]] = []
    by_identity: dict[str, dict[str, Any]] = {}

    def quality(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
        text = str(row.get("text", row.get("content", ""))).strip()
        hashes = sum(bool(str(row.get(key, "")).strip()) for key in ("content_sha256", "normalized_content_sha256"))
        provenance = sum(bool(str(row.get(key, "")).strip()) for key in ("source_name", "source_revision", "source_record_id", "document_id", "task_id", "split_group"))
        tier = int(bool(str(row.get("tier", row.get("split", ""))).strip()))
        return (int(bool(text)), hashes, provenance, tier)

    def path_priority(path: Path) -> tuple[int, str]:
        normalized = path.as_posix().casefold()
        if normalized.endswith(".jsonl") and ("/inputs/prepared/" in normalized or "/inputs/prepared-final/" in normalized):
            return (0, normalized)
        if normalized.endswith("corpus-v2.2.jsonl"):
            return (1, normalized)
        if "corpus-v2.2" in normalized:
            return (2, normalized)
        return (3, normalized)

    for path_value in sorted({Path(path) for path in paths}, key=path_priority):
        path = Path(path_value)
        if not path.exists():
            continue
        file_rows = list(_iter_json_rows(path))
        metadata.append({"path": str(path), "sha256": sha256_file(path), "records": len(file_rows)})
        for row in file_rows:
            identity = str(row.get("id", row.get("stable_id", ""))).strip() or _canonical_id(row)
            previous = by_identity.get(identity)
            if previous is None or quality(row) > quality(previous):
                by_identity[identity] = row
    return list(by_identity.values()), metadata


def audit_v22_overlap(
    rows: Iterable[Mapping[str, Any]],
    *,
    historical_rows: Iterable[Mapping[str, Any]] = (),
    historical_paths: Iterable[Path] = (),
    near_duplicate_threshold: float = 0.80,
    max_conflicts: int = 10_000,
) -> dict[str, Any]:
    """Audit fresh rows against every supplied V2.2 identity ledger.

    Identity matches are checked independently for source, task, tree,
    trajectory, document, repository-path+revision, normalized text, source
    record, and explicit group identity.  Near duplicates use MinHash bands as
    a screening index and exact shingle Jaccard for the final decision.
    """

    fresh = [dict(row) for row in rows]
    historical = [dict(row) for row in historical_rows]
    dimensions: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for dimension in IDENTITY_DIMENSIONS:
        old_index: dict[str, list[str]] = defaultdict(list)
        for old in historical:
            for identity in _identity_values(old, dimension):
                old_index[identity].append(str(old.get("id", old.get("stable_id", _canonical_id(old)))))
        dimension_conflicts: list[dict[str, Any]] = []
        for row in fresh:
            row_id = str(row.get("id", row.get("stable_id", _canonical_id(row))))
            for identity in _identity_values(row, dimension):
                matches = old_index.get(identity, [])
                if matches:
                    item = {"dimension": dimension, "identity": identity, "fresh_id": row_id, "historical_ids": sorted(set(matches))}
                    dimension_conflicts.append(item)
                    if len(conflicts) < max_conflicts:
                        conflicts.append(item)
        dimensions[dimension] = {
            "fresh_identities": sum(len(_identity_values(row, dimension)) for row in fresh),
            "historical_identities": len(old_index),
            "conflicts": dimension_conflicts,
            "status": "PASS" if not dimension_conflicts else "FAIL",
        }

    signature_cache: dict[str, tuple[int, ...]] = {}

    def signature(text: str) -> tuple[int, ...]:
        normalized = _normalise_text(text)
        cached = signature_cache.get(normalized)
        if cached is None:
            cached = _minhash_signature(normalized)
            signature_cache[normalized] = cached
        return cached

    historical_bands: dict[tuple[int, tuple[int, ...]], list[tuple[str, str]]] = defaultdict(list)
    for old in historical:
        text = _normalise_text(str(old.get("text", old.get("content", ""))))
        if not text:
            continue
        old_id = str(old.get("id", old.get("stable_id", _canonical_id(old))))
        old_signature = signature(text)
        for band in range(0, len(old_signature), 4):
            historical_bands[(band // 4, old_signature[band : band + 4])].append((old_id, text))
    near: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in fresh:
        text = _normalise_text(str(row.get("text", row.get("content", ""))))
        if not text:
            continue
        row_id = str(row.get("id", row.get("stable_id", _canonical_id(row))))
        row_signature = signature(text)
        candidates: dict[str, str] = {}
        for band in range(0, len(row_signature), 4):
            for old_id, old_text in historical_bands.get((band // 4, row_signature[band : band + 4]), []):
                candidates[old_id] = old_text
        for old_id, old_text in candidates.items():
            pair = tuple(sorted((row_id, old_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            similarity = _shingle_jaccard(text, old_text)
            if similarity >= near_duplicate_threshold:
                item = {"fresh_id": row_id, "historical_id": old_id, "jaccard": similarity}
                near.append(item)
                if len(conflicts) < max_conflicts:
                    conflicts.append({"dimension": "near_duplicate", **item})
                if len(near) >= max_conflicts:
                    break
        if len(near) >= max_conflicts:
            break
    dimensions["near_duplicate"] = {"conflicts": near, "threshold": float(near_duplicate_threshold), "status": "PASS" if not near else "FAIL"}
    paths = [{"path": str(path), "sha256": sha256_file(path)} for path in historical_paths if Path(path).exists()]
    return {
        "schema_version": 1,
        "audit_type": "dense2moe-v23-v22-overlap",
        "status": "PASS" if not conflicts else "FAIL",
        "fresh_records_checked": len(fresh),
        "historical_records_checked": len(historical),
        "historical_ledgers": paths,
        "dimensions": dimensions,
        "conflicts": conflicts,
        "zero_forbidden_overlap": not conflicts,
    }


def audit_v23_tier_disjointness(rows: Iterable[Mapping[str, Any]], *, near_duplicate_threshold: float = 0.80) -> dict[str, Any]:
    """Audit all required group identities between FIT-TRAIN and FIT-DEV."""

    values = [dict(row) for row in rows]
    conflicts: list[dict[str, Any]] = []
    by_dimension: dict[str, dict[str, dict[str, list[str]]]] = {}
    for dimension in IDENTITY_DIMENSIONS:
        index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for row in values:
            tier = str(row.get("tier", row.get("split", "")))
            row_id = str(row.get("id", row.get("stable_id", _canonical_id(row))))
            for identity in _identity_values(row, dimension):
                index[identity][tier].append(row_id)
        dimension_conflicts: list[dict[str, Any]] = []
        for identity, tiers in index.items():
            active = {tier: sorted(set(ids)) for tier, ids in tiers.items() if tier in DEVELOPMENT_TIERS}
            if len(active) > 1:
                item = {"dimension": dimension, "identity": identity, "tiers": active}
                dimension_conflicts.append(item)
                conflicts.append(item)
        by_dimension[dimension] = {identity: {tier: sorted(set(ids)) for tier, ids in tiers.items()} for identity, tiers in index.items()}

    signature_cache: dict[str, tuple[int, ...]] = {}

    def signature(text: str) -> tuple[int, ...]:
        normalized = _normalise_text(text)
        cached = signature_cache.get(normalized)
        if cached is None:
            cached = _minhash_signature(normalized)
            signature_cache[normalized] = cached
        return cached

    bands: dict[tuple[int, tuple[int, ...]], list[tuple[str, str, str]]] = defaultdict(list)
    for row in values:
        tier = str(row.get("tier", row.get("split", "")))
        text = _normalise_text(str(row.get("text", row.get("content", ""))))
        if tier not in DEVELOPMENT_TIERS or not text:
            continue
        row_id = str(row.get("id", row.get("stable_id", _canonical_id(row))))
        row_signature = signature(text)
        for band in range(0, len(row_signature), 4):
            bands[(band // 4, row_signature[band : band + 4])].append((row_id, tier, text))
    near: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for bucket in bands.values():
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                if left[1] == right[1]:
                    continue
                pair = tuple(sorted((left[0], right[0])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                similarity = _shingle_jaccard(left[2], right[2])
                if similarity >= near_duplicate_threshold:
                    near.append({"left_id": left[0], "left_tier": left[1], "right_id": right[0], "right_tier": right[1], "jaccard": similarity})
    return {
        "schema_version": 1,
        "audit_type": "dense2moe-v23-grouped-tier-disjointness",
        "status": "PASS" if not conflicts and not near else "FAIL",
        "checked_tiers": list(DEVELOPMENT_TIERS),
        "records_checked": len(values),
        "group_conflicts": conflicts,
        "near_duplicate_conflicts": near,
        "near_duplicate_threshold": float(near_duplicate_threshold),
        "zero_forbidden_overlap": not conflicts and not near,
        "identity_index_sizes": {dimension: len(index) for dimension, index in ((dimension, by_dimension[dimension]) for dimension in by_dimension)},
    }


def _load_tokenizer(path: Path | None) -> Any | None:
    if path is None:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tokenizers is required when --tokenizer-path is supplied") from exc
    return Tokenizer.from_file(str(path))


def build_corpus(
    *,
    output: Path,
    sources: Iterable[Mapping[str, Any]] = DEFAULT_SOURCES,
    historical_ledgers: Iterable[Path] = (),
    max_chars: int = 12_000,
    max_records_per_source: int | None = None,
    seed: int = 23,
    acquired_at: str | None = None,
    tokenizer_path: Path | None = None,
    opener: Callable[..., Any] | None = None,
    capture_commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Acquire sources and write ``raw/`` plus ``prepared/`` run artifacts."""

    normalized_sources = [_as_source(source) for source in sources]
    source_tiers = _assign_source_tiers(normalized_sources, seed=seed)
    timestamp = acquired_at or _now_utc()
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("acquired_at must be an ISO-8601 timestamp") from exc
    output = Path(output)
    raw_dir, prepared_dir = output / "raw", output / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    for source in sorted(normalized_sources, key=lambda item: str(item["source_id"])):
        item = download_source(source, raw_dir=raw_dir, opener=opener)
        item["tier"] = source_tiers[item["source_id"]]
        item["acquisition_time_utc"] = timestamp
        item["capture_command"] = str(source.get("capture_command", ""))
        downloaded.append(item)

    historical_paths = [Path(path) for path in historical_ledgers]
    historical_rows, historical_metadata = load_v22_rows(historical_paths)
    tokenizer = _load_tokenizer(tokenizer_path)
    prepared: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    discarded_short_records = 0
    for source in downloaded:
        raw_text = Path(source["raw_path"]).read_text(encoding="utf-8", errors="replace")
        raw_text = _prepare_source_text(source, raw_text)
        for record_index, text in enumerate(_chunks(raw_text, max_chars=max_chars)):
            # Gutenberg exports contain many standalone headings/markers
            # (e.g. ``Contents``, ``Yes.``, ``THE END``).  They are not useful
            # calibration records and, when repeated across books, would
            # create a false cross-tier identity/near-duplicate collision.
            # Keep all substantive prose; record the deterministic filter in
            # the acquisition receipt for auditability.
            if str(source["source_id"]).casefold().startswith("gutenberg-") and len(_normalise_text(text).split()) < 8:
                discarded_short_records += 1
                continue
            normalized_hash = normalized_text_hash(text)
            if not text.strip() or normalized_hash in seen_normalized:
                continue
            seen_normalized.add(normalized_hash)
            source_id = str(source["source_id"])
            row: dict[str, Any] = {
                "text": text,
                "tier": str(source["tier"]),
                "split": str(source["tier"]),
                "source_id": source_id,
                "source_name": str(source["name"]),
                "source_url": str(source["source_url"]),
                "source_revision": str(source["source_revision"]),
                "source_license": str(source["source_license"]),
                "download_sha256": str(source["download_sha256"]),
                "source_file_sha256": str(source["source_file_sha256"]),
                "acquisition_time_utc": timestamp,
                "source_record_id": f"{source_id}:record:{record_index:06d}",
                "source_record_index": record_index,
                "source_file": str(Path(source["raw_path"]).relative_to(output).as_posix()),
                "content_sha256": content_hash(text),
                "normalized_content_sha256": normalized_hash,
                "group_identity": f"v23-source:{source_id}",
                "split_group": f"v23-source:{source_id}",
                "source_lineage": f"v23-source:{source_id}",
                # A prepared chunk is an independent sampling task/document
                # while ``group_identity``/``source_lineage`` retain the
                # source-level split boundary.  Keeping task identities per
                # record lets the activation planner apply its 4k/task cap
                # without collapsing an entire book into one tiny budget.
                "task_id": f"v23-task:{source_id}:{record_index:06d}",
                "tree_id": f"v23-tree:{source_id}:{record_index:06d}",
                "trajectory_id": f"v23-trajectory:{source_id}:{record_index:06d}",
                "source_family": f"v23-source-family:{source_id}",
                "document_id": f"v23-document:{source_id}:{record_index:06d}",
                "repository_path_revision": f"{source_id}|{source['source_revision']}",
                "domain": str(source["domain"]),
                "task_family": str(source["domain"]),
                "selection_rationale": str(source["rationale"]),
                "benchmark_membership": [],
                "benchmark_context": [],
                "v23_method_version": V23_METHOD_VERSION,
                "v23_parent": "none",
            }
            row["id"] = _canonical_id(row)
            if tokenizer is not None:
                encoded = tokenizer.encode(text, add_special_tokens=False)
                row["token_count"] = len(getattr(encoded, "ids", encoded))
            prepared.append(row)

    if not prepared:
        raise ValueError("public sources produced no usable records")
    overlap = audit_v22_overlap(prepared, historical_rows=historical_rows, historical_paths=historical_paths)
    if overlap["status"] != "PASS":
        raise ValueError(f"fresh V2.3 acquisition overlaps V2.2 identity ledgers: {overlap['conflicts'][:3]}")
    tier_audit = audit_v23_tier_disjointness(prepared)
    if tier_audit["status"] != "PASS":
        raise ValueError(f"source/group assignment is not tier-disjoint: {tier_audit['group_conflicts'][:3]}")

    prepared.sort(key=lambda row: (str(row["tier"]), str(row["source_id"]), int(row["source_record_index"]), str(row["id"])))
    for index, row in enumerate(prepared):
        row["prepared_record_index"] = index
    prepared_path = prepared_dir / "v23-prepared.jsonl"
    prepared_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in prepared)
    prepared_hash = write_immutable_text(prepared_path, prepared_text)
    source_receipts = []
    for source in downloaded:
        source_receipts.append(
            {
                "source_id": source["source_id"],
                "name": source["name"],
                "url": source["url"],
                "source_url": source["source_url"],
                "source_revision": source["source_revision"],
                "source_license": source["source_license"],
                "domain": source["domain"],
                "tier": source["tier"],
                "raw_path": str(Path(source["raw_path"]).relative_to(output).as_posix()),
                "raw_sha256": source["raw_sha256"],
                "download_sha256": source["download_sha256"],
                "source_file_sha256": source["source_file_sha256"],
                "bytes": source["bytes"],
                "record_count": sum(1 for row in prepared if row["source_id"] == source["source_id"]),
                "acquisition_time_utc": timestamp,
                "rationale": source["rationale"],
            }
        )
    receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-v23-acquisition-receipt",
        "status": "V23_PUBLIC_SOURCES_ACQUIRED",
        "corpus_version": "v2.3",
        "method_version": V23_METHOD_VERSION,
        "acquisition_time_utc": timestamp,
        "sources": sorted(source_receipts, key=lambda item: str(item["source_id"])),
        "prepared": {"path": str(prepared_path.relative_to(output).as_posix()), "sha256": prepared_hash, "records": len(prepared)},
        "tiers": {tier: sum(1 for row in prepared if row["tier"] == tier) for tier in DEVELOPMENT_TIERS},
        "historical_v22_ledgers": historical_metadata,
        "historical_overlap_audit": overlap,
        "tier_disjointness_audit": tier_audit,
        "preparation": {
            "gutenberg_wrapper_trimmed": True,
            "short_gutenberg_records_discarded": discarded_short_records,
            "short_gutenberg_record_policy": "discard normalized chunks with fewer than 8 whitespace tokens after wrapper trim",
        },
        "exact_capture_commands": list(capture_commands),
        "policy": "public-only; permissive licenses; whole-source groups; no V2.2 evaluation or promotion data",
    }
    receipt_path = prepared_dir / "acquisition-receipt.json"
    receipt_hash = write_immutable_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output": str(output),
        "prepared_path": str(prepared_path),
        "prepared_sha256": prepared_hash,
        "prepared_records": len(prepared),
        "tier_counts": receipt["tiers"],
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": receipt_hash,
        "historical_overlap": overlap,
        "tier_disjointness": tier_audit,
    }


def _load_source_specs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("sources", payload.get("records", []))
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("source spec must be a JSON list or an object with a sources list")
    return [dict(item) for item in payload]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-specs", type=Path)
    parser.add_argument("--v22-ledger", type=Path, action="append", default=None)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--max-records-per-source", type=int)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--acquired-at")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    source_specs = _load_source_specs(args.source_specs) if args.source_specs else list(DEFAULT_SOURCES)
    historical = args.v22_ledger if args.v22_ledger is not None else discover_v22_ledgers(ROOT)
    command = "python " + " ".join(json.dumps(str(item)) for item in ([Path(__file__).resolve(), *(argv or sys.argv[1:])]))
    result = build_corpus(
        output=args.output,
        sources=source_specs,
        historical_ledgers=historical,
        max_chars=args.max_chars,
        max_records_per_source=args.max_records_per_source,
        seed=args.seed,
        acquired_at=args.acquired_at,
        tokenizer_path=args.tokenizer_path,
        capture_commands=(command,),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
