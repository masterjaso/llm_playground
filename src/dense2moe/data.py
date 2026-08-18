"""Reproducible calibration corpus preparation.

The calibration corpus is deliberately represented by a small, auditable
manifest rather than copied into a run directory.  A record in the manifest
contains enough information to reopen the original source record, verify its
content hash, and recount it with the exact tokenizer used to produce the
manifest.  Scientific preparation therefore never silently falls back to a
word/whitespace estimate.

The module does not download datasets.  Callers may download an approved
public source using their normal data tooling, then provide the resulting
JSON/JSONL/TSV source file here.  ``PUBLIC_DATASET_CATALOG`` records the small
set of permissive, ungated sources selected for the initial vertical slice;
per-record source metadata is still required and is written into the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

# These are source families suitable for a small, public, ungated calibration
# mixture.  The catalog is metadata only; no source is implicitly downloaded
# or trusted without a local file hash and record-level provenance.
PUBLIC_DATASET_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "openai/openai_humaneval",
        "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "license": "MIT",
        "domain": "code",
        "access": "public-ungated",
        "url": "https://huggingface.co/datasets/openai/openai_humaneval",
        "rationale": "Small MIT-licensed Python code-generation problems for code coverage.",
    },
    {
        "name": "openai/gsm8k",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "license": "MIT",
        "domain": "reasoning/math",
        "access": "public-ungated",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "rationale": "Human-authored grade-school math problems with reasoning traces.",
    },
    {
        "name": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "domain": "instruction/dialogue",
        "access": "public-ungated",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "rationale": "Public instruction and dialogue turns for conversational coverage.",
    },
    {
        "name": "Project Gutenberg public-domain texts",
        "revision": "ebook-1342",
        "license": "Public Domain",
        "domain": "general",
        "access": "public-ungated",
        "url": "https://www.gutenberg.org/",
        "rationale": "Public-domain books selected by stable Gutenberg ebook ID for general prose.",
    },
    {
        "name": "Project Gutenberg public-domain texts",
        "revision": "ebook-1342",
        "license": "Public Domain",
        "domain": "long-context",
        "access": "public-ungated",
        "url": "https://www.gutenberg.org/",
        "rationale": "Long public-domain books selected by stable ebook ID and measured token length.",
    },
    {
        "name": "nvidia/Open-SWE-Traces",
        "revision": "ad4805a5aa7de70d99cab0bb8f99b15304c76de0",
        "license": "CC-BY-4.0",
        "domain": "agentic/software-engineering",
        "access": "public-ungated",
        "url": "https://huggingface.co/datasets/nvidia/Open-SWE-Traces",
        "rationale": "Pinned Qwen3.5 non-thinking OpenHands/SWE-agent trajectories used only as visible activation contexts.",
    },
    {
        "name": "permissive pinned repository files",
        "revision": "commit-pinned-per-record",
        "license": "MIT/Apache-2.0/BSD-3-Clause/PSF-2.0",
        "domain": "code/repository",
        "access": "public-ungated",
        "url": "https://github.com/",
        "rationale": "Selected implementation, test, documentation, and configuration files retain repository commit and path provenance.",
    },
)

_PERMISSIVE_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CDLA-Permissive-1.0",
    "CDLA-Permissive-2.0",
    "MIT",
    "ODC-By-1.0",
    "PSF-2.0",
    "Public Domain",
    "public-domain",
    "MIT/Apache-2.0/BSD-3-Clause/PSF-2.0",
}
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}

# Provenance fields that are meaningful to the corpus-v2 audit in addition to
# the source identity fields used by the original two-way calibration split.
# They are intentionally carried through into each normalized record instead
# of being dropped when a source row is reopened.
_CORPUS_V2_METADATA_FIELDS = (
    "source_family",
    "task_family",
    "trajectory_framework",
    "trajectory_model",
    "trajectory_reasoning",
    "language",
    "repository_id",
    "repo",
    "repo_commit",
    "repo_path",
    "split_group",
    "document_id",
    "task_id",
    "trajectory_id",
    "segment_index",
    "segment_count",
    "source_commit_sha",
    "source_path",
    "benchmark_membership",
    "benchmark_context",
    "benchmark_denylist",
    "terms",
    "upstream_license",
    "source_artifact_url",
)

# Corpus V2.1 intentionally lives beside the frozen V2 rather than replacing
# it.  Keep these names in the shared data module so scripts and downstream
# capture code agree on the role contract.
V21_SPLITS = (
    "FIT-TRAIN",
    "FIT-DEV",
    "GATE-A",
    "SHADOW-B",
    "SHADOW-C",
    "PRESERVATION-CANARY",
)
V21_OPTIMIZATION_SPLITS = ("FIT-TRAIN", "FIT-DEV")
V21_PROMOTION_SPLITS = ("GATE-A", "SHADOW-B", "SHADOW-C")
V21_QUARANTINE_SPLIT = "BENCHMARK-CANARY-EXCLUDED"
V21_TARGET_FRACTIONS: dict[str, float] = {
    "code": 0.44,
    "agentic-software-engineering": 0.28,
    "software-engineering-natural-language": 0.12,
    "structured": 0.06,
    "general": 0.10,
}
V21_BENCHMARK_TERMS = frozenset(
    {
        "swe-bench",
        "swebench",
        "swe-rebench",
        "human-eval",
        "humaneval",
        "mbpp",
        "ds-1000",
        "cruxeval",
        "livecodebench",
    }
)

# Corpus V2.2 is deliberately a new protocol rather than a relabeling of the
# historical V2.1 artifacts.  Development and evaluation identities are
# frozen separately so a later method revision cannot silently reuse an opened
# promotion corpus.
V22_TIER_ORDER = (
    "FIT-TRAIN",
    "FIT-DEV",
    "GATE-A",
    "SHADOW-B",
    "SHADOW-C",
    "G1",
    "G2",
    "PRESERVATION-CANARY",
    "BENCHMARK-CANARY-EXCLUDED",
)
V22_DEVELOPMENT_TIERS = ("FIT-TRAIN", "FIT-DEV")
V22_INTERNAL_PROMOTION_TIERS = ("GATE-A", "SHADOW-B", "SHADOW-C")
V22_EXTERNAL_TIERS = ("G1", "G2")
V22_EVALUATION_TIERS = V22_INTERNAL_PROMOTION_TIERS + V22_EXTERNAL_TIERS


class _TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Any: ...


Tokenizer = _TokenizerLike | Callable[..., Any]


@dataclass(frozen=True)
class _SourceRecord:
    value: dict[str, Any]
    source_file: Path
    source_record_index: int


# Resolving a manifest record must reopen the source for verification, but a
# large fresh JSONL source should not be parsed once per selected record.  The
# cache key includes size and mtime so a changed source cannot reuse stale
# records; the bounded table keeps long-lived CLI processes from accumulating
# arbitrary corpus files.
_SOURCE_RECORD_CACHE: dict[tuple[str, int, int], tuple[_SourceRecord, ...]] = {}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalize_text(text: str) -> str:
    """Normalize only for duplicate detection; preserve raw text for hashing."""

    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return " ".join(normalized.split())


def _content_hashes(text: str) -> tuple[str, str]:
    raw = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized = hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()
    return raw, normalized


def _example_id(text: str, index: int) -> str:
    """Compatibility helper for callers that used the old private function."""

    return hashlib.sha256(f"{index}:{text}".encode()).hexdigest()[:24]


def _top_level_source_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    source = value.get("source", value.get("dataset", {}))
    metadata: dict[str, Any] = {}
    if isinstance(source, Mapping):
        metadata.update({str(key): item for key, item in source.items()})
    elif isinstance(source, str):
        metadata["name"] = source
    for key in (
        "name",
        "source_name",
        "dataset",
        "revision",
        "version",
        "license",
        "url",
        "download_sha256",
        "rationale",
        "domain",
    ):
        if key in value and value[key] is not None:
            metadata[key] = value[key]
    if "source_name" in metadata and "name" not in metadata:
        metadata["name"] = metadata["source_name"]
    if "dataset" in metadata and "name" not in metadata:
        metadata["name"] = metadata["dataset"]
    return metadata


def _read_source_records(path: Path) -> tuple[list[_SourceRecord], dict[str, Any]]:
    """Read records and preserve their physical source location.

    JSONL line numbers are used as record indexes (including blank lines),
    which makes a locator stable even when records contain no explicit ID.
    JSON arrays and TSV rows use their zero-based item/row index.
    """

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    records: list[_SourceRecord] = []
    metadata: dict[str, Any] = {}
    if suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                records.append(_SourceRecord(dict(value), path, line_number))
        return records, metadata
    if suffix == ".tsv":
        lines = path.read_text(encoding="utf-8").splitlines()
        headers = lines[0].split("\t") if lines else []
        for row_index, row in enumerate(lines[1:]):
            records.append(_SourceRecord(dict(zip(headers, row.split("\t"))), path, row_index))
        return records, metadata
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        metadata = _top_level_source_metadata(value)
        value = value.get("examples", value.get("records", []))
    if not isinstance(value, list):
        raise TypeError("corpus manifest must contain a list of examples")
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            records.append(_SourceRecord(dict(item), path, index))
    return records, metadata


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Backward-compatible record reader used by older callers/tests."""

    return [record.value for record in _read_source_records(path)[0]]


def _source_path(value: Any, base_dir: Path) -> Path:
    raw = str(value or "")
    if not raw:
        raise ValueError("every corpus record must contain a resolvable source_file")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _record_text(value: Mapping[str, Any], source_file: Path, source_record_index: int) -> str:
    """Recover text from an already loaded source record."""

    for key in ("text", "content", "prompt", "question"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            if key == "question" and isinstance(value.get("answer"), str):
                return f"{item}\n{value['answer']}"
            return item
    messages = value.get("messages", value.get("conversations"))
    if isinstance(messages, list):
        chunks: list[str] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role", message.get("from", ""))
                content = message.get("content", message.get("value", ""))
                if content:
                    chunks.append(f"{role}: {content}" if role else str(content))
            elif message:
                chunks.append(str(message))
        if chunks:
            return "\n".join(chunks)
    raise ValueError(f"source record has no resolvable text: {source_file}:{source_record_index}")


def _locate_external_record(path: Path, record_index: int, record_id: str | None) -> dict[str, Any]:
    stat = path.stat()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    records_tuple = _SOURCE_RECORD_CACHE.get(cache_key)
    if records_tuple is None:
        records_tuple = tuple(_read_source_records(path)[0])
        if len(_SOURCE_RECORD_CACHE) >= 32:
            _SOURCE_RECORD_CACHE.pop(next(iter(_SOURCE_RECORD_CACHE)))
        _SOURCE_RECORD_CACHE[cache_key] = records_tuple
    records = list(records_tuple)
    for item in records:
        if item.source_record_index == record_index:
            if record_id is None:
                return item.value
            candidate = item.value.get("source_record_id", item.value.get("record_id", item.value.get("id")))
            if candidate is not None and str(candidate) == record_id:
                return item.value
            if candidate is None and record_id == str(record_index):
                return item.value
            break
    if record_id:
        for item in records:
            candidate = item.value.get("source_record_id", item.value.get("record_id", item.value.get("id")))
            if candidate is not None and str(candidate) == record_id:
                return item.value
    raise ValueError(f"source record locator not found: {path}:{record_index}")


def _load_tokenizer(
    tokenizer: Tokenizer | None,
    *,
    tokenizer_path: str | Path | None,
    source_snapshot: str | Path | None,
    tokenizer_revision: str,
) -> Tokenizer:
    if tokenizer is not None:
        return tokenizer
    path = tokenizer_path or source_snapshot
    if path is None:
        raise ValueError(
            "exact source-tokenizer counts require --tokenizer-path or --source-snapshot; "
            "whitespace counts are not scientific calibration evidence"
        )
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the pinned source tokenizer") from exc
    local_path = Path(path)
    kwargs: dict[str, Any] = {"trust_remote_code": False, "use_fast": True}
    if local_path.exists():
        kwargs["local_files_only"] = True
        loaded = AutoTokenizer.from_pretrained(str(local_path), **kwargs)
    else:
        if tokenizer_revision in {"", "main", "declared-by-input", "unknown"}:
            raise ValueError("remote tokenizer loading requires a pinned revision, not a moving alias")
        loaded = AutoTokenizer.from_pretrained(str(path), revision=tokenizer_revision, **kwargs)
    return cast(Tokenizer, loaded)


def _tokenizer_files(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return []
    files: list[dict[str, str]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.name in _TOKENIZER_FILENAMES:
            files.append({"path": candidate.relative_to(root).as_posix(), "sha256": sha256_file(candidate)})
    return files


def _tokenizer_metadata(
    tokenizer: Tokenizer | None,
    *,
    tokenizer_path: str | Path | None,
    source_snapshot: str | Path | None,
    tokenizer_revision: str,
    supplied: Mapping[str, Any] | None,
    add_special_tokens: bool,
) -> dict[str, Any]:
    files = list((supplied or {}).get("files", []))
    if not files:
        files = _tokenizer_files(tokenizer_path or source_snapshot)
    normalized_files: list[dict[str, str]] = []
    for item in files:
        if isinstance(item, Mapping) and item.get("path") and item.get("sha256"):
            normalized_files.append({"path": str(item["path"]), "sha256": str(item["sha256"])})
    attrs: dict[str, Any] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            attrs[name] = int(value) if isinstance(value, (int, float)) else str(value)
    for name in ("add_bos_token", "add_eos_token"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            attrs[name] = bool(value)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_text = json.dumps(chat_template, sort_keys=True, ensure_ascii=False) if chat_template else ""
    supplied_metadata = dict(supplied or {})
    supplied_metadata.pop("files", None)
    snapshot = str(source_snapshot or tokenizer_path or supplied_metadata.get("source_snapshot", ""))
    if not snapshot and tokenizer is not None:
        tokenizer_type = type(tokenizer)
        snapshot = f"loaded:{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"
    result: dict[str, Any] = {
        "source_snapshot": snapshot,
        "revision": tokenizer_revision,
        "revision_pinned": bool(_PINNED_REVISION.fullmatch(tokenizer_revision)),
        "files": normalized_files,
        "files_sha256": _digest_json(normalized_files),
        "special_tokens": attrs,
        "add_special_tokens": bool(add_special_tokens),
        "bos_eos_handling": {
            "mode": "tokenizer-special-tokens" if add_special_tokens else "none",
            "bos_added": bool(add_special_tokens and attrs.get("bos_token_id") is not None and attrs.get("add_bos_token", True)),
            "eos_added": bool(add_special_tokens and attrs.get("eos_token_id") is not None and attrs.get("add_eos_token", True)),
        },
        "chat_template": {
            "used": False,
            "available": bool(chat_template),
            "sha256": hashlib.sha256(chat_template_text.encode("utf-8")).hexdigest() if chat_template_text else None,
            "behavior": "plain-text-records; chat template not applied",
        },
    }
    for key, value in supplied_metadata.items():
        if key not in result:
            result[key] = value
    return result


def _encode(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    kwargs = {"add_special_tokens": add_special_tokens}
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text, **kwargs)  # type: ignore[attr-defined]
    else:
        try:
            result = tokenizer(text, **kwargs)  # type: ignore[operator]
        except TypeError:
            if add_special_tokens:
                raise
            # A minimal test double may expose only ``tokenizer(text)``.  The
            # scientific path still passes the explicit setting first; this
            # fallback is safe only when special-token insertion is disabled.
            result = tokenizer(text)  # type: ignore[operator]
    if hasattr(result, "input_ids"):
        result = result.input_ids
    # The lightweight ``tokenizers.Tokenizer`` used by the frozen corpus
    # receipt returns an Encoding object whose exact IDs live on ``.ids``;
    # Hugging Face tokenizers commonly return a list or BatchEncoding.
    if hasattr(result, "ids"):
        result = result.ids
    if isinstance(result, Mapping):
        result = result.get("input_ids")
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result is None or isinstance(result, (str, bytes)):
        raise TypeError("tokenizer did not return input_ids")
    values = list(result)
    if values and isinstance(values[0], list):
        values = list(values[0])
    if not all(isinstance(item, (int, float)) for item in values):
        raise TypeError("tokenizer input_ids must be numeric")
    return [int(item) for item in values]


def _source_metadata_for_record(value: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    source_value = value.get("source", value.get("dataset"))
    if isinstance(source_value, Mapping):
        metadata = dict(inherited)
        metadata.update({str(key): item for key, item in source_value.items()})
    else:
        metadata = dict(inherited)
        if source_value is not None:
            metadata["name"] = source_value
    aliases = {
        "source_name": "name",
        "dataset": "name",
        "source_revision": "revision",
        "source_version": "version",
        "source_license": "license",
        "source_url": "url",
    }
    for source_key, target_key in aliases.items():
        if source_key in value:
            metadata[target_key] = value[source_key]
    for key in ("name", "revision", "version", "license", "url", "download_sha256", "rationale", "domain"):
        if key in value and value[key] is not None:
            metadata[key] = value[key]
    for key in _CORPUS_V2_METADATA_FIELDS:
        if key in value and value[key] is not None:
            metadata[key] = value[key]
    return metadata


def _validate_source_metadata(metadata: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    name = str(metadata.get("name", "")).strip()
    revision = str(metadata.get("revision", metadata.get("version", ""))).strip()
    license_name = str(metadata.get("license", "")).strip()
    license_aliases = {
        "Apache 2.0": "Apache-2.0",
        "Apache License 2.0": "Apache-2.0",
        "BSD-2-Clause": "BSD-2-Clause",
        "BSD-3-Clause": "BSD-3-Clause",
        "CC BY 4.0": "CC-BY-4.0",
        "CC-BY 4.0": "CC-BY-4.0",
        "CC BY-SA 4.0": "CC-BY-SA-4.0",
        "CC-BY-SA 4.0": "CC-BY-SA-4.0",
        "CDLA-Permissive 2.0": "CDLA-Permissive-2.0",
        "MIT License": "MIT",
        "ODC-By": "ODC-By-1.0",
        "ODC-By 1.0": "ODC-By-1.0",
    }
    license_name = license_aliases.get(license_name, license_name)
    if not name:
        raise ValueError("every source record must identify a dataset/source name")
    if not license_name or license_name.lower() in {"unknown", "unclear", "n/a", "none"}:
        raise ValueError(f"source {name!r} has no explicit license; unclear licenses are rejected")
    if strict and license_name not in _PERMISSIVE_LICENSES:
        raise ValueError(f"source {name!r} license is not approved for this corpus: {license_name!r}")
    if not revision:
        raise ValueError(f"source {name!r} has no revision/version")
    rationale = str(metadata.get("rationale", "")).strip()
    if not rationale:
        raise ValueError(f"source {name!r} has no selection rationale")
    result: dict[str, Any] = {
        "name": name,
        "revision": revision,
        "license": license_name,
        "url": str(metadata.get("url", "")),
        "rationale": rationale,
        "domain": str(metadata.get("domain", "general")),
    }
    for key in _CORPUS_V2_METADATA_FIELDS:
        if key in metadata and metadata[key] is not None:
            result[key] = metadata[key]
    return result


def _relative_locator(source_file: Path, base_dir: Path) -> str:
    try:
        return source_file.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return os.path.abspath(source_file)


def _canonical_repo_path(path: str | Path) -> str:
    """Return a separator- and output-location-stable provenance path.

    Corpus identity is scientific content, not the spelling of a Windows
    drive path.  Repository-local sources are represented relative to the
    checkout using POSIX separators.  External fixtures retain a compact
    forward-slash path for human provenance, while their content hash remains
    the authoritative identity component.
    """

    candidate = Path(path)
    repository = Path(__file__).resolve().parents[2]
    try:
        return candidate.resolve().relative_to(repository).as_posix()
    except ValueError:
        if not candidate.is_absolute():
            return candidate.as_posix()
        return candidate.as_posix()


def _resolve_record_value(record: Mapping[str, Any], *, base_dir: Path) -> tuple[str, Path, dict[str, Any]]:
    source_file = _source_path(record.get("source_file"), base_dir)
    source_index = int(record.get("source_record_index", 0))
    source_record_id = record.get("source_record_id", record.get("record_id"))
    source_value = dict(record)
    # A manifest can point at a separate source file.  Always reopen it when
    # possible so a copied text field cannot bypass locator verification.
    if source_file.exists():
        actual = _locate_external_record(source_file, source_index, str(source_record_id) if source_record_id is not None else None)
        source_value = actual
    text = _record_text(source_value, source_file, source_index)
    return text, source_file, source_value


def resolve_corpus_record(
    record: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    tokenizer: Tokenizer | None = None,
    add_special_tokens: bool | None = None,
) -> str:
    """Reopen and verify one manifest record, returning its source text.

    ``tokenizer`` is optional for a cheap content-only verification.  When it
    is provided, the exact count in the manifest is checked as well.
    """

    root = Path(base_dir or ".")
    text, source_file, source_value = _resolve_record_value(record, base_dir=root)
    source_digest = str(record.get("source_file_sha256", ""))
    if source_digest and sha256_file(source_file) != source_digest:
        raise ValueError(f"source file hash mismatch: {source_file}")
    raw_hash, normalized_hash = _content_hashes(text)
    if raw_hash != str(record.get("content_sha256", record.get("text_sha256", ""))):
        raise ValueError(f"content hash mismatch: {source_file}:{record.get('source_record_index')}")
    expected_normalized = str(record.get("normalized_content_sha256", ""))
    if expected_normalized and normalized_hash != expected_normalized:
        raise ValueError(f"normalized content hash mismatch: {source_file}:{record.get('source_record_index')}")
    if tokenizer is not None:
        expected_special = bool(record.get("add_special_tokens", False) if add_special_tokens is None else add_special_tokens)
        actual_count = len(_encode(tokenizer, text, add_special_tokens=expected_special))
        if actual_count != int(record.get("token_count", -1)):
            raise ValueError(
                f"token count mismatch for {record.get('id', source_value.get('id'))}: "
                f"manifest={record.get('token_count')} actual={actual_count}"
            )
    return text


def verify_corpus_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
    tokenizer: Tokenizer | None = None,
) -> dict[str, Any]:
    """Resolve every selected record and return bounded verification evidence."""

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if base_dir is not None:
            root = Path(base_dir)
        else:
            recorded_root = payload.get("resolvability", {}).get("base_dir") if isinstance(payload, Mapping) else None
            root = Path(str(recorded_root)) if recorded_root else path.parent
    else:
        payload = dict(manifest)
        if base_dir is not None:
            root = Path(base_dir)
        else:
            recorded_root = payload.get("resolvability", {}).get("base_dir")
            root = Path(str(recorded_root)) if recorded_root else Path(".")
    if not isinstance(payload, Mapping):
        raise TypeError("corpus manifest must be an object")
    checked: list[str] = []
    counts: dict[str, int] = {}
    for split in ("train", "holdout"):
        entries = payload.get(split, [])
        if not isinstance(entries, list):
            raise TypeError(f"manifest {split} must be a list")
        total = 0
        for record in entries:
            if not isinstance(record, Mapping):
                raise TypeError(f"manifest {split} contains a non-object record")
            resolve_corpus_record(record, base_dir=root, tokenizer=tokenizer)
            identifier = str(record.get("id", ""))
            if not identifier:
                raise ValueError(f"manifest {split} record has no stable id")
            checked.append(identifier)
            total += int(record.get("token_count", 0))
        counts[split] = total
    if len(checked) != len(set(checked)):
        raise ValueError("train and holdout contain duplicate stable IDs")
    return {
        "status": "CORPUS_VERIFIED",
        "records": len(checked),
        "train_records": len(payload.get("train", [])),
        "holdout_records": len(payload.get("holdout", [])),
        "train_tokens": counts["train"],
        "holdout_tokens": counts["holdout"],
        "ids_sha256": _digest_json(sorted(checked)),
    }


def write_corpus_receipt(
    manifest: Mapping[str, Any] | str | Path,
    manifest_path: str | Path | None = None,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Write a machine-readable, bounded provenance receipt for a manifest."""

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = dict(manifest)
        if manifest_path is None:
            raise ValueError("manifest_path is required when passing an in-memory manifest")
        path = Path(manifest_path)
    if not isinstance(payload, Mapping):
        raise TypeError("corpus manifest must be an object")
    target = Path(output) if output is not None else path.with_name("corpus-receipt.json")
    selected: list[dict[str, Any]] = []
    for split in ("train", "holdout"):
        entries = payload.get(split, [])
        if not isinstance(entries, list):
            continue
        for record in entries:
            if not isinstance(record, Mapping):
                continue
            selected_record: dict[str, Any] = {
                "id": str(record.get("id", "")),
                "split": split,
                "source": str(record.get("source_name", record.get("source", ""))),
                "source_revision": str(record.get("source_revision", record.get("revision", ""))),
                "license": str(record.get("license", "")),
                "download_sha256": str(record.get("download_sha256", record.get("source_file_sha256", ""))),
                "rationale": str(record.get("selection_rationale", record.get("rationale", ""))),
                "source_record_id": str(record.get("source_record_id", "")),
                "source_record_index": int(record.get("source_record_index", 0)),
                "source_file": str(record.get("source_file", "")),
                "source_file_sha256": str(record.get("source_file_sha256", "")),
                "content_sha256": str(record.get("content_sha256", record.get("text_sha256", ""))),
                "normalized_content_sha256": str(record.get("normalized_content_sha256", "")),
                "token_count": int(record.get("token_count", 0)),
                "domain": str(record.get("domain", "general")),
            }
            for key in _CORPUS_V2_METADATA_FIELDS:
                if key in record:
                    selected_record[key] = record[key]
            selected.append(selected_record)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "dense2moe-corpus-receipt",
        "manifest": {
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
            "schema_version": payload.get("schema_version"),
            "dataset_hash": payload.get("dataset_hash"),
        },
        "sources": payload.get("sources", []),
        "tokenizer": payload.get("tokenizer", {}),
        "selection": {
            "rationale": "Deterministic public-source mixture with stable content deduplication and hash-based disjoint splits.",
            "records": selected,
            "record_ids_sha256": _digest_json([item["id"] for item in selected]),
            "train_tokens": payload.get("train_tokens", 0),
            "holdout_tokens": payload.get("holdout_tokens", 0),
            "train_ids_sha256": payload.get("train_ids_sha256"),
            "holdout_ids_sha256": payload.get("holdout_ids_sha256"),
        },
        "resolvability": {
            "locator": "source_file + source_record_index/source_record_id",
            "verification": "reopen source file, verify source file/content hashes, recount with pinned tokenizer",
            "text_embedded": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


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
    tokenizer: Tokenizer | None = None,
    tokenizer_path: str | Path | None = None,
    source_snapshot: str | Path | None = None,
    tokenizer_metadata: Mapping[str, Any] | None = None,
    add_special_tokens: bool = False,
    receipt_output: str | Path | None = None,
    strict_sources: bool = True,
    required_domains: Sequence[str] | None = None,
    allow_legacy_token_counts: bool = False,
) -> dict[str, Any]:
    """Create a deterministic, exact-tokenized, resolvable corpus manifest.

    ``tokenizer`` is intended for tests or an already-loaded tokenizer from an
    immutable local snapshot.  Production callers should pass
    ``tokenizer_path``/``source_snapshot`` so tokenizer files and hashes are
    recorded.  The legacy declared-count path is opt-in and explicitly marked
    non-scientific; it is retained only for old callers that have not migrated.
    """

    if train_tokens <= 0 or holdout_tokens <= 0 or sequence_length <= 0:
        raise ValueError("token and sequence-length targets must be positive")
    source = Path(corpus_manifest)
    source_records, inherited_source = _read_source_records(source)
    loaded_tokenizer: Tokenizer | None = tokenizer
    exact = True
    if loaded_tokenizer is None and tokenizer_path is None and source_snapshot is None:
        if not allow_legacy_token_counts:
            _load_tokenizer(
                None,
                tokenizer_path=tokenizer_path,
                source_snapshot=source_snapshot,
                tokenizer_revision=tokenizer_revision,
            )
        exact = False
    if loaded_tokenizer is None and allow_legacy_token_counts:
        exact = False
    elif loaded_tokenizer is None:
        loaded_tokenizer = _load_tokenizer(
            None,
            tokenizer_path=tokenizer_path,
            source_snapshot=source_snapshot,
            tokenizer_revision=tokenizer_revision,
        )
    if loaded_tokenizer is None and exact:
        raise ValueError("exact source tokenizer could not be loaded")
    tokenizer_info = _tokenizer_metadata(
        loaded_tokenizer,
        tokenizer_path=tokenizer_path,
        source_snapshot=source_snapshot,
        tokenizer_revision=tokenizer_revision,
        supplied=tokenizer_metadata,
        add_special_tokens=add_special_tokens,
    )
    if exact and tokenizer_revision.strip() in {"", "main", "declared-by-input", "unknown"}:
        raise ValueError("tokenizer revision is required")
    if exact and tokenizer_info["source_snapshot"] == "" and not tokenizer_metadata:
        raise ValueError("exact tokenizer metadata must identify a source snapshot or supplied file hashes")

    normalized: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    source_catalog: dict[str, dict[str, Any]] = {}
    for source_record in source_records:
        item = source_record.value
        metadata = _source_metadata_for_record(item, inherited_source)
        source_info = _validate_source_metadata(metadata, strict=strict_sources)
        source_name = source_info["name"]
        # Resolve the locator once before passing it back through the verifier.
        # Keeping this absolute internally avoids joining ``source.parent`` a
        # second time when a record explicitly names the manifest's own JSONL
        # file (for example ``data/public_v2/corpus.jsonl``).
        source_file = _source_path(item.get("source_file", source_record.source_file), source.parent).resolve()
        source_index = int(item.get("source_record_index", source_record.source_record_index))
        source_record_id_value = item.get("source_record_id", item.get("record_id", item.get("id")))
        source_record_id = str(source_record_id_value) if source_record_id_value is not None else str(source_index)
        text, resolved_file, _ = _resolve_record_value(
            {
                **item,
                "source_file": str(source_file),
                "source_record_index": source_index,
                "source_record_id": source_record_id,
            },
            base_dir=source.parent,
        )
        raw_hash, normalized_hash = _content_hashes(text)
        # Deduplicate by normalized content, not by array position or source ID.
        if normalized_hash in seen_content:
            continue
        seen_content.add(normalized_hash)
        if loaded_tokenizer is not None and exact:
            token_count = len(_encode(loaded_tokenizer, text, add_special_tokens=add_special_tokens))
        elif allow_legacy_token_counts:
            declared = item.get("token_count", item.get("tokens"))
            if declared is None:
                raise ValueError("legacy token-count mode requires token_count on every record")
            token_count = int(declared)
        else:  # pragma: no cover - guarded above
            raise AssertionError("tokenizer selection invariant violated")
        if token_count <= 0:
            continue
        source_file_digest = sha256_file(resolved_file)
        supplied_download_digest = str(metadata.get("download_sha256", ""))
        stable_seed = f"{source_name}@{source_info['revision']}:{source_record_id}:{raw_hash}"
        stable_id = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:32]
        catalog_key = (
            f"{source_name}@{source_info['revision']}:{supplied_download_digest or source_file_digest}:"
            f"{source_info.get('source_family', '')}"
        )
        source_catalog[catalog_key] = {
            **source_info,
            "download_sha256": supplied_download_digest or source_file_digest,
        }
        normalized_record: dict[str, Any] = {
            "id": stable_id,
            "stable_id": stable_id,
            "source_name": source_name,
            "source_revision": source_info["revision"],
            "license": source_info["license"],
            "source_url": source_info["url"],
            "selection_rationale": source_info["rationale"],
            "rationale": source_info["rationale"],
            "source_record_id": source_record_id,
            "source_record_index": source_index,
            "source_file": _relative_locator(resolved_file, source.parent),
            "source_file_sha256": source_file_digest,
            "download_sha256": supplied_download_digest or source_file_digest,
            "content_sha256": raw_hash,
            "text_sha256": raw_hash,
            "normalized_content_sha256": normalized_hash,
            "token_count": token_count,
            "domain": source_info["domain"],
            "sequence_length": min(sequence_length, int(item.get("sequence_length", sequence_length))),
            "add_special_tokens": bool(add_special_tokens),
        }
        for key in _CORPUS_V2_METADATA_FIELDS:
            if key in source_info:
                normalized_record[key] = source_info[key]
        normalized.append(normalized_record)
    if not normalized:
        raise ValueError("corpus contains no non-empty, licensed examples")
    required = {str(item) for item in (required_domains or ())}
    observed = {str(item["domain"]) for item in normalized}
    if not required.issubset(observed):
        raise ValueError(f"corpus is missing required domains: {sorted(required - observed)}")

    by_id = {str(item["id"]): item for item in normalized}
    ids = sorted(by_id)
    # Hash ordering avoids dependence on source array order while retaining a
    # reproducible seed and leaves enough records for the holdout target.
    train_order = sorted(ids, key=lambda identifier: hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest())
    holdout_order = sorted(ids, key=lambda identifier: hashlib.sha256(f"{holdout_seed}:{identifier}".encode()).hexdigest())
    remaining_total = sum(int(item["token_count"]) for item in normalized)
    train: list[dict[str, Any]] = []
    used: set[str] = set()
    total = 0
    for identifier in train_order:
        if total >= train_tokens:
            break
        item = by_id[identifier]
        candidate_remaining = remaining_total - total - int(item["token_count"])
        # Preserve enough unselected tokens for holdout whenever possible.
        if candidate_remaining < holdout_tokens and total < train_tokens:
            continue
        train.append({**item, "split": "train"})
        used.add(identifier)
        total += int(item["token_count"])
    if total < train_tokens:
        # If a large record made the reserve heuristic too conservative, fill
        # deterministically and report the ordinary insufficiency below.
        for identifier in train_order:
            if identifier in used:
                continue
            item = by_id[identifier]
            train.append({**item, "split": "train"})
            used.add(identifier)
            total += int(item["token_count"])
            if total >= train_tokens:
                break
    holdout: list[dict[str, Any]] = []
    total_holdout = 0
    for identifier in holdout_order:
        if identifier in used or total_holdout >= holdout_tokens:
            continue
        item = by_id[identifier]
        holdout.append({**item, "split": "holdout"})
        total_holdout += int(item["token_count"])
    if total < train_tokens or total_holdout < holdout_tokens:
        raise ValueError(f"corpus cannot satisfy disjoint token targets: train={total}, holdout={total_holdout}")

    output_path = Path(output)
    source_descriptor = {
        "path": _canonical_repo_path(source),
        "sha256": sha256_file(source),
        "format": source.suffix.lower().lstrip("."),
    }
    payload: dict[str, Any] = {
        "schema_version": 3,
        "manifest_type": "dense2moe-calibration-corpus",
        "status": "CALIBRATION_READY",
        "source": source_descriptor,
        "sources": sorted(source_catalog.values(), key=lambda item: item["name"]),
        "tokenizer": tokenizer_info,
        "tokenizer_files": tokenizer_info["files"],
        "tokenizer_files_sha256": tokenizer_info["files_sha256"],
        "tokenizer_revision": tokenizer_revision,
        "tokenization_method": "source_tokenizer_exact" if exact else "declared_count_legacy_non_scientific",
        "sequence_length": sequence_length,
        "truncation": {"policy": "none", "note": "records are resolved intact; sequence_length is a capture bound"},
        "chat_template": tokenizer_info["chat_template"],
        "bos_eos_handling": tokenizer_info["bos_eos_handling"],
        "deduplication": "normalized-content-sha256",
        "seed": seed,
        "holdout_seed": holdout_seed,
        "train": train,
        "holdout": holdout,
        "train_tokens": total,
        "holdout_tokens": total_holdout,
        "train_ids_sha256": _digest_json([item["id"] for item in train]),
        "holdout_ids_sha256": _digest_json([item["id"] for item in holdout]),
        "domains": sorted(observed),
        "required_domains": sorted(required),
        "resolvability": {
            "base_dir": _canonical_repo_path(source.parent),
            "locator_fields": ["source_file", "source_record_index", "source_record_id"],
            "verification": ["source_file_sha256", "content_sha256", "normalized_content_sha256", "token_count"],
            "text_embedded": False,
        },
    }
    payload["receipt_path"] = str(Path(receipt_output) if receipt_output is not None else output_path.with_name("corpus-receipt.json"))
    payload["dataset_hash"] = _digest_json({key: value for key, value in payload.items() if key not in {"dataset_hash", "receipt_path"}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_corpus_receipt(payload, output_path, output=receipt_output)
    # receipt_path is intentionally not part of dataset_hash: changing output
    # locations must not change scientific corpus identity.
    return payload


# ---------------------------------------------------------------------------
# Corpus V2.1 derivation and audit helpers
# ---------------------------------------------------------------------------


def _v21_values(record: Mapping[str, Any], *keys: str) -> list[str]:
    """Return non-empty scalar/list metadata values in stable form."""

    values: list[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            values.append(str(value).strip())
    return values


def _v21_identity(record: Mapping[str, Any], kind: str) -> str:
    """Resolve a repository/task/document identity without using record text."""

    if kind == "repository":
        values = _v21_values(
            record,
            "repository_id",
            "repository",
            "repo",
            "repo_name",
            "repository_name",
        )
        if not values:
            group = str(record.get("split_group", "")).strip()
            if group.lower().startswith("repo:"):
                values = [group[5:]]
    elif kind == "task":
        values = _v21_values(record, "task_id", "issue_id", "instance_id", "task", "issue")
        if not values and str(record.get("source_family", "")).lower().find("agent") >= 0:
            values = _v21_values(record, "trajectory_id", "source_record_id")
    elif kind == "document":
        values = _v21_values(record, "document_id", "document", "source_document_id")
        if not values:
            values = _v21_values(record, "source_record_id", "id", "content_sha256", "text_sha256")
    else:  # pragma: no cover - internal callers pass one of the three kinds
        raise ValueError(f"unknown V2.1 identity kind: {kind}")
    if not values:
        return ""
    # Identity matching is case-insensitive and separator-stable.  Do not use
    # the text itself here: a copied paragraph is a content duplicate, not a
    # repository/task/document identity.
    return " ".join(values[0].replace("\\", "/").split()).casefold()


def stable_corpus_record_id(record: Mapping[str, Any]) -> str:
    """Derive the content-addressed V2.1 row ID used by all receipts."""

    existing = str(record.get("id", record.get("stable_id", ""))).strip()
    if existing and re.fullmatch(r"[0-9a-f]{32,64}", existing, re.IGNORECASE):
        return existing.lower()
    text = str(record.get("text", record.get("content", "")))
    content_hash = str(record.get("content_sha256", "")) or hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = "|".join(
        (
            str(record.get("source_name", record.get("source", ""))),
            str(record.get("source_revision", record.get("revision", ""))),
            str(record.get("source_record_id", record.get("record_id", ""))),
            content_hash,
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def is_benchmark_derived(
    record: Mapping[str, Any],
    *,
    benchmark_terms: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return benchmark provenance markers that require quarantine.

    Explicit membership/context tags always win.  The fallback term scan is
    deliberately restricted to provenance metadata and identifiers; scanning
    the trajectory text would quarantine ordinary code/docs that merely
    mention an evaluation name.
    """

    terms = {str(item).casefold() for item in (benchmark_terms or V21_BENCHMARK_TERMS)}
    matches: set[str] = set()
    for key in ("benchmark_membership", "benchmark_context", "benchmark_denylist", "benchmarks"):
        for value in _v21_values(record, key):
            if value.casefold() not in {"none", "[]", "false", "unknown"}:
                matches.add(value)
    provenance_keys = (
        "source_name",
        "source_url",
        "source_artifact_url",
        "source_record_id",
        "task_id",
        "issue_id",
        "task_family",
        "document_id",
        "repo",
        "repo_path",
    )
    haystack = " ".join(_v21_values(record, *provenance_keys)).casefold()
    for term in terms:
        if term and term in haystack:
            matches.add(term)
    return tuple(sorted(matches))


def quarantine_benchmark_records(
    records: Iterable[Mapping[str, Any]],
    *,
    quarantine_split: str = V21_QUARANTINE_SPLIT,
    optimization_splits: Sequence[str] = V21_OPTIMIZATION_SPLITS + V21_PROMOTION_SPLITS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move benchmark-derived rows to an explicit diagnostic-only bucket.

    The input is never mutated.  Every quarantined row retains its original
    role and benchmark markers so diagnostic analyses can explain why it was
    excluded from gradients, checkpoint selection, and promotion.
    """

    allowed = {str(item) for item in optimization_splits}
    output: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw in records:
        row = dict(raw)
        original_split = str(row.get("split", "")).strip()
        matches = list(is_benchmark_derived(row))
        if matches:
            row["split"] = quarantine_split
            row["benchmark_quarantine"] = True
            row["benchmark_quarantine_reason"] = matches
            row["benchmark_original_split"] = original_split
            excluded.append(
                {
                    "id": stable_corpus_record_id(row),
                    "original_split": original_split,
                    "source_name": str(row.get("source_name", "")),
                    "source_record_id": str(row.get("source_record_id", "")),
                    "task_id": str(row.get("task_id", "")),
                    "repository": _v21_identity(row, "repository"),
                    "document": _v21_identity(row, "document"),
                    "matches": matches,
                }
            )
        elif original_split == quarantine_split:
            row["benchmark_quarantine"] = True
        output.append(row)
    # An original optimization role is expected for a row being quarantined;
    # the invariant is about its *final* role, which is set above.  The
    # ``forbidden`` list is retained as a receipt field for callers that want
    # to inspect a post-transform failure, but a moved row is not a failure.
    forbidden: list[dict[str, Any]] = []
    audit = {
        "status": "PASS" if not forbidden else "FAIL",
        "quarantine_split": quarantine_split,
        "optimization_splits": sorted(allowed),
        "excluded_records": len(excluded),
        "excluded_ids": sorted(item["id"] for item in excluded),
        "by_original_split": {
            split: sum(1 for item in excluded if item["original_split"] == split)
            for split in sorted({str(item["original_split"]) for item in excluded})
        },
        "forbidden_benchmark_records": forbidden,
        "provenance_retained": True,
        "promotion_excluded": True,
    }
    return output, audit


def audit_split_disjointness(
    records: Iterable[Mapping[str, Any]],
    *,
    splits: Sequence[str] = V21_SPLITS,
    excluded_splits: Sequence[str] = (V21_QUARANTINE_SPLIT, "PRESERVATION-CANARY"),
) -> dict[str, Any]:
    """Audit repository, task/issue, and document overlap across split roles."""

    selected_splits = {str(item) for item in splits} - {str(item) for item in excluded_splits}
    identities: dict[str, dict[str, dict[str, set[str]]]] = {
        kind: {split: {} for split in sorted(selected_splits)}
        for kind in ("repository", "task", "document")
    }
    row_counts = Counter()
    for raw in records:
        split = str(raw.get("split", "")).strip()
        if split not in selected_splits:
            continue
        row_counts[split] += 1
        for kind in identities:
            identity = _v21_identity(raw, kind)
            if identity:
                identities[kind][split].setdefault(identity, set()).add(stable_corpus_record_id(raw))
    overlaps: dict[str, dict[str, dict[str, list[str]]]] = {}
    for kind, by_split in identities.items():
        all_values: dict[str, dict[str, list[str]]] = defaultdict(dict)
        for split, values in by_split.items():
            for identity, row_ids in values.items():
                all_values[identity][split] = sorted(row_ids)
        conflicts = {
            identity: split_rows
            for identity, split_rows in all_values.items()
            if len(split_rows) > 1
        }
        if conflicts:
            overlaps[kind] = conflicts
    return {
        "status": "PASS" if not overlaps else "FAIL",
        "checked_splits": sorted(selected_splits),
        "excluded_splits": sorted({str(item) for item in excluded_splits}),
        "row_counts": dict(sorted(row_counts.items())),
        "overlap": overlaps,
        "repository_overlap": overlaps.get("repository", {}),
        "task_issue_overlap": overlaps.get("task", {}),
        "document_overlap": overlaps.get("document", {}),
        "zero_forbidden_overlap": not bool(overlaps),
    }


def _tier_identity(record: Mapping[str, Any], kind: str) -> str:
    """Resolve a V2.2 grouping identity without falling back to row text."""

    if kind == "source_record":
        values = _v21_values(record, "source_name", "source", "dataset")
        revision = _v21_values(record, "source_revision", "revision", "repo_commit", "source_commit_sha")
        record_id = _v21_values(record, "source_record_id", "record_id", "id")
        if values and revision and record_id:
            return "|".join((values[0].casefold(), revision[0], record_id[0])).casefold()
        return ""
    if kind == "trajectory":
        values = _v21_values(record, "trajectory_id", "trajectory", "source_trajectory_id")
        return values[0].casefold() if values else ""
    if kind == "split_group":
        values = _v21_values(record, "split_group", "group_id", "lineage_group")
        return values[0].casefold() if values else ""
    if kind == "source_family_group":
        # A source *family* is a balancing label and may occur in several
        # tiers.  Only an explicit family-group/lineage identity is a
        # forbidden cross-tier grouping key.
        values = _v21_values(record, "source_family_group", "source_lineage", "source_family_id")
        return values[0].casefold() if values else ""
    if kind in {"repository", "task", "document"}:
        return _v21_identity(record, kind)
    raise ValueError(f"unknown tier identity kind: {kind}")


def _normalized_record_text(record: Mapping[str, Any]) -> str:
    return _normalize_text(str(record.get("text", record.get("content", ""))))


def _minhash_signature(text: str, *, shingle_size: int = 5, permutations: int = 32) -> tuple[int, ...]:
    """Return a deterministic, bounded signature for near-duplicate checks.

    This is intentionally not a probabilistic external dependency.  The
    signature is only a screening index; candidate pairs are verified with
    exact shingle Jaccard similarity before being reported.
    """

    tokens = text.split()
    if len(tokens) < shingle_size:
        tokens = list(text)
        shingle_size = min(shingle_size, max(1, len(tokens)))
    if not tokens:
        return tuple(0 for _ in range(permutations))
    shingles = {
        " ".join(tokens[index : index + shingle_size])
        for index in range(max(1, len(tokens) - shingle_size + 1))
    }
    values = [int(hashlib.sha256(shingle.encode("utf-8")).hexdigest()[:16], 16) for shingle in shingles]
    signature: list[int] = []
    for permutation in range(permutations):
        salt = f"d2m-minhash-{permutation}:".encode()
        signature.append(min(int(hashlib.sha256(salt + value.to_bytes(8, "big")).hexdigest()[:16], 16) for value in values))
    return tuple(signature)


def _shingle_jaccard(left: str, right: str, *, shingle_size: int = 5) -> float:
    def shingles(value: str) -> set[str]:
        tokens = value.split()
        if len(tokens) < shingle_size:
            tokens = list(value)
            size = min(shingle_size, max(1, len(tokens)))
        else:
            size = shingle_size
        return {" ".join(tokens[index : index + size]) for index in range(max(1, len(tokens) - size + 1))}

    first, second = shingles(left), shingles(right)
    if not first and not second:
        return 1.0
    return len(first & second) / max(1, len(first | second))


def audit_corpus_tier_disjointness(
    records: Iterable[Mapping[str, Any]],
    *,
    tiers: Sequence[str] = V22_TIER_ORDER,
    near_duplicate_threshold: float = 0.80,
    max_near_duplicate_pairs: int = 10_000,
) -> dict[str, Any]:
    """Audit exact, grouped, source-record, and near-duplicate tier leakage.

    The audit is fail-closed: any normalized-content collision, source record
    reuse, lineage/group reuse, or verified near duplicate across two active
    tiers is a failure.  Rows without an explicit tier are reported as
    malformed instead of being guessed into development data.
    """

    allowed = {str(item) for item in tiers}
    rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    for raw in records:
        row = dict(raw)
        tier = str(row.get("tier", row.get("split", ""))).strip()
        row_id = stable_corpus_record_id(row)
        if tier not in allowed:
            malformed.append(row_id)
            continue
        text = _normalized_record_text(row)
        raw_hash, normalized_hash = _content_hashes(str(row.get("text", row.get("content", ""))))
        row.update(
            {
                "tier": tier,
                "_row_id": row_id,
                "_raw_content_sha256": raw_hash,
                "_normalized_content_sha256": normalized_hash,
                "_minhash": _minhash_signature(text),
                "_normalized_text": text,
            }
        )
        rows.append(row)

    identity_kinds = ("repository", "task", "trajectory", "document", "source_record", "split_group", "source_family_group")
    indexed: dict[str, dict[str, dict[str, set[str]]]] = {kind: defaultdict(lambda: defaultdict(set)) for kind in identity_kinds}
    for row in rows:
        for kind in identity_kinds:
            identity = _tier_identity(row, kind)
            if identity:
                indexed[kind][identity][row["tier"]].add(row["_row_id"])

    grouped_conflicts: list[dict[str, Any]] = []
    for kind, values in indexed.items():
        for identity, by_tier in values.items():
            if len(by_tier) > 1:
                grouped_conflicts.append(
                    {"kind": kind, "identity": identity, "tiers": {tier: sorted(ids) for tier, ids in sorted(by_tier.items())}}
                )

    exact_indexes: dict[str, dict[str, dict[str, set[str]]]] = {
        "raw_content_sha256": defaultdict(lambda: defaultdict(set)),
        "normalized_content_sha256": defaultdict(lambda: defaultdict(set)),
    }
    for row in rows:
        for key in exact_indexes:
            exact_indexes[key][row[f"_{key}"]][row["tier"]].add(row["_row_id"])
    exact_conflicts: list[dict[str, Any]] = []
    for key, values in exact_indexes.items():
        for digest, by_tier in values.items():
            if len(by_tier) > 1:
                exact_conflicts.append({"kind": key, "digest": digest, "tiers": {tier: sorted(ids) for tier, ids in sorted(by_tier.items())}})

    # MinHash bands reduce comparisons on large manifests while the final
    # Jaccard calculation remains deterministic and exact for each candidate.
    bands: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        signature = row["_minhash"]
        for band in range(0, len(signature), 4):
            bands[(band // 4, signature[band : band + 4])].append(row)
    near_duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for bucket in bands.values():
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                if left["tier"] == right["tier"]:
                    continue
                pair = tuple(sorted((left["_row_id"], right["_row_id"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                similarity = _shingle_jaccard(left["_normalized_text"], right["_normalized_text"])
                if similarity >= float(near_duplicate_threshold):
                    near_duplicates.append(
                        {
                            "left_id": left["_row_id"],
                            "right_id": right["_row_id"],
                            "left_tier": left["tier"],
                            "right_tier": right["tier"],
                            "jaccard": similarity,
                        }
                    )
                    if len(near_duplicates) >= max_near_duplicate_pairs:
                        break
            if len(near_duplicates) >= max_near_duplicate_pairs:
                break
        if len(near_duplicates) >= max_near_duplicate_pairs:
            break

    conflicts = grouped_conflicts + exact_conflicts + near_duplicates
    return {
        "schema_version": 2,
        "status": "PASS" if not malformed and not conflicts else "FAIL",
        "checked_tiers": sorted(allowed),
        "records_checked": len(rows),
        "malformed_record_ids": sorted(malformed),
        "group_conflicts": grouped_conflicts,
        "exact_conflicts": exact_conflicts,
        "near_duplicate_conflicts": near_duplicates,
        "near_duplicate_threshold": float(near_duplicate_threshold),
        "zero_forbidden_overlap": not malformed and not conflicts,
    }


def audit_grouped_split_disjointness(
    records: Iterable[Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    """Compatibility alias with an explicit grouped-split name."""

    return audit_corpus_tier_disjointness(records, **kwargs)


def assign_grouped_tiers(
    records: Iterable[Mapping[str, Any]],
    *,
    tier_fractions: Mapping[str, float],
    seed: int = 17,
    explicit_tier_key: str = "tier",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign whole lineage groups to deterministic, disjoint tiers.

    Existing explicit tier labels are honored but conflicting labels for one
    group fail closed.  Unlabeled groups are assigned by a stable hash bucket;
    no row-level random split is used.
    """

    fractions = {str(key): float(value) for key, value in tier_fractions.items()}
    if not fractions or any(value < 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("tier_fractions must be non-negative and sum to one")
    rows = [dict(row) for row in records]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    explicit: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        group = (
            _tier_identity(row, "split_group")
            or _tier_identity(row, "trajectory")
            or _tier_identity(row, "task")
            or _tier_identity(row, "repository")
            or _tier_identity(row, "document")
            or f"row:{stable_corpus_record_id(row)}"
        )
        groups[group].append(row)
        label = str(row.get(explicit_tier_key, row.get("split", ""))).strip()
        if label:
            if label not in fractions:
                raise ValueError(f"explicit tier {label!r} is not in tier_fractions")
            previous = explicit.get(group)
            if previous is not None and previous != label:
                conflicts.append({"group": group, "tiers": sorted({previous, label})})
            explicit[group] = label
    if conflicts:
        raise ValueError(f"conflicting grouped tier assignments: {conflicts[:3]}")

    ordered = sorted(fractions.items())
    cumulative: list[tuple[str, int]] = []
    total = 10_000
    cursor = 0
    for tier, fraction in ordered:
        cursor += round(fraction * total)
        cumulative.append((tier, cursor))
    cumulative[-1] = (cumulative[-1][0], total)
    assignment: dict[str, str] = {}
    for group in sorted(groups):
        if group in explicit:
            assignment[group] = explicit[group]
            continue
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % total
        assignment[group] = next(tier for tier, limit in cumulative if bucket < limit)
    output: list[dict[str, Any]] = []
    for row in rows:
        group = (
            _tier_identity(row, "split_group")
            or _tier_identity(row, "trajectory")
            or _tier_identity(row, "task")
            or _tier_identity(row, "repository")
            or _tier_identity(row, "document")
            or f"row:{stable_corpus_record_id(row)}"
        )
        row["tier"] = assignment[group]
        row["split"] = assignment[group]
        row["split_group"] = group
        output.append(row)
    counts = Counter(assignment.values())
    return output, {
        "schema_version": 1,
        "method": "sha256-group-bucket",
        "seed": int(seed),
        "tier_fractions": fractions,
        "groups": len(groups),
        "assignment_counts": dict(sorted(counts.items())),
        "group_assignments": {group: assignment[group] for group in sorted(assignment)},
        "status": "PASS",
    }


def validate_pinned_source_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Require source revision, license, and record identity for every row."""

    failures: list[dict[str, Any]] = []
    checked = 0
    for raw in records:
        checked += 1
        missing = []
        if not str(raw.get("source_revision", raw.get("revision", ""))).strip():
            missing.append("source_revision")
        if not str(raw.get("source_license", raw.get("license", raw.get("upstream_license", "")))).strip():
            missing.append("source_license")
        if not _tier_identity(raw, "source_record"):
            missing.append("source_record_identity")
        if missing:
            failures.append({"id": stable_corpus_record_id(raw), "missing": missing})
    return {"status": "PASS" if not failures else "FAIL", "records_checked": checked, "failures": failures}


def new_contamination_ledger(
    *,
    method_version: str,
    code_commit: str,
    thresholds_fingerprint: str,
    runtime_lock_sha256: str,
    corpus_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Create an empty one-way evaluation-tier contamination ledger."""

    if not method_version.strip() or not code_commit.strip() or not thresholds_fingerprint.strip():
        raise ValueError("method_version, code_commit, and thresholds_fingerprint are required")
    return {
        "schema_version": 1,
        "ledger_type": "dense2moe-evaluation-contamination",
        "status": "SEALED",
        "method_version": method_version,
        "code_commit": code_commit,
        "thresholds_fingerprint": thresholds_fingerprint,
        "runtime_lock_sha256": runtime_lock_sha256,
        "corpus_hashes": {str(key): str(value) for key, value in sorted(corpus_hashes.items())},
        "tiers": {},
        "history": [],
    }


def open_evaluation_tier(
    ledger: Mapping[str, Any],
    *,
    tier: str,
    dataset_hash: str,
    method_version: str,
    code_commit: str,
    thresholds_fingerprint: str,
) -> dict[str, Any]:
    """Open a promotion/generalization tier exactly once.

    Reopening an already opened or retired tier is rejected, even when the
    caller supplies the same method.  This prevents repeated tuning against a
    supposedly untouched corpus.
    """

    name = str(tier)
    if name not in V22_EVALUATION_TIERS:
        raise ValueError(f"only promotion/external tiers may be opened: {name}")
    result = json.loads(json.dumps(dict(ledger), sort_keys=True))
    tiers = dict(result.get("tiers", {}))
    existing = tiers.get(name)
    if existing is not None:
        raise ValueError(f"evaluation tier {name} is already opened and cannot be reused")
    locked = {"method_version": str(result.get("method_version", "")), "code_commit": str(result.get("code_commit", "")), "thresholds_fingerprint": str(result.get("thresholds_fingerprint", ""))}
    supplied = {"method_version": str(method_version), "code_commit": str(code_commit), "thresholds_fingerprint": str(thresholds_fingerprint)}
    if supplied != locked:
        raise ValueError("method/threshold identity differs from the sealed contamination ledger")
    entry = {
        "tier": name,
        "dataset_hash": str(dataset_hash),
        **supplied,
        "status": "OPENED",
        "sequence": len(result.get("history", [])) + 1,
    }
    tiers[name] = entry
    result["tiers"] = tiers
    result.setdefault("history", []).append({"event": "OPENED", **entry})
    result["status"] = "OPENED"
    return result


def retire_evaluation_tier(ledger: Mapping[str, Any], *, tier: str, reason: str) -> dict[str, Any]:
    """Retire an opened tier permanently after a failure or method change."""

    result = json.loads(json.dumps(dict(ledger), sort_keys=True))
    entry = dict(result.get("tiers", {}).get(str(tier), {}))
    if not entry:
        raise ValueError(f"cannot retire unopened evaluation tier {tier}")
    if entry.get("status") == "RETIRED":
        raise ValueError(f"evaluation tier {tier} is already retired")
    entry["status"] = "RETIRED"
    entry["retirement_reason"] = str(reason)
    result["tiers"][str(tier)] = entry
    result.setdefault("history", []).append({"event": "RETIRED", "tier": str(tier), "reason": str(reason), "sequence": len(result.get("history", [])) + 1})
    result["status"] = "RETIRED"
    return result


def validate_contamination_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that the ledger is internally consistent and fail-closed."""

    failures: list[str] = []
    if ledger.get("ledger_type") != "dense2moe-evaluation-contamination":
        failures.append("ledger_type")
    for tier, entry in dict(ledger.get("tiers", {})).items():
        if tier not in V22_EVALUATION_TIERS:
            failures.append(f"unknown_tier:{tier}")
        if entry.get("status") not in {"OPENED", "RETIRED"}:
            failures.append(f"invalid_status:{tier}")
        for key in ("dataset_hash", "method_version", "code_commit", "thresholds_fingerprint"):
            if not str(entry.get(key, "")):
                failures.append(f"missing:{tier}:{key}")
        if any(entry.get(key) != ledger.get(key) for key in ("method_version", "code_commit", "thresholds_fingerprint")):
            failures.append(f"method_identity_mismatch:{tier}")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "opened_tiers": sorted(ledger.get("tiers", {}))}


def audit_agent_task_diversity(
    records: Iterable[Mapping[str, Any]],
    *,
    minimum_tasks: int = 96,
    eligible_splits: Sequence[str] = V21_OPTIMIZATION_SPLITS + V21_PROMOTION_SPLITS,
) -> dict[str, Any]:
    """Count independent non-benchmark agent tasks and concentration signals."""

    allowed = {str(item) for item in eligible_splits}
    tasks: dict[str, dict[str, Any]] = {}
    skipped_benchmark = 0
    for raw in records:
        if str(raw.get("split", "")).strip() not in allowed:
            continue
        if is_benchmark_derived(raw):
            skipped_benchmark += 1
            continue
        source_family = str(raw.get("source_family", "")).casefold()
        domain = str(raw.get("domain", "")).casefold()
        if "agent" not in source_family and not domain.startswith("agentic"):
            continue
        task = _v21_identity(raw, "task")
        if not task:
            continue
        item = tasks.setdefault(
            task,
            {
                "task_id": task,
                "repositories": set(),
                "languages": set(),
                "frameworks": set(),
                "splits": set(),
                "records": 0,
                "tokens": 0,
            },
        )
        repository = _v21_identity(raw, "repository")
        if repository:
            item["repositories"].add(repository)
        language = str(raw.get("language", "")).strip()
        if language:
            item["languages"].add(language)
        framework = str(raw.get("trajectory_framework", "")).strip()
        if framework:
            item["frameworks"].add(framework)
        item["splits"].add(str(raw.get("split", "")))
        item["records"] += 1
        item["tokens"] += max(0, int(raw.get("token_count", 0) or 0))
    serializable = []
    for task in sorted(tasks):
        item = tasks[task]
        serializable.append(
            {
                **item,
                "repositories": sorted(item["repositories"]),
                "languages": sorted(item["languages"]),
                "frameworks": sorted(item["frameworks"]),
                "splits": sorted(item["splits"]),
            }
        )
    count = len(serializable)
    return {
        "status": "PASS" if count >= minimum_tasks else "ACQUISITION_BLOCKED",
        "minimum_tasks": int(minimum_tasks),
        "independent_non_benchmark_tasks": count,
        "shortfall": max(0, int(minimum_tasks) - count),
        "eligible_splits": sorted(allowed),
        "benchmark_records_ignored": skipped_benchmark,
        "repositories": sorted({repo for item in serializable for repo in item["repositories"]}),
        "languages": sorted({language for item in serializable for language in item["languages"]}),
        "frameworks": sorted({framework for item in serializable for framework in item["frameworks"]}),
        "tasks": serializable,
        "acquisition_blocker": None
        if count >= minimum_tasks
        else {
            "reason": "insufficient independent non-benchmark agent tasks in supplied sources",
            "required": int(minimum_tasks),
            "observed": count,
            "impact": "Phase 1 teacher capture must not be promoted until acquisition closes this gap or the epic owner accepts reduced assurance.",
        },
    }


def _v21_encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    values = _encode(tokenizer, text, add_special_tokens=add_special_tokens)
    return values


def _stable_path_label(path: Path) -> str:
    """Return a host-independent label for repository-local artifacts."""

    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return resolved.as_posix()


def audit_tokenizer_records(
    records: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_revision: str = "",
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Perform exact token recount and tokenizer-behavior audit for V2.1."""

    loaded = tokenizer
    path = Path(tokenizer_path) if tokenizer_path is not None else None
    if loaded is None:
        if path is None:
            raise ValueError("V2.1 tokenizer audit requires tokenizer or tokenizer_path")
        try:
            from tokenizers import Tokenizer as FastTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise RuntimeError("tokenizers is required for the V2.1 tokenizer audit") from exc
        loaded = FastTokenizer.from_file(str(path))
    recounts: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    special_deltas: list[int] = []
    chat_rows = 0
    chat_failures: list[str] = []
    for raw in records:
        row_id = stable_corpus_record_id(raw)
        text = str(raw.get("text", raw.get("content", "")))
        if not text.strip():
            mismatches.append({"id": row_id, "reason": "empty_text"})
            continue
        normal_count = len(_v21_encode(loaded, text, add_special_tokens=False))
        expected = raw.get("token_count")
        expected_count = int(expected) if expected is not None else None
        special_count: int | None
        try:
            special_count = len(_v21_encode(loaded, text, add_special_tokens=True))
        except (TypeError, ValueError):
            special_count = None
        if special_count is not None:
            special_deltas.append(special_count - normal_count)
        item = {
            "id": row_id,
            "expected_token_count": expected_count,
            "recount_token_count": normal_count,
            "special_token_count": special_count,
            "special_token_delta": None if special_count is None else special_count - normal_count,
        }
        recounts.append(item)
        if expected_count is not None and expected_count != normal_count:
            mismatches.append(
                {"id": row_id, "reason": "token_count_mismatch", "expected": expected_count, "actual": normal_count}
            )
        messages = raw.get("messages", raw.get("conversations"))
        if messages is not None:
            chat_rows += 1
            apply_template = getattr(loaded, "apply_chat_template", None)
            if callable(apply_template):
                try:
                    first = apply_template(messages, tokenize=True, add_generation_prompt=False)
                    second = apply_template(messages, tokenize=True, add_generation_prompt=False)
                    first_ids = list(first if isinstance(first, (list, tuple)) else getattr(first, "input_ids", first))
                    second_ids = list(second if isinstance(second, (list, tuple)) else getattr(second, "input_ids", second))
                    if first_ids != second_ids:
                        chat_failures.append(row_id)
                except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - tokenizer implementation dependent
                    chat_failures.append(f"{row_id}:{type(exc).__name__}")
            else:
                chat_failures.append(f"{row_id}:apply_chat_template_unavailable")
    if chat_rows == 0:
        chat_template_audit = {
            "status": "NOT_APPLICABLE",
            "rows": 0,
            "behavior": "plain-text records; chat template not applied",
        }
    else:
        chat_template_audit = {
            "status": "PASS" if not chat_failures else "FAIL",
            "rows": chat_rows,
            "failures": chat_failures,
            "behavior": "apply_chat_template tokenize=True, add_generation_prompt=False",
        }
    files: list[dict[str, str]] = []
    if path is not None:
        roots = [path] if path.is_file() else list(path.rglob("*"))
        for candidate in sorted(item for item in roots if item.is_file()):
            if candidate.name in _TOKENIZER_FILENAMES or candidate.name in {"chat_template.jinja"}:
                files.append({"path": candidate.name if path.is_file() else candidate.relative_to(path).as_posix(), "sha256": sha256_file(candidate)})
    return {
        "status": "PASS" if not mismatches and not chat_failures else "FAIL",
        "method": "tokenizers.Tokenizer.encode(add_special_tokens=False)",
        "tokenizer_revision": str(tokenizer_revision),
        "tokenizer_path": _stable_path_label(path) if path is not None else "",
        "tokenizer_files": files,
        "tokenizer_files_sha256": _digest_json(files),
        "records_checked": len(recounts),
        "record_recounts": recounts,
        "mismatches": mismatches,
        "special_tokens": {
            "policy": "manifest counts exclude special tokens",
            "requested_add_special_tokens": bool(add_special_tokens),
            "observed_deltas": sorted(set(special_deltas)),
            "behavior_exact": True,
        },
        "chat_template": chat_template_audit,
    }


def build_balanced_activation_plan(
    records: Iterable[Mapping[str, Any]],
    *,
    planned_tokens: int = 750_000,
    target_fractions: Mapping[str, float] | None = None,
    task_token_cap: int = 4_096,
    repository_token_cap: int = 65_536,
    source_family_token_cap: int | None = None,
    trajectory_token_cap: int = 4_096,
    eligible_splits: Sequence[str] = V21_OPTIMIZATION_SPLITS,
    seed: int = 17,
) -> dict[str, Any]:
    """Build a deterministic domain-balanced, concentration-bounded plan."""

    if planned_tokens <= 0 or task_token_cap <= 0 or repository_token_cap <= 0:
        raise ValueError("activation plan budgets and caps must be positive")
    fractions = dict(target_fractions or V21_TARGET_FRACTIONS)
    if not fractions or any(float(value) < 0 for value in fractions.values()):
        raise ValueError("target fractions must be non-negative")
    total_fraction = sum(float(value) for value in fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("target fractions must sum to one")
    rows = [dict(row) for row in records]
    eligible = {str(item) for item in eligible_splits}
    targets = {domain: round(planned_tokens * float(fraction)) for domain, fraction in fractions.items()}
    if targets:
        first = next(iter(targets))
        targets[first] += int(planned_tokens) - sum(targets.values())
    capacities: dict[str, list[dict[str, Any]]] = {domain: [] for domain in fractions}
    for row in rows:
        if str(row.get("split", "")) not in eligible or is_benchmark_derived(row):
            continue
        domain = str(row.get("domain", "")).strip()
        if domain not in capacities:
            continue
        count = max(0, int(row.get("token_count", 0) or 0))
        if not count:
            continue
        task = _v21_identity(row, "task") or f"row:{stable_corpus_record_id(row)}"
        repository_identity = _v21_identity(row, "repository")
        # Repository concentration is meaningful only for repository-backed
        # records. Collapsing every dialogue/task record from one public
        # dataset into a fake repository duplicates the source-family cap and
        # discards independent task diversity. Non-repository rows remain
        # bounded by both their task identity and their source family.
        repository = repository_identity or f"nonrepo-task:{task}"
        family = str(row.get("source_family", "unknown")) or "unknown"
        if "agent" in str(row.get("source_family", "")).casefold() or domain.startswith("agentic"):
            count = min(count, trajectory_token_cap)
        capacities[domain].append(
            {
                "row": row,
                "capacity": count,
                "task": task,
                "repository": repository,
                "source_family": family,
            }
        )
    selected: list[dict[str, Any]] = []
    available_by_domain: Counter[str] = Counter()
    selected_by_domain: Counter[str] = Counter()
    task_totals: Counter[str] = Counter()
    repository_totals: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    for domain, candidates in capacities.items():
        for item in candidates:
            available_by_domain[domain] += item["capacity"]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{domain}:{stable_corpus_record_id(item['row'])}".encode()
            ).hexdigest()
        )
        remaining = targets[domain]
        family_limit = source_family_token_cap if source_family_token_cap is not None else max(planned_tokens // 2, 1)
        for item in candidates:
            if remaining <= 0:
                break
            amount = min(
                int(item["capacity"]),
                remaining,
                task_token_cap - task_totals[item["task"]],
                repository_token_cap - repository_totals[item["repository"]],
                family_limit - family_totals[item["source_family"]],
            )
            if amount <= 0:
                continue
            row = item["row"]
            row_id = stable_corpus_record_id(row)
            selected.append(
                {
                    "id": row_id,
                    "split": str(row.get("split", "")),
                    "domain": domain,
                    "source_family": item["source_family"],
                    "task_id": item["task"],
                    "repository": item["repository"],
                    "document_id": _v21_identity(row, "document"),
                    "available_tokens": int(row.get("token_count", 0) or 0),
                    "sample_tokens": int(amount),
                    "window_policy": "deterministic-prefix-suffix; cap applied per task/repository",
                }
            )
            selected_by_domain[domain] += amount
            task_totals[item["task"]] += amount
            repository_totals[item["repository"]] += amount
            family_totals[item["source_family"]] += amount
            remaining -= amount
    selected_total = sum(selected_by_domain.values())
    domain_plan = {}
    for domain, fraction in fractions.items():
        available = int(available_by_domain[domain])
        selected_count = int(selected_by_domain[domain])
        target = int(targets[domain])
        domain_plan[domain] = {
            "target_tokens": target,
            "selected_tokens": selected_count,
            "available_tokens_after_caps": available,
            "target_fraction": float(fraction),
            "selected_fraction": selected_count / selected_total if selected_total else 0.0,
            "status": "PASS" if selected_count >= target else "INSUFFICIENT_SOURCE",
        }
    def concentration(counter: Counter[str]) -> dict[str, Any]:
        largest = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
        return {
            "groups": len(counter),
            "max_tokens": int(largest[0][1]) if largest else 0,
            "max_fraction": (largest[0][1] / selected_total) if largest and selected_total else 0.0,
            "largest": [{"id": key, "tokens": int(value)} for key, value in largest],
        }
    missing = [domain for domain, item in domain_plan.items() if item["status"] != "PASS"]
    return {
        "status": "READY_FOR_BALANCED_CAPTURE" if not missing and selected_total == planned_tokens else "REBALANCE_REQUIRED",
        "planned_tokens": int(planned_tokens),
        "selected_tokens": int(selected_total),
        "target_by_domain": domain_plan,
        "sampling_unit": "stable row IDs with per-task, per-repository, and per-source-family caps",
        "caps": {
            "task_tokens": int(task_token_cap),
            "repository_tokens": int(repository_token_cap),
            "source_family_tokens": int(source_family_token_cap or max(planned_tokens // 2, 1)),
            "trajectory_tokens": int(trajectory_token_cap),
        },
        "eligible_splits": sorted(eligible),
        "selected_rows": sorted(selected, key=lambda item: (item["domain"], item["id"])),
        "concentration": {
            "task": concentration(task_totals),
            "repository": concentration(repository_totals),
            "source_family": concentration(family_totals),
        },
        "missing_domains": missing,
        "preservation_canary": "excluded_from_optimization_and_capture",
        "teacher_capture": "NOT_STARTED",
    }


def write_immutable_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """Write canonical JSON once and refuse mutation on subsequent writes."""

    target = Path(path)
    data = _canonical_json(payload) + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    if target.exists():
        existing = target.read_bytes()
        if existing != data:
            raise ValueError(f"immutable artifact mismatch: {target}")
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return digest


def write_immutable_text(path: str | Path, text: str) -> str:
    """Write immutable UTF-8 text (used for canonical JSONL manifests)."""

    target = Path(path)
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"immutable artifact mismatch: {target}")
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return digest


def verify_immutable_artifacts(root: str | Path, artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Verify content-addressed artifact hashes from a V2.1 receipt."""

    base = Path(root)
    failures: list[dict[str, Any]] = []
    checked = 0
    for name, item in artifacts.items():
        relative = str(item.get("path", name))
        expected = str(item.get("sha256", ""))
        path = base / relative
        checked += 1
        if not path.exists():
            failures.append({"name": name, "reason": "missing", "path": relative})
        elif sha256_file(path) != expected:
            failures.append({"name": name, "reason": "hash_mismatch", "path": relative})
    return {"status": "PASS" if not failures else "FAIL", "checked": checked, "failures": failures}


__all__ = [
    "PUBLIC_DATASET_CATALOG",
    "V21_BENCHMARK_TERMS",
    "V21_OPTIMIZATION_SPLITS",
    "V21_PROMOTION_SPLITS",
    "V21_QUARANTINE_SPLIT",
    "V21_SPLITS",
    "V21_TARGET_FRACTIONS",
    "V22_DEVELOPMENT_TIERS",
    "V22_EVALUATION_TIERS",
    "V22_EXTERNAL_TIERS",
    "V22_INTERNAL_PROMOTION_TIERS",
    "V22_TIER_ORDER",
    "assign_grouped_tiers",
    "audit_agent_task_diversity",
    "audit_corpus_tier_disjointness",
    "audit_grouped_split_disjointness",
    "audit_split_disjointness",
    "audit_tokenizer_records",
    "build_balanced_activation_plan",
    "is_benchmark_derived",
    "new_contamination_ledger",
    "open_evaluation_tier",
    "prepare_calibration_manifest",
    "quarantine_benchmark_records",
    "resolve_corpus_record",
    "retire_evaluation_tier",
    "sha256_file",
    "stable_corpus_record_id",
    "validate_contamination_ledger",
    "validate_pinned_source_records",
    "verify_corpus_manifest",
    "verify_immutable_artifacts",
    "write_corpus_receipt",
    "write_immutable_json",
    "write_immutable_text",
]
