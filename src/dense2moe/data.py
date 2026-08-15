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
import unicodedata
from collections.abc import Callable, Mapping, Sequence
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
)

_PERMISSIVE_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CDLA-Permissive-1.0",
    "CDLA-Permissive-2.0",
    "MIT",
    "ODC-By-1.0",
    "Public Domain",
    "public-domain",
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


class _TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Any: ...


Tokenizer = _TokenizerLike | Callable[..., Any]


@dataclass(frozen=True)
class _SourceRecord:
    value: dict[str, Any]
    source_file: Path
    source_record_index: int


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
    records, _ = _read_source_records(path)
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
    return metadata


def _validate_source_metadata(metadata: Mapping[str, Any], *, strict: bool) -> dict[str, str]:
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
    return {
        "name": name,
        "revision": revision,
        "license": license_name,
        "url": str(metadata.get("url", "")),
        "rationale": rationale,
        "domain": str(metadata.get("domain", "general")),
    }


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
            selected.append(
                {
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
            )
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
    source_catalog: dict[str, dict[str, str]] = {}
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
        source_catalog[f"{source_name}@{source_info['revision']}"] = {
            **source_info,
            "download_sha256": supplied_download_digest or source_file_digest,
        }
        normalized.append(
            {
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
        )
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


__all__ = [
    "PUBLIC_DATASET_CATALOG",
    "prepare_calibration_manifest",
    "resolve_corpus_record",
    "sha256_file",
    "verify_corpus_manifest",
    "write_corpus_receipt",
]
