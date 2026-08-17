#!/usr/bin/env python3
"""Materialize the bounded, production-weighted public corpus-v2 source set.

The resulting JSONL is intentionally self-describing: every line carries the
source revision, license, URL, rationale, domain, source family, and stable
record/document identifiers.  Qwen3.5 non-thinking Open-SWE traces and pinned
permissive repository files provide software-engineering coverage; old
benchmark sources are opt-in so they cannot silently contaminate downstream
evaluation.  ``dense2moe.data.prepare_calibration_manifest`` reopens this
same file through its source locator and records the aggregate file hash in a
receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
OASST_URL = "https://datasets-server.huggingface.co/rows?dataset=OpenAssistant%2Foasst1&config=default&split=train"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"

# The Qwen-only configs are pinned to the data upload commit rather than a
# moving ``main`` alias.  The repository contains separate OpenHands and
# SWE-agent parquet collections; a small deterministic sample of each gives
# us agent/tool contexts without asking the local dense model to generate them.
OPEN_SWE_DATASET = "nvidia/Open-SWE-Traces"
OPEN_SWE_REVISION = "ad4805a5aa7de70d99cab0bb8f99b15304c76de0"
OPEN_SWE_BASE_URL = "https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/ad4805a5aa7de70d99cab0bb8f99b15304c76de0"
OPEN_SWE_SHARDS = (
    {
        "config": "qwen35_openhands",
        "framework": "OpenHands",
        "model": "Qwen3.5-122B-A10B",
        "path": "data/qwen35_openhands_trajectories/train-00000-of-00023.parquet",
    },
    {
        "config": "qwen35_sweagent",
        "framework": "SWE-agent",
        "model": "Qwen3.5-122B-A10B",
        "path": "data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet",
    },
)

# These are deliberately small source slices, not whole repository snapshots.
# Each commit was resolved from the repository default branch when this source
# manifest was authored and is retained here so a future run cannot silently
# drift to a different tree.  All selected repositories advertise a permissive
# SPDX license.
PERMISSIVE_REPOSITORY_FILES = (
    {
        "name": "psf/requests",
        "revision": "8068356288978c4f54661ae6f95afe0e0831885e",
        "license": "Apache-2.0",
        "language": "Python",
        "files": ("src/requests/api.py", "tests/test_requests.py", "pyproject.toml"),
    },
    {
        "name": "pallets/flask",
        "revision": "d318b683471101618febed18996405ad26462110",
        "license": "BSD-3-Clause",
        "language": "Python",
        "files": ("src/flask/app.py", "tests/test_basic.py", "pyproject.toml"),
    },
    {
        "name": "gin-gonic/gin",
        "revision": "dcaa4296d111981ffb31ac3eba90bb63e1eb5ab9",
        "license": "MIT",
        "language": "Go",
        "files": ("gin.go", "context.go", "go.mod"),
    },
    {
        "name": "rust-lang/rustlings",
        "revision": "02b22b49e7349b36305dbe97cd5d6e198d2f0537",
        "license": "MIT",
        "language": "Rust",
        "files": ("src/main.rs", "Cargo.toml", "README.md"),
    },
    {
        "name": "tokio-rs/tokio",
        "revision": "625954f365727668cb02d04172b34f1149637728",
        "license": "MIT",
        "language": "Rust",
        "files": ("tokio/src/lib.rs", "tokio/Cargo.toml", "README.md"),
    },
)


def _fetch(url: str, *, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-public-corpus/2.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500 or attempt + 1 >= attempts:
                raise
    raise RuntimeError(f"source fetch failed after {attempts} attempts: {url}: {last_error}")


def _fetch_to_path(url: str, path: Path) -> str:
    """Stream a source artifact to disk and return its content hash."""

    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-public-corpus/2.0"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_text(trajectory: Any) -> tuple[str, bool]:
    """Render visible trajectory messages and report private-reasoning use.

    Open-SWE-Traces stores ``reasoning_content`` and ``think`` alongside the
    visible message text.  The latter are not part of the intended activation
    context.  We reject a row containing non-empty private fields rather than
    silently claiming it is a non-thinking trajectory.
    """

    if not isinstance(trajectory, list):
        return "", False
    chunks: list[str] = []
    private_reasoning = False
    for raw_message in trajectory:
        if not isinstance(raw_message, dict):
            continue
        if raw_message.get("reasoning_content"):
            private_reasoning = True
        if raw_message.get("think") is True:
            private_reasoning = True
        role = str(raw_message.get("role", "")).strip()
        content = raw_message.get("content", "")
        if content is not None and not isinstance(content, str):
            content = str(content)
        if isinstance(content, str) and content:
            # Protect against providers embedding a hidden block in the
            # visible field.  Keep the rest of the visible tool transcript.
            content = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
            if content:
                chunks.append(f"{role}: {content}" if role else content)
        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            chunks.append(
                f"{role or 'assistant'} tool_calls: "
                + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
            )
    return "\n".join(chunks).strip(), private_reasoning


def _text_segments(text: str, *, max_chars: int = 32_000, max_segments: int = 1) -> list[str]:
    """Bound long trajectories while exposing the truncation policy."""

    if len(text) <= max_chars:
        return [text]
    if max_segments == 1:
        marker = "\n[... trajectory middle omitted ...]\n"
        half = max(1, (max_chars - len(marker)) // 2)
        return [text[:half] + marker + text[-(max_chars - len(marker) - half) :]]
    return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]


def _read_parquet_sample(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Read a bounded deterministic sample without materializing a full shard."""

    if limit <= 0:
        return []
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "pyarrow is required for Open-SWE-Traces acquisition; install the data extra or "
            "pass --open-swe-limit 0"
        ) from exc
    reader = parquet.ParquetFile(path)
    total = int(reader.metadata.num_rows)
    # Evenly spaced physical rows avoid a first-page-only sample while keeping
    # memory bounded by a small Arrow batch.
    target_count = min(limit, total)
    targets = {
        int(round(index * (total - 1) / max(target_count - 1, 1)))
        for index in range(target_count)
    }
    selected: list[dict[str, Any]] = []
    offset = 0
    for batch in reader.iter_batches(batch_size=32):
        for local_index in range(batch.num_rows):
            row_index = offset + local_index
            if row_index not in targets:
                continue
            selected.append({name: batch[name][local_index].as_py() for name in batch.column_names})
        offset += batch.num_rows
        if len(selected) >= target_count:
            break
    return selected


def _open_swe_records(*, limit_per_framework: int = 16, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """Fetch Qwen non-thinking Open-SWE rows and normalize visible traces."""

    if limit_per_framework <= 0:
        return []
    cache_root = cache_dir or Path(tempfile.gettempdir()) / "dense2moe-open-swe"
    cache_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for shard in OPEN_SWE_SHARDS:
        relative = Path(str(shard["path"]))
        local_path = cache_root / f"{shard['config']}-{relative.name}"
        source_url = f"https://huggingface.co/datasets/{OPEN_SWE_DATASET}/resolve/{OPEN_SWE_REVISION}/{relative.as_posix()}"
        if local_path.exists():
            download_digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        else:
            download_digest = _fetch_to_path(source_url, local_path)
        rows = _read_parquet_sample(local_path, limit=limit_per_framework)
        for row in rows:
            text, private_reasoning = _trajectory_text(row.get("trajectory"))
            if private_reasoning or not text:
                continue
            instance_id = str(row.get("instance_id", "")).strip()
            trajectory_id = str(row.get("trajectory_id", "")).strip()
            repository = str(row.get("repo", "")).strip()
            if not instance_id or not trajectory_id or not repository:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            category = str(metadata.get("category", "software-engineering")).strip() or "software-engineering"
            model_patch = metadata.get("model_patch") if isinstance(metadata, dict) else None
            patch_text = model_patch.get("patch") if isinstance(model_patch, dict) else ""
            if isinstance(patch_text, str) and patch_text.strip():
                text = f"{text}\nassistant patch:\n{patch_text.strip()}"
            original_chars = len(text)
            segments = _text_segments(text, max_segments=1)
            for segment_index, segment in enumerate(segments):
                records.append(
                    {
                        "source_file": "corpus.jsonl",
                        "source_name": OPEN_SWE_DATASET,
                        "source_revision": OPEN_SWE_REVISION,
                        "source_license": "CC-BY-4.0",
                        "upstream_license": str(row.get("license", "")),
                        "terms": "Open-SWE-Traces CC-BY-4.0; retain the upstream repository license and provenance.",
                        "source_url": f"{OPEN_SWE_BASE_URL}/{relative.as_posix()}",
                        "download_sha256": download_digest,
                        "selection_rationale": (
                            "Visible Qwen3.5 non-thinking software-agent trajectory sampled from "
                            f"the pinned {shard['framework']} shard; reasoning_content and think=true "
                            "fields were excluded."
                        ),
                        "rationale": "Agent/tool contexts for production-weighted software-engineering coverage.",
                        "domain": "agentic/software-engineering",
                        "source_family": "agent-trajectory",
                        "task_family": category,
                        "trajectory_framework": shard["framework"],
                        "trajectory_model": shard["model"],
                        "trajectory_reasoning": "non-thinking-visible-transcript-only",
                        "language": str(row.get("language", "unknown")),
                        "repository_id": repository,
                        "document_id": f"{repository}::{instance_id}",
                        "task_id": instance_id,
                        "trajectory_id": trajectory_id,
                        "segment_index": segment_index,
                        "segment_count": len(segments),
                        "trajectory_original_chars": original_chars,
                        "trajectory_truncated": original_chars > len(segment),
                        "trajectory_truncation_policy": "prefix+suffix character cap" if original_chars > len(segment) else "none",
                        "benchmark_membership": ["SWE-rebench-V2-derived", "SWE-Bench-style"],
                        "benchmark_context": ["SWE-rebench-V2-derived", "SWE-Bench-style"],
                        "source_record_id": f"{shard['config']}::{trajectory_id}::{segment_index:03d}",
                        "text": segment,
                    }
                )
    return records


def _repository_records() -> list[dict[str, Any]]:
    """Fetch selected files from pinned permissive repository commits."""

    records: list[dict[str, Any]] = []
    for repository in PERMISSIVE_REPOSITORY_FILES:
        name = str(repository["name"])
        revision = str(repository["revision"])
        license_name = str(repository["license"])
        language = str(repository["language"])
        for relative_path in repository["files"]:
            file_path = str(relative_path)
            source_url = f"https://github.com/{name}/blob/{revision}/{file_path}"
            raw_url = f"https://raw.githubusercontent.com/{name}/{revision}/{file_path}"
            content_bytes = _fetch(raw_url)
            text = content_bytes.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            content_digest = hashlib.sha256(content_bytes).hexdigest()
            records.append(
                {
                    "source_file": "corpus.jsonl",
                    "source_name": f"github.com/{name}",
                    "source_revision": revision,
                    "source_license": license_name,
                    "source_url": source_url,
                    "download_sha256": content_digest,
                    "selection_rationale": "Pinned permissive repository file retained for implementation, test, and build coverage.",
                    "rationale": "Real maintained repository context across languages and file roles.",
                    "domain": "code/repository",
                    "source_family": "permissive-repository",
                    "task_family": "tests" if "/test" in file_path or file_path.startswith("tests/") else "implementation/configuration",
                    "language": language,
                    "repository_id": name,
                    "document_id": f"{name}@{revision}:{file_path}",
                    "source_commit_sha": revision,
                    "source_path": file_path,
                    "benchmark_membership": [],
                    "source_record_id": f"{name}@{revision}:{file_path}",
                    "text": text,
                }
            )
    return records


def _deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop exact normalized-content duplicates while preserving first provenance."""

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", ""))
        normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        record["normalized_content_sha256"] = digest
        record["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        unique.append(record)
    return unique, len(records) - len(unique)


def _clean_book(raw: str) -> str:
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    if start:
        raw = raw[start.end() :]
    if end:
        raw = raw[: end.start()]
    return raw.strip()


def _book_chunks(text: str, *, max_chars: int = 5000) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            size = 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _oasst_rows(limit: int) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        length = min(100, limit - len(rows))
        url = f"{OASST_URL}&offset={offset}&length={length}"
        payload = json.loads(_fetch(url).decode("utf-8"))
        page = payload.get("rows", [])
        if not isinstance(page, list) or not page:
            break
        for item in page:
            row = item.get("row", {}) if isinstance(item, dict) else {}
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            if row.get("lang") not in (None, "en"):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
        offset += len(page)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return rows, digest


def build_records(
    *,
    oasst_limit: int = 512,
    gsm_limit: int = 0,
    open_swe_limit: int = 8,
    open_swe_cache: Path | None = None,
    include_benchmark_sources: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    common = {"source_file": "corpus.jsonl"}

    if include_benchmark_sources:
        humaneval_bytes = _fetch(HUMANEVAL_URL)
        humaneval = gzip.decompress(humaneval_bytes).decode("utf-8")
        humaneval_digest = hashlib.sha256(humaneval_bytes).hexdigest()
        for index, line in enumerate(humaneval.splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", ""))
            solution = str(item.get("canonical_solution", ""))
            records.append(
                {
                    **common,
                    "source_name": "openai/openai_humaneval",
                    "source_revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
                    "source_license": "MIT",
                    "source_url": "https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d",
                    "download_sha256": humaneval_digest,
                    "selection_rationale": "MIT-licensed code-generation prompts and reference solutions for code coverage.",
                    "rationale": "MIT-licensed code-generation prompts and reference solutions for code coverage.",
                    "domain": "code",
                    "source_family": "benchmark",
                    "task_family": "code-generation",
                    "language": "Python",
                    "document_id": f"HumanEval/{item.get('task_id', index)}",
                    "benchmark_membership": ["HumanEval"],
                    "source_record_id": str(item.get("task_id", f"HumanEval/{index}")),
                    "text": f"{prompt}\n{solution}",
                }
            )

    if include_benchmark_sources and gsm_limit > 0:
        gsm_bytes = _fetch(GSM8K_URL)
        gsm = gsm_bytes.decode("utf-8")
        gsm_digest = hashlib.sha256(gsm_bytes).hexdigest()
        for index, line in enumerate(gsm.splitlines()[:gsm_limit]):
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    **common,
                    "source_name": "openai/gsm8k",
                    "source_revision": "3101c7d5072418e28b9008a6636bde82a006892c",
                    "source_license": "MIT",
                    "source_url": "https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c",
                    "download_sha256": gsm_digest,
                    "selection_rationale": "MIT-licensed grade-school math questions with human-written reasoning traces.",
                    "rationale": "MIT-licensed grade-school math questions with human-written reasoning traces.",
                    "domain": "reasoning/math",
                    "source_family": "benchmark",
                    "task_family": "mathematical-reasoning",
                    "language": "English",
                    "document_id": f"GSM8K/train/{index}",
                    "benchmark_membership": ["GSM8K"],
                    "source_record_id": f"train/{index}",
                    "text": f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}",
                }
            )

    oasst_rows, oasst_digest = _oasst_rows(oasst_limit)
    for index, item in enumerate(oasst_rows):
        record_id = str(item.get("message_id", item.get("id", f"oasst/{index}")))
        records.append(
            {
                **common,
                "source_name": "OpenAssistant/oasst1",
                "source_revision": OASST_REVISION,
                "source_license": "Apache-2.0",
                "source_url": f"https://huggingface.co/datasets/OpenAssistant/oasst1/tree/{OASST_REVISION}",
                "download_sha256": oasst_digest,
                "selection_rationale": "Apache-2.0 public instruction and dialogue turns for conversational coverage.",
                "rationale": "Apache-2.0 public instruction and dialogue turns for conversational coverage.",
                "domain": "instruction/dialogue",
                "source_family": "instruction-dialogue",
                "task_family": "general-instruction",
                "language": "English",
                "document_id": f"OpenAssistant/oasst1/{record_id}",
                "benchmark_membership": [],
                "source_record_id": record_id,
                "text": str(item["text"]),
            }
        )

    book_bytes = _fetch(GUTENBERG_URL)
    book = _clean_book(book_bytes.decode("utf-8", errors="replace"))
    book_digest = hashlib.sha256(book_bytes).hexdigest()
    for index, chunk in enumerate(_book_chunks(book)):
        records.append(
            {
                **common,
                "source_name": "Project Gutenberg public-domain texts",
                "source_revision": "ebook-1342",
                "source_license": "Public Domain",
                "source_url": GUTENBERG_URL,
                "download_sha256": book_digest,
                "selection_rationale": "Stable Project Gutenberg ebook 1342; U.S. public-domain prose selected for general and long-context coverage.",
                "rationale": "Stable Project Gutenberg ebook 1342; U.S. public-domain prose selected for general and long-context coverage.",
                "domain": "general" if index < 8 else "long-context",
                "source_family": "general-public-domain",
                "task_family": "technical-preservation-canary",
                "language": "English",
                "repository_id": "gutenberg:1342",
                "document_id": "gutenberg:1342",
                "benchmark_membership": [],
                "source_record_id": f"1342-chunk-{index:04d}",
                "text": chunk,
            }
        )
    records.extend(_repository_records())
    records.extend(_open_swe_records(limit_per_framework=open_swe_limit, cache_dir=open_swe_cache))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/public_v2/corpus-v2-source.jsonl"))
    parser.add_argument("--oasst-limit", type=int, default=512)
    parser.add_argument("--open-swe-limit", type=int, default=8, help="rows per Qwen framework (0 disables traces)")
    parser.add_argument("--open-swe-cache", type=Path, default=None)
    parser.add_argument(
        "--include-benchmark-sources",
        action="store_true",
        help="include HumanEval/GSM8K; disabled by default because they are evaluation benchmarks",
    )
    parser.add_argument("--gsm-limit", type=int, default=512, help="legacy benchmark rows when explicitly enabled")
    args = parser.parse_args()
    records = build_records(
        oasst_limit=args.oasst_limit,
        gsm_limit=args.gsm_limit,
        open_swe_limit=args.open_swe_limit,
        open_swe_cache=args.open_swe_cache,
        include_benchmark_sources=args.include_benchmark_sources,
    )
    records, duplicate_count = _deduplicate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Every locator points back at this materialized JSONL.  Using its actual
    # basename keeps alternate output locations resolvable by the verifier.
    for record in records:
        record["source_file"] = args.output.name
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            record["source_record_index"] = index
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    source_counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("source_family", "unknown"))
        source_counts[key] = source_counts.get(key, 0) + 1
    print(
        json.dumps(
            {
                "status": "CORPUS_MATERIALIZED",
                "path": str(args.output),
                "records": len(records),
                "duplicates_removed": duplicate_count,
                "source_families": source_counts,
                "open_swe_revision": OPEN_SWE_REVISION,
                "benchmarks_included": bool(args.include_benchmark_sources),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
