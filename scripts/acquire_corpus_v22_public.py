#!/usr/bin/env python3
"""Prepare pinned public sources for the Corpus V2.2 freeze gate.

This adapter never weakens V2.2 validation. It converts two already-downloaded,
permissively licensed source files into development and untouched internal
JSONL components. Whole task/tree groups receive one deterministic tier.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.data import _minhash_signature, _shingle_jaccard, sha256_file

CODEAGENT_NAME = "krzysztofwos/CodeAgent-Trajectories"
CODEAGENT_REVISION = "2215a62ec56c9d8e7b6927f727635340a5238d76"
CODEAGENT_LICENSE = "Apache-2.0"
OASST_NAME = "OpenAssistant/oasst1"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OASST_LICENSE = "Apache-2.0"
VSCODE_NAME = "microsoft/vscode"
VSCODE_REVISION = "8c629223a4a6ebf48878194a6a990037c17823ba"
VSCODE_LICENSE = "MIT"


def _bucket(group: str, choices: tuple[tuple[str, int], ...], *, seed: int = 22) -> str:
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 10_000
    cursor = 0
    for tier, weight in choices:
        cursor += weight
        if value < cursor:
            return tier
    raise AssertionError("tier weights must total 10000")


def _codeagent_tier(group: str) -> str:
    # The shared CodeAgent system/tool scaffold makes otherwise distinct tasks
    # near-duplicates. Keep that entire source family in FIT-TRAIN; OASST
    # supplies the independently grouped FIT-DEV and promotion tiers.
    del group
    return "FIT-TRAIN"


def _oasst_tier(group: str) -> str:
    return _bucket(
        group,
        (("FIT-TRAIN", 6800), ("FIT-DEV", 1200), ("GATE-A", 800), ("SHADOW-B", 600), ("SHADOW-C", 600)),
    )


def _oasst_domain(text: str) -> tuple[str, str]:
    value = text.casefold()
    structured = ("json", "yaml", "xml", "csv", "schema", "regular expression", "regex", "sql")
    code = ("python", "javascript", "typescript", "java ", "c++", "rust", "golang", "write code", "function", "programming")
    software = ("software", "api", "database", "git ", "docker", "linux", "debug", "developer", "documentation")
    if any(term in value for term in structured):
        return "structured", "structured-data-or-query"
    if any(term in value for term in code):
        return "code", "programming-dialogue"
    if any(term in value for term in software):
        return "software-engineering-natural-language", "software-engineering-dialogue"
    return "general", "general-dialogue"


def _tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tokenizers is required") from exc
    return Tokenizer.from_file(str(path))


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping):
                yield dict(value)


def _chunks(tokenizer: Any, text: str, *, size: int = 2_048, limit: int | None = None) -> Iterable[tuple[int, str, int]]:
    token_ids = tokenizer.encode(text).ids
    if limit is not None:
        token_ids = token_ids[:limit]
    for index in range(0, len(token_ids), size):
        selected = token_ids[index : index + size]
        if len(selected) < 32:
            continue
        yield index // size, tokenizer.decode(selected), len(selected)


def prepare(
    *,
    codeagent_source: Path,
    oasst_source: Path,
    v21_source: Path | None,
    gutenberg_source: Path | None,
    vscode_archive: Path | None,
    tokenizer_path: Path,
    output: Path,
) -> dict[str, Any]:
    tokenizer = _tokenizer(tokenizer_path)
    development: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    seen_oasst_content: set[str] = set()
    oasst_bands: dict[tuple[int, tuple[int, ...]], list[str]] = {}

    for index, raw in enumerate(_jsonl(codeagent_source)):
        task = str(raw.get("task", "")).strip()
        messages = raw.get("messages", [])
        visible = []
        if task:
            visible.append(f"[task]\n{task}")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, Mapping) and str(message.get("content", "")).strip():
                    visible.append(f"[{message.get('role', 'message')}]\n{message['content']}")
        text = "\n\n".join(visible).strip()
        if not text:
            continue
        task_id = hashlib.sha256(f"codeagent:{index}:{task}".encode()).hexdigest()[:24]
        row = {
            "text": text,
            "tier": _codeagent_tier(task_id),
            "source_name": CODEAGENT_NAME,
            "source_revision": CODEAGENT_REVISION,
            "source_license": CODEAGENT_LICENSE,
            "source_url": f"https://huggingface.co/datasets/{CODEAGENT_NAME}",
            "source_record_id": f"train:{index}",
            "source_family": "agent-trajectory-public",
            "task_id": task_id,
            "trajectory_id": task_id,
            "split_group": f"task:{task_id}",
            "document_id": f"codeagent:{task_id}",
            "domain": "agentic-software-engineering",
            "task_family": "tool-using-agent-task",
            "trajectory_framework": "smolagents-codeagent",
            "language": "en",
            "selection_rationale": "Pinned public visible CodeAgent task transcript; grouped by complete task.",
            "benchmark_membership": [],
            "token_count": len(tokenizer.encode(text).ids),
        }
        development.append(row)

    if v21_source is not None:
        for raw in _jsonl(v21_source):
            if str(raw.get("source_family", "")) != "real-repository":
                continue
            if str(raw.get("split", "")) not in {"FIT-TRAIN", "FIT-DEV"}:
                continue
            if raw.get("benchmark_quarantine") or raw.get("benchmark_membership"):
                continue
            row = dict(raw)
            row["tier"] = str(row["split"])
            row["v22_parent"] = str(row.get("id", ""))
            row["selection_rationale"] = (
                "Provenance-preserving V2.1 real-repository ancestry reused only in V2.2 development tiers."
            )
            development.append(row)

    if gutenberg_source is not None:
        book = gutenberg_source.read_text(encoding="utf-8-sig", errors="replace")
        for index, text, token_count in _chunks(tokenizer, book, limit=90_000):
            record_id = f"ebook-1342:chunk-{index:04d}"
            development.append(
                {
                    "text": text,
                    "tier": "FIT-TRAIN",
                    "source_name": "Project Gutenberg public-domain texts",
                    "source_revision": "ebook-1342",
                    "source_license": "Public Domain",
                    "source_url": "https://www.gutenberg.org/ebooks/1342",
                    "source_record_id": record_id,
                    "source_family": "public-domain-book",
                    "task_id": record_id,
                    "split_group": record_id,
                    "document_id": record_id,
                    "domain": "general",
                    "task_family": "general-prose",
                    "language": "en",
                    "selection_rationale": "Pinned public-domain book chunks for independent general-language coverage.",
                    "benchmark_membership": [],
                    "token_count": token_count,
                }
            )

    if vscode_archive is not None:
        budgets = {"structured": 50_000, "software-engineering-natural-language": 30_000}
        used = Counter()
        with tarfile.open(vscode_archive, "r:gz") as archive:
            members = sorted((member for member in archive.getmembers() if member.isfile()), key=lambda item: item.name)
            for member in members:
                relative = member.name.split("/", 1)[-1]
                lower = relative.casefold()
                if lower == "package-lock.json":
                    domain = "structured"
                elif lower.endswith(".md") and not lower.startswith(("node_modules/", ".build/")):
                    domain = "software-engineering-natural-language"
                else:
                    continue
                if used[domain] >= budgets[domain]:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                content = handle.read().decode("utf-8", errors="replace")
                remaining = budgets[domain] - used[domain]
                for index, text, token_count in _chunks(tokenizer, content, limit=remaining):
                    used[domain] += token_count
                    record_id = f"{relative}:chunk-{index:04d}"
                    development.append(
                        {
                            "text": text,
                            "tier": "FIT-TRAIN",
                            "source_name": VSCODE_NAME,
                            "source_revision": VSCODE_REVISION,
                            "source_license": VSCODE_LICENSE,
                            "source_url": f"https://github.com/{VSCODE_NAME}",
                            "source_record_id": record_id,
                            "source_family": "repository-vscode",
                            "repo": VSCODE_NAME,
                            "repo_commit": VSCODE_REVISION,
                            "repo_path": relative,
                            "task_id": record_id,
                            "split_group": f"repo:{VSCODE_NAME}:{record_id}",
                            "document_id": record_id,
                            "domain": domain,
                            "task_family": "repository-configuration" if domain == "structured" else "software-engineering-documentation",
                            "language": "en",
                            "selection_rationale": "Pinned MIT repository file chunks for structured and engineering-documentation coverage.",
                            "benchmark_membership": [],
                            "token_count": token_count,
                        }
                    )
                    if used[domain] >= budgets[domain]:
                        break

    with gzip.open(oasst_source, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("text", "")).strip()
            if not text or raw.get("deleted") or raw.get("review_result") is False or str(raw.get("lang", "")) != "en":
                continue
            normalized = " ".join(text.split()).casefold()
            content_identity = hashlib.sha256(normalized.encode()).hexdigest()
            if content_identity in seen_oasst_content:
                continue
            signature = _minhash_signature(normalized)
            candidates: set[str] = set()
            for band in range(0, len(signature), 4):
                candidates.update(oasst_bands.get((band // 4, signature[band : band + 4]), []))
            if any(_shingle_jaccard(normalized, candidate) >= 0.80 for candidate in candidates):
                continue
            seen_oasst_content.add(content_identity)
            for band in range(0, len(signature), 4):
                oasst_bands.setdefault((band // 4, signature[band : band + 4]), []).append(normalized)
            tree_id = str(raw.get("message_tree_id", "")).strip()
            message_id = str(raw.get("message_id", "")).strip()
            if not tree_id or not message_id:
                continue
            tier = _oasst_tier(tree_id)
            domain, task_family = _oasst_domain(text)
            token_count = len(tokenizer.encode(text).ids) + 3
            if token_count < 32:
                continue
            row = {
                "text": f"[{raw.get('role', 'message')}]\n{text}",
                "tier": tier,
                "source_name": OASST_NAME,
                "source_revision": OASST_REVISION,
                "source_license": OASST_LICENSE,
                "source_url": f"https://huggingface.co/datasets/{OASST_NAME}",
                "source_record_id": message_id,
                "source_family": "human-assistant-dialogue-public",
                "task_id": f"oasst-tree:{tree_id}",
                "split_group": f"oasst-tree:{tree_id}",
                "document_id": f"oasst-message:{message_id}",
                "domain": domain,
                "task_family": task_family,
                "language": "en",
                "selection_rationale": "Pinned reviewed OpenAssistant visible message; grouped by complete conversation tree.",
                "benchmark_membership": [],
                "token_count": token_count,
            }
            (development if tier in {"FIT-TRAIN", "FIT-DEV"} else internal).append(row)

    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "development.jsonl"
    internal_path = output / "internal.jsonl"
    for path, rows in ((development_path, development), (internal_path, internal)):
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8", newline="\n")
    summary = {
        "status": "PUBLIC_V22_SOURCES_PREPARED",
        "sources": {
            "codeagent": {"revision": CODEAGENT_REVISION, "sha256": sha256_file(codeagent_source)},
            "oasst1": {"revision": OASST_REVISION, "sha256": sha256_file(oasst_source)},
            "v21_real_repository": None
            if v21_source is None
            else {"path": str(v21_source), "sha256": sha256_file(v21_source)},
            "gutenberg_1342": None
            if gutenberg_source is None
            else {"revision": "ebook-1342", "sha256": sha256_file(gutenberg_source)},
            "vscode": None
            if vscode_archive is None
            else {"revision": VSCODE_REVISION, "sha256": sha256_file(vscode_archive)},
        },
        "outputs": {
            "development": {"path": str(development_path), "sha256": sha256_file(development_path), "records": len(development)},
            "internal": {"path": str(internal_path), "sha256": sha256_file(internal_path), "records": len(internal)},
        },
        "tiers": dict(sorted(Counter(str(row["tier"]) for row in development + internal).items())),
        "domains": dict(sorted(Counter(str(row["domain"]) for row in development + internal).items())),
        "policy": "public-only; whole task/tree grouping; internal tiers untouched; freeze gate remains authoritative",
    }
    (output / "acquisition-receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codeagent-source", type=Path, required=True)
    parser.add_argument("--oasst-source", type=Path, required=True)
    parser.add_argument("--v21-source", type=Path)
    parser.add_argument("--gutenberg-source", type=Path)
    parser.add_argument("--vscode-archive", type=Path)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = prepare(
        codeagent_source=args.codeagent_source,
        oasst_source=args.oasst_source,
        v21_source=args.v21_source,
        gutenberg_source=args.gutenberg_source,
        vscode_archive=args.vscode_archive,
        tokenizer_path=args.tokenizer_path,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
