#!/usr/bin/env python3
"""Materialize the small public, permissive calibration mixture used by V2.

The resulting JSONL is intentionally self-describing: every line carries the
source revision, license, URL, rationale, domain, and a stable source record
ID.  ``dense2moe.data.prepare_calibration_manifest`` reopens this same file
through its source locator and records the aggregate file hash in the receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
OASST_URL = "https://datasets-server.huggingface.co/rows?dataset=OpenAssistant%2Foasst1&config=default&split=train"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-public-corpus/2.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


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


def build_records(*, oasst_limit: int = 512, gsm_limit: int = 512) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    common = {"source_file": "corpus.jsonl"}

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
                "source_record_id": str(item.get("task_id", f"HumanEval/{index}")),
                "text": f"{prompt}\n{solution}",
            }
        )

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
                "source_record_id": f"1342-chunk-{index:04d}",
                "text": chunk,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/public_v2/corpus.jsonl"))
    parser.add_argument("--oasst-limit", type=int, default=512)
    parser.add_argument("--gsm-limit", type=int, default=512)
    args = parser.parse_args()
    records = build_records(oasst_limit=args.oasst_limit, gsm_limit=args.gsm_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            record["source_record_index"] = index
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "CORPUS_MATERIALIZED", "path": str(args.output), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
