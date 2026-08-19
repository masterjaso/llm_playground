from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.acquire_corpus_v23_public import (
    _prepare_source_text,
    audit_v22_overlap,
    audit_v23_tier_disjointness,
    build_corpus,
    normalized_text_hash,
)


def _source(tmp_path: Path, name: str, text: str, *, tier: str = "") -> dict[str, str]:
    path = tmp_path / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    item = {
        "source_id": name,
        "name": f"Fresh {name}",
        "url": f"https://example.test/{name}.txt",
        "license": "MIT",
        "domain": "general",
        "local_path": str(path),
    }
    if tier:
        item["tier"] = tier
    return item


def test_build_corpus_assigns_whole_sources_and_stable_hashes(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, "alpha", "Alpha paragraph. " * 150),
        _source(tmp_path, "beta", "Beta paragraph. " * 150),
    ]
    first = build_corpus(output=tmp_path / "one", sources=sources, max_chars=500, acquired_at="2026-08-18T20:00:00Z")
    second = build_corpus(output=tmp_path / "two", sources=sources, max_chars=500, acquired_at="2026-08-18T20:00:00Z")
    assert first["tier_counts"]["FIT-TRAIN"] > 0
    assert first["tier_counts"]["FIT-DEV"] > 0
    assert first["prepared_sha256"] == second["prepared_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "one" / "prepared" / "v23-prepared.jsonl").read_text().splitlines()]
    assert {row["tier"] for row in rows} == {"FIT-TRAIN", "FIT-DEV"}
    for row in rows:
        assert row["normalized_content_sha256"] == normalized_text_hash(row["text"])
        assert row["group_identity"].startswith("v23-source:")
        assert row["source_family"].startswith("v23-source-family:")
        assert row["source_file_sha256"] == row["download_sha256"]
    by_source = {row["source_id"]: {item["tier"] for item in rows if item["source_id"] == row["source_id"]} for row in rows}
    assert all(len(tiers) == 1 for tiers in by_source.values())


def test_tier_audit_rejects_source_task_and_near_duplicate_crossing(tmp_path: Path) -> None:
    base = {
        "source_name": "fresh",
        "source_url": "https://example.test/fresh",
        "source_revision": "sha256:" + "1" * 64,
        "source_record_id": "r1",
        "source_file": "raw/fresh.txt",
        "source_file_sha256": "1" * 64,
        "group_identity": "group-1",
        "task_id": "task-1",
        "tree_id": "tree-1",
        "trajectory_id": "trajectory-1",
        "document_id": "document-1",
        "repository_path_revision": "repo|path|rev",
    }
    left = {**base, "id": "a" * 32, "tier": "FIT-TRAIN", "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"}
    right = {**base, "id": "b" * 32, "tier": "FIT-DEV", "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen changed"}
    audit = audit_v23_tier_disjointness([left, right])
    assert audit["status"] == "FAIL"
    assert audit["group_conflicts"]
    assert audit["near_duplicate_conflicts"]


def test_historical_overlap_audit_checks_normalized_text_and_source_record() -> None:
    fresh = {"id": "a" * 32, "text": "Same content here", "source_name": "new", "source_record_id": "new-1"}
    historical = {"id": "b" * 32, "text": "Same   content here", "source_name": "old", "source_record_id": "old-1"}
    audit = audit_v22_overlap([fresh], historical_rows=[historical])
    assert audit["status"] == "FAIL"
    assert audit["dimensions"]["normalized_text"]["status"] == "FAIL"


def test_unapproved_source_license_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path, "bad", "text")
    source["license"] = "unknown"
    with pytest.raises(ValueError, match="license"):
        build_corpus(output=tmp_path / "out", sources=[source, _source(tmp_path, "ok", "other text")])


def test_gutenberg_boilerplate_is_removed_only_from_prepared_text() -> None:
    source = {"source_id": "gutenberg-example", "url": "https://www.gutenberg.org/cache/epub/1/pg1.txt"}
    text = "before\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nwork\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nafter\n"
    assert _prepare_source_text(source, text) == "work"
    assert _prepare_source_text({"source_id": "other", "url": "https://example.test/raw"}, text) == text
