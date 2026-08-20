from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.acquire_corpus_v23_public import _canonical_id, normalized_text_hash
from scripts.freeze_corpus_v23 import freeze_v23


def _row(index: int, tier: str) -> dict[str, object]:
    text = f"Fresh V23 source {index}. " + ("independent content token " * 8)
    source_name = f"fresh-source-{index}"
    source_revision = f"sha256:{index:064x}"
    source_record_id = f"source-{index}:record-0"
    source_hash = hashlib.sha256(f"raw-{index}".encode()).hexdigest()
    row: dict[str, object] = {
        "text": text,
        "tier": tier,
        "split": tier,
        "source_id": source_name,
        "source_name": source_name,
        "source_url": f"https://example.test/{source_name}.txt",
        "source_revision": source_revision,
        "source_license": "MIT",
        "download_sha256": source_hash,
        "source_file_sha256": source_hash,
        "acquisition_time_utc": "2026-08-18T20:00:00Z",
        "source_record_id": source_record_id,
        "source_record_index": 0,
        "source_file": f"raw/{source_name}.txt",
        "group_identity": f"v23-source:{source_name}",
        "split_group": f"v23-source:{source_name}",
        "source_lineage": f"v23-source:{source_name}",
        "task_id": f"task-{index}",
        "tree_id": f"tree-{index}",
        "trajectory_id": f"trajectory-{index}",
        "document_id": f"document-{index}",
        "repository_path_revision": f"repo-{index}|path-{index}|{source_revision}",
        "domain": "general",
        "task_family": "general",
        "source_family": source_name,
        "token_count": 20,
        "benchmark_membership": [],
    }
    row["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    row["normalized_content_sha256"] = normalized_text_hash(text)
    row["id"] = _canonical_id(row)
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_freeze_writes_required_immutable_artifacts_and_plan(tmp_path: Path) -> None:
    source = tmp_path / "prepared.jsonl"
    _write_rows(source, [_row(0, "FIT-TRAIN"), _row(1, "FIT-TRAIN"), _row(2, "FIT-TRAIN"), _row(3, "FIT-DEV")])
    result = freeze_v23(sources=[source], output=tmp_path / "out", planned_tokens=60, capture_commands=["acquire", "freeze"])
    assert result["phase_gate"] == "PASS"
    assert result["overlap_status"] == "PASS"
    required = {
        "v23-corpus.jsonl",
        "v23-tier-ledger.json",
        "v23-overlap-audit.json",
        "v23-source-receipt.json",
        "v23-freeze-receipt.json",
        "v23-activation-plan.json",
    }
    assert required.issubset({path.name for path in (tmp_path / "out").iterdir()})
    receipt = json.loads((tmp_path / "out" / "v23-freeze-receipt.json").read_text())
    assert receipt["evaluation_tiers_opened"] == []
    assert receipt["promotion_data_used"] is False
    assert receipt["token_counts"]["FIT-TRAIN"] == 60
    assert receipt["exact_capture_commands"] == ["acquire", "freeze"]


def test_freeze_rejects_evaluation_or_retired_tiers(tmp_path: Path) -> None:
    source = tmp_path / "prepared.jsonl"
    _write_rows(source, [_row(0, "GATE-A"), _row(1, "FIT-DEV")])
    with pytest.raises(ValueError, match="only FIT-TRAIN/FIT-DEV"):
        freeze_v23(sources=[source], output=tmp_path / "out", planned_tokens=10)


def test_freeze_rejects_cross_tier_historical_identity(tmp_path: Path) -> None:
    source = tmp_path / "prepared.jsonl"
    fresh = _row(0, "FIT-TRAIN")
    fresh["source_record_id"] = "historical-record"
    fresh["id"] = _canonical_id(fresh)
    _write_rows(source, [fresh, _row(1, "FIT-DEV"), _row(2, "FIT-TRAIN"), _row(3, "FIT-TRAIN")])
    historical = tmp_path / "v22.jsonl"
    old = {"id": "old-id", "text": "not overlapping", "source_record_id": "historical-record"}
    historical.write_text(json.dumps(old) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps V2.2"):
        freeze_v23(sources=[source], output=tmp_path / "out", v22_ledgers=[historical], planned_tokens=60)
