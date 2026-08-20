from __future__ import annotations

import json
from pathlib import Path

import pytest

from dense2moe.lineage import build_lineage_index, classify_reuse, lineage_identity
from scripts.attach_corpus_v22_external import attach_external
from scripts.freeze_corpus_v22 import freeze_v22


def _row(index: int, tier: str, domain: str, tokens: int = 8) -> dict[str, object]:
    return {
        "id": f"record-{tier}-{index}",
        "text": f"independent source text {tier} {index} {domain}",
        "tier": tier,
        "domain": domain,
        "task_family": "agentic" if domain == "agentic-software-engineering" else "general",
        "source_family": domain,
        "source_name": f"source-{tier}",
        "source_revision": "a" * 40,
        "source_license": "MIT",
        "source_record_id": f"source-record-{tier}-{index}",
        "task_id": f"task-{tier}-{index}",
        "repository_id": f"repo-{tier}-{index}",
        "document_id": f"doc-{tier}-{index}",
        "token_count": tokens,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _development_rows() -> list[dict[str, object]]:
    return [
        _row(0, "FIT-TRAIN", "code", 44),
        _row(1, "FIT-TRAIN", "agentic-software-engineering", 28),
        _row(2, "FIT-TRAIN", "software-engineering-natural-language", 12),
        _row(3, "FIT-TRAIN", "general", 10),
        _row(4, "FIT-TRAIN", "structured", 6),
        _row(8, "FIT-DEV", "general", 1),
        _row(5, "GATE-A", "general"),
        _row(6, "SHADOW-B", "general"),
        _row(7, "SHADOW-C", "general"),
    ]


def test_lineage_reuse_states_are_explicit() -> None:
    expected = lineage_identity(artifact_kind="activation", dataset_hash="d", layer=0, dtype="bf16")
    assert classify_reuse(None, expected)["state"] == "RECOMPUTE"
    assert classify_reuse(expected, expected)["state"] == "REUSE"
    assert classify_reuse(expected, lineage_identity(artifact_kind="activation", dataset_hash="other"))["state"] == "RECOMPUTE"
    assert classify_reuse(expected, expected, historical=True)["state"] == "BASELINE_ONLY"
    assert classify_reuse(expected, expected, blocked_reason="tier retired")["state"] == "BLOCKED"


def test_lineage_index_counts_and_hashes() -> None:
    identity = lineage_identity(artifact_kind="checkpoint", profile="p16", seed=17)
    result = build_lineage_index(
        run_id="run-1",
        method_version="m01",
        runtime_lock_sha256="lock",
        artifacts=[{"name": "checkpoint", "path": "layer.json", "identity": identity, "existing_identity": identity}],
    )
    assert result["counts"]["REUSE"] == 1
    assert result["index_sha256"]


def test_staged_freeze_passes_without_external_tiers(tmp_path: Path) -> None:
    source = _write_rows(tmp_path / "development.jsonl", _development_rows())
    result = freeze_v22(sources=[source], output=tmp_path / "base", require_agent_tasks=0, planned_tokens=100, component="development-internal")
    assert result["phase_gate"] == "PASS"
    receipt = json.loads((tmp_path / "base" / "corpus-v2.2-receipt.json").read_text(encoding="utf-8"))
    assert receipt["external_attachment_required"] is True
    assert receipt["tier_counts"]["G1"] == 0
    assert receipt["tier_counts"]["G2"] == 0


def test_external_attachment_requires_lock_and_is_one_way(tmp_path: Path) -> None:
    source = _write_rows(tmp_path / "development.jsonl", _development_rows())
    base_dir = tmp_path / "base"
    freeze_v22(sources=[source], output=base_dir, require_agent_tasks=0, planned_tokens=100, component="development-internal", method_version="m01")
    external = _write_rows(tmp_path / "g1.jsonl", [_row(20, "G1", "general", 8)])
    lock = tmp_path / "method-lock.json"
    lock.write_text(json.dumps({"status": "METHOD_LOCKED", "method_version": "m01", "threshold_fingerprint": "sealed-qwen38-promotion-v1", "external_tuning_forbidden": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="method lock"):
        attach_external(parent_receipt=base_dir / "corpus-v2.2-receipt.json", sources=[external], output=tmp_path / "g1", tier="G1", method_lock=tmp_path / "missing-lock.json", method_version="m01")
    first = attach_external(parent_receipt=base_dir / "corpus-v2.2-receipt.json", sources=[external], output=tmp_path / "g1", tier="G1", method_lock=lock, method_version="m01")
    assert first["status"] == "CORPUS_V22_EXTERNAL_ATTACHED"
    with pytest.raises(ValueError, match="already exists"):
        attach_external(parent_receipt=tmp_path / "g1" / "corpus-v2.2-receipt.json", sources=[external], output=tmp_path / "g1-repeat", tier="G1", method_lock=lock, method_version="m01")
