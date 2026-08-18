from __future__ import annotations

import hashlib

import pytest

from dense2moe.data import (
    V21_QUARANTINE_SPLIT,
    audit_agent_task_diversity,
    audit_split_disjointness,
    audit_tokenizer_records,
    build_balanced_activation_plan,
    quarantine_benchmark_records,
    verify_immutable_artifacts,
    write_immutable_json,
)


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = [ord(char) for char in text]
        return [self.bos_token_id, *values, self.eos_token_id] if add_special_tokens else values


def _row(index: int, *, task: str = "", repo: str = "", split: str = "FIT-TRAIN", benchmark: bool = False) -> dict[str, object]:
    text = f"record-{index}"
    row: dict[str, object] = {
        "id": hashlib.sha256(text.encode()).hexdigest()[:32],
        "text": text,
        "token_count": len(text),
        "split": split,
        "domain": "agentic-software-engineering" if task else "code",
        "source_family": "agent-trajectory" if task else "real-repository",
        "source_name": "fixture",
        "source_revision": "v1",
        "source_license": "MIT",
        "source_record_id": f"record-{index}",
        "document_id": f"document-{index}",
        "repo": repo,
        "task_id": task,
    }
    if benchmark:
        row["benchmark_membership"] = ["SWE-Bench-style"]
    return row


def test_benchmark_rows_are_quarantined_with_provenance() -> None:
    rows, audit = quarantine_benchmark_records([_row(1, task="task-1", repo="repo-1", benchmark=True)])
    assert rows[0]["split"] == V21_QUARANTINE_SPLIT
    assert rows[0]["benchmark_original_split"] == "FIT-TRAIN"
    assert audit["status"] == "PASS"
    assert audit["excluded_records"] == 1
    assert audit["promotion_excluded"] is True


def test_overlap_audit_reports_repository_task_and_document_conflicts() -> None:
    rows = [
        _row(1, task="task-a", repo="repo-a", split="FIT-TRAIN"),
        _row(2, task="task-a", repo="repo-a", split="GATE-A"),
        _row(3, task="task-b", repo="repo-b", split="FIT-DEV"),
        _row(4, task="task-c", repo="repo-c", split="SHADOW-B"),
    ]
    rows[1]["document_id"] = rows[0]["document_id"]
    audit = audit_split_disjointness(rows)
    assert audit["status"] == "FAIL"
    assert "repository" in audit["overlap"]
    assert "task" in audit["overlap"]
    assert "document" in audit["overlap"]


def test_agent_diversity_excludes_benchmark_tasks() -> None:
    rows = [_row(index, task=f"task-{index}", repo=f"repo-{index}") for index in range(96)]
    rows.append(_row(100, task="benchmark-task", repo="benchmark-repo", benchmark=True))
    result = audit_agent_task_diversity(rows, minimum_tasks=96)
    assert result["status"] == "PASS"
    assert result["independent_non_benchmark_tasks"] == 96


def test_tokenizer_audit_recounts_special_tokens_and_detects_mismatch() -> None:
    rows = [_row(1)]
    audit = audit_tokenizer_records(rows, tokenizer=CharacterTokenizer(), tokenizer_revision="fixture-v1")
    assert audit["status"] == "PASS"
    assert audit["record_recounts"][0]["recount_token_count"] == len("record-1")
    assert audit["record_recounts"][0]["special_token_delta"] == 2
    rows[0]["token_count"] = 1
    failed = audit_tokenizer_records(rows, tokenizer=CharacterTokenizer(), tokenizer_revision="fixture-v1")
    assert failed["status"] == "FAIL"
    assert failed["mismatches"][0]["reason"] == "token_count_mismatch"


def test_balanced_plan_is_deterministic_and_caps_task_tokens() -> None:
    rows = [
        {
            **_row(index, task=f"task-{index}", repo=f"repo-{index}"),
            "domain": "agentic-software-engineering",
            "token_count": 100,
        }
        for index in range(4)
    ]
    rows.extend({**_row(10 + index, repo=f"repo-{index}"), "domain": "code", "token_count": 100} for index in range(4))
    first = build_balanced_activation_plan(
        rows,
        planned_tokens=800,
        target_fractions={"agentic-software-engineering": 0.5, "code": 0.5},
        task_token_cap=50,
        repository_token_cap=100,
        source_family_token_cap=800,
    )
    second = build_balanced_activation_plan(
        rows,
        planned_tokens=800,
        target_fractions={"agentic-software-engineering": 0.5, "code": 0.5},
        task_token_cap=50,
        repository_token_cap=100,
        source_family_token_cap=800,
    )
    assert first["selected_rows"] == second["selected_rows"]
    assert first["selected_tokens"] == 400
    assert first["concentration"]["task"]["max_tokens"] <= 50
    assert first["status"] == "REBALANCE_REQUIRED"


def test_non_repository_records_do_not_share_a_fake_repository_cap() -> None:
    rows = [
        {
            **_row(index, task=f"dialogue-{index}", repo=""),
            "domain": "general",
            "source_name": "dialogue/source",
            "source_family": "dialogue-public",
            "token_count": 100,
        }
        for index in range(4)
    ]
    result = build_balanced_activation_plan(
        rows,
        planned_tokens=200,
        target_fractions={"general": 1.0},
        task_token_cap=50,
        repository_token_cap=50,
        source_family_token_cap=200,
    )
    assert result["status"] == "READY_FOR_BALANCED_CAPTURE"
    assert result["selected_tokens"] == 200
    assert result["concentration"]["repository"]["groups"] == 4


def test_immutable_artifacts_refuse_mutation_and_verify(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    digest = write_immutable_json(path, {"status": "frozen", "rows": [1, 2]})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert write_immutable_json(path, {"status": "frozen", "rows": [1, 2]}) == digest
    with pytest.raises(ValueError, match="immutable artifact mismatch"):
        write_immutable_json(path, {"status": "changed"})
    evidence = verify_immutable_artifacts(tmp_path, {"artifact": {"path": path.name, "sha256": digest}})
    assert evidence["status"] == "PASS"
