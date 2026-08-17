from __future__ import annotations

import json

import pytest

from dense2moe.method_proof import (
    MethodProofBlocked,
    prepare_method_proof_data,
    select_method_proof_rows,
)


def _row(identifier: str, *, domain: str, split: str = "FIT-TRAIN", tokens: int = 20_000, benchmark: bool = False) -> dict[str, object]:
    return {
        "text": f"method proof row {identifier}",
        "token_count": tokens,
        "domain": domain,
        "split": split,
        "source_name": "fixture",
        "source_revision": "a" * 40,
        "source_record_id": identifier,
        "source_family": "fixture",
        "benchmark_membership": ["SWE-Bench-style"] if benchmark else [],
    }


def test_selection_is_clean_split_bound_and_diverse() -> None:
    rows = [
        _row("z-benchmark", domain="code", tokens=90_000, benchmark=True),
        _row("b-code", domain="code", tokens=12_000),
        _row("a-agent", domain="agentic-software-engineering", tokens=12_000),
        _row("eval", domain="code", split="FIT-DEV", tokens=90_000),
    ]

    selected, facts = select_method_proof_rows(rows, min_tokens=20_000)

    assert {item["source_record_id"] for item in selected} == {"a-agent", "b-code"}
    assert facts["selected_tokens"] == 24_000
    assert facts["excluded_benchmark_rows"] == 1
    assert all(item["split"] == "FIT-TRAIN" for item in selected)
    assert all(item["method_proof_split"] == "FIT-TRAIN" for item in selected)


def test_selection_blocks_without_required_diversity() -> None:
    with pytest.raises(MethodProofBlocked, match="METHOD_PROOF_DIVERSITY_BLOCKED"):
        select_method_proof_rows([_row("only-code", domain="code", tokens=40_000)])


def test_selection_blocks_when_clean_rows_are_too_short() -> None:
    with pytest.raises(MethodProofBlocked, match="METHOD_PROOF_TOKEN_BUDGET_BLOCKED"):
        select_method_proof_rows(
            [_row("code", domain="code", tokens=8_192), _row("technical", domain="structured", tokens=8_192)],
            min_tokens=32_768,
        )


def test_selection_blocks_optimizer_evaluation_identity_overlap() -> None:
    train = _row("train", domain="code", tokens=20_000)
    train["split_group"] = "repo:shared"
    eval_row = _row("eval", domain="structured", split="FIT-DEV", tokens=20_000)
    eval_row["split_group"] = "repo:shared"
    with pytest.raises(MethodProofBlocked, match="METHOD_PROOF_DATA_OVERLAP_BLOCKED"):
        select_method_proof_rows([train, eval_row], min_tokens=20_000)


def test_prepare_writes_manifest_and_receipt_without_mutating_source(tmp_path) -> None:
    source = tmp_path / "corpus-v2.1.jsonl"
    source_rows = [_row("code", domain="code", tokens=20_000), _row("technical", domain="structured", tokens=20_000)]
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows), encoding="utf-8")
    before = source.read_bytes()

    result = prepare_method_proof_data(source, tmp_path / "method-proof", min_tokens=32_768)

    assert result["status"] == "METHOD_PROOF_READY"
    assert (tmp_path / "method-proof" / "manifest.jsonl").exists()
    assert (tmp_path / "method-proof" / "receipt.json").exists()
    assert source.read_bytes() == before
    receipt = json.loads((tmp_path / "method-proof" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["manifest"]["sha256"] == result["manifest"]["sha256"]
    assert receipt["method_proof_policy"]["eligible_split"] == "FIT-TRAIN"
