from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_public_corpus_v2.py"
_SPEC = importlib.util.spec_from_file_location("fetch_public_corpus_v2", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FETCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FETCH)


def test_trajectory_renderer_excludes_private_reasoning_and_keeps_tools() -> None:
    text, private = _FETCH._trajectory_text(
        [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "visible answer",
                "reasoning_content": "private chain",
                "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}],
            },
        ]
    )
    assert private is True
    assert "visible answer" in text
    assert "tool_calls" in text
    assert "private chain" not in text


def test_source_deduplication_records_hashes() -> None:
    rows, dropped = _FETCH._deduplicate_records(
        [{"text": " alpha  beta "}, {"text": "alpha beta"}, {"text": "different"}]
    )
    assert dropped == 1
    assert len(rows) == 2
    assert all(len(row["content_sha256"]) == 64 for row in rows)
    assert all(len(row["normalized_content_sha256"]) == 64 for row in rows)


def test_trajectory_segments_are_bounded() -> None:
    segments = _FETCH._text_segments("x" * 65, max_chars=32, max_segments=3)
    assert [len(item) for item in segments] == [32, 32, 1]
