"""Resumable machine-readable build state (v4): atomic JSON writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def default_state(recipe_name: str = "", recipe_hash: str = "") -> dict:
    return {
        "recipe_name": recipe_name,
        "recipe_hash": recipe_hash,
        "source_cursors": {},
        "selected_documents": 0,
        "rejected_documents": {},
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "published_shards": [],
        "published_bytes": 0,
        "estimated_tokens_by_domain": {},
        "hf_revision": "",
        "last_successful_operation": "",
        "errors": [],
        "corpus_fingerprint": "",
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return default_state()
    return json.loads(path.read_text())


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_successful_operation"] = state.get("last_successful_operation", "")
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
