"""Source registry loading + validation (v4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REDISTRIBUTION_CLASSES = {
    "mirror_allowed", "recipe_only", "gated_recipe_only",
    "review_required", "generated_owned",
}

# 40-hex-char immutable commit required for decisive corpora.
_HEX40 = set("0123456789abcdef")


def _is_hex40(value: str) -> bool:
    return len(value) == 40 and all(c in _HEX40 for c in value.lower())


def load_registry(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError("registry must contain a 'sources' mapping")
    sources = raw["sources"]
    if not isinstance(sources, dict):
        raise ValueError("registry 'sources' must be a mapping")
    for sid, src in sources.items():
        _validate_source(sid, src)
    return raw


def _validate_source(sid: str, src: dict) -> None:
    if not isinstance(src, dict):
        raise ValueError(f"source {sid}: must be a mapping")
    for field in ("dataset_id", "domain", "redistribution_class"):
        if field not in src:
            raise ValueError(f"source {sid}: missing required field '{field}'")
    if src["redistribution_class"] not in REDISTRIBUTION_CLASSES:
        raise ValueError(f"source {sid}: unknown redistribution class "
                         f"{src['redistribution_class']!r}")
    rev = src.get("revision")
    if src.get("decisive") and not (isinstance(rev, str) and _is_hex40(rev)):
        raise ValueError(f"source {sid}: decisive sources require immutable 40-hex revision")


def registry_hash(registry: dict) -> str:
    return hashlib.sha256(
        json.dumps(registry, sort_keys=True).encode("utf-8")).hexdigest()


def source_lock_entry(source_id: str, resolved: dict) -> dict:
    """Build an immutable lock record from a resolved source probe."""
    return {
        "source_id": source_id,
        "dataset_id": resolved["dataset_id"],
        "revision": resolved["revision"],
        "config": resolved.get("config"),
        "split": resolved.get("split"),
        "license": resolved.get("license"),
        "gated": bool(resolved.get("gated", False)),
        "redistribution_class": resolved.get("redistribution_class"),
        "schema": resolved.get("schema"),
        "card_sha256": resolved.get("card_sha256"),
    }
