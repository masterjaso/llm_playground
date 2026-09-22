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


def load_registry(path: Path, *, lock_path: Path | None = None,
                  require_immutable: bool = False) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError("registry must contain a 'sources' mapping")
    sources = raw["sources"]
    if not isinstance(sources, dict):
        raise TypeError("registry 'sources' must be a mapping")
    for sid, src in sources.items():
        _validate_source(sid, src)
    if lock_path is not None:
        lock = json.loads(Path(lock_path).read_text())
        raw = merge_source_lock(raw, lock, require_immutable=require_immutable)
    elif require_immutable:
        raise ValueError("production registry requires source_snapshot.lock.json")
    return raw


def _validate_source(sid: str, src: dict) -> None:
    if not isinstance(src, dict):
        raise TypeError(f"source {sid}: must be a mapping")
    for field in ("dataset_id", "domain", "redistribution_class"):
        if field not in src:
            raise ValueError(f"source {sid}: missing required field '{field}'")
    if src["redistribution_class"] not in REDISTRIBUTION_CLASSES:
        raise ValueError(f"source {sid}: unknown redistribution class "
                         f"{src['redistribution_class']!r}")
    rev = src.get("revision")
    if src.get("decisive") and not (isinstance(rev, str) and _is_hex40(rev)):
        raise ValueError(f"source {sid}: decisive sources require immutable 40-hex revision")


def merge_source_lock(registry: dict, lock: dict, *, require_immutable: bool = True) -> dict:
    """Overlay immutable probe facts onto the human-edited source registry."""
    if not isinstance(lock, dict):
        raise TypeError("source lock must be a mapping")
    merged = json.loads(json.dumps(registry))
    for sid, source in merged.get("sources", {}).items():
        entry = lock.get(sid)
        if not isinstance(entry, dict):
            if require_immutable:
                raise ValueError(f"source lock missing entry for {sid}")
            continue
        if entry.get("dataset_id") and entry["dataset_id"] != source.get("dataset_id"):
            raise ValueError(f"source lock dataset mismatch for {sid}")
        revision = entry.get("revision") or source.get("revision")
        if require_immutable and not (isinstance(revision, str) and _is_hex40(revision)):
            raise ValueError(f"source lock for {sid} lacks immutable revision")
        source.update({
            "revision": revision,
            "config": entry.get("config", source.get("config")),
            "split": entry.get("split", source.get("split", "train")),
            "license": entry.get("license", source.get("license", "")),
            "gated": bool(entry.get("gated", source.get("gated", False))),
            "lock_card_sha256": entry.get("card_sha256", ""),
        })
        source["immutable_lock"] = True
    return merged


def validate_source_lock(registry: dict, lock: dict) -> dict:
    """Return a machine-readable lock health report without mutating input."""
    report = {"valid": True, "missing": [], "mutable": [], "mismatched": []}
    entries = lock if isinstance(lock, dict) else {}
    for sid, source in registry.get("sources", {}).items():
        entry = entries.get(sid)
        if not isinstance(entry, dict):
            report["missing"].append(sid)
            continue
        if entry.get("dataset_id") != source.get("dataset_id"):
            report["mismatched"].append(sid)
        if source.get("revision") and entry.get("revision") != source.get("revision"):
            report["mismatched"].append(sid)
        if not _is_hex40(str(entry.get("revision", ""))):
            report["mutable"].append(sid)
    report["valid"] = not any(report[key] for key in ("missing", "mutable", "mismatched"))
    return report


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
