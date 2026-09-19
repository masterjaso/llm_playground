"""Licensing/provenance gate: fail-closed redistribution check (v4)."""

from __future__ import annotations

# Only these classes may have content bytes published to our HF repo.
PUBLISHABLE_CONTENT = {"mirror_allowed", "generated_owned"}


def classify_for_publish(redistribution_class: str) -> tuple[bool, str]:
    if redistribution_class in PUBLISHABLE_CONTENT:
        return True, "content_publish_allowed"
    return False, (
        f"class {redistribution_class!r} is recipe-only: store provenance, "
        "not content bytes")


def required_attribution(source: dict) -> str:
    parts = [str(source.get("dataset_id", ""))]
    if source.get("revision"):
        parts.append(f"rev:{source['revision'][:12]}")
    if source.get("license"):
        parts.append(f"license:{source['license']}")
    return " | ".join(p for p in parts if p)
