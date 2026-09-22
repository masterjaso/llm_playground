"""Licensing/provenance gate: fail-closed redistribution check (v4)."""

from __future__ import annotations

# Only these classes may have content bytes published to our HF repo.
PUBLISHABLE_CONTENT = {"mirror_allowed", "generated_owned"}
UPSTREAM_ONLY = {"recipe_only", "gated_recipe_only", "review_required"}


def classify_for_publish(redistribution_class: str) -> tuple[bool, str]:
    if redistribution_class in PUBLISHABLE_CONTENT:
        return True, "content_publish_allowed"
    return False, (
        f"class {redistribution_class!r} is recipe-only: store provenance, "
        "not content bytes")


def training_source_policy(source: dict) -> dict:
    """Return an explicit policy for a source used by a training view."""
    cls = str(source.get("redistribution_class", "review_required"))
    if cls in PUBLISHABLE_CONTENT:
        return {"mode": "mirror", "publish_content": True, "class": cls}
    if cls in UPSTREAM_ONLY:
        return {
            "mode": "upstream", "publish_content": False, "class": cls,
            "requires_revision": True, "requires_runtime_access": True,
        }
    return {
        "mode": "blocked", "publish_content": False, "class": cls,
        "reason": "unknown redistribution classification",
    }


def assert_training_resolvable(source: dict) -> None:
    """Fail closed when a manifest would point at unavailable held content."""
    policy = training_source_policy(source)
    if policy["mode"] == "blocked":
        raise ValueError(policy["reason"])
    if policy.get("requires_revision") and not source.get("revision"):
        raise ValueError(
            f"source {source.get('source_id', source.get('dataset_id', ''))} "
            "requires a pinned upstream revision")


def required_attribution(source: dict) -> str:
    parts = [str(source.get("dataset_id", ""))]
    if source.get("revision"):
        parts.append(f"rev:{source['revision'][:12]}")
    if source.get("license"):
        parts.append(f"license:{source['license']}")
    return " | ".join(p for p in parts if p)
