"""Content-addressed science artifact lineage and reuse decisions.

Lineage is intentionally independent of any model provider.  A descendant can
be reused only when its complete scientific identity matches the requested
identity.  Historical or otherwise non-promotable evidence remains useful for
diagnostics, but is never silently promoted into a new experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

LINEAGE_SCHEMA_VERSION = 1
REUSE_STATES = frozenset({"REUSE", "BASELINE_ONLY", "RECOMPUTE", "BLOCKED"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def lineage_identity(*, artifact_kind: str, **fields: Any) -> dict[str, Any]:
    """Return a canonical identity and digest for one scientific artifact."""

    if not artifact_kind.strip():
        raise ValueError("artifact_kind must be non-empty")
    normalized = {str(key): value for key, value in fields.items() if value is not None}
    payload = {"schema_version": LINEAGE_SCHEMA_VERSION, "artifact_kind": artifact_kind, "fields": normalized}
    payload["identity_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def classify_reuse(
    existing: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    historical: bool = False,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    """Classify reuse without mutating the existing artifact.

    ``historical`` is an explicit caller assertion for old evidence.  It is
    never inferred from a missing field, preventing accidental promotion of
    legacy receipts.
    """

    if blocked_reason:
        return {"state": "BLOCKED", "reason": blocked_reason, "expected_identity": expected.get("identity_sha256")}
    if existing is None:
        return {"state": "RECOMPUTE", "reason": "artifact_missing", "expected_identity": expected.get("identity_sha256")}
    if historical:
        return {
            "state": "BASELINE_ONLY",
            "reason": "historical_evidence_requires_explicit_baseline_use",
            "existing_identity": existing.get("identity_sha256"),
            "expected_identity": expected.get("identity_sha256"),
        }
    if str(existing.get("identity_sha256", "")) == str(expected.get("identity_sha256", "")):
        return {"state": "REUSE", "reason": "identity_match", "existing_identity": existing.get("identity_sha256"), "expected_identity": expected.get("identity_sha256")}
    return {
        "state": "RECOMPUTE",
        "reason": "identity_mismatch_descendants_invalidated",
        "existing_identity": existing.get("identity_sha256"),
        "expected_identity": expected.get("identity_sha256"),
    }


def build_lineage_index(
    *,
    run_id: str,
    method_version: str,
    runtime_lock_sha256: str,
    artifacts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic index receipt for a science run."""

    if not run_id.strip() or not method_version.strip() or not runtime_lock_sha256.strip():
        raise ValueError("run_id, method_version, and runtime_lock_sha256 are required")
    entries: list[dict[str, Any]] = []
    for item in artifacts:
        name = str(item.get("name", "")).strip()
        identity = item.get("identity")
        if not name or not isinstance(identity, Mapping):
            raise ValueError("each lineage artifact requires name and identity")
        decision = classify_reuse(
            item.get("existing_identity") if isinstance(item.get("existing_identity"), Mapping) else None,
            identity,
            historical=bool(item.get("historical", False)),
            blocked_reason=str(item.get("blocked_reason")) if item.get("blocked_reason") else None,
        )
        if decision["state"] not in REUSE_STATES:
            raise ValueError("unknown lineage state")
        entries.append({"name": name, "decision": decision, "path": str(item.get("path", "")), "artifact_kind": identity.get("artifact_kind")})
    entries.sort(key=lambda value: value["name"])
    payload: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "receipt_type": "dense2moe-science-lineage-index",
        "run_id": run_id,
        "method_version": method_version,
        "runtime_lock_sha256": runtime_lock_sha256,
        "artifacts": entries,
        "counts": {state: sum(1 for entry in entries if entry["decision"]["state"] == state) for state in sorted(REUSE_STATES)},
    }
    payload["index_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


__all__ = ["LINEAGE_SCHEMA_VERSION", "REUSE_STATES", "build_lineage_index", "classify_reuse", "lineage_identity"]
