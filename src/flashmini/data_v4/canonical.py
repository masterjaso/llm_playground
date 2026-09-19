"""Conservative canonicalization + stable document identity (v4)."""

from __future__ import annotations

import hashlib
import re

_WS_LINE = re.compile(r"[ \t]+", re.MULTILINE)

COOKIE_PATTERNS = (
    "accept cookies",
    "cookie policy",
    "subscribe to our newsletter",
    "click here to subscribe",
    "privacy policy | terms of service | cookie",
)


def canonicalize_text(text: str) -> str:
    """Normalize junk whitespace while preserving technical structure."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces per line; keep headings, code indent, equations.
    lines = [ln.rstrip() for ln in t.split("\n")]
    t = "\n".join(lines)
    # Collapse 3+ blank lines to exactly two newlines (one blank line).
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    t = re.sub(r"\n{3}", "\n\n", t)
    return t.strip("\n")


def content_hash(text: str) -> str:
    return hashlib.sha256(canonicalize_text(text).encode("utf-8")).hexdigest()


def document_id(source_id: str, source_revision: str, record_id: str,
                normalized_content_hash: str) -> str:
    """Stable identity independent of ingestion order."""
    payload = "\0".join([source_id.strip(), source_revision.strip(),
                         record_id.strip(), normalized_content_hash.strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def looks_junk(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return True, "empty"
    low = text.lower()
    for pat in COOKIE_PATTERNS:
        if pat in low and len(text) < 2000:
            return True, "boilerplate"
    # Pathological repetition: one distinct 50-char window dominating.
    if len(text) > 2000:
        window = text[:50]
        if text.count(window) * len(window) > 0.5 * len(text):
            return True, "pathological_repetition"
    if "\x00" in text:
        return True, "binary"
    return False, ""
