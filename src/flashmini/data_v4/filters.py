"""Lightweight conservative quality filters (v4).

Deliberately NOT n-gram-bigram removers: natural linguistic recurrence is
signal. Only document-level junk / pathological repetition is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import looks_junk

FILTER_VERSION = "flashmini-v4-filters-v1"


@dataclass
class FilterResult:
    keep: bool
    reason: str = ""


def filter_document(text: str, *, min_chars: int = 200,
                    max_chars: int = 1_000_000,
                    min_words: int = 20) -> FilterResult:
    n = len(text)
    if n < min_chars:
        return FilterResult(False, "too_short")
    if n > max_chars:
        return FilterResult(False, "too_long")
    if len(text.split()) < min_words:
        return FilterResult(False, "too_few_words")
    junk, reason = looks_junk(text)
    if junk:
        return FilterResult(False, reason)
    # Pathological char repetition (e.g. "aaaa..."): reject only extreme case.
    if n > 500:
        from collections import Counter
        top = Counter(text).most_common(1)[0][1] / n
        if top > 0.5:
            return FilterResult(False, "char_repetition")
    return FilterResult(True, "")
