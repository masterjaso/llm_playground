"""Bounded immutable validation selection, separate from training targets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


def validation_priority(document_id: str, salt: str = "flashmini-validation-v1") -> int:
    return int.from_bytes(
        hashlib.sha256(f"{salt}\0{document_id}".encode()).digest()[:16], "big"
    )


@dataclass
class ValidationBudget:
    target_tokens: int
    salt: str = "flashmini-validation-v1"
    selected_tokens: int = 0
    selected_documents: int = 0

    def __post_init__(self) -> None:
        if int(self.target_tokens) < 0:
            raise ValueError("validation token budget cannot be negative")
        self.target_tokens = int(self.target_tokens)

    def consider(self, document_id: str, token_count: int) -> str:
        """Assign one document without ever exceeding the explicit budget."""
        tokens = max(0, int(token_count))
        if tokens and self.selected_tokens + tokens <= self.target_tokens:
            # Streaming fallback: caller may use this after a deterministic
            # hash split has already identified validation candidates.
            self.selected_tokens += tokens
            self.selected_documents += 1
            return "val"
        return "train"

    def snapshot(self) -> dict:
        return {
            "version": 1,
            "target_tokens": self.target_tokens,
            "salt": self.salt,
            "selected_tokens": self.selected_tokens,
            "selected_documents": self.selected_documents,
        }


def select_validation(documents: Iterable[dict], *, target_tokens: int,
                       salt: str = "flashmini-validation-v1") -> tuple[list[dict], list[dict]]:
    """Select a globally deterministic, bounded validation set.

    ``documents`` must contain ``document_id`` and ``token_count``.  Hash rank
    makes the result independent of source iteration order; the greedy budget
    ensures validation is additional to (and never deducted from) train.
    """
    rows = sorted(
        documents,
        key=lambda row: validation_priority(str(row["document_id"]), salt),
    )
    val: list[dict] = []
    train: list[dict] = []
    used = 0
    for row in rows:
        tokens = max(0, int(row.get("token_count", row.get("exact_tokens", 0))))
        if tokens and used + tokens <= int(target_tokens):
            val.append(row)
            used += tokens
        else:
            train.append(row)
    return train, val
