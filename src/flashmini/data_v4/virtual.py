"""Deterministic virtual-corpus view for production pretraining.

The view is a small immutable description of remote sources.  It never
materializes the complete 100B-token corpus: a worker opens one pinned source,
reads a bounded window, tokenizes it, and checkpoints the source/document/token
cursor.  The same state therefore resumes identically after a Kaggle session
ends.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .packing import encode_text
from .recipes import domain_token_targets, recipe_hash
from .scheduler import DeficitTokenScheduler
from .source import SourceCursor

VIRTUAL_VIEW_VERSION = "flashmini-virtual-view-v1"
VIRTUAL_STREAM_VERSION = "flashmini-virtual-stream-v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def document_selection_key(seed: int, source_id: str, document_id: str) -> int:
    """Stable rank independent of Python hash randomization or stream order."""
    digest = hashlib.sha256(f"{int(seed)}\0{source_id}\0{document_id}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


@dataclass(frozen=True)
class VirtualSource:
    source_id: str
    dataset_id: str
    revision: str
    split: str = "train"
    config: str | None = None
    text_field: str = "text"
    domain: str = ""
    weight: float = 1.0
    gated: bool = False
    redistribution_class: str = "mirror_allowed"

    @classmethod
    def from_mapping(cls, source_id: str, value: Mapping[str, Any], *, weight: float = 1.0) -> VirtualSource:
        revision = str(value.get("revision", ""))
        if len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision):
            raise ValueError(f"virtual source {source_id} requires a pinned 40-hex revision")
        return cls(
            source_id=str(source_id), dataset_id=str(value.get("dataset_id", "")),
            revision=revision, split=str(value.get("split", "train")),
            config=value.get("config"), text_field=str(value.get("text_field", "text")),
            domain=str(value.get("domain", "")), weight=float(weight),
            gated=bool(value.get("gated", False)),
            redistribution_class=str(value.get("redistribution_class", "mirror_allowed")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "dataset_id": self.dataset_id,
            "revision": self.revision, "split": self.split, "config": self.config,
            "text_field": self.text_field, "domain": self.domain,
            "weight": self.weight, "gated": self.gated,
            "redistribution_class": self.redistribution_class,
        }


@dataclass
class VirtualCursor:
    """All state needed to resume one virtual-source stream."""

    domain: str = ""
    source_id: str = ""
    source_offset: int = 0
    document_id: str = ""
    token_offset: int = 0
    exact_tokens: int = 0
    source_cursors: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain, "source_id": self.source_id,
            "source_offset": int(self.source_offset), "document_id": self.document_id,
            "token_offset": int(self.token_offset), "exact_tokens": int(self.exact_tokens),
            "source_cursors": {
                str(k): dict(v) for k, v in sorted(self.source_cursors.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> VirtualCursor:
        value = value or {}
        return cls(
            domain=str(value.get("domain", "")), source_id=str(value.get("source_id", "")),
            source_offset=int(value.get("source_offset", 0)),
            document_id=str(value.get("document_id", "")),
            token_offset=int(value.get("token_offset", 0)),
            exact_tokens=int(value.get("exact_tokens", 0)),
            source_cursors={str(k): dict(v) for k, v in (value.get("source_cursors") or {}).items()},
        )


class VirtualCorpus:
    """Immutable recipe/source selection with mutable resumable cursor."""

    def __init__(self, recipe: Mapping[str, Any], source_lock: Mapping[str, Mapping[str, Any]], *,
                 tokenizer_identity: str, seq_len: int = 2048, seed: int | None = None,
                 view_id: str = "flashmini-1b-virtual-foundation-v1", stage: str = "foundation",
                 cursor: VirtualCursor | None = None) -> None:
        self.recipe = dict(recipe)
        self.source_lock = {str(k): dict(v) for k, v in source_lock.items()}
        self.tokenizer_identity = str(tokenizer_identity)
        self.seq_len = int(seq_len)
        self.seed = int(self.recipe.get("seed", 0) if seed is None else seed)
        self.view_id = str(view_id)
        self.stage = str(stage)
        if self.seq_len <= 0 or not self.tokenizer_identity:
            raise ValueError("virtual corpus needs a positive sequence length and tokenizer identity")
        self.domain_targets = domain_token_targets(self.recipe)
        self.sources_by_domain: dict[str, list[VirtualSource]] = {}
        for domain, entry in self.recipe["domains"].items():
            sources = []
            for source_id in entry.get("sources", []):
                if source_id not in self.source_lock:
                    raise ValueError(f"recipe source {source_id!r} missing from source lock")
                source = VirtualSource.from_mapping(
                    source_id, self.source_lock[source_id], weight=float(entry.get("weight", 1.0)))
                sources.append(source)
            if not sources:
                raise ValueError(f"domain {domain!r} has no virtual sources")
            self.sources_by_domain[str(domain)] = sorted(sources, key=lambda item: item.source_id)
        self.scheduler = DeficitTokenScheduler(self.domain_targets, seed=self.seed)
        self.cursor = cursor or VirtualCursor()

    @property
    def fingerprint(self) -> str:
        return _sha({
            "version": VIRTUAL_VIEW_VERSION, "view_id": self.view_id,
            "stage": self.stage, "recipe_hash": recipe_hash(self.recipe),
            "tokenizer_identity": self.tokenizer_identity, "seq_len": self.seq_len,
            "seed": self.seed,
            "sources": {sid: value for sid, value in sorted(self.source_lock.items())},
        })

    def source_for(self, domain: str, ordinal: int | None = None) -> VirtualSource:
        """Select a source deterministically, without depending on stream order."""
        sources = self.sources_by_domain[str(domain)]
        ordinal = self.cursor.source_offset if ordinal is None else int(ordinal)
        digest = hashlib.sha256(f"{self.seed}\0{domain}\0{ordinal}".encode()).digest()
        # Weighted source selection uses exact decimal weights only as a stable
        # ranking hint; it never changes the domain token contract.
        return sources[int.from_bytes(digest[:8], "big") % len(sources)]

    def ordered_records(self, source: VirtualSource, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return a deterministic source window order from any provider."""
        normalized = []
        for index, row in enumerate(records):
            row = dict(row)
            document_id = str(row.get("document_id", row.get("id", row.get("record_id", index))))
            normalized.append((document_selection_key(self.seed, source.source_id, document_id), document_id, row))
        normalized.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in normalized]

    def state(self) -> dict[str, Any]:
        return {
            "version": VIRTUAL_VIEW_VERSION, "fingerprint": self.fingerprint,
            "view_id": self.view_id, "stage": self.stage, "seed": self.seed,
            "recipe_hash": recipe_hash(self.recipe), "tokenizer_identity": self.tokenizer_identity,
            "seq_len": self.seq_len, "scheduler": self.scheduler.snapshot(),
            "cursor": self.cursor.as_dict(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        if state.get("version") not in (None, VIRTUAL_VIEW_VERSION):
            raise ValueError("virtual data state version mismatch")
        if state.get("fingerprint") and state["fingerprint"] != self.fingerprint:
            raise ValueError("virtual data-view fingerprint mismatch")
        if state.get("recipe_hash") and state["recipe_hash"] != recipe_hash(self.recipe):
            raise ValueError("virtual recipe changed on resume")
        if state.get("tokenizer_identity") and state["tokenizer_identity"] != self.tokenizer_identity:
            raise ValueError("virtual tokenizer identity mismatch")
        self.scheduler = DeficitTokenScheduler.restore(dict(state.get("scheduler") or {}))
        if self.scheduler.targets != self.domain_targets:
            raise ValueError("virtual scheduler domain targets changed on resume")
        self.cursor = VirtualCursor.from_dict(state.get("cursor"))

    def record(self, *, domain: str, source_id: str, document_id: str, token_count: int,
               token_offset: int = 0) -> None:
        if domain not in self.sources_by_domain:
            raise KeyError(domain)
        if source_id not in {source.source_id for source in self.sources_by_domain[domain]}:
            raise ValueError(f"source {source_id!r} is not configured for domain {domain!r}")
        count = int(token_count)
        if count < 0 or count > self.scheduler.deficits[domain]:
            raise ValueError("record would exceed the exact domain token target")
        self.scheduler.record(domain, count)
        self.cursor.domain, self.cursor.source_id = domain, source_id
        self.cursor.source_offset += 1
        self.cursor.document_id = str(document_id)
        self.cursor.token_offset = int(token_offset)
        self.cursor.exact_tokens += count

    def next_source_request(self) -> tuple[str, VirtualSource, SourceCursor]:
        domain = self.scheduler.choose_domain(self.sources_by_domain)
        if domain is None:
            raise StopIteration
        source = self.source_for(domain)
        source_state = SourceCursor.from_dict(self.cursor.source_cursors.get(source.source_id))
        return domain, source, source_state

    def accept_window(self, domain: str, source: VirtualSource, records: Iterable[Mapping[str, Any]],
                      *, tokenizer: Any, max_documents: int = 64) -> list[dict[str, Any]]:
        """Tokenize and accept a bounded deterministic source window.

        Documents that would cross the exact domain target are clipped at the
        remaining token budget.  The returned rows retain provenance and the
        cursor records the precise token offset, so a restart does not guess.
        """
        accepted: list[dict[str, Any]] = []
        for row in self.ordered_records(source, records)[:max_documents]:
            document_id = str(row.get("document_id", row.get("id", row.get("record_id", ""))))
            text = row.get(source.text_field, row.get("text", ""))
            if not isinstance(text, str):
                continue
            ids = encode_text(text, tokenizer)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                ids = ids + [int(eos)]
            remaining = self.scheduler.deficits[domain]
            if remaining <= 0:
                break
            start = int(self.cursor.token_offset) if self.cursor.document_id == document_id else 0
            ids = ids[start:start + remaining]
            if not ids:
                continue
            self.record(domain=domain, source_id=source.source_id, document_id=document_id,
                        token_count=len(ids), token_offset=start + len(ids))
            accepted.append({
                "domain": domain, "source_id": source.source_id,
                "dataset_id": source.dataset_id, "revision": source.revision,
                "document_id": document_id, "token_offset": start,
                "token_count": len(ids), "tokens": np.asarray(ids, dtype=np.int64),
            })
            if self.scheduler.complete():
                break
        return accepted

    def iter_windows(self, reader: Callable[[VirtualSource, SourceCursor, int], tuple[Iterable[Mapping[str, Any]], SourceCursor]],
                     *, tokenizer: Any, window_documents: int = 64) -> Iterator[list[dict[str, Any]]]:
        """Yield accepted windows from a bounded remote reader."""
        while not self.scheduler.complete():
            domain, source, cursor = self.next_source_request()
            records, next_cursor = reader(source, cursor, window_documents)
            rows = self.accept_window(domain, source, records, tokenizer=tokenizer,
                                      max_documents=window_documents)
            if not rows:
                self.cursor.source_cursors[source.source_id] = next_cursor.as_dict()
                raise RuntimeError(f"source {source.source_id} produced no usable records")
            self.cursor.source_cursors[source.source_id] = next_cursor.as_dict()
            yield rows


class VirtualBatchStream:
    """Bounded exact-token packer over :class:`VirtualCorpus`.

    The reader owns remote access and returns one bounded source window.  This
    class keeps only a small pending token buffer, emits fixed ``(B, T)``
    batches, and serializes that buffer with the corpus cursor so a restart
    cannot skip or replay a partially consumed document.
    """

    def __init__(self, corpus: VirtualCorpus, *, tokenizer: Any,
                 reader: Callable[[VirtualSource, SourceCursor, int],
                                  tuple[Iterable[Mapping[str, Any]], SourceCursor]],
                 window_documents: int = 64) -> None:
        if int(window_documents) <= 0:
            raise ValueError("window_documents must be positive")
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.reader = reader
        self.window_documents = int(window_documents)
        self.pending = np.empty((0,), dtype=np.int64)
        self.sequences_emitted = 0
        self.consumed_exact_tokens = 0
        self.metrics = {
            "data_wait_seconds": 0.0,
            "network_errors": 0,
            "windows_read": 0,
            "ready_tokens": 0,
        }

    @property
    def fingerprint(self) -> str:
        return _sha({
            "version": VIRTUAL_STREAM_VERSION,
            "corpus": self.corpus.fingerprint,
            "seq_len": self.corpus.seq_len,
            "window_documents": self.window_documents,
        })

    def _fill(self, required_tokens: int) -> None:
        while int(self.pending.size) < int(required_tokens):
            started = time.monotonic()
            domain, source, cursor = self.corpus.next_source_request()
            try:
                records, next_cursor = self.reader(source, cursor, self.window_documents)
                rows = self.corpus.accept_window(
                    domain, source, records, tokenizer=self.tokenizer,
                    max_documents=self.window_documents,
                )
            except Exception:
                self.metrics["network_errors"] += 1
                raise
            self.metrics["data_wait_seconds"] += max(0.0, time.monotonic() - started)
            self.metrics["windows_read"] += 1
            self.corpus.cursor.source_cursors[source.source_id] = next_cursor.as_dict()
            if not rows:
                raise RuntimeError(f"virtual source {source.source_id} produced no usable rows")
            values = np.concatenate([row["tokens"] for row in rows]).astype(np.int64, copy=False)
            self.pending = np.concatenate([self.pending, values])
            self.metrics["ready_tokens"] = int(self.pending.size)

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        needed = batch_size * (self.corpus.seq_len + 1)
        self._fill(needed)
        window = self.pending[:needed].reshape(batch_size, self.corpus.seq_len + 1)
        self.pending = self.pending[needed:]
        self.sequences_emitted += batch_size
        self.consumed_exact_tokens += batch_size * self.corpus.seq_len
        self.metrics["ready_tokens"] = int(self.pending.size)
        return window[:, :-1].copy(), window[:, 1:].copy()

    def state(self) -> dict[str, Any]:
        return {
            "version": VIRTUAL_STREAM_VERSION,
            "fingerprint": self.fingerprint,
            "corpus": self.corpus.state(),
            "pending_tokens": self.pending.tolist(),
            "sequences_emitted": int(self.sequences_emitted),
            "consumed_exact_tokens": int(self.consumed_exact_tokens),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        if state.get("version") not in (None, VIRTUAL_STREAM_VERSION):
            raise ValueError("virtual stream state version mismatch")
        if state.get("fingerprint") and state["fingerprint"] != self.fingerprint:
            raise ValueError("virtual stream fingerprint mismatch")
        self.corpus.restore(dict(state.get("corpus") or {}))
        self.pending = np.asarray(state.get("pending_tokens") or [], dtype=np.int64)
        self.sequences_emitted = int(state.get("sequences_emitted", 0))
        self.consumed_exact_tokens = int(state.get("consumed_exact_tokens", 0))
        self.metrics["ready_tokens"] = int(self.pending.size)


__all__ = [
    "VIRTUAL_STREAM_VERSION", "VIRTUAL_VIEW_VERSION", "VirtualBatchStream",
    "VirtualCorpus", "VirtualCursor", "VirtualSource", "document_selection_key",
]
