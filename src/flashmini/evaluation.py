"""Bounded, deterministic evaluation slices for the v3 PoC gates.

An evaluation slice is a contiguous run of validation sequences that begins
after the reserved prefix and is bounded by an explicit sequence count. Every
report records the exact first sequence, final sequence, number of sequences,
number of scored tokens, the data manifest hash, and the checkpoint hash so a
gate report is fully reproducible.

The official slices are:

- 2.1M gate: skip the first 1024 validation sequences, evaluate the next 8192.
- 100M gate: skip the first 1024, evaluate the next 32768.
- 250M final: skip the first 1024, evaluate the entire remaining validation set.

A/B/C must always use the identical slice at a given gate.
"""

from __future__ import annotations

from typing import Any

import torch

from .eval import compute_validation_nll

RESERVED_PREFIX_SEQUENCES = 1024
GATE_2P1M_MAX_SEQUENCES = 8192
GATE_100M_MAX_SEQUENCES = 32768


class _BoundedSlice:
    """A contiguous slice of a dataset addressed by absolute sequence index."""

    def __init__(self, data, start: int, end: int):
        self.data = data
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    def get_batch(self, indices):
        return self.data.get_batch(indices + self.start)


def evaluate_bounded_slice(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    skip_sequences: int,
    max_sequences: int | None,
    block_sequences: int,
    ple_enabled: bool | None = None,
    checkpoint_sha256: str | None = None,
    data_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate a bounded slice and return a reproducible report.

    ``skip_sequences`` must be at least the reserved prefix. ``max_sequences``
    bounds the number of sequences evaluated; ``None`` evaluates the entire
    remaining validation set.
    """
    if skip_sequences < RESERVED_PREFIX_SEQUENCES:
        raise ValueError(
            f"skip_sequences must be at least the reserved prefix "
            f"({RESERVED_PREFIX_SEQUENCES})"
        )
    if block_sequences <= 0:
        raise ValueError("block_sequences must be positive")
    total = len(dataset)
    start = skip_sequences
    if start >= total:
        raise ValueError("skip_sequences exceeds the validation set")
    end = total if max_sequences is None else min(start + max_sequences, total)
    if end <= start:
        raise ValueError("evaluation slice is empty")

    blocks: list[dict[str, Any]] = []
    for block_start in range(start, end, block_sequences):
        block_end = min(block_start + block_sequences, end)
        block = compute_validation_nll(
            model, _BoundedSlice(dataset, block_start, block_end), device,
            ple_enabled=ple_enabled,
        )
        blocks.append(block)

    tokens = sum(b["tokens"] for b in blocks)
    if tokens <= 0:
        raise ValueError("evaluation slice has no scored tokens")
    nll = sum(b["tokens"] * b["nll"] for b in blocks) / tokens
    accuracy = sum(b["correct_tokens"] for b in blocks) / tokens
    return {
        "first_sequence": start,
        "final_sequence": end - 1,
        "num_sequences": end - start,
        "scored_tokens": int(tokens),
        "nll": nll,
        "perplexity": _perplexity(nll),
        "top1_accuracy": accuracy,
        "blocks": blocks,
        "block_sequences": block_sequences,
        "skip_sequences": skip_sequences,
        "max_sequences": max_sequences,
        "checkpoint_sha256": checkpoint_sha256,
        "data_manifest_sha256": data_manifest_sha256,
    }


def _perplexity(nll: float) -> float:
    import math

    return math.exp(min(nll, math.log(torch.finfo(torch.float64).max)))


def official_slice(gate: str) -> tuple[int, int | None]:
    """Return ``(skip_sequences, max_sequences)`` for a named official gate."""
    if gate == "2p1m":
        return RESERVED_PREFIX_SEQUENCES, GATE_2P1M_MAX_SEQUENCES
    if gate == "100m":
        return RESERVED_PREFIX_SEQUENCES, GATE_100M_MAX_SEQUENCES
    if gate == "250m":
        return RESERVED_PREFIX_SEQUENCES, None
    raise ValueError(f"unknown official gate {gate!r}")
