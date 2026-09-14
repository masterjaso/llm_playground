"""Validated data accounting and deterministic fresh-corpus sampling."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch


def validate_document_boundaries(manifest, config):
    if (config.architecture_version >= 3 and manifest is not None
            and manifest.get("eos_id") != config.ple.eos_id):
        raise ValueError("dataset EOS differs from the configured PLE/document boundary EOS")


def validate_data_contract(dataset, config, seq_len, total_tokens, *, allow_repeated=False):
    if config.architecture_version >= 3 and getattr(dataset, "split", "train") != "train":
        raise ValueError("training requires the train split, not validation data")
    inputs, labels = dataset.get_batch(np.array([0]))
    if inputs.ndim != 2 or labels.shape != inputs.shape:
        raise ValueError("dataset input/label shapes must match and be rank two")
    actual = inputs.shape[1]
    if actual != seq_len or actual != config.max_seq_len:
        raise ValueError(f"sequence length mismatch: dataset={actual}, caller={seq_len}, config={config.max_seq_len}")
    if getattr(dataset, "seq_len", actual) != actual:
        raise ValueError("dataset sequence length metadata disagrees with actual data")
    manifest = {}
    if hasattr(dataset, "data_dir"):
        path = Path(dataset.data_dir) / "data_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("seq_len") != actual:
                raise ValueError("manifest sequence length mismatch")
            validate_document_boundaries(manifest, config)
    frozen = manifest.get("splits", {}).get("train", {}).get("scored_tokens")
    decisive = getattr(config, "experiment_mode", "screening") == "decisive"
    if decisive:
        if not hasattr(dataset, "verify_integrity"):
            raise ValueError("decisive training requires verifiable frozen data")
        dataset.verify_integrity()
        if manifest.get("format_version", 0) < 3 or not frozen:
            raise ValueError("decisive v3 training requires a version-3 frozen corpus manifest")
        for key in ("dataset_revision", "tokenizer_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get(key, ""))):
                raise ValueError(f"decisive dataset requires immutable {key}")
        if frozen < 250_000_000 and not allow_repeated:
            raise ValueError("decisive PoC requires at least 250M scored frozen training tokens")
    # Array fixtures have no provenance: their physical positions bound reuse,
    # but they can only supply mechanism-screening evidence.
    capacity = int(frozen) if frozen else len(dataset) * actual
    if capacity > len(dataset) * actual:
        raise ValueError("frozen token count exceeds physical training data capacity")
    if capacity <= 0:
        raise ValueError("frozen training corpus must contain scored tokens")
    repeated = total_tokens > capacity
    if config.architecture_version >= 3 and repeated and not allow_repeated:
        raise ValueError("training budget repeats the frozen corpus; use --allow-repeated-corpus for non-decisive screening")
    return {
        "actual_seq_len": actual,
        "frozen_scored_train_tokens": frozen,
        "intended_training_tokens": total_tokens,
        "implied_corpus_passes": total_tokens / capacity,
        "non_decisive_repeated_corpus": repeated,
        "corpus_reuse_override": bool(allow_repeated),
        "experiment_mode": "decisive" if decisive and not repeated and not allow_repeated else "screening",
        "sampling": "seeded_epoch_permutation_v1" if config.architecture_version >= 3 else "replacement_v2",
        "long_context_validated": False,
    }


class EpochSampler:
    """Rebuild the current permutation from seed/epoch; resume needs only position."""

    def __init__(self, rows: int, seed: int, consumed: int = 0):
        self.rows, self.seed, self.consumed = rows, seed, consumed
        self.epoch = -1
        self.order = None

    def take(self, count: int):
        epoch, offset = divmod(self.consumed, self.rows)
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63 - 1))
            self.order = torch.randperm(self.rows, generator=generator).numpy()
            self.epoch = epoch
        count = min(count, self.rows - offset)
        result = self.order[offset:offset + count]
        self.consumed += count
        return result


def remaining_sequences(tokens_seen, total_tokens, seq_len, batch_size):
    return min(batch_size, math.ceil((total_tokens - tokens_seen) / seq_len))
