"""Tests for bounded, deterministic evaluation slices."""

from __future__ import annotations

import unittest

import torch

from flashmini.evaluation import (
    GATE_2P1M_MAX_SEQUENCES,
    GATE_100M_MAX_SEQUENCES,
    RESERVED_PREFIX_SEQUENCES,
    evaluate_bounded_slice,
    official_slice,
)


class _FakeDataset:
    """A minimal dataset exposing get_batch and __len__ for slice tests."""

    def __init__(self, num_sequences: int, seq_len: int = 4):
        self.num_sequences = num_sequences
        self.seq_len = seq_len

    def __len__(self):
        return self.num_sequences

    def get_batch(self, indices):
        import numpy as np

        indices = np.asarray(indices)
        inputs = np.zeros((len(indices), self.seq_len), dtype=np.int64)
        labels = np.full((len(indices), self.seq_len), -100, dtype=np.int64)
        labels[:, 1:] = 0  # one scored token per sequence
        return inputs, labels


class _FakeModel(torch.nn.Module):
    """A model whose forward returns uniform logits for a fixed vocab."""

    def __init__(self, vocab: int = 8):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, labels=None, **kwargs):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab)
        return {"logits": logits, "loss": torch.zeros(())}


class EvaluationSliceTests(unittest.TestCase):
    def test_official_slices(self):
        self.assertEqual(official_slice("2p1m"), (RESERVED_PREFIX_SEQUENCES, GATE_2P1M_MAX_SEQUENCES))
        self.assertEqual(official_slice("100m"), (RESERVED_PREFIX_SEQUENCES, GATE_100M_MAX_SEQUENCES))
        self.assertEqual(official_slice("250m"), (RESERVED_PREFIX_SEQUENCES, None))
        with self.assertRaises(ValueError):
            official_slice("bogus")

    def test_bounded_slice_reports_exact_boundaries(self):
        dataset = _FakeDataset(num_sequences=2048, seq_len=4)
        model = _FakeModel()
        result = evaluate_bounded_slice(
            model, dataset, torch.device("cpu"),
            skip_sequences=RESERVED_PREFIX_SEQUENCES,
            max_sequences=10, block_sequences=4,
            checkpoint_sha256="ckpt", data_manifest_sha256="manifest",
        )
        self.assertEqual(result["first_sequence"], RESERVED_PREFIX_SEQUENCES)
        self.assertEqual(result["final_sequence"], RESERVED_PREFIX_SEQUENCES + 9)
        self.assertEqual(result["num_sequences"], 10)
        self.assertEqual(result["scored_tokens"], 10 * 3)  # 3 scored tokens per seq
        self.assertEqual(result["checkpoint_sha256"], "ckpt")
        self.assertEqual(result["data_manifest_sha256"], "manifest")

    def test_skip_below_reserved_prefix_rejected(self):
        dataset = _FakeDataset(num_sequences=2048, seq_len=4)
        model = _FakeModel()
        with self.assertRaises(ValueError):
            evaluate_bounded_slice(
                model, dataset, torch.device("cpu"),
                skip_sequences=100, max_sequences=10, block_sequences=4,
            )

    def test_full_remaining_slice(self):
        dataset = _FakeDataset(num_sequences=2048, seq_len=4)
        model = _FakeModel()
        result = evaluate_bounded_slice(
            model, dataset, torch.device("cpu"),
            skip_sequences=RESERVED_PREFIX_SEQUENCES,
            max_sequences=None, block_sequences=16,
        )
        self.assertEqual(result["num_sequences"], 2048 - RESERVED_PREFIX_SEQUENCES)
        self.assertEqual(result["final_sequence"], 2047)


if __name__ == "__main__":
    unittest.main()
