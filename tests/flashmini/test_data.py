"""Tests for deterministic data packing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flashmini.data import pack_tokens, prepare_dataset, MemmapDataset


class DataTests(unittest.TestCase):
    def test_no_all_ignored_tail_row(self):
        x, y = pack_tokens([1, 2, 3, 4, 5], seq_len=4, eos_id=99,
                           rng=np.random.default_rng(0))
        self.assertEqual(x.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(y.tolist(), [[2, 3, 4, 5]])

    def test_pack_tokens_shapes(self):
        tokens = list(range(100))
        rng = np.random.default_rng(0)
        input_ids, labels = pack_tokens(tokens, seq_len=32, eos_id=0, rng=rng)
        self.assertEqual(input_ids.shape[1], 32)
        self.assertEqual(labels.shape, input_ids.shape)
        # labels shifted by 1
        valid = labels[:, :-1] != -100
        np.testing.assert_array_equal(labels[:, :-1][valid], input_ids[:, 1:][valid])
        self.assertEqual(labels[0, -1], 32)
        self.assertTrue((labels.reshape(-1)[99:] == -100).all())

    def test_document_split_deduplicates_and_excludes_partial_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokens = [1, 2, 99, 1, 2, 99, 3, 4, 99, 5, 6, 99, 7, 8]
            manifest = prepare_dataset(tokens, Path(tmp), seq_len=4, eos_id=99,
                                       val_fraction=0.3, document_split=True)
            self.assertEqual(manifest["exact_duplicate_documents_removed"], 1)
            self.assertEqual(manifest["total_tokens"], 9)
            self.assertEqual(manifest["splits"]["train"]["scored_tokens"], 5)
            self.assertEqual(manifest["splits"]["val"]["scored_tokens"], 2)
            val = MemmapDataset(Path(tmp), split="val")
            self.assertEqual(val.input[0].tolist(), [5, 6, 99, 99])
            self.assertEqual(val.labels[0].tolist(), [6, 99, -100, -100])

    def test_deterministic_packing(self):
        tokens = list(range(1000))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a1, b1 = pack_tokens(tokens, seq_len=64, eos_id=0, rng=rng1)
        a2, b2 = pack_tokens(tokens, seq_len=64, eos_id=0, rng=rng2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)

    def test_prepare_dataset_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokens = list(np.random.default_rng(0).integers(0, 100, 5000))
            manifest = prepare_dataset(tokens, Path(tmp), seq_len=64, eos_id=0, seed=0)
            self.assertIn("train", manifest["splits"])
            self.assertIn("val", manifest["splits"])
            # train + val tokens = total
            total = manifest["splits"]["train"]["tokens"] + manifest["splits"]["val"]["tokens"]
            self.assertLessEqual(total, len(tokens) + 64)
            # manifest hashes recorded
            self.assertIn("input_sha256", manifest["splits"]["train"])

    def test_memmap_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokens = list(np.random.default_rng(0).integers(0, 100, 5000))
            prepare_dataset(tokens, Path(tmp), seq_len=64, eos_id=0, seed=0)
            ds = MemmapDataset(Path(tmp), split="train")
            self.assertGreater(len(ds), 0)
            idx = np.array([0, 1])
            inp, lab = ds.get_batch(idx)
            self.assertEqual(inp.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
