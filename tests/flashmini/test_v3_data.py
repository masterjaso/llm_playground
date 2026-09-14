"""Regression tests for the bounded, reproducible v3 data preparation path."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flashmini.data import MemmapDataset, prepare_streaming_documents


def _documents(count: int = 16):
    # Each document has a unique body while the final entry is a deliberate
    # duplicate.  Documents intentionally omit EOS; the preparer owns it.
    docs = [[100 + i, 200 + i, 300 + i, 400 + i] for i in range(count)]
    docs.append(list(docs[3]))
    return docs


def _prepare(path: Path, *, seed: int = 17, target: int | None = None) -> dict:
    return prepare_streaming_documents(
        _documents(),
        path,
        seq_len=4,
        eos_id=99,
        target_train_tokens=target,
        seed=seed,
        split_salt=f"test-salt-{seed}",
        val_fraction=0.5,
        provenance={
            "dataset_id": "test/frozen",
            "dataset_revision": "a" * 40,
            "tokenizer_id": "test-tokenizer",
            "tokenizer_revision": "b" * 40,
        },
    )


class V3DataTests(unittest.TestCase):
    def test_streaming_dedup_split_manifest_and_compact_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _prepare(Path(tmp))
            self.assertEqual(manifest["format_version"], 3)
            self.assertEqual(manifest["duplicate_doc_count"], 1)
            self.assertEqual(manifest["unique_doc_count"], 16)
            self.assertEqual(manifest["dataset_revision"], "a" * 40)
            self.assertEqual(manifest["tokenizer_revision"], "b" * 40)
            self.assertEqual(manifest["split_method"], "sha256_document_hash_threshold_v1")
            self.assertGreaterEqual(manifest["splits"]["train"]["scored_tokens"], 20)
            self.assertEqual(manifest["splits"]["train"]["input_dtype"], "int32")
            self.assertEqual(manifest["splits"]["val"]["labels_dtype"], "int32")

            train = MemmapDataset(Path(tmp), "train")
            val = MemmapDataset(Path(tmp), "val")
            self.assertTrue(train.verify_integrity()["valid"])
            self.assertEqual(train.input.dtype, np.dtype("int32"))
            self.assertEqual(train.labels.dtype, np.dtype("int32"))
            batch, labels = train.get_batch(np.array([0]))
            self.assertEqual(batch.dtype, np.dtype("int64"))
            self.assertEqual(labels.dtype, np.dtype("int64"))
            self.assertEqual(train.seq_len, 4)
            self.assertGreaterEqual(val.scored_tokens, 1)

            with sqlite3.connect(Path(tmp) / "document_dedupe.sqlite3") as db:
                rows = db.execute("SELECT split, COUNT(*) FROM seen_docs GROUP BY split").fetchall()
                self.assertEqual(sum(row[1] for row in rows), 16)
                overlap = db.execute(
                    "SELECT COUNT(*) FROM seen_docs GROUP BY digest HAVING COUNT(DISTINCT split) > 1"
                ).fetchall()
                self.assertEqual(overlap, [])

    def test_same_provenance_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            m1 = _prepare(first, seed=23)
            m2 = _prepare(second, seed=23)
            for split in ("train", "val"):
                for field in ("input_sha256", "labels_sha256", "scored_tokens", "raw_tokens"):
                    self.assertEqual(m1["splits"][split][field], m2["splits"][split][field])
                np.testing.assert_array_equal(
                    np.load(first / f"{split}_input.npy"),
                    np.load(second / f"{split}_input.npy"),
                )
            self.assertEqual(m1["document_dedupe_db"]["sha256"], m2["document_dedupe_db"]["sha256"])

    def test_changed_salt_changes_partition_without_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            _prepare(first, seed=1)
            _prepare(second, seed=2)
            def assignment(path: Path):
                with sqlite3.connect(path / "document_dedupe.sqlite3") as db:
                    return dict(db.execute("SELECT hex(digest), split FROM seen_docs"))
            a1, a2 = assignment(first), assignment(second)
            self.assertEqual(set(a1), set(a2))
            self.assertTrue(any(a1[digest] != a2[digest] for digest in a1))
            self.assertTrue(all(split in {"train", "val"} for split in a1.values()))

    def test_permutation_batches_cover_each_sequence_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            _prepare(Path(tmp), seed=7)
            data = MemmapDataset(Path(tmp), "train")
            batches = list(data.iter_epoch_batches(seed=17, epoch=2, batch_size=3))
            indices = np.concatenate(batches)
            np.testing.assert_array_equal(np.sort(indices), np.arange(len(data)))
            self.assertEqual(len(indices), len(data))
            self.assertFalse(np.array_equal(indices, np.arange(len(data))))

    def test_eos_in_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "omit the EOS"):
            prepare_streaming_documents(
                [[1, 99], [2, 3]],
                Path(tmp),
                seq_len=4,
                eos_id=99,
                val_fraction=0.5,
            )

    def test_no_fully_ignored_last_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare_streaming_documents([range(1, 9), range(9, 17)], Path(tmp),
                seq_len=8, eos_id=31, seed=0, val_fraction=0.5)
            for split in ("train", "val"):
                data = MemmapDataset(Path(tmp), split)
                self.assertTrue(np.all(np.any(data.labels != -100, axis=1)))
                self.assertEqual(manifest["splits"][split]["scored_tokens"], 8)
                self.assertEqual(len(data), 1)
            self.assertTrue(data.verify_integrity()["valid"])

    def test_source_token_limit_does_not_admit_oversized_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare_streaming_documents([range(1, 9), range(9, 17), range(17, 25)], Path(tmp),
                seq_len=8, eos_id=31, seed=0, val_fraction=0.5, max_source_tokens=17)
            self.assertEqual(manifest["source_body_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
