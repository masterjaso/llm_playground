from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense2moe.capture import (
    capture_activation_shards,
    capture_text_teacher_activations,
    fixed_split_metadata,
    is_pinned_source_revision,
    iter_activation_shards,
    resolve_corpus_records,
    tokenize_corpus_records,
)


class _FakeTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class TeacherCaptureContractTests(unittest.TestCase):
    def test_pinned_revision_requires_commit_sha(self) -> None:
        self.assertTrue(is_pinned_source_revision("a" * 40))
        self.assertFalse(is_pinned_source_revision("main"))
        self.assertFalse(is_pinned_source_revision("a" * 39))

    def test_missing_teacher_snapshot_is_truthful_and_writes_no_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "corpus.jsonl"
            text = "one two three"
            source.write_text(json.dumps({"id": "a", "text": text}) + "\n", encoding="utf-8")
            digest = hashlib.sha256(text.encode()).hexdigest()
            manifest = root / "dataset.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "CALIBRATION_READY",
                        "source": {"path": str(source)},
                        "tokenizer_revision": "tokenizer-rev",
                        "sequence_length": 8,
                        "train": [{"id": "a", "text_sha256": digest, "token_count": 3}],
                        "holdout": [{"id": "b", "text": "four five six", "text_sha256": hashlib.sha256(b"four five six").hexdigest(), "token_count": 3}],
                    }
                ),
                encoding="utf-8",
            )
            result = capture_text_teacher_activations(
                manifest,
                root / "missing-snapshot",
                root / "capture",
                layers=(0,),
                source_revision="a" * 40,
                tokenizer=_FakeTokenizer(),
            )
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["blocker_code"], "SOURCE_SNAPSHOT_MISSING")
            self.assertFalse((root / "capture").exists())

    def test_fixed_records_are_reopened_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "corpus.jsonl"
            text = "alpha beta"
            source.write_text(json.dumps({"id": "source-a", "text": text}) + "\n", encoding="utf-8")
            digest = hashlib.sha256(text.encode()).hexdigest()
            manifest = root / "dataset.json"
            manifest.write_text(
                json.dumps({"status": "CALIBRATION_READY", "source": {"path": str(source)}, "train": [{"id": "a", "text_sha256": digest, "token_count": 2}]}),
                encoding="utf-8",
            )
            records = resolve_corpus_records(manifest, "train")
            self.assertEqual(records[0]["text"], text)
            tokenized = tokenize_corpus_records(records, _FakeTokenizer(), split="train", sequence_length=1)
            self.assertEqual([item.input_ids for item in tokenized], [(0,), (1,)])
            metadata = fixed_split_metadata(records, split="train", tokenizer_revision="rev", tokenizer_hashes={"tokenizer.json": "hash"})
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["split_record_count"], 1)

    def test_split_shard_manifest_is_resumable_and_split_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            values = np.arange(12, dtype=np.float32).reshape(4, 3)
            metadata = {"dataset_hash": "dataset", "source_revision": "a" * 40, "split": "holdout"}
            first = capture_activation_shards(values, tmp, layer=0, split="holdout", manifest_name="layer-0000-holdout.json", shard_tokens=2, metadata=metadata)
            self.assertEqual(first["split"], "holdout")
            self.assertTrue(all(item["metadata"]["split"] == "holdout" for item in first["shards"]))
            resumed = capture_activation_shards(values, tmp, layer=0, split="holdout", manifest_name="layer-0000-holdout.json", shard_tokens=2, resume=True, metadata=metadata)
            self.assertEqual(resumed["status"], "CAPTURE_RESUMED")
            np.testing.assert_array_equal(np.concatenate(list(iter_activation_shards(Path(tmp) / "layer-0000-holdout.json", expected_split="holdout"))), values)


if __name__ == "__main__":
    unittest.main()
