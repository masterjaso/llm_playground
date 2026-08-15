from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


class _CountingMLP:
    def __init__(self, torch: object, width: int) -> None:
        from torch import nn

        self._module = nn.Module()
        self._module.gate_proj = nn.Linear(width, width, bias=False)
        self._module.up_proj = nn.Linear(width, width, bias=False)
        self._module.down_proj = nn.Linear(width, width, bias=False)
        self._module.act_fn = nn.SiLU()

        def forward(value: object) -> object:
            gate = self._module.gate_proj(value)
            up = self._module.up_proj(value)
            return self._module.down_proj(self._module.act_fn(gate) * up)

        self._module.forward = forward  # type: ignore[method-assign]

    def module(self) -> object:
        return self._module


def _counting_teacher(torch: object, *, layers: int = 2, width: int = 4) -> object:
    from torch import nn

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _CountingMLP(torch, width).module()

        def forward(self, hidden: object) -> object:
            return hidden + self.mlp(hidden)

    class Teacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, width)
            self.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.config = SimpleNamespace(num_hidden_layers=layers)
            self.forward_calls = 0

        def forward(self, input_ids: object, attention_mask: object | None = None, use_cache: bool = False) -> object:
            del attention_mask, use_cache
            self.forward_calls += 1
            hidden = self.embed(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    return Teacher()


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

    def test_multi_layer_capture_fanout_counts_one_forward_per_microbatch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "corpus.jsonl"
            records = [
                {"id": f"train-{index}", "text": "alpha beta", "token_count": 2}
                for index in range(4)
            ]
            source.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            digest = hashlib.sha256(b"alpha beta").hexdigest()
            manifest = root / "dataset.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "CALIBRATION_READY",
                        "source": {"path": str(source)},
                        "tokenizer_revision": "a" * 40,
                        "sequence_length": 8,
                        "train": [
                            {"id": item["id"], "source_record_index": index, "text_sha256": digest, "token_count": 2}
                            for index, item in enumerate(records)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model = _counting_teacher(torch)
            result = capture_text_teacher_activations(
                manifest,
                root / "source-snapshot",
                root / "capture",
                layers=(0, 1),
                source_revision="a" * 40,
                split="train",
                microbatch=2,
                tokenizer=_FakeTokenizer(),
                model=model,
            )
            self.assertEqual(result["status"], "CAPTURE_COMPLETE")
            self.assertEqual(result["teacher_forward_count"], 2)
            self.assertEqual(result["expected_capture_forward_count"], 2)
            self.assertEqual(model.forward_calls, 3)  # one verification + two capture forwards
            self.assertTrue(result["fanout"]["single_forward_per_microbatch"])
            for layer in (0, 1):
                self.assertTrue((root / "capture" / f"layer-{layer:04d}-train.json").is_file())


if __name__ == "__main__":
    unittest.main()
