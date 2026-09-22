from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flashmini.data_v4 import (
    contamination,
    dedupe,
    overlay,
    packing,
    recipes,
    scheduler,
    shards,
    source,
    validation,
)
from flashmini.data_v4.tokenizer import TokenizerSpec, token_dtype_for_vocab


class _Tokenizer:
    eos_token_id = 9
    pad_token_id = 0
    vocab_size = 100

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(ch) % 50) for ch in text]}


class ProductionPrimitiveTests(unittest.TestCase):
    def test_stage_aggregates_are_exact(self):
        one = recipes.load_recipe(Path("training_data/recipes/flashmini_1b_full_v1.yaml"))
        self.assertEqual(sum(recipes.domain_token_targets(one).values()), 100_000_000_000)
        fifty = recipes.load_recipe(Path("training_data/recipes/flashmini_50b_full_v1.yaml"))
        self.assertEqual(sum(recipes.domain_token_targets(fifty).values()), 8_000_000_000_000)

    def test_sqlite_exact_dedupe_survives_restart_without_hash_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact.sqlite"
            with dedupe.ExactDedupe(path) as index:
                self.assertTrue(index.check("a" * 64))
                self.assertFalse(index.check("a" * 64))
            with dedupe.ExactDedupe(path) as index:
                self.assertFalse(index.check("a" * 64))
                self.assertEqual(len(index), 1)

    def test_sqlite_near_dedupe_survives_restart(self):
        text = "alpha beta gamma delta epsilon " * 20
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "near.sqlite"
            with dedupe.NearDedupeIndex(path, threshold=0.99) as index:
                self.assertTrue(index.check(text, "a" * 64))
            with dedupe.NearDedupeIndex(path, threshold=0.99) as index:
                self.assertFalse(index.check(text, "b" * 64))

    def test_minhash_bounds_long_document_shingles(self):
        text = " ".join(f"word-{i}" for i in range(100_000))
        first = dedupe.minhash_signature(text)
        second = dedupe.minhash_signature(text)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128)
        self.assertLessEqual(len(dedupe._shingles(text)), dedupe.MAX_SHINGLES)

    def test_scheduler_prioritizes_normalized_deficit(self):
        sched = scheduler.DeficitTokenScheduler({"web": 100, "code": 100}, seed=1)
        sched.record("web", 90)
        self.assertEqual(sched.choose_domain(), "code")
        state = sched.snapshot()
        self.assertEqual(scheduler.DeficitTokenScheduler.restore(state).deficits, sched.deficits)

    def test_contiguous_packing_consumes_long_document(self):
        tok = _Tokenizer()
        texts = ["a" * 3, "b" * 3, "c" * 80]
        packed, metadata = packing.pack_documents(
            texts, ["a", "b", "c"], tokenizer=tok, seq_len=16, return_metadata=True)
        self.assertEqual(packed.dtype, np.uint16)
        self.assertGreater(len(packed), len(texts))
        self.assertTrue(all(row["content_tokens"] <= 16 for row in metadata))

    def test_coherent_packing_keeps_short_documents_separate(self):
        packed, metadata = packing.pack_documents(
            ["a" * 3, "b" * 3], ["a", "b"], tokenizer=_Tokenizer(),
            seq_len=16, policy="coherent", return_metadata=True)
        self.assertEqual(len(packed), 2)
        self.assertEqual([row["document_indices"] for row in metadata], [[0], [1]])

    def test_validation_budget_never_exceeds_target(self):
        rows = [{"document_id": f"{i:064x}", "token_count": 7} for i in range(10)]
        train, val = validation.select_validation(rows, target_tokens=20)
        self.assertLessEqual(sum(x["token_count"] for x in val), 20)
        self.assertEqual(len(train) + len(val), len(rows))

    def test_cursor_round_trip_keeps_native_position(self):
        cursor = source.SourceCursor(
            config="c", split="train", offset=4, source_file="part-2.parquet",
            row_group=3, row_index=12, revision="a" * 40)
        self.assertEqual(source.SourceCursor.from_dict(cursor.as_dict()), cursor)

    def test_tokenizer_dtype_contract(self):
        self.assertEqual(token_dtype_for_vocab(65536), "uint16")
        self.assertEqual(token_dtype_for_vocab(65537), "uint32")
        with self.assertRaises(ValueError):
            TokenizerSpec("gpt2", "main", 50257, 50256)

    def test_benchmark_exclusion_loads_frozen_corpus_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "arc.json").write_text(json.dumps({"documents": ["benchmark prompt"]}))
            (root / "rules.yaml").write_text(
                "benchmarks: [ARC]\ncorpus_files:\n  ARC: arc.json\n")
            excluder = contamination.BenchmarkExcluder.from_path(root / "rules.yaml")
            self.assertEqual(excluder.loaded_documents, 1)
            self.assertTrue(excluder.check("benchmark prompt").excluded)
            report = excluder.report()
            self.assertTrue(report["complete"])
            self.assertEqual(report["missing_benchmarks"], [])

    def test_benchmark_exclusion_allows_partial_ingestion_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rules.yaml").write_text(
                "benchmarks: [ARC, FlashMiniEval]\ncorpus_files:\n  ARC: arc.json\n")
            (root / "arc.json").write_text(json.dumps({"documents": ["arc prompt"]}))
            excluder = contamination.BenchmarkExcluder.from_path(root / "rules.yaml")
            report = excluder.report()
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_benchmarks"], ["FlashMiniEval"])
            self.assertTrue(excluder.check("ordinary training text").excluded is False)

    def test_overlay_filters_published_canonical_shard_without_reingestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "canonical.parquet"
            docs = [
                {"document_id": "a", "text": "benchmark prompt", "content_hash": "a",
                 "source_id": "s", "domain": "web", "split": "train"},
                {"document_id": "b", "text": "ordinary training text", "content_hash": "b",
                 "source_id": "s", "domain": "web", "split": "train"},
            ]
            shards.write_shard(docs, shard, shard_id="canonical", exact_token_counts=[3, 4])
            rules = root / "rules.yaml"
            corpus = root / "arc.json"
            corpus.write_text(json.dumps({"documents": ["benchmark prompt"]}))
            rules.write_text("benchmarks: [ARC]\ncorpus_files:\n  ARC: arc.json\n")
            excluder = contamination.BenchmarkExcluder.from_path(rules)
            kept, counts, excluded, tokens = overlay._filter_canonical(shard, excluder)
            self.assertEqual([row["document_id"] for row in kept], ["b"])
            self.assertEqual(counts, [4])
            self.assertEqual((excluded, tokens), (1, 3))


if __name__ == "__main__":
    unittest.main()
