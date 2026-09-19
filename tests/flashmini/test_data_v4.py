"""v4 data contract tests: determinism, dedupe, cache, sampler, secrets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flashmini.data_v4 import canonical as C
from flashmini.data_v4 import dedupe as D
from flashmini.data_v4 import provenance as P
from flashmini.data_v4 import recipes as R
from flashmini.data_v4 import registry as REG
from flashmini.data_v4 import splits as S
from flashmini.data_v4.cache import (
    atomic_write_bytes, enforce_bound, check_watermark, verify_sha256,
)


class RegistryTests(unittest.TestCase):
    def test_valid_registry_accepted(self):
        reg = REG.load_registry(Path("training_data/registry/sources.yaml"))
        self.assertIn("fineweb_edu", reg["sources"])

    def test_bad_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            p.write_text("sources:\n  bad:\n    dataset_id: x/y\n")
            with self.assertRaises(ValueError):
                REG.load_registry(p)

    def test_unknown_class_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            p.write_text("sources:\n  s:\n    dataset_id: x/y\n    domain: code\n"
                         "    redistribution_class: nope\n")
            with self.assertRaises(ValueError):
                REG.load_registry(p)

    def test_mutable_decisive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            p.write_text("sources:\n  s:\n    dataset_id: x/y\n    domain: code\n"
                         "    redistribution_class: mirror_allowed\n"
                         "    decisive: true\n    revision: main\n")
            with self.assertRaises(ValueError):
                REG.load_registry(p)

class DeterminismTests(unittest.TestCase):
    def test_same_document_same_id(self):
        t = "Hello world. " * 50
        ch = C.content_hash(t)
        self.assertEqual(C.document_id("src", "r" * 40, "1", ch),
                         C.document_id("src", "r" * 40, "1", ch))

    def test_reorder_does_not_change_split(self):
        texts = [f"document number {i} " * 40 for i in range(20)]
        first = [S.assign_split(C.document_id("s", "r" * 40, str(i),
                                             C.content_hash(t))) for i, t in enumerate(texts)]
        rev = list(enumerate(texts))[::-1]
        second = [S.assign_split(C.document_id("s", "r" * 40, str(i),
                                              C.content_hash(t))) for i, t in rev]
        self.assertEqual(sorted(first), sorted(second))

    def test_recipe_weights_sum(self):
        recipe = R.load_recipe(Path("training_data/recipes/flashmini_1b_full_v1.yaml"))
        self.assertAlmostEqual(sum(d["weight"] for d in recipe["domains"].values()), 1.0)


class DedupeTests(unittest.TestCase):
    def test_exact_duplicates_removed(self):
        dd = D.ExactDedupe()
        self.assertTrue(dd.check("a" * 64))
        self.assertFalse(dd.check("a" * 64))
        self.assertEqual(dd.duplicates, 1)

    def test_near_dedupe_deterministic(self):
        t = "the quick brown fox jumps over the lazy dog " * 30
        self.assertEqual(D.minhash_signature(t), D.minhash_signature(t))

    def test_partition_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = "alpha beta gamma delta epsilon zeta eta theta " * 40
            recs = [{"document_id": f"{i:064x}", "buckets": ["ab-00000000"],
                     "signature": D.minhash_signature(t)} for i in range(3)]
            paths = D.partition_signatures(recs, Path(tmp))
            self.assertEqual(len(paths), 1)
            out = D.resolve_partition(paths[0], threshold=0.99)
            self.assertEqual(out["members"], 3)


class CacheTests(unittest.TestCase):
    def test_bound_respected(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                (root / f"f{i}.bin").write_bytes(b"x" * 1024)
                os.utime(root / f"f{i}.bin", (i, i))
            evicted = enforce_bound(root, 2048)
            self.assertGreaterEqual(len(evicted), 2)

    def test_active_not_evicted(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "active.bin").write_bytes(b"x" * 3000)
            (root / "old.bin").write_bytes(b"y" * 3000)
            os.utime(root / "old.bin", (1, 1))
            os.utime(root / "active.bin", (2, 2))
            evicted = enforce_bound(root, 3500, active={"active.bin"})
            self.assertTrue((root / "active.bin").exists())
            self.assertIn(str(root / "old.bin"), evicted)

    def test_partial_not_valid_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            atomic_write_bytes(p, b"hello")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            with self.assertRaises(ValueError):
                verify_sha256(p, "0" * 64)

    def test_watermark(self):
        with self.assertRaises(RuntimeError):
            check_watermark(Path("/nonexistent-xyz-flashmini"), min_avail_bytes=10 ** 18)


class SamplerTests(unittest.TestCase):
    def _manifest(self, tmp: str) -> Path:
        m = {"recipe_hash": "r", "corpus_fingerprint": "f",
             "shards": [{"shard_id": f"s{i}", "sha256": "a" * 64,
                         "document_count": 10, "sequence_count": 10,
                         "split": "train"} for i in range(3)]}
        p = Path(tmp) / "corpus_manifest.json"
        p.write_text(json.dumps(m))
        return p

    def test_same_seed_same_order(self):
        from flashmini.data_v4.sampler import RemoteShardDataset
        with tempfile.TemporaryDirectory() as tmp:
            p = self._manifest(tmp)
            a = RemoteShardDataset(p, seed=1, epoch=0)
            b = RemoteShardDataset(p, seed=1, epoch=0)
            self.assertEqual(list(a.epoch_order(seed=1, epoch=0)),
                             list(b.epoch_order(seed=1, epoch=0)))

    def test_resume_continuation(self):
        from flashmini.data_v4.sampler import RemoteShardDataset
        with tempfile.TemporaryDirectory() as tmp:
            p = self._manifest(tmp)
            a = RemoteShardDataset(p, seed=5, epoch=1)
            batches = list(a.iter_epoch_batches(seed=5, epoch=1, batch_size=4))
            flat = [int(x) for b in batches[:2] for x in b]
            st = a.sampler_state()
            b = RemoteShardDataset(p, seed=5, epoch=1)
            full = list(b.epoch_order(seed=5, epoch=1))
            self.assertEqual(flat, list(full[:len(flat)]))
            b.restore_sampler_state(st)
            self.assertEqual(b.sampler_state()["position"], st["position"])

    def test_no_skip_no_dupe(self):
        from flashmini.data_v4.sampler import RemoteShardDataset
        with tempfile.TemporaryDirectory() as tmp:
            p = self._manifest(tmp)
            a = RemoteShardDataset(p, seed=9, epoch=2)
            order = list(a.epoch_order(seed=9, epoch=2))
            self.assertEqual(len(order), len(set(order)))
            self.assertEqual(len(order), len(a))


class ProvenanceTests(unittest.TestCase):
    def test_fail_closed(self):
        ok, _ = P.classify_for_publish("recipe_only")
        self.assertFalse(ok)
        ok, _ = P.classify_for_publish("mirror_allowed")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

