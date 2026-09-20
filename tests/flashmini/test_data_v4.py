"""v4 data contract tests: determinism, dedupe, cache, sampler, secrets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


class _StubTokenizer:
    """Deterministic offline tokenizer for hermetic tests (no Hub download)."""

    eos_token_id = 1
    pad_token_id = 0
    vocab_size = 512

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 500) for c in text]}

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}


def _build_local_corpus(tmp: str, *, docs: int = 6, val_every: int = 3):
    """Write one canonical shard + frozen manifest readable without the Hub."""
    from flashmini.data_v4 import shards as SH
    root = Path(tmp)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    for i in range(docs):
        documents.append({
            "document_id": f"{i:064x}",
            "text": f"document {i} body text. " * 12,
            "source_id": "unit_source",
            "domain": "general_web",
            "language": "en",
            "license": "ODC-By-1.0",
            "redistribution_class": "mirror_allowed",
            "content_hash": f"{i:064x}",
            "split": "val" if i % val_every == 0 else "train",
        })
    shard = SH.write_shard(documents, shard_dir / "unit-shard-000000.parquet",
                           shard_id="unit-shard-000000", recipe_name="unit")
    shard["split"] = "train"
    shard["sequence_count"] = shard["document_count"]
    shard["remote_path"] = f"shards/{shard['path']}"
    manifest = {"recipe_hash": "recipehash", "corpus_fingerprint": "fp",
                "format_version": 4, "hf_repo": "local/unit",
                "hf_revision": "rev0", "shards": [shard]}
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path, shard_dir, shard


def _dataset(tmp: str, **kwargs):
    from flashmini.data_v4.sampler import RemoteShardDataset
    manifest_path, shard_dir, shard = _build_local_corpus(tmp)
    ds = RemoteShardDataset(
        manifest_path, seq_len=kwargs.pop("seq_len", 32),
        cache_dir=Path(tmp) / "cache", local_base=shard_dir,
        cache_gb=kwargs.pop("cache_gb", 0.05), **kwargs)
    ds._tokenizer = _StubTokenizer()
    return ds, shard


class PackingTests(unittest.TestCase):
    def test_pack_is_deterministic_and_eos_terminated(self):
        from flashmini.data_v4 import packing as PK
        ids = [5] * 10
        a = PK.pack_document(ids, seq_len=16, document_id="d", epoch=0,
                             eos_token_id=1, pad_token_id=0)
        b = PK.pack_document(ids, seq_len=16, document_id="d", epoch=0,
                             eos_token_id=1, pad_token_id=0)
        self.assertEqual(list(a), list(b))
        self.assertEqual(len(a), 16)
        self.assertEqual(a[10], 1)                    # EOS after content
        self.assertEqual(set(a[11:].tolist()), {0})   # pad tail

    def test_long_document_window_depends_on_doc_and_epoch(self):
        from flashmini.data_v4 import packing as PK
        long_ids = list(range(1000))
        e0 = PK.pack_document(long_ids, seq_len=64, document_id="doc-a",
                              epoch=0, eos_token_id=1, pad_token_id=0)
        other = PK.pack_document(long_ids, seq_len=64, document_id="doc-b",
                                 epoch=0, eos_token_id=1, pad_token_id=0)
        self.assertNotEqual(e0[0], other[0])

    def test_packed_cache_keyed_by_tokenizer_revision(self):
        from flashmini.data_v4 import packing as PK
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok = _StubTokenizer()
            texts, ids = ["hello there " * 5], ["d1"]
            first = PK.load_or_build_packed(
                cache_root=root, shard_sha256="a" * 64, texts=texts,
                document_ids=ids, tokenizer=tok, tokenizer_id="stub",
                tokenizer_revision="r1", seq_len=16, epoch=0)
            self.assertEqual(len(list((root / "packed").glob("*.npy"))), 1)
            second = PK.load_or_build_packed(
                cache_root=root, shard_sha256="a" * 64, texts=texts,
                document_ids=ids, tokenizer=tok, tokenizer_id="stub",
                tokenizer_revision="r1", seq_len=16, epoch=0)
            self.assertEqual(list(first.reshape(-1)), list(second.reshape(-1)))
            PK.load_or_build_packed(
                cache_root=root, shard_sha256="a" * 64, texts=texts,
                document_ids=ids, tokenizer=tok, tokenizer_id="stub",
                tokenizer_revision="r2", seq_len=16, epoch=0)
            self.assertEqual(len(list((root / "packed").glob("*.npy"))), 2)


class RemoteConsumeTests(unittest.TestCase):
    def test_batch_is_real_corpus_tokens_not_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds, _shard = _dataset(tmp)
            self.assertEqual(len(ds), 4)          # 6 docs, 2 are val
            order = ds.epoch_order(seed=0, epoch=0)
            x, y = ds.get_batch(order[:2])
            self.assertEqual(x.shape, (2, 32))
            self.assertTrue((x >= 2).all())       # stub tokenizer content range
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(list(y[0][:-1]), list(x[0][1:]))

    def test_split_isolation_uses_document_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_train, _ = _dataset(tmp)
            from flashmini.data_v4.sampler import RemoteShardDataset
            ds_val = RemoteShardDataset(
                Path(tmp) / "corpus_manifest.json", split="val", seq_len=32,
                cache_dir=Path(tmp) / "cache2", local_base=Path(tmp) / "shards")
            ds_val._tokenizer = _StubTokenizer()
            self.assertEqual(len(ds_train), 4)
            self.assertEqual(len(ds_val), 2)

    def test_identity_provenance_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds, _ = _dataset(tmp)
            ident = ds.dataset_identity()
            for key in ("manifest_sha256", "recipe_hash", "corpus_fingerprint",
                        "hf_repo", "hf_revision", "tokenizer_identity",
                        "packing_version", "seq_len", "format_version"):
                self.assertIn(key, ident)
            self.assertEqual(ident["format_version"], 4)
            self.assertEqual(ident["tokenizer_identity"], "gpt2@main")

    def test_corrupt_local_shard_is_rejected_and_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds, shard = _dataset(tmp)
            local = ds.local_shard_path(shard)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"truncated-not-a-parquet")
            report = ds.verify_integrity()
            self.assertIn(shard["shard_id"], report["corrupt_local"])
            self.assertFalse(local.exists())
            x, _y = ds.get_batch(np.asarray([0]))
            self.assertEqual(x.shape, (1, 32))

    def test_resume_equivalence_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_a, _ = _dataset(tmp)
            sampler_a = ds_a.sampler_for(seed=7)
            straight = [ds_a.get_batch(sampler_a.take(2))[0].tolist()
                        for _ in range(4)]
            ds_b, _ = _dataset(tmp)
            sampler_b = ds_b.sampler_for(seed=7)
            for _ in range(2):
                ds_b.get_batch(sampler_b.take(2))
            checkpoint = sampler_b.state()
            ds_c, _ = _dataset(tmp)
            sampler_c = ds_c.sampler_for(seed=7, consumed=checkpoint["consumed"])
            resumed = [ds_c.get_batch(sampler_c.take(2))[0].tolist()
                       for _ in range(2)]
            self.assertEqual(straight[2:], resumed)

    def test_small_cache_bound_evicts_inactive_and_keeps_reading(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            ds, _ = _dataset(tmp, cache_gb=0.0002)
            stray = ds.cache_root / "stray.bin"
            stray.write_bytes(b"z" * 300_000)
            os.utime(stray, (1, 1))
            x, _y = ds.get_batch(np.asarray([0]))
            self.assertEqual(x.shape, (1, 32))
            self.assertFalse(stray.exists())
            used = sum(p.stat().st_size for p in ds.cache_root.rglob("*")
                       if p.is_file())
            self.assertLessEqual(used, ds.max_bytes)


class SecretSafetyTests(unittest.TestCase):
    def test_no_token_in_generated_files(self):
        from flashmini.data_v4 import hf_store
        token = hf_store.load_token()
        if not token:
            self.skipTest("no HF token available")
        for pattern in ("**/*.json", "**/*.yaml"):
            for path in Path("training_data").glob(pattern):
                self.assertNotIn(token, path.read_text(errors="ignore"),
                                 f"token leaked into {path}")
        self.assertNotIn(token, Path(".gitignore").read_text())


class ProvenanceTests(unittest.TestCase):
    def test_fail_closed(self):
        ok, _ = P.classify_for_publish("recipe_only")
        self.assertFalse(ok)
        ok, _ = P.classify_for_publish("mirror_allowed")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

