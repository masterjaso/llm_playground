"""End-to-end smoke for the production data paths.

The smoke uses the real Parquet writer, SQLite dedupe indexes, contiguous token
store/packing, cache, distributed sampler, and resume logic.  Remote Hub
verification is opt-in and always targets an explicitly supplied temporary
repository; the immutable pilot is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from flashmini.data_v4 import cache as cache_mod
from flashmini.data_v4 import dedupe, hf_store, materialize, packing, shards
from flashmini.data_v4.sampler import RemoteShardDataset
from flashmini.data_v4.tokenizer import TokenizerSpec, spec_fingerprint


class SmokeTokenizer:
    vocab_size = 512
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + ord(char) % 500 for char in text]}

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}


def _remote_smoke(repo: str, root: Path, shard_path: Path, manifest_path: Path,
                  shard: dict, *, tokenizer, tokenizer_spec,
                  delete_after: bool) -> dict:
    """Upload, verify metadata, redownload, and optionally delete a temp repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_store.load_token())
    existed = True
    try:
        api.dataset_info(repo)
    except Exception:  # noqa: BLE001 - not-found is the expected new-repo path
        existed = False
    hf_store.ensure_repo(repo, private=True)
    remote_prefix = "smoke/flashmini-data-v4"
    remote_shard = f"{remote_prefix}/{shard_path.name}"
    remote_manifest = f"{remote_prefix}/manifest.json"
    try:
        hf_store.upload_file(repo, shard_path, remote_shard,
                             commit_message="flashmini data v4 smoke shard")
        hf_store.upload_file(repo, manifest_path, remote_manifest,
                             commit_message="flashmini data v4 smoke manifest")
        info = hf_store.remote_file_info(repo, remote_shard)
        if info is None:
            raise RuntimeError("remote smoke shard metadata is unavailable")
        remote_size = getattr(info, "size", None)
        lfs = getattr(info, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None) if lfs else None
        if remote_size != shard["bytes"] or (remote_sha and remote_sha != shard["sha256"]):
            raise ValueError(
                f"remote smoke identity mismatch size={remote_size} sha={remote_sha}")
        revision = hf_store.remote_head_sha(repo) or None
        redownload = root / "redownload.parquet"
        hf_store.download_file(repo, remote_shard, redownload, revision=revision,
                                cache_dir=root / "hf-cache")
        cache_mod.verify_sha256(redownload, shard["sha256"])
        remote_manifest_data = json.loads(manifest_path.read_text())
        remote_manifest_data["hf_repo"] = repo
        remote_manifest_data["hf_revision"] = revision or ""
        remote_manifest_data["shards"][0]["remote_path"] = remote_shard
        remote_manifest_path = root / "remote-manifest.json"
        remote_manifest_path.write_text(json.dumps(remote_manifest_data, sort_keys=True))
        remote_dataset = RemoteShardDataset(
            remote_manifest_path, seq_len=32, seed=7, cache_dir=root / "remote-cache",
            cache_gb=0.01, min_avail_gb=0, packing_policy="document_mix",
            tokenizer_id=tokenizer_spec.tokenizer_id,
            tokenizer_revision=tokenizer_spec.revision, view_id="smoke-1b",
            stage="foundation")
        remote_dataset._tokenizer = tokenizer
        remote_batch, _ = remote_dataset.get_batch(np.asarray([0]))
        return {
            "repo": repo,
            "revision": revision,
            "uploaded": [remote_shard, remote_manifest],
            "remote_size": remote_size,
            "remote_sha256": remote_sha,
            "redownload_verified": True,
            "training_loader_batch_shape": list(remote_batch.shape),
            "training_loader_telemetry": remote_dataset.telemetry_snapshot(),
            "repository_created_for_smoke": not existed,
        }
    finally:
        if delete_after and not existed and "flashmini-data-smoke-" in repo:
            api.delete_repo(repo_id=repo, repo_type="dataset")


def run_smoke(output: str | Path | None = None, *, remote_repo: str | None = None,
              delete_remote_after: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix="flashmini-data-smoke-") as tmp:
        root = Path(tmp)
        shard_dir = root / "shards"
        docs = []
        for i in range(24):
            docs.append({
                "document_id": f"{i:064x}",
                "text": (f"domain {i % 3} source {i % 2} coherent document " * (8 + i)),
                "source_id": f"source_{i % 2}", "domain": f"domain_{i % 3}",
                "language": "en", "license": "MIT",
                "redistribution_class": "mirror_allowed",
                "content_hash": hashlib.sha256(str(i).encode()).hexdigest(),
                "split": "val" if i in {0, 7} else "train",
            })
        with dedupe.ExactDedupe(root / "exact.sqlite") as exact, \
                dedupe.NearDedupeIndex(root / "near.sqlite") as near:
            accepted = [doc for doc in docs
                        if exact.check(doc["content_hash"], doc["document_id"])
                        and near.check(doc["text"], doc["document_id"])]
            dedupe_counts = {"accepted": len(accepted), "exact_duplicates": exact.duplicates,
                             "near_duplicates": near.duplicates}
        tok = SmokeTokenizer()
        spec = TokenizerSpec(
            "smoke-tokenizer", "a" * 40, tok.vocab_size, tok.eos_token_id,
            fingerprint="")
        spec = TokenizerSpec(**{**spec.__dict__, "fingerprint": spec_fingerprint(spec)})
        shard_dir.mkdir(parents=True)
        tokens = [packing.encode_text(doc["text"], tok) + [tok.eos_token_id] for doc in accepted]
        counts = [len(row) for row in tokens]
        shard = shards.write_shard(
            accepted, shard_dir / "smoke-000000.parquet", shard_id="smoke-000000",
            recipe_name="smoke", recipe_hash="a" * 64, exact_token_counts=counts,
            tokenizer=spec.as_dict(), remote_path="shards/smoke-000000.parquet")
        shard["published"] = True
        shard["publishable_content"] = True
        manifest = {
            "release_id": "smoke-v1", "view_id": "smoke-1b",
            "recipe_hash": "a" * 64, "corpus_fingerprint": "b" * 64,
            "training_view_fingerprint": "c" * 64, "format_version": 4,
            "hf_repo": "local/smoke", "hf_revision": "local",
            "tokenizer": spec.as_dict(), "shards": [shard],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        materialized = materialize.materialize_canonical_shard(
            shard_dir / shard["path"], root / "tokens" / "smoke",
            tokenizer=tok, tokenizer_spec=spec, sequence_length=32)
        ds = RemoteShardDataset(
            manifest_path, seq_len=32, seed=7, cache_dir=root / "cache",
            cache_gb=0.01, local_base=shard_dir, packing_policy="document_mix",
            tokenizer_id=spec.tokenizer_id, tokenizer_revision=spec.revision,
            view_id="smoke-1b", stage="foundation")
        ds._tokenizer = tok
        sampler = ds.sampler_for(seed=19, world_size=2, rank=0)
        first = sampler.take(3)
        x, y = ds.get_batch(first)
        checkpoint = sampler.state()
        resumed = ds.sampler_for(seed=19, consumed=checkpoint["consumed"], world_size=2, rank=0)
        next_batch = resumed.take(3)
        cache_probe = root / "cache-probe"
        cache_probe.mkdir()
        cache_mod.atomic_write_bytes(cache_probe / "active.bin", b"a" * 64)
        cache_mod.atomic_write_bytes(cache_probe / "evictable.bin", b"b" * 64)
        evicted = cache_mod.enforce_bound(cache_probe, 64, active={"active.bin"})
        result = {
            "dedupe": dedupe_counts, "shard_bytes": shard["bytes"],
            "exact_tokens": shard["exact_token_count"],
            "materialized_tokens": materialized["token_count"],
            "materialized_sequences": materialized["packed"]["sequence_count"],
            "dataset_sequences": len(ds), "batch_shape": list(x.shape),
            "label_shape": list(y.shape), "resume_consumed": resumed.consumed,
            "next_batch_sha256": hashlib.sha256(next_batch.tobytes()).hexdigest(),
            "cache_telemetry": ds.telemetry_snapshot(),
            "cache_eviction": {"evicted": evicted, "active_preserved":
                                (cache_probe / "active.bin").is_file()},
            "remote_upload": "not attempted (pass --remote-repo for temporary Hub smoke)",
        }
        if remote_repo:
            result["remote_upload"] = _remote_smoke(
                remote_repo, root, shard_dir / shard["path"], manifest_path, shard,
                tokenizer=tok, tokenizer_spec=spec,
                delete_after=delete_remote_after)
        if output:
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, sort_keys=True))
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None)
    parser.add_argument("--remote-repo", default=None,
                        help="temporary Hub dataset repo for upload/redownload smoke")
    parser.add_argument("--delete-remote-after", action="store_true",
                        help="delete a newly-created flashmini-data-smoke-* repo after smoke")
    args = parser.parse_args()
    print(json.dumps(run_smoke(
        args.report, remote_repo=args.remote_repo,
        delete_remote_after=args.delete_remote_after), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
