"""Bounded build loop: stream window -> validate -> shard -> publish -> evict."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import cache as cache_mod
from . import dedupe as dedupe_mod
from . import filters as filters_mod
from . import hf_store, manifests, provenance
from . import packing as packing_mod
from . import recipes as recipes_mod
from . import registry as registry_mod
from . import shards as shards_mod
from . import source as source_mod
from .canonical import canonicalize_text, content_hash, document_id
from .splits import SPLIT_SALT_DEFAULT, assign_split


def _doc_from_record(rec: dict, src: dict, salt: str) -> dict | None:
    text = canonicalize_text(rec.get("text", ""))
    verdict = filters_mod.filter_document(text)
    if not verdict.keep:
        return {"_reject": verdict.reason}
    ch = content_hash(text)
    did = document_id(rec["source_id"], rec.get("revision", ""),
                      rec["record_id"], ch)
    split = assign_split(did, salt=salt)
    return {
        "document_id": did, "text": text, "content_hash": ch,
        "source_id": rec["source_id"], "domain": src.get("domain", ""),
        "language": src.get("language", "en"), "license": src.get("license", ""),
        "redistribution_class": src.get("redistribution_class", "review_required"),
        "split": split,
    }


def _open_or_reuse(streams: dict, sid: str, src: dict, cursor) -> tuple[object, object]:
    """Open a source iterator once per process; reuse across windows."""
    entry = streams.get(sid)
    if entry is None or cursor.offset != entry[1]:
        src_with_offset = dict(src)
        src_with_offset["_offset"] = cursor.offset
        opened = source_mod.open_source_stream(src_with_offset)
        if isinstance(opened, source_mod.SourceResult):
            return opened, cursor.offset
        _result, iterator = opened
        entry = (iterator, cursor.offset)
        streams[sid] = entry
    return entry[0], entry[1]


def cmd_build(args) -> int:
    hf_store.load_token()  # set HF_TOKEN for datasets streaming rate limits
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    reg = registry_mod.load_registry(Path(args.registry))
    rhash = recipes_mod.recipe_hash(recipe)
    state_path = Path(args.state)
    state = manifests.load_state(state_path)
    if state.get("recipe_hash") and state["recipe_hash"] != rhash:
        print("resume: recipe hash changed; recording new hash")
    state["recipe_name"] = recipe["name"]
    state["recipe_hash"] = rhash
    if not state.get("source_cursors"):
        state["source_cursors"] = {}
    cache_dir = cache_mod.cache_root(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = cache_mod.cache_max_bytes(args.cache_gb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exact = dedupe_mod.ExactDedupe()
    for h in state.get("seen_hashes", []):
        exact._seen.add(h)
    seen_hashes = set(exact._seen)
    buffer: list[dict] = []
    shard_idx = len(state.get("published_shards", []))
    total_docs = 0
    salt = getattr(args, "split_salt", SPLIT_SALT_DEFAULT)
    max_docs = getattr(args, "max_docs", 20000)
    window = getattr(args, "window", 500)
    shard_docs = getattr(args, "shard_docs", 2000)
    per_source = getattr(args, "per_source_docs", max_docs)
    streams: dict = {}
    for domain, dom in recipe["domains"].items():
        for sid in dom["sources"]:
            if total_docs >= max_docs:
                break
            src = dict(reg["sources"][sid])
            src["source_id"] = sid
            cursor = source_mod.SourceCursor(
                config=src.get("config"), split=src.get("split", "train"),
                offset=int(state["source_cursors"].get(sid, 0)))
            opened, start_offset = _open_or_reuse(streams, sid, src, cursor)
            if isinstance(opened, source_mod.SourceResult):
                res = opened
            else:
                res = source_mod.stream_records(
                    opened, src,
                    limit=min(window, max_docs - total_docs, per_source),
                    start_offset=start_offset)
            state["source_cursors"][sid] = cursor.offset
            if res.status != "OK":
                state.setdefault("errors", []).append(
                    {"source": sid, "status": res.status, "reason": res.reason})
                print(f"build: {sid} {res.status} {res.reason} (continuing)")
                continue
            for rec in res.records:
                if total_docs >= max_docs:
                    break
                doc = _doc_from_record(rec, src, salt)
                if doc is None:
                    continue
                if "_reject" in doc:
                    rej = state.setdefault("rejected_documents", {})
                    rej[doc["_reject"]] = rej.get(doc["_reject"], 0) + 1
                    continue
                if not exact.check(doc["content_hash"]):
                    state["exact_duplicates"] = state.get("exact_duplicates", 0) + 1
                    continue
                seen_hashes.add(doc["content_hash"])
                buffer.append(doc)
                total_docs += 1
                toks = state.setdefault("estimated_tokens_by_domain", {})
                toks[domain] = toks.get(domain, 0) + max(1, len(doc["text"]) // 4)
                if len(buffer) >= shard_docs:
                    shard_idx = _flush(buffer, out_dir, shard_idx, recipe, rhash,
                                       state, src, args)
                    buffer.clear()
            manifests.save_state(state_path, state)
    if buffer:
        _flush(buffer, out_dir, shard_idx, recipe, rhash, state,
               {"redistribution_class": "review_required"}, args)
        buffer.clear()
    state["selected_documents"] = state.get("selected_documents", 0) + total_docs
    state["seen_hashes"] = sorted(seen_hashes)
    state["last_successful_operation"] = "build"
    manifests.save_state(state_path, state)
    try:
        cache_mod.enforce_bound(cache_dir, max_bytes)
    except RuntimeError as exc:
        print(f"build: cache bound: {exc}")
        return 2
    print(f"build: docs_this_run={total_docs} shards={len(state.get('published_shards', []))}")
    return 0

def verify_remote_shard(repo: str, local_path: Path, manifest: dict,
                        hf_prefix: str = "shards") -> None:
    """Upload + verify remote size/sha; raises on any mismatch (fail-closed)."""
    from huggingface_hub import HfApi
    hf_store.upload_file(repo, local_path, f"{hf_prefix}/{local_path.name}",
                         commit_message=f"add {manifest['shard_id']}")
    api = HfApi(token=hf_store.load_token())
    info = api.get_paths_info(repo, f"{hf_prefix}/{local_path.name}",
                              repo_type="dataset")
    if isinstance(info, list):
        info = info[0] if info else None
    if info is None:
        raise ValueError("remote file not found after upload")
    remote_size = getattr(info, "size", None)
    lfs = getattr(info, "lfs", None)
    remote_sha = getattr(lfs, "sha256", None) if lfs else None
    ok = remote_size == manifest["bytes"]
    if remote_sha is not None:
        ok = ok and remote_sha == manifest["sha256"]
    if not ok:
        raise ValueError(
            f"remote identity mismatch size={remote_size} sha={remote_sha}")


def _flush(buffer: list[dict], out_dir: Path, idx: int, recipe: dict,
           rhash: str, state: dict, src: dict, args) -> int:
    """Write one buffer as per-class shards: publishable content separated from
    held (recipe-only) content so licensing gating is document-level, not
    shard-level."""
    publishable = [d for d in buffer
                   if d.get("redistribution_class") in provenance.PUBLISHABLE_CONTENT]
    held = [d for d in buffer
            if d.get("redistribution_class") not in provenance.PUBLISHABLE_CONTENT]
    next_idx = idx
    if publishable:
        next_idx = _write_and_maybe_publish(
            publishable, out_dir, next_idx, recipe, rhash, state, args, held=False)
    if held:
        next_idx = _write_and_maybe_publish(
            held, out_dir, next_idx, recipe, rhash, state, args, held=True)
    return next_idx


def _write_and_maybe_publish(docs: list[dict], out_dir: Path, idx: int,
                             recipe: dict, rhash: str, state: dict, args,
                             *, held: bool) -> int:
    shard_id = f"{recipe['name']}-shard-{idx:06d}" + ("-held" if held else "")
    out_path = out_dir / f"{shard_id}.parquet"
    manifest = shards_mod.write_shard(docs, out_path, shard_id=shard_id,
                                      recipe_name=recipe["name"], recipe_hash=rhash)
    manifest["split"] = "train"
    manifest["sequence_count"] = manifest["document_count"]
    hf_prefix = getattr(args, "hf_prefix", "shards")
    manifest["remote_path"] = f"{hf_prefix}/{out_path.name}"
    ok, reason = provenance.classify_for_publish(
        "review_required" if held else "mirror_allowed")
    manifest["publishable_content"] = ok
    manifest["publish_note"] = reason
    if not held and not getattr(args, "no_publish", False):
        try:
            info = hf_store.whoami()
            repo = hf_store.repo_id_for("data", info.get("name", ""))
            hf_store.ensure_repo(repo)
            verify_remote_shard(repo, out_path, manifest,
                                hf_prefix=getattr(args, "hf_prefix", "shards"))
            state["hf_revision"] = hf_store.remote_head_sha(repo)
            out_path.unlink(missing_ok=True)
            manifest["published"] = True
            manifest["evicted_local"] = True
        except Exception as exc:
            manifest["published"] = False
            manifest["publish_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            state.setdefault("errors", []).append(
                {"shard": shard_id, "error": manifest["publish_error"]})
    else:
        manifest["published"] = False
    state.setdefault("published_shards", []).append(manifest)
    state["published_bytes"] = state.get("published_bytes", 0) + manifest["bytes"]
    manifests.save_state(Path(args.state), state)
    print(f"build: shard {shard_id} docs={manifest['document_count']} "
          f"published={manifest['published']}")
    return idx + 1



def cmd_publish(args) -> int:
    state = manifests.load_state(Path(args.state))
    try:
        info = hf_store.whoami()
    except RuntimeError as exc:
        print(f"publish: {exc}")
        return 2
    repo = hf_store.repo_id_for("data", info.get("name", ""))
    hf_store.ensure_repo(repo)
    shard_dir = Path(getattr(args, "shard_dir", "training_data/manifests/shards"))
    published = 0
    failed = 0
    for entry in state.get("published_shards", []):
        if entry.get("published") or not entry.get("publishable_content"):
            continue
        local = shard_dir / entry["path"]
        if not local.exists():
            print(f"publish: SKIP {entry['shard_id']} (local shard absent)")
            failed += 1
            continue
        try:
            cache_mod.verify_sha256(local, entry["sha256"])
            remote_path = entry.get("remote_path") or f"shards/{local.name}"
            hf_store.upload_file(repo, local, remote_path,
                                 commit_message=f"add {entry['shard_id']}")
        except Exception as exc:
            entry["publish_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            state.setdefault("errors", []).append(
                {"shard": entry["shard_id"], "error": entry["publish_error"]})
            failed += 1
            manifests.save_state(Path(args.state), state)
            print(f"publish: FAIL {entry['shard_id']} {entry['publish_error']}")
            continue
        # Remote verification: size + sha via Hub metadata
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_store.load_token())
            info_f = api.get_paths_info(repo, remote_path, repo_type="dataset")
            if isinstance(info_f, list):
                info_f = info_f[0] if info_f else None
            if info_f is None:
                raise ValueError("remote file not found after upload")
            remote_size = getattr(info_f, "size", None)
            lfs = getattr(info_f, "lfs", None)
            remote_sha = getattr(lfs, "sha256", None) if lfs else None
            ok = remote_size == entry["bytes"]
            if remote_sha is not None:
                ok = ok and remote_sha == entry["sha256"]
            if not ok:
                raise ValueError(
                    f"remote identity mismatch size={remote_size} sha={remote_sha}")
        except Exception as exc:
            entry["publish_error"] = f"verify: {type(exc).__name__}: {str(exc)[:120]}"
            state.setdefault("errors", []).append(
                {"shard": entry["shard_id"], "error": entry["publish_error"]})
            failed += 1
            manifests.save_state(Path(args.state), state)
            print(f"publish: VERIFY-FAIL {entry['shard_id']}")
            continue
        entry["published"] = True
        entry["evicted_local"] = True
        local.unlink()
        published += 1
        print(f"publish: OK {entry['shard_id']} verified remotely, evicted local")
    state["hf_revision"] = hf_store.remote_head_sha(repo)
    manifests.save_state(Path(args.state), state)
    print(f"publish: done published={published} failed={failed} "
          f"pending={[s['shard_id'] for s in state.get('published_shards', []) if not s.get('published') and s.get('publishable_content')]}")
    return 1 if failed else 0



def cmd_verify(args) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("verify: FAIL no corpus manifest; run freeze first")
        return 1
    manifest = json.loads(manifest_path.read_text())
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    ok = manifest.get("recipe_hash") == recipes_mod.recipe_hash(recipe)
    shards = manifest.get("shards", [])
    valid = bool(shards) and all(s.get("sha256") for s in shards)
    print(f"verify: recipe_match={ok} shards={len(shards)} hashes_valid={valid}")
    print(f"verify: fingerprint={manifest.get('corpus_fingerprint','')}")
    return 0 if (ok and valid) else 1


def cmd_freeze(args) -> int:
    state = manifests.load_state(Path(args.state))
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    reg = registry_mod.load_registry(Path(getattr(args, "registry", "training_data/registry/sources.yaml")))
    manifest_path = Path(args.manifest)
    shards = state.get("published_shards", [])
    shard_hashes = [s["sha256"] for s in shards if s.get("sha256")]
    fp = shards_mod.corpus_fingerprint(
        registry_hash=registry_mod.registry_hash(reg),
        recipe_hash=recipes_mod.recipe_hash(recipe),
        filter_version=filters_mod.FILTER_VERSION,
        dedupe_version=dedupe_mod.DEDUPE_VERSION,
        split_salt=getattr(args, "split_salt", SPLIT_SALT_DEFAULT),
        shard_hashes=shard_hashes,
        tokenizer_identity="corpus-is-tokenizer-independent")
    state["corpus_fingerprint"] = fp
    manifests.save_state(Path(args.state), state)
    hf_repo = ""
    hf_revision = state.get("hf_revision", "")
    try:
        hf_repo = hf_store.repo_id_for("data", hf_store.whoami().get("name", ""))
    except RuntimeError:
        hf_repo = ""
    split_totals: dict[str, int] = {}
    for s in shards:
        for split, count in (s.get("split_distribution") or {}).items():
            split_totals[split] = split_totals.get(split, 0) + int(count)
    corpus = {
        "recipe_name": recipe["name"],
        "recipe_hash": recipes_mod.recipe_hash(recipe),
        "corpus_fingerprint": fp,
        "format_version": 4,
        "packing_version": packing_mod.PACKING_VERSION,
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "shards": shards,
        "split_totals": split_totals,
        "tokenizer": {
            "note": "corpus is tokenizer-independent; training declares its own "
                    "tokenizer identity in dataset_identity()",
            "default_tokenizer_id": getattr(args, "tokenizer", "gpt2"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(corpus, indent=2, sort_keys=True))
    print(f"freeze: fingerprint={fp} shards={len(shards)} "
          f"splits={split_totals} rev={hf_revision[:12]} -> {manifest_path}")
    return 0


def _smoke_dataset(args):
    from .sampler import RemoteShardDataset
    return RemoteShardDataset(
        Path(args.manifest), split=args.split, seq_len=args.seq_len,
        seed=args.seed, epoch=args.epoch,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        cache_gb=args.cache_gb, hf_repo=args.hf_repo,
        revision=args.revision, tokenizer_id=args.tokenizer,
        local_base=Path(args.local_base) if args.local_base else None,
        max_open_shards=args.max_open_shards)


def cmd_resume_check(args) -> int:
    """Prove resume-equivalence with the trainer's own index source.

    A run is advanced ``--batches`` steps; a second run stops halfway, keeps
    only the sampler state, and a *fresh* dataset/sampler restores that state
    and continues.  Every subsequent batch digest must be identical.
    """
    steps = max(2, int(args.batches))
    half = steps // 2

    def digests(dataset, sampler, count):
        out = []
        for _ in range(count):
            indices = sampler.take(args.batch_size)
            if len(indices) == 0:
                break
            x, _y = dataset.get_batch(indices)
            out.append(hashlib.sha256(x.tobytes()).hexdigest())
        return out

    uninterrupted = _smoke_dataset(args)
    a = digests(uninterrupted, uninterrupted.sampler_for(args.seed), steps)

    part = _smoke_dataset(args)
    sampler = part.sampler_for(args.seed)
    digests(part, sampler, half)
    checkpoint = sampler.state()
    del part, sampler

    fresh = _smoke_dataset(args)
    resumed_sampler = fresh.sampler_for(args.seed, consumed=checkpoint["consumed"])
    resumed = digests(fresh, resumed_sampler, steps - half)

    ok = a[half:] == resumed
    print(f"resume-check: batches={steps} half={half} identical={ok}")
    if not ok:
        for i, (left, right) in enumerate(zip(a[half:], resumed)):
            if left != right:
                print(f"resume-check: MISMATCH at batch {half + i}")
                break
        return 1
    print(f"resume-check: consumed={resumed_sampler.consumed} "
          f"next_batch_sha256={resumed[0][:16]} "
          f"tokens_after_resume={(len(resumed)) * args.batch_size * args.seq_len}")
    return 0


def cmd_train_smoke(args) -> int:
    ds = _smoke_dataset(args)
    integrity = ds.verify_integrity()
    print(f"smoke: sequences={len(ds)} integrity={integrity}")
    print(f"smoke: identity={ds.dataset_identity()}")
    sampler = ds.sampler_for(args.seed, consumed=args.consumed_batches * args.batch_size)
    seen = 0
    digest = hashlib.sha256()
    for _ in range(args.batches):
        batch = sampler.take(args.batch_size)
        if len(batch) == 0:
            break
        x, y = ds.get_batch(batch)
        assert x.shape == (len(batch), args.seq_len), x.shape
        assert y.shape == x.shape
        assert not (x < 0).any()
        digest.update(x.tobytes())
        seen += len(batch)
    print(f"smoke: consumed_sequences={seen} tokens={seen * args.seq_len} "
          f"downloads={ds.downloads} evictions={ds.evictions} "
          f"batch_sha256={digest.hexdigest()[:16]}")
    print(f"smoke: cache_root={ds.cache_root} notes={ds.notes[:3]}")
    return 0

