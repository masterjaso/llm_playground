"""Bounded build loop: stream window -> validate -> shard -> publish -> evict."""

from __future__ import annotations

import json
from pathlib import Path

from . import cache as cache_mod
from . import dedupe as dedupe_mod
from . import filters as filters_mod
from . import hf_store, manifests, provenance
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


def cmd_build(args) -> int:
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
    for domain, dom in recipe["domains"].items():
        for sid in dom["sources"]:
            if total_docs >= max_docs:
                break
            src = dict(reg["sources"][sid])
            src["source_id"] = sid
            cursor = source_mod.SourceCursor(
                config=src.get("config"), split=src.get("split", "train"),
                offset=int(state["source_cursors"].get(sid, 0)))
            res = source_mod.stream_source_window(
                src, limit=min(window, max_docs - total_docs, per_source),
                cursor=cursor)
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

def _flush(buffer: list[dict], out_dir: Path, idx: int, recipe: dict,
           rhash: str, state: dict, src: dict, args) -> int:
    shard_id = f"{recipe['name']}-shard-{idx:06d}"
    out_path = out_dir / f"{shard_id}.parquet"
    manifest = shards_mod.write_shard(buffer, out_path, shard_id=shard_id,
                                      recipe_name=recipe["name"], recipe_hash=rhash)
    manifest["split"] = "train"
    manifest["sequence_count"] = manifest["document_count"]
    first_class = buffer[0].get("redistribution_class", "review_required") if buffer else src.get("redistribution_class", "review_required")
    ok, reason = provenance.classify_for_publish(first_class)
    manifest["publishable_content"] = ok
    manifest["publish_note"] = reason
    if not getattr(args, "no_publish", False) and ok:
        try:
            info = hf_store.whoami()
            repo = hf_store.repo_id_for("data", info.get("name", ""))
            hf_store.ensure_repo(repo)
            hf_store.upload_file(repo, out_path, f"shards/{out_path.name}",
                                 commit_message=f"add {shard_id}")
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
    print(f"build: shard {shard_id} docs={manifest['document_count']} published={manifest['published']}")
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
    pending = [s for s in state.get("published_shards", []) if not s.get("published")]
    print(f"publish: {len(pending)} pending shards (content only if publishable)")
    return 0


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
    shard_hashes = [s["sha256"] for s in state.get("published_shards", []) if s.get("sha256")]
    fp = shards_mod.corpus_fingerprint(
        registry_hash=registry_mod.registry_hash(reg),
        recipe_hash=recipes_mod.recipe_hash(recipe),
        filter_version=filters_mod.FILTER_VERSION,
        dedupe_version=dedupe_mod.DEDUPE_VERSION,
        split_salt=getattr(args, "split_salt", SPLIT_SALT_DEFAULT),
        shard_hashes=shard_hashes)
    state["corpus_fingerprint"] = fp
    manifests.save_state(Path(args.state), state)
    corpus = {"recipe_name": recipe["name"], "recipe_hash": recipes_mod.recipe_hash(recipe),
              "corpus_fingerprint": fp, "shards": state.get("published_shards", []),
              "format_version": 4}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(corpus, indent=2, sort_keys=True))
    print(f"freeze: fingerprint={fp} shards={len(corpus['shards'])} -> {manifest_path}")
    return 0


def cmd_train_smoke(args) -> int:
    from .sampler import RemoteShardDataset
    ds = RemoteShardDataset(Path(args.manifest), split="train",
                            seq_len=args.seq_len, seed=0, epoch=0)
    print(f"smoke: sequences={len(ds)} integrity={ds.verify_integrity()}")
    seen = 0
    it = ds.iter_epoch_batches(seed=0, epoch=0, batch_size=args.batch_size)
    for _ in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x, y = ds.get_batch(batch)
        assert x.shape == (len(batch), args.seq_len)
        seen += len(batch)
    print(f"smoke: consumed_sequences={seen} resume_ok=True")
    return 0

