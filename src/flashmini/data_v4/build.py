"""Bounded build loop: stream window -> validate -> shard -> publish -> evict."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import cache as cache_mod
from . import contamination as contamination_mod
from . import dedupe as dedupe_mod
from . import filters as filters_mod
from . import hf_store, manifests, provenance
from . import packing as packing_mod
from . import recipes as recipes_mod
from . import registry as registry_mod
from . import scheduler as scheduler_mod
from . import shards as shards_mod
from . import source as source_mod
from . import tokenizer as tokenizer_mod
from . import validation as validation_mod
from .canonical import canonicalize_text, content_hash, document_id
from .splits import SPLIT_SALT_DEFAULT, assign_split


def _add_timing(state: dict, key: str, elapsed: float) -> None:
    telemetry = state.setdefault("telemetry", {})
    telemetry[key] = float(telemetry.get(key, 0.0)) + max(0.0, float(elapsed))


def _publish_repo(args) -> str:
    info = hf_store.whoami()
    return getattr(args, "hf_repo", None) or hf_store.repo_id_for(
        "production" if getattr(args, "production", False) else "data",
        info.get("name", ""))


def _doc_from_record(rec: dict, src: dict, salt: str,
                     telemetry: dict | None = None) -> dict | None:
    telemetry = telemetry if telemetry is not None else {}
    canonical_started = time.perf_counter()
    text = canonicalize_text(rec.get("text", ""))
    _add_timing({"telemetry": telemetry}, "canonicalization_seconds",
                time.perf_counter() - canonical_started)
    filter_started = time.perf_counter()
    verdict = filters_mod.filter_document(text)
    _add_timing({"telemetry": telemetry}, "filter_seconds",
                time.perf_counter() - filter_started)
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
        "source_revision": rec.get("revision", src.get("revision", "")),
        "record_id": rec.get("record_id", ""),
        "source_cursor": rec.get("source_cursor", {}),
        "split": split,
    }


def _open_or_reuse(streams: dict, sid: str, src: dict, cursor) -> tuple[object, object]:
    """Open a source iterator once per process; reuse across windows."""
    entry = streams.get(sid)
    if entry is None or cursor.offset != entry[1]:
        src_with_offset = dict(src)
        src_with_offset["_offset"] = cursor.offset
        src_with_offset["_cursor"] = cursor.as_dict()
        opened = source_mod.open_source_stream(src_with_offset)
        if isinstance(opened, source_mod.SourceResult):
            return opened, cursor.offset
        _result, iterator = opened
        entry = (iterator, cursor.offset)
        streams[sid] = entry
    return entry[0], entry[1]


def _load_cursor(value, src: dict) -> source_mod.SourceCursor:
    if isinstance(value, dict):
        return source_mod.SourceCursor.from_dict(value)
    return source_mod.SourceCursor(
        config=src.get("config"), split=src.get("split", "train"),
        offset=int(value or 0), revision=src.get("revision"))


def _exact_token_count(text: str, tokenizer) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(packing_mod.encode_text(text, tokenizer)) + 1  # explicit EOS boundary


def cmd_build(args) -> int:
    """Deficit-aware, restartable build loop.

    ``max_docs`` is only an operational chunk limit.  With no limit the loop
    terminates only when every exact domain target is met or all sources for a
    deficit are exhausted.
    """
    hf_store.load_token()
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    lock_path = Path(args.source_lock) if getattr(args, "source_lock", None) else None
    production = bool(getattr(args, "production", False))
    reg = registry_mod.load_registry(
        Path(args.registry), lock_path=lock_path, require_immutable=production)
    rhash = recipes_mod.recipe_hash(recipe)
    state_path = Path(args.state)
    state = manifests.load_state(state_path)
    if state.get("recipe_hash") and state["recipe_hash"] != rhash:
        print("resume: recipe hash changed; recording new hash")
    state["recipe_name"] = recipe["name"]
    state["recipe_hash"] = rhash
    state.setdefault("source_cursors", {})
    cache_dir = cache_mod.cache_root(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir = cache_dir / "hf-datasets"
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = cache_mod.cache_max_bytes(
        int(args.cache_gb * 1024 ** 3) if args.cache_gb else None)
    watermark_bytes = int(
        float(getattr(args, "free_space_watermark_gib", 10.0)) * 1024 ** 3)
    state["storage_contract"] = {
        "cache_max_bytes": max_bytes,
        "free_space_watermark_bytes": watermark_bytes,
        "evict_after_remote_verify": True,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedupe_path = Path(getattr(args, "dedupe_db", "") or state_path.with_suffix(".dedupe.sqlite3"))
    near_path = Path(getattr(args, "near_dedupe_db", "") or state_path.with_suffix(".near.sqlite3"))
    exact = dedupe_mod.ExactDedupe(dedupe_path)
    legacy = state.pop("seen_hashes", [])
    if legacy:
        exact.import_hashes(legacy)
        state.setdefault("migration_notes", []).append(
            f"migrated {len(legacy)} legacy hashes to {dedupe_path}")
    near = dedupe_mod.NearDedupeIndex(near_path)
    state["dedupe_db"] = str(dedupe_path)
    state["near_dedupe_db"] = str(near_path)
    state["near_dedupe_version"] = dedupe_mod.DEDUPE_VERSION
    contamination_path = getattr(args, "contamination_config", None)
    excluder = (contamination_mod.BenchmarkExcluder.from_path(contamination_path)
                if contamination_path and Path(contamination_path).exists() else None)
    if production and excluder is None:
        raise ValueError("production build requires a contamination config")
    if excluder is not None:
        state["benchmark_exclusion_version"] = contamination_mod.CONTAMINATION_VERSION
        state["benchmark_exclusion_config"] = str(contamination_path)
        report = excluder.report()
        state["benchmark_exclusion_status"] = (
            "complete" if report["complete"] else "partial_pending_exports")
        state["benchmark_exclusion_report"] = report
    elif production:
        state["benchmark_exclusion_status"] = "missing_config"

    tokenizer = None
    tok_spec = None
    tok_revision = getattr(args, "tokenizer_revision", None)
    tok_spec_path = getattr(args, "tokenizer_spec", None)
    if tok_spec_path:
        tok_spec = tokenizer_mod.load_spec(tok_spec_path)
        tok_revision = tok_spec.revision
        args.tokenizer = tok_spec.tokenizer_id
    if tok_revision:
        tokenizer = tokenizer_mod.load_tokenizer(
            getattr(args, "tokenizer", "gpt2"), tok_revision, production=production)
        runtime_fingerprint = packing_mod.tokenizer_fingerprint(tokenizer)
        state["tokenizer"] = {
            "identity": tokenizer_mod.tokenizer_identity(getattr(args, "tokenizer", "gpt2"), tok_revision),
            "fingerprint": runtime_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "spec_fingerprint": tok_spec.fingerprint if tok_spec else "",
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "eos_token_id": int(tokenizer.eos_token_id),
            "dtype": str(packing_mod.token_dtype_for_tokenizer(tokenizer)),
        }
    elif production:
        raise ValueError("production build requires --tokenizer-spec or --tokenizer-revision")

    targets = recipes_mod.domain_token_targets(recipe)
    saved_scheduler = state.get("scheduler") or {}
    scheduler = (scheduler_mod.DeficitTokenScheduler.restore(saved_scheduler)
                 if saved_scheduler.get("targets") else
                 scheduler_mod.DeficitTokenScheduler(targets, seed=int(recipe.get("seed", 0))))
    if scheduler.targets != targets:
        scheduler = scheduler_mod.DeficitTokenScheduler(targets, seed=int(recipe.get("seed", 0)))
    validation = validation_mod.ValidationBudget(
        int(recipe.get("validation_tokens", 0)),
        salt=str(recipe.get("validation_salt", "flashmini-validation-v1")),
        selected_tokens=int(state.get("validation_tokens", 0)),
        selected_documents=int(state.get("validation_documents", 0)),
    )
    state["validation_budget"] = validation.target_tokens
    state["training_target_tokens"] = int(recipe["target_tokens"])
    state["published_bytes"] = sum(
        int(row.get("bytes", 0)) for row in state.get("published_shards", [])
        if row.get("published"))
    state["published_documents"] = sum(
        int(row.get("document_count", 0)) for row in state.get("published_shards", [])
        if row.get("published"))
    state["published_exact_tokens"] = sum(
        int(row.get("exact_token_count") or 0)
        for row in state.get("published_shards", []) if row.get("published"))
    state["published_train_tokens"] = sum(
        int((row.get("exact_tokens_by_split") or {}).get("train", 0))
        for row in state.get("published_shards", []) if row.get("published"))
    state["published_validation_tokens"] = sum(
        int((row.get("exact_tokens_by_split") or {}).get("val", 0))
        for row in state.get("published_shards", []) if row.get("published"))

    uploader = None
    if not getattr(args, "no_publish", False):
        uploader = _UploadCoordinator(
            repo=_publish_repo(args), args=args, state=state, out_dir=out_dir,
            workers=int(getattr(args, "upload_workers", 1) or 1),
            max_pending=int(getattr(args, "max_pending_shards", 1) or 1))
        try:
            _retry_pending_shards(uploader, state, args, out_dir)
        except RuntimeError as exc:
            print(f"build: pending shard retry blocked: {exc}")
            exact.close()
            near.close()
            uploader.finish()
            return 2

    buffer: list[dict] = []
    buffer_tokens: list[int] = []
    shard_idx = len(state.get("published_shards", []))
    total_docs = 0
    salt = getattr(args, "split_salt", recipe.get("split_salt", SPLIT_SALT_DEFAULT))
    max_docs = getattr(args, "max_docs", None)
    max_docs = int(max_docs) if max_docs else None
    max_records = getattr(args, "max_records", None)
    max_records = int(max_records) if max_records else None
    if max_docs is None and max_records is None:
        state["completion"] = "running"
    window = max(1, int(getattr(args, "window", 500)))
    max_shard_docs = int(getattr(args, "shard_docs", 0) or 0) or None
    target_shard_bytes = int(getattr(args, "shard_bytes", 512 * 1024 * 1024))
    target_shard_tokens = int(getattr(args, "shard_tokens", 0) or 0) or None
    per_source = getattr(args, "per_source_docs", None)
    per_source = int(per_source) if per_source else None
    streams: dict = {}
    live_cursors: dict[str, source_mod.SourceCursor] = {}
    pending_cursors: dict[str, dict] = {}
    source_index = {name: int(state.get("source_indices", {}).get(name, 0))
                    for name in recipe["domains"]}
    source_filter = {
        str(sid).strip() for sid in (getattr(args, "source_ids", None) or [])
        if str(sid).strip()
    }
    known_sources = {sid for dom in recipe["domains"].values() for sid in dom["sources"]}
    unknown_sources = sorted(source_filter - known_sources)
    if unknown_sources:
        raise ValueError(f"source allowlist contains sources outside recipe: {unknown_sources}")
    state["source_filter"] = sorted(source_filter) if source_filter else None
    source_done = set(state.get("source_done", []))
    source_paused: set[str] = set()
    source_docs_this_run: dict[str, int] = {}
    source_records_this_run = 0
    telemetry = state.setdefault("telemetry", {})
    telemetry.setdefault("started_at", int(time.time()))

    try:
        while (not scheduler.complete()
               and (max_docs is None or total_docs < max_docs)
               and (max_records is None or source_records_this_run < max_records)):
            cache_mod.check_watermark(cache_dir, watermark_bytes)
            available: list[str] = []
            for domain, dom in recipe["domains"].items():
                if any((not source_filter or sid in source_filter)
                       and sid not in source_done and sid not in source_paused
                       for sid in dom["sources"]):
                    available.append(domain)
            if not available and source_paused:
                # ``per_source_docs`` is a fairness/chunking slice, never an
                # exhaustion signal. Start another round once every live
                # source has yielded its slice.
                source_paused.clear()
                source_docs_this_run.clear()
                continue
            domain = scheduler.choose_domain(available)
            if domain is None:
                break
            source_ids = [sid for sid in recipe["domains"][domain]["sources"]
                          if not source_filter or sid in source_filter]
            if not source_ids:
                continue
            start = source_index.get(domain, 0) % len(source_ids)
            sid = None
            for offset in range(len(source_ids)):
                candidate = source_ids[(start + offset) % len(source_ids)]
                if candidate not in source_done and candidate not in source_paused:
                    sid = candidate
                    source_index[domain] = (start + offset + 1) % len(source_ids)
                    break
            if sid is None:
                continue
            src = dict(reg["sources"][sid])
            src["source_id"] = sid
            src["_cache_dir"] = str(source_cache_dir)
            try:
                provenance.assert_training_resolvable(src)
            except ValueError as exc:
                source_done.add(sid)
                state.setdefault("errors", []).append(
                    {"source": sid, "status": "SOURCE_BLOCKED", "reason": str(exc)})
                print(f"build: {sid} SOURCE_BLOCKED {exc} (continuing)")
                continue
            cursor = live_cursors.get(sid)
            if cursor is None:
                cursor = _load_cursor(state["source_cursors"].get(sid), src)
                live_cursors[sid] = cursor
            open_started = time.perf_counter()
            opened, start_offset = _open_or_reuse(streams, sid, src, cursor)
            _add_timing(state, "source_open_seconds", time.perf_counter() - open_started)
            if isinstance(opened, source_mod.SourceResult):
                res = opened
            else:
                remaining = (max_docs - total_docs) if max_docs is not None else window
                limit = min(window, remaining)
                if max_records is not None:
                    limit = min(limit, max_records - source_records_this_run)
                if per_source is not None:
                    limit = min(limit, max(0, per_source - source_docs_this_run.get(sid, 0)))
                stream_started = time.perf_counter()
                res = source_mod.stream_records(opened, src, limit=max(1, limit),
                                                start_offset=start_offset)
                _add_timing(state, "source_stream_seconds",
                             time.perf_counter() - stream_started)
                _add_timing(state, "record_decode_seconds", res.decode_seconds)
                telemetry["source_records"] = telemetry.get("source_records", 0) + len(res.records)
                source_records_this_run += len(res.records)
                telemetry["source_bytes"] = telemetry.get("source_bytes", 0) + sum(
                    len(str(row.get("text", "")).encode("utf-8")) for row in res.records)
            if res.status == "OK" and sid in streams:
                # The iterator has already consumed the complete bounded
                # window.  Keep its physical position separate from the
                # durable checkpoint; a partial max-docs window must reopen
                # from the last record actually processed on resume.
                streams[sid] = (streams[sid][0], int(res.cursor.offset))
            if res.status != "OK":
                source_done.add(sid)
                state.setdefault("errors", []).append(
                    {"source": sid, "status": res.status, "reason": res.reason})
                print(f"build: {sid} {res.status} {res.reason} (continuing)")
                continue
            if not res.records:
                source_done.add(sid)
                continue
            source_docs_this_run[sid] = source_docs_this_run.get(sid, 0) + len(res.records)
            processed_records = 0
            for rec in res.records:
                if max_docs is not None and total_docs >= max_docs:
                    break
                processed_records += 1
                filter_started = time.perf_counter()
                doc = _doc_from_record(rec, src, salt, telemetry=telemetry)
                _add_timing(state, "canonical_filter_seconds",
                             time.perf_counter() - filter_started)
                if doc is None:
                    continue
                if "_reject" in doc:
                    rej = state.setdefault("rejected_documents", {})
                    rej[doc["_reject"]] = rej.get(doc["_reject"], 0) + 1
                    continue
                token_started = time.perf_counter()
                exact_tokens = _exact_token_count(doc["text"], tokenizer)
                _add_timing(state, "exact_tokenization_seconds",
                             time.perf_counter() - token_started)
                doc["exact_token_count"] = exact_tokens if tokenizer is not None else None
                if excluder is not None:
                    contamination_started = time.perf_counter()
                    contamination = excluder.check(doc["text"], token_count=exact_tokens)
                    _add_timing(state, "benchmark_exclusion_seconds",
                                 time.perf_counter() - contamination_started)
                    if contamination.excluded:
                        rejected = state.setdefault("rejected_documents", {})
                        reason = f"benchmark:{contamination.benchmark}"
                        rejected[reason] = rejected.get(reason, 0) + 1
                        state["benchmark_excluded_documents"] = (
                            state.get("benchmark_excluded_documents", 0) + 1)
                        state["benchmark_excluded_tokens"] = (
                            state.get("benchmark_excluded_tokens", 0) + exact_tokens)
                        continue
                candidate_split = assign_split(doc["document_id"], salt=salt,
                                               val_fraction=float(recipe.get("val_fraction", 0.005)))
                if candidate_split == "val" and validation.target_tokens:
                    doc["split"] = validation.consider(doc["document_id"], exact_tokens)
                else:
                    doc["split"] = "train"
                dedupe_started = time.perf_counter()
                if not exact.check(doc["content_hash"], doc["document_id"]):
                    _add_timing(state, "exact_dedupe_seconds",
                                 time.perf_counter() - dedupe_started)
                    state["exact_duplicates"] = state.get("exact_duplicates", 0) + 1
                    continue
                _add_timing(state, "exact_dedupe_seconds",
                             time.perf_counter() - dedupe_started)
                near_started = time.perf_counter()
                if not near.check(doc["text"], doc["document_id"]):
                    _add_timing(state, "near_dedupe_seconds",
                                 time.perf_counter() - near_started)
                    state["near_duplicates"] = state.get("near_duplicates", 0) + 1
                    continue
                _add_timing(state, "near_dedupe_seconds",
                             time.perf_counter() - near_started)
                buffer.append(doc)
                buffer_tokens.append(exact_tokens if tokenizer is not None else 0)
                total_docs += 1
                state["selected_documents"] = state.get("selected_documents", 0) + 1
                est = state.setdefault("estimated_tokens_by_domain", {})
                est[domain] = est.get(domain, 0) + max(1, len(doc["text"]) // 4)
                if tokenizer is not None:
                    if doc["split"] == "train":
                        if doc.get("redistribution_class") in provenance.PUBLISHABLE_CONTENT:
                            scheduler.record(domain, exact_tokens)
                            state["training_tokens"] = state.get("training_tokens", 0) + exact_tokens
                        else:
                            state["held_training_tokens"] = (
                                state.get("held_training_tokens", 0) + exact_tokens)
                    else:
                        if doc.get("redistribution_class") in provenance.PUBLISHABLE_CONTENT:
                            state["validation_tokens"] = validation.selected_tokens
                            state["validation_documents"] = validation.selected_documents
                        else:
                            state["held_validation_tokens"] = (
                                state.get("held_validation_tokens", 0) + exact_tokens)
                    exact_by_dom = state.setdefault("exact_tokens_by_domain", {})
                    exact_by_dom[domain] = exact_by_dom.get(domain, 0) + exact_tokens
                    exact_by_src = state.setdefault("exact_tokens_by_source", {})
                    exact_by_src[sid] = exact_by_src.get(sid, 0) + exact_tokens
                if shards_mod.should_rollover(
                    buffer, target_bytes=target_shard_bytes,
                    target_tokens=target_shard_tokens,
                    max_documents=max_shard_docs,
                    exact_token_counts=buffer_tokens if tokenizer is not None else None,
                ):
                    shard_idx = _flush(buffer, out_dir, shard_idx, recipe, rhash,
                                       state, src, args, exact_token_counts=buffer_tokens,
                                       tokenizer=state.get("tokenizer", {}),
                                       uploader=uploader)
                    buffer.clear(); buffer_tokens.clear()
            # Publish every completed source window before making its cursor
            # durable.  With batched Hub commits, a partial upload queue can
            # span windows; live cursors advance in memory while durable
            # checkpoints wait for the batch to verify.  Exact/near dedupe
            # makes a crash before that checkpoint replay-safe.
            if buffer:
                shard_idx = _flush(
                    buffer, out_dir, shard_idx, recipe, rhash, state, src, args,
                    exact_token_counts=buffer_tokens if tokenizer is not None else None,
                    tokenizer=state.get("tokenizer", {}), uploader=uploader)
                buffer.clear(); buffer_tokens.clear()
            if res.status == "OK":
                checkpoint = res.cursor.as_dict()
                checkpoint["offset"] = int(cursor.offset) + processed_records
                if checkpoint.get("row_index") is not None:
                    base_row = cursor.row_index if cursor.row_index is not None else cursor.offset
                    checkpoint["row_index"] = int(base_row) + processed_records
                live_cursors[sid] = _load_cursor(checkpoint, src)
                if uploader is not None:
                    pending_cursors[sid] = checkpoint
                else:
                    state["source_cursors"][sid] = checkpoint
            else:
                live_cursors[sid] = cursor
            state["source_indices"] = source_index
            state["source_done"] = sorted(source_done)
            state["scheduler"] = scheduler.snapshot()
            if excluder is not None:
                report = excluder.report()
                state["benchmark_exclusion_report"] = report
                state["benchmark_exclusion_status"] = (
                    "complete" if report["complete"] else "partial_pending_exports")
            if uploader is not None:
                uploader.flush(force=False)
            if uploader is None or not uploader.has_pending:
                if pending_cursors:
                    state["source_cursors"].update(pending_cursors)
                    pending_cursors.clear()
                manifests.save_state(state_path, state)
            # A per-source cap is a chunking control, not a completion gate;
            # move to another source on the next scheduler turn.
            if per_source is not None and source_docs_this_run.get(sid, 0) >= per_source:
                source_paused.add(sid)
            if uploader is not None:
                uploader.drain_ready()
    except BaseException:
        # The final buffer is only flushed on a normal loop exit.  On an
        # exceptional exit, close the upload queue here so worker threads and
        # verified results are drained without masking the original error.
        if uploader is not None:
            try:
                uploader.finish()
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve original failure
                state.setdefault("errors", []).append({
                    "upload_shutdown": f"{type(cleanup_exc).__name__}: {str(cleanup_exc)[:160]}"
                })
        raise
    finally:
        for entry in streams.values():
            try:
                closer = getattr(entry[0], "close", None)
                if closer is not None:
                    closer()
            except Exception:  # noqa: BLE001, S110 - cleanup is best effort
                pass
        streams.clear()
        exact.close()
        near.close()

    if buffer:
        _flush(buffer, out_dir, shard_idx, recipe, rhash, state,
               {"redistribution_class": "review_required"}, args,
               exact_token_counts=buffer_tokens if tokenizer is not None else None,
               tokenizer=state.get("tokenizer", {}), uploader=uploader)
    if uploader is not None:
        uploader.finish()
    if pending_cursors:
        state["source_cursors"].update(pending_cursors)
        pending_cursors.clear()
    state["scheduler"] = scheduler.snapshot()
    state["source_done"] = sorted(source_done)
    if excluder is not None:
        report = excluder.report()
        state["benchmark_exclusion_report"] = report
        state["benchmark_exclusion_status"] = (
            "complete" if report["complete"] else "partial_pending_exports")
    state["last_successful_operation"] = "build"
    if scheduler.complete():
        state["completion"] = "all_exact_domain_targets_met"
    elif source_done:
        state["completion"] = "sources_exhausted_or_operational_chunk_limit"
    elif max_docs is not None or max_records is not None:
        state["completion"] = "operational_chunk_limit"
    state["source_records_this_run"] = source_records_this_run
    manifests.save_state(state_path, state)
    if uploader is not None:
        try:
            _publish_progress(uploader.repo, state, args, out_dir)
        except Exception as exc:  # noqa: BLE001 - local state remains authoritative
            state.setdefault("errors", []).append({
                "progress": f"{type(exc).__name__}: {str(exc)[:160]}"})
            manifests.save_state(state_path, state)
    try:
        cache_mod.enforce_bound(cache_dir, max_bytes)
    except RuntimeError as exc:
        print(f"build: cache bound: {exc}")
        return 2
    print(f"build: docs_this_run={total_docs} source_records={source_records_this_run} "
          f"shards={len(state.get('published_shards', []))} "
          f"training_tokens={state.get('training_tokens', 0)} unresolved={scheduler.unresolved()}")
    return 0

def _remote_sha(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None) if lfs else None


def verify_remote_shards_batch(
    repo: str,
    items: list[tuple[Path, dict]],
    hf_prefix: str = "shards",
) -> dict:
    """Upload one atomic batch and verify every remote shard identity."""
    from huggingface_hub import HfApi

    if not items:
        raise ValueError("at least one shard is required for a batch upload")
    started = time.perf_counter()
    remote_paths = [f"{hf_prefix}/{local.name}" for local, _ in items]
    upload_started = time.perf_counter()
    revision = hf_store.upload_files(
        repo,
        [(local, remote) for (local, _), remote in zip(items, remote_paths)],
        commit_message=(f"add {len(items)} FlashMini production shards"),
    )
    upload_elapsed = time.perf_counter() - upload_started
    verify_started = time.perf_counter()
    api = HfApi(token=hf_store.load_token())
    infos = api.get_paths_info(
        repo, remote_paths, repo_type="dataset", revision=revision or None)
    info_by_path = {
        getattr(info, "path", ""): info for info in (infos or [])
    }
    missing = [path for path in remote_paths if path not in info_by_path]
    if missing:
        # The Hub API can briefly omit entries from get_paths_info immediately
        # after a commit. Resolve the immutable tree before failing closed.
        entries = api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, expand=True,
            revision=revision or None)
        info_by_path.update({
            getattr(info, "path", ""): info for info in entries
            if getattr(info, "path", "") in missing
        })
    for (local_path, manifest), remote_path in zip(items, remote_paths):
        info = info_by_path.get(remote_path)
        if info is None:
            raise ValueError(f"remote file not found after batch upload: {remote_path}")
        remote_size = getattr(info, "size", None)
        remote_sha = _remote_sha(info)
        if remote_size is None:
            raise ValueError(f"remote metadata did not expose file size: {remote_path}")
        if remote_size != manifest["bytes"]:
            raise ValueError(
                f"remote identity mismatch for {remote_path}: "
                f"size={remote_size} expected={manifest['bytes']}")
        if remote_sha is not None and remote_sha != manifest["sha256"]:
            raise ValueError(
                f"remote identity mismatch for {remote_path}: "
                f"sha={remote_sha} expected={manifest['sha256']}")
    verify_elapsed = time.perf_counter() - verify_started
    return {
        "revision": revision or hf_store.remote_head_sha(repo),
        "elapsed": time.perf_counter() - started,
        "upload_seconds": upload_elapsed,
        "remote_verification_seconds": verify_elapsed,
    }


def verify_remote_shard(repo: str, local_path: Path, manifest: dict,
                        hf_prefix: str = "shards") -> dict:
    """Upload + verify one remote shard using the batch path."""
    return verify_remote_shards_batch(
        repo, [(local_path, manifest)], hf_prefix=hf_prefix)


class _UploadCoordinator:
    """Bounded upload queue that publishes shards in atomic Hub batches.

    Hugging Face limits repository commits, not bytes.  A batch is kept local
    until ``max_pending`` shards are ready, then one commit is made and every
    file is verified before local eviction.  A forced flush is used at a
    process boundary; a non-forced flush lets source windows accumulate a
    batch while their cursors remain non-durable.
    """

    def __init__(self, *, repo: str, args, state: dict, out_dir: Path,
                 workers: int = 1, max_pending: int = 1) -> None:
        self.repo = repo
        self.args = args
        self.state = state
        self.out_dir = out_dir
        self.hf_prefix = str(getattr(args, "hf_prefix", "shards"))
        self.max_pending = max(1, int(max_pending))
        self.workers = max(1, int(workers))
        self.pending: list[tuple[Path, dict]] = []
        self.failed_shards: list[str] = []
        self.batch_count = 0
        self.closed = False
        hf_store.ensure_repo(repo)

    def submit(self, local_path: Path, manifest: dict) -> None:
        if self.closed:
            raise RuntimeError("upload coordinator is already closed")
        self.pending.append((local_path, manifest))
        if len(self.pending) >= self.max_pending:
            self.flush(force=False)

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)

    def _commit_batch(self, batch: list[tuple[Path, dict]]) -> None:
        try:
            result = verify_remote_shards_batch(
                self.repo, batch, hf_prefix=self.hf_prefix)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
            for local_path, manifest in batch:
                manifest["published"] = False
                manifest["publish_error"] = error
                self.state.setdefault("errors", []).append({
                    "shard": manifest["shard_id"], "error": error,
                })
                _commit_manifest(self.state, self.args, manifest, local_path,
                                 published=False, evict=False)
                self.failed_shards.append(manifest["shard_id"])
                print(f"build: shard {manifest['shard_id']} publish=FAILED")
            raise RuntimeError(
                f"production shard batch failed ({len(batch)} shards): {error}") from exc

        revision = result.get("revision", "")
        for local_path, manifest in batch:
            manifest["published"] = True
            manifest["evicted_local"] = True
            manifest["remote_revision"] = revision
            _commit_manifest(self.state, self.args, manifest, local_path,
                             published=True, evict=True)
            _add_timing(self.state, "upload_and_verify_seconds",
                        result.get("elapsed", 0.0) / len(batch))
            _add_timing(self.state, "hf_upload_seconds",
                        result.get("upload_seconds", 0.0) / len(batch))
            _add_timing(self.state, "remote_verification_seconds",
                        result.get("remote_verification_seconds", 0.0) / len(batch))
            print(f"build: shard {manifest['shard_id']} published=True evicted=True")
        self.batch_count += 1
        # Progress is advisory and itself costs a Hub commit. Publish it only
        # periodically; local state is authoritative and saved per shard.
        if self.batch_count % 16 == 0:
            try:
                _publish_progress(self.repo, self.state, self.args, self.out_dir)
            except Exception as exc:  # noqa: BLE001 - progress is advisory
                self.state.setdefault("errors", []).append({
                    "progress": f"{type(exc).__name__}: {str(exc)[:160]}"})
                manifests.save_state(Path(self.args.state), self.state)

    def flush(self, *, force: bool = True) -> bool:
        """Commit queued shards; non-forced flushes only a full batch."""
        committed = False
        while self.pending and (force or len(self.pending) >= self.max_pending):
            batch = self.pending[:self.max_pending]
            del self.pending[:len(batch)]
            self._commit_batch(batch)
            committed = True
        return committed

    def finish(self) -> None:
        if self.closed:
            return
        self.flush(force=True)
        self.closed = True

    def drain_ready(self) -> None:
        # Kept as a compatibility hook for the bounded build loop. Batch
        # commits are synchronous so there is no background future to poll.
        self.flush(force=False)


def _retry_pending_shards(uploader: _UploadCoordinator, state: dict, args, out_dir: Path) -> None:
    """Retry durable local shards whose previous upload did not verify.

    Failed rows stay in the state ledger with their local Parquet file.  They
    must be repaired before new source cursors advance; otherwise a transient
    Hub/API failure could strand documents behind a supposedly resumed cursor.
    """
    pending = []
    for row in state.get("published_shards", []):
        if row.get("published") or not row.get("publishable_content"):
            continue
        local_path = out_dir / str(row.get("path", ""))
        if not local_path.is_file():
            raise RuntimeError(
                f"pending shard {row.get('shard_id')} has no local retry artifact")
        pending.append((local_path, row))
    for local_path, row in pending:
        try:
            cache_mod.verify_sha256(local_path, str(row.get("sha256", "")))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"pending shard {row.get('shard_id')} checksum is invalid: {exc}") from exc
        row.pop("publish_error", None)
        row["published"] = False
        uploader.submit(local_path, row)
    uploader.flush(force=True)
    remaining = [row.get("shard_id") for _, row in pending if not row.get("published")]
    if remaining:
        raise RuntimeError(f"pending shard uploads remain unverified: {remaining}")


def _commit_manifest(state: dict, args, manifest: dict, local_path: Path, *,
                     published: bool, evict: bool) -> None:
    """Atomically record one shard result, then evict only verified artifacts."""
    if published:
        state["hf_revision"] = manifest.get("remote_revision", state.get("hf_revision", ""))
        state["published_bytes"] = state.get("published_bytes", 0) + int(manifest["bytes"])
        state["published_documents"] = state.get("published_documents", 0) + int(
            manifest.get("document_count", 0))
        state["published_exact_tokens"] = state.get("published_exact_tokens", 0) + int(
            manifest.get("exact_token_count") or 0)
        split_tokens = manifest.get("exact_tokens_by_split") or {}
        state["published_train_tokens"] = state.get("published_train_tokens", 0) + int(
            split_tokens.get("train", 0))
        state["published_validation_tokens"] = state.get(
            "published_validation_tokens", 0) + int(split_tokens.get("val", 0))
    rows = state.setdefault("published_shards", [])
    replaced = False
    for index, row in enumerate(rows):
        if row.get("shard_id") == manifest.get("shard_id"):
            rows[index] = manifest
            replaced = True
            break
    if not replaced:
        rows.append(manifest)
    persist_started = time.perf_counter()
    manifests.save_state(Path(args.state), state)
    persist_elapsed = time.perf_counter() - persist_started
    _add_timing(state, "state_persistence_seconds", persist_elapsed)
    manifests.save_state(Path(args.state), state)
    if evict:
        local_path.unlink(missing_ok=True)


def _publish_progress(repo: str, state: dict, args, out_dir: Path) -> None:
    """Publish a non-secret resumable progress snapshot after verified shards."""
    payload = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    progress = out_dir / ".flashmini-build-progress.json"
    cache_mod.atomic_write_bytes(progress, payload)
    try:
        remote = f"{getattr(args, 'hf_prefix', 'shards')}/BUILD_STATE.json"
        hf_store.upload_file(repo, progress, remote,
                             commit_message="update production build state")
        info = hf_store.remote_file_info(repo, remote)
        if info is None or getattr(info, "size", None) != len(payload):
            raise ValueError("remote progress state size verification failed")
        progress_revision = hf_store.remote_head_sha(repo)
        if progress_revision:
            state["hf_progress_revision"] = progress_revision
            manifests.save_state(Path(args.state), state)
    finally:
        progress.unlink(missing_ok=True)


def _flush(buffer: list[dict], out_dir: Path, idx: int, recipe: dict,
           rhash: str, state: dict, src: dict, args, *,
           exact_token_counts: list[int] | None = None,
           tokenizer: dict | None = None,
           uploader: _UploadCoordinator | None = None) -> int:
    """Write one buffer as per-class shards: publishable content separated from
    held (recipe-only) content so licensing gating is document-level, not
    shard-level."""
    publishable = [d for d in buffer
                   if d.get("redistribution_class") in provenance.PUBLISHABLE_CONTENT]
    held = [d for d in buffer
            if d.get("redistribution_class") not in provenance.PUBLISHABLE_CONTENT]
    next_idx = idx
    counts = list(exact_token_counts or [])
    if counts and len(counts) != len(buffer):
        raise ValueError("exact token counts must match the flush buffer")
    publish_counts = ([counts[i] for i, doc in enumerate(buffer)
                       if doc.get("redistribution_class") in provenance.PUBLISHABLE_CONTENT]
                      if counts else None)
    held_counts = ([counts[i] for i, doc in enumerate(buffer)
                    if doc.get("redistribution_class") not in provenance.PUBLISHABLE_CONTENT]
                   if counts else None)
    if publishable:
        next_idx = _write_and_maybe_publish(
            publishable, out_dir, next_idx, recipe, rhash, state, args, held=False,
            exact_token_counts=publish_counts, tokenizer=tokenizer,
            uploader=uploader)
    if held:
        next_idx = _write_and_maybe_publish(
            held, out_dir, next_idx, recipe, rhash, state, args, held=True,
            exact_token_counts=held_counts, tokenizer=tokenizer,
            uploader=uploader)
    return next_idx


def _write_and_maybe_publish(docs: list[dict], out_dir: Path, idx: int,
                             recipe: dict, rhash: str, state: dict, args,
                             *, held: bool, exact_token_counts: list[int] | None = None,
                             tokenizer: dict | None = None,
                             uploader: _UploadCoordinator | None = None) -> int:
    shard_id = f"{recipe['name']}-shard-{idx:06d}" + ("-held" if held else "")
    out_path = out_dir / f"{shard_id}.parquet"
    prior = next((row for row in state.get("published_shards", [])
                  if row.get("shard_id") == shard_id), None)
    if prior is not None:
        if prior.get("published") and not out_path.exists():
            return idx + 1
        if out_path.exists():
            try:
                cache_mod.verify_sha256(out_path, prior.get("sha256", ""))
                return idx + 1
            except (OSError, ValueError):
                out_path.unlink(missing_ok=True)
    encode_started = time.perf_counter()
    manifest = shards_mod.write_shard(
        docs, out_path, shard_id=shard_id, recipe_name=recipe["name"],
        recipe_hash=rhash, exact_token_counts=exact_token_counts,
        tokenizer=tokenizer or {},
        remote_path=f"{getattr(args, 'hf_prefix', 'shards')}/{out_path.name}",
        hf_revision=state.get("hf_revision", ""))
    _add_timing(state, "parquet_encode_compress_seconds",
                 time.perf_counter() - encode_started)
    manifest["split"] = "train"
    manifest["sequence_count"] = manifest["document_count"]
    hf_prefix = getattr(args, "hf_prefix", "shards")
    manifest["remote_path"] = f"{hf_prefix}/{out_path.name}"
    ok, reason = provenance.classify_for_publish(
        "review_required" if held else "mirror_allowed")
    manifest["publishable_content"] = ok
    manifest["publish_note"] = reason
    manifest["benchmark_exclusion_version"] = state.get(
        "benchmark_exclusion_version", contamination_mod.CONTAMINATION_VERSION)
    if held:
        manifest["published"] = False
        manifest["evicted_local"] = True
        manifest["eviction_note"] = "not_publishable_provenance"
        _commit_manifest(state, args, manifest, out_path,
                         published=False, evict=True)
    elif getattr(args, "no_publish", False):
        manifest["published"] = False
        _commit_manifest(state, args, manifest, out_path,
                         published=False, evict=False)
    elif uploader is not None:
        uploader.submit(out_path, manifest)
    else:
        try:
            repo = _publish_repo(args)
            timings = verify_remote_shard(
                repo, out_path, manifest,
                hf_prefix=getattr(args, "hf_prefix", "shards"))
            elapsed = timings.get("elapsed", 0.0)
            manifest["published"] = True
            manifest["evicted_local"] = True
            manifest["remote_revision"] = hf_store.remote_head_sha(repo)
            _commit_manifest(state, args, manifest, out_path,
                             published=True, evict=True)
            _add_timing(state, "upload_and_verify_seconds", elapsed)
            _add_timing(state, "hf_upload_seconds",
                        timings.get("upload_seconds", 0.0))
            _add_timing(state, "remote_verification_seconds",
                        timings.get("remote_verification_seconds", 0.0))
            try:
                _publish_progress(repo, state, args, out_dir)
            except Exception as exc:  # noqa: BLE001 - progress is advisory after verified shard
                state.setdefault("errors", []).append({
                    "progress": f"{type(exc).__name__}: {str(exc)[:160]}"})
                manifests.save_state(Path(args.state), state)
        except Exception as exc:  # noqa: BLE001 - external publish errors are recorded
            manifest["published"] = False
            manifest["publish_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            state.setdefault("errors", []).append(
                {"shard": shard_id, "error": manifest["publish_error"]})
            _commit_manifest(state, args, manifest, out_path,
                             published=False, evict=False)
    print(f"build: shard {shard_id} docs={manifest['document_count']} "
          f"published={manifest.get('published', False)}")
    return idx + 1



def cmd_publish(args) -> int:
    state = manifests.load_state(Path(args.state))
    try:
        info = hf_store.whoami()
    except RuntimeError as exc:
        print(f"publish: {exc}")
        return 2
    repo = getattr(args, "hf_repo", None) or hf_store.repo_id_for(
        "production" if getattr(args, "production", False) else "data",
        info.get("name", ""))
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
        except Exception as exc:  # noqa: BLE001 - external publish errors are recorded
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
        except Exception as exc:  # noqa: BLE001 - remote verification is fail-closed
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
    held = [s.get("shard_id") for s in shards if not s.get("published", True)
            or not s.get("publishable_content", True)]
    exact_ready = bool(manifest.get("exact_token_ready", False))
    print(f"verify: recipe_match={ok} shards={len(shards)} hashes_valid={valid} "
          f"held_refs={len(held)} exact_token_ready={exact_ready}")
    print(f"verify: fingerprint={manifest.get('corpus_fingerprint','')}")
    return 0 if (ok and valid and not held) else 1


def cmd_freeze(args) -> int:
    state = manifests.load_state(Path(args.state))
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    registry_path = Path(getattr(args, "registry", "training_data/registry/sources.yaml"))
    lock_path = Path(getattr(args, "source_lock", registry_path.parent / "source_snapshot.lock.json"))
    reg = registry_mod.load_registry(
        registry_path, lock_path=lock_path if lock_path.exists() else None,
        require_immutable=bool(getattr(args, "production", False)))
    manifest_path = Path(args.manifest)
    # A frozen training view may reference only content that was both allowed
    # by provenance and verified as present remotely.  Held/unpublished rows
    # remain in build state as audit evidence but never enter the view.
    shards = [s for s in state.get("published_shards", [])
              if s.get("published") and s.get("publishable_content", True)]
    shard_hashes = [s["sha256"] for s in shards if s.get("sha256")]
    tokenizer_identity = ""
    tokenizer_meta = {}
    spec_path = getattr(args, "tokenizer_spec", None)
    if spec_path:
        spec = tokenizer_mod.load_spec(spec_path)
        tokenizer_identity = spec.identity
        tokenizer_meta = spec.as_dict()
    elif getattr(args, "tokenizer_revision", None):
        tokenizer_identity = tokenizer_mod.tokenizer_identity(
            getattr(args, "tokenizer", "gpt2"), args.tokenizer_revision)
        tokenizer_meta = {
            "tokenizer_id": getattr(args, "tokenizer", "gpt2"),
            "revision": args.tokenizer_revision,
            "identity": tokenizer_identity,
        }
    release_id = getattr(args, "release_id", "flashmini-pretrain-production-v1")
    fp = shards_mod.corpus_fingerprint(
        registry_hash=registry_mod.registry_hash(reg),
        recipe_hash=recipes_mod.recipe_hash(recipe),
        filter_version=filters_mod.FILTER_VERSION,
        dedupe_version=dedupe_mod.DEDUPE_VERSION,
        split_salt=getattr(args, "split_salt", SPLIT_SALT_DEFAULT),
        shard_hashes=shard_hashes,
        tokenizer_identity=tokenizer_identity,
        release_id=release_id,
        token_format_version=packing_mod.TOKEN_FORMAT_VERSION,
        benchmark_exclusion_version=state.get("benchmark_exclusion_version", ""),
        validation_tokens=int(recipe.get("validation_tokens", 0)),
        validation_salt=str(recipe.get("validation_salt", "flashmini-validation-v1")))
    state["corpus_fingerprint"] = fp
    manifests.save_state(Path(args.state), state)
    hf_repo = ""
    hf_revision = state.get("hf_revision", "")
    try:
        info = hf_store.whoami()
        hf_repo = getattr(args, "hf_repo", None) or hf_store.repo_id_for(
            "production" if getattr(args, "production", False) else "data",
            info.get("name", ""))
    except RuntimeError:
        hf_repo = ""
    split_totals: dict[str, int] = {}
    exact_split_totals: dict[str, int] = {}
    exact_domain_totals: dict[str, int] = {}
    exact_source_totals: dict[str, int] = {}
    for s in shards:
        for split, count in (s.get("split_distribution") or {}).items():
            split_totals[split] = split_totals.get(split, 0) + int(count)
        # Exact token counts are document-level in the production writer.  A
        # shard without them is deliberately marked estimate-only.
        for split, count in (s.get("exact_tokens_by_split") or {}).items():
            exact_split_totals[split] = exact_split_totals.get(split, 0) + int(count)
        for domain, count in (s.get("exact_tokens_by_domain") or {}).items():
            exact_domain_totals[domain] = exact_domain_totals.get(domain, 0) + int(count)
        for source, count in (s.get("exact_tokens_by_source") or {}).items():
            exact_source_totals[source] = exact_source_totals.get(source, 0) + int(count)
    stage_targets = {}
    for stage_name in recipe.get("stages", []):
        stage_path = Path(args.recipe).parent / f"{stage_name}.yaml"
        if stage_path.exists():
            stage_targets[stage_name] = int(recipes_mod.load_recipe(stage_path)["target_tokens"])
    exact_ready = bool(shards) and bool(tokenizer_meta) and all(
        s.get("exact_token_count") is not None for s in shards)
    if getattr(args, "production", False) and not tokenizer_meta:
        raise ValueError("production freeze requires --tokenizer-spec or immutable revision")
    if getattr(args, "production", False) and not exact_ready:
        raise ValueError("production freeze requires exact-token manifests for every shard")
    if getattr(args, "production", False) and state.get("benchmark_exclusion_status") != "complete":
        raise ValueError(
            "production freeze requires complete benchmark exclusion exports; "
            "canonical ingestion may continue while this gate is pending")
    if getattr(args, "production", False) and (
            not hf_repo or not tokenizer_mod._IMMUTABLE_REVISION.fullmatch(hf_revision or "")):
        raise ValueError("production freeze requires a verified immutable Hub revision")
    source_lock_hash = ""
    if lock_path.exists():
        source_lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    corpus = {
        "release_id": release_id,
        "recipe_name": recipe["name"],
        "recipe_hash": recipes_mod.recipe_hash(recipe),
        "corpus_fingerprint": fp,
        "format_version": 4,
        "packing_version": packing_mod.PACKING_VERSION,
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "shards": shards,
        "split_totals": split_totals,
        "tokenizer": tokenizer_meta or {"status": "not_frozen"},
        "token_format_version": packing_mod.TOKEN_FORMAT_VERSION,
        "exact_token_ready": exact_ready,
        "exact_split_tokens": exact_split_totals,
        "exact_domain_tokens": exact_domain_totals,
        "exact_source_tokens": exact_source_totals,
        "stage_target_tokens": stage_targets,
        "validation_tokens_target": int(recipe.get("validation_tokens", 0)),
        "source_lock_sha256": source_lock_hash,
        "dedupe_version": dedupe_mod.DEDUPE_VERSION,
        "filter_version": filters_mod.FILTER_VERSION,
        "benchmark_exclusion_version": state.get("benchmark_exclusion_version", ""),
        "status": "exact-token-ready" if exact_ready else "estimate-only",
        "view_status": {
            "train_target_tokens": int(recipe["target_tokens"]),
            "train_exact_tokens": int(exact_split_totals.get("train", 0)),
            "validation_exact_tokens": int(exact_split_totals.get("val", 0)),
            "train_target_met": exact_ready and exact_split_totals.get("train", 0) == int(recipe["target_tokens"]),
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
        max_open_shards=args.max_open_shards,
        production=bool(getattr(args, "production", False)))


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
