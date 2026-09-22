"""Bounded late benchmark-exclusion overlay for an already published corpus."""

from __future__ import annotations

import json
from pathlib import Path

from . import cache as cache_mod
from . import contamination, hf_store, manifests, shards


def _row_to_document(row: dict) -> dict:
    cursor = row.get("source_cursor", {})
    if isinstance(cursor, str):
        try:
            cursor = json.loads(cursor)
        except json.JSONDecodeError:
            cursor = {}
    return {
        "document_id": row.get("document_id", ""),
        "text": row.get("text", ""),
        "source_id": row.get("source_id", ""),
        "source_revision": row.get("source_revision", ""),
        "record_id": row.get("record_id", ""),
        "source_cursor": cursor if isinstance(cursor, dict) else {},
        "domain": row.get("domain", ""),
        "language": row.get("language", "en"),
        "license": row.get("license", ""),
        "redistribution_class": row.get("redistribution_class", "review_required"),
        "split": row.get("split", "train"),
        "content_hash": row.get("content_hash", ""),
    }


def _filter_canonical(path: Path, excluder: contamination.BenchmarkExcluder):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    documents = []
    counts = []
    excluded = 0
    excluded_tokens = 0
    for row in rows:
        count = row.get("exact_token_count")
        result = excluder.check(row.get("text", ""), token_count=int(count or 0))
        if result.excluded:
            excluded += 1
            excluded_tokens += int(count or 0)
            continue
        documents.append(_row_to_document(row))
        if count is not None:
            counts.append(int(count))
    exact_counts = counts if len(counts) == len(documents) else None
    return documents, exact_counts, excluded, excluded_tokens


def _verify_upload(repo: str, local: Path, manifest: dict, remote_path: str) -> str:
    cache_mod.verify_sha256(local, manifest["sha256"])
    hf_store.upload_file(repo, local, remote_path,
                         commit_message=f"add benchmark overlay {manifest['shard_id']}")
    info = hf_store.remote_file_info(repo, remote_path)
    if info is None or getattr(info, "size", None) != manifest["bytes"]:
        raise ValueError(f"remote overlay size mismatch for {remote_path}")
    remote_sha = getattr(getattr(info, "lfs", None), "sha256", None)
    if remote_sha is not None and remote_sha != manifest["sha256"]:
        raise ValueError(f"remote overlay SHA mismatch for {remote_path}")
    return hf_store.remote_head_sha(repo)


def _publish_json(repo: str, payload: dict, out_dir: Path, remote_path: str) -> str:
    path = out_dir / ".overlay-state.json"
    cache_mod.atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode())
    try:
        hf_store.upload_file(repo, path, remote_path,
                             commit_message="update benchmark overlay progress")
        info = hf_store.remote_file_info(repo, remote_path)
        if info is None or getattr(info, "size", None) != path.stat().st_size:
            raise ValueError(f"remote overlay state size mismatch for {remote_path}")
        return hf_store.remote_head_sha(repo)
    finally:
        path.unlink(missing_ok=True)


def run_overlay(*, state_path: str | Path, contamination_config: str | Path,
                repo: str, output_prefix: str, out_dir: str | Path,
                cache_dir: str | Path, cache_gb: float | None = None,
                revision: str | None = None, overlay_state_path: str | Path | None = None,
                max_shards: int | None = None, free_space_watermark_gib: float = 10.0) -> dict:
    """Rebuild a filtered training view from published canonical shards.

    Only one source shard is downloaded and one overlay shard is staged at a
    time. The canonical source release is never modified. The command refuses
    to run until every configured benchmark export is present.
    """
    state_path = Path(state_path)
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = cache_mod.cache_max_bytes(
        int(cache_gb * 1024 ** 3) if cache_gb is not None else None)
    watermark = int(float(free_space_watermark_gib) * 1024 ** 3)
    excluder = contamination.BenchmarkExcluder.from_path(contamination_config)
    report = excluder.report()
    if not report["complete"]:
        raise ValueError(
            "benchmark overlay requires complete immutable exports; "
            f"missing={report['missing_benchmarks']}")
    source_state = manifests.load_state(state_path)
    overlay_state_path = (Path(overlay_state_path) if overlay_state_path
                          else state_path.with_suffix(".overlay.json"))
    overlay_state = manifests.load_state(overlay_state_path)
    overlay_state.update({
        "kind": "benchmark_exclusion_overlay",
        "source_state": str(state_path),
        "repository": repo,
        "source_revision": revision or source_state.get("hf_revision", ""),
        "output_prefix": output_prefix,
        "benchmark_exclusion_version": contamination.CONTAMINATION_VERSION,
        "benchmark_exclusion_report": report,
        "published_shards": overlay_state.get("published_shards", []),
        "processed_source_shards": overlay_state.get("processed_source_shards", []),
        "excluded_documents": overlay_state.get("excluded_documents", 0),
        "excluded_tokens": overlay_state.get("excluded_tokens", 0),
    })
    hf_store.ensure_repo(repo)
    source_rows = [row for row in source_state.get("published_shards", [])
                   if row.get("published") and row.get("publishable_content", True)]
    completed = set(overlay_state["processed_source_shards"])
    processed = 0
    for source_row in source_rows:
        source_shard_id = source_row.get("shard_id", "")
        if source_shard_id in completed:
            continue
        if max_shards is not None and processed >= int(max_shards):
            break
        cache_mod.check_watermark(cache_dir, watermark)
        remote_source = source_row.get("remote_path")
        if not remote_source:
            raise ValueError(f"canonical shard has no remote path: {source_shard_id}")
        source_local = out_dir / "source" / Path(remote_source).name
        overlay_local = out_dir / "overlay" / Path(remote_source).name
        hf_store.download_file(
            repo, remote_source, source_local,
            revision=revision or source_state.get("hf_revision") or None,
            cache_dir=cache_dir, min_avail_bytes=watermark)
        cache_mod.verify_sha256(source_local, source_row["sha256"])
        docs, exact_counts, excluded, excluded_tokens = _filter_canonical(source_local, excluder)
        source_local.unlink(missing_ok=True)
        if docs:
            overlay_id = f"{source_shard_id}-decontaminated"
            remote_path = f"{output_prefix}/{overlay_id}.parquet"
            manifest = shards.write_shard(
                docs, overlay_local, shard_id=overlay_id,
                recipe_name=source_row.get("recipe_name", ""),
                recipe_hash=source_row.get("recipe_hash", ""),
                exact_token_counts=exact_counts,
                tokenizer=source_row.get("tokenizer", {}),
                remote_path=remote_path,
                hf_revision=overlay_state.get("hf_revision", ""))
            manifest.update({
                "source_shard_id": source_shard_id,
                "published": True,
                "publishable_content": True,
                "remote_path": remote_path,
                "benchmark_exclusion_version": contamination.CONTAMINATION_VERSION,
            })
            manifest["remote_revision"] = _verify_upload(repo, overlay_local, manifest, remote_path)
            overlay_local.unlink(missing_ok=True)
            overlay_state["published_shards"].append(manifest)
        overlay_state["excluded_documents"] += excluded
        overlay_state["excluded_tokens"] += excluded_tokens
        overlay_state["processed_source_shards"].append(source_shard_id)
        overlay_state["hf_revision"] = hf_store.remote_head_sha(repo)
        manifests.save_state(overlay_state_path, overlay_state)
        try:
            _publish_json(repo, overlay_state, out_dir,
                          f"{output_prefix}/BUILD_STATE.json")
        finally:
            cache_mod.enforce_bound(cache_dir, max_bytes)
        processed += 1
    overlay_state["complete"] = all(
        row.get("shard_id") in set(overlay_state["processed_source_shards"])
        for row in source_rows)
    overlay_state["training_view_exact_tokens"] = sum(
        int((row.get("exact_tokens_by_split") or {}).get("train", 0))
        for row in overlay_state["published_shards"])
    overlay_state["shards"] = overlay_state["published_shards"]
    overlay_state["exact_token_ready"] = all(
        row.get("exact_token_count") is not None
        for row in overlay_state["published_shards"])
    manifests.save_state(overlay_state_path, overlay_state)
    manifest_path = out_dir / "MANIFEST.json"
    cache_mod.atomic_write_bytes(manifest_path,
                                 json.dumps(overlay_state, indent=2, sort_keys=True).encode())
    try:
        remote_manifest = f"{output_prefix}/MANIFEST.json"
        hf_store.upload_file(repo, manifest_path, remote_manifest,
                             commit_message="publish benchmark-decontaminated view manifest")
        info = hf_store.remote_file_info(repo, remote_manifest)
        if info is None or getattr(info, "size", None) != manifest_path.stat().st_size:
            raise ValueError("remote overlay manifest size verification failed")
        overlay_state["remote_manifest"] = remote_manifest
        overlay_state["remote_revision"] = hf_store.remote_head_sha(repo)
    finally:
        manifest_path.unlink(missing_ok=True)
    manifests.save_state(overlay_state_path, overlay_state)
    return overlay_state
