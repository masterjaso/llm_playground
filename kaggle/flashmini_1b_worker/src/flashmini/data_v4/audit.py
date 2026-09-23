"""Live Hugging Face release audit (metadata-first, no bulk download)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import hf_store


def _download_json(repo: str, path: str, revision: str | None, cache_dir: Path | None):
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                            cache_dir=str(cache_dir) if cache_dir else None)
    return json.loads(Path(local).read_text())


def audit_hf_repository(repo: str, *, revision: str | None = None,
                        cache_dir: Path | None = None) -> dict:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_store.load_token())
    info = api.dataset_info(repo, revision=revision)
    pinned = getattr(info, "sha", None) or revision or ""
    entries = list(api.list_repo_tree(repo, repo_type="dataset", recursive=True,
                                      expand=True, revision=pinned or None))
    files = [entry for entry in entries if entry.__class__.__name__ == "RepoFile"]
    remote_by_path = {entry.path: entry for entry in files}
    manifest = None
    state = None
    try:
        manifest = _download_json(repo, "manifests/corpus_manifest.json", pinned, cache_dir)
    except Exception:  # noqa: BLE001, S110 - missing optional audit files are tolerated
        pass
    try:
        state = _download_json(repo, "manifests/build_state.json", pinned, cache_dir)
    except Exception:  # noqa: BLE001, S110 - missing optional audit files are tolerated
        pass
    shard_rows = list((manifest or {}).get("shards", []))
    manifest_paths = {row.get("remote_path") or f"shards/{row.get('path', '')}": row
                      for row in shard_rows}
    remote_shards = {path: item for path, item in remote_by_path.items()
                     if path.endswith(".parquet") and path.startswith("shards/")}
    published_rows = [row for row in shard_rows if row.get("published")]
    domains = Counter(); sources = Counter(); languages = Counter(); licenses = Counter()
    for row in published_rows:
        for field, counter in (("domain_distribution", domains),
                               ("source_distribution", sources),
                               ("language_distribution", languages),
                               ("license_distribution", licenses)):
            for key, value in (row.get(field) or {}).items():
                counter[key] += int(value)
    mismatches = []
    for path, row in manifest_paths.items():
        remote = remote_by_path.get(path)
        if remote is None:
            continue
        lfs = getattr(remote, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None) if lfs else None
        remote_size = getattr(remote, "size", None)
        if (remote_size is not None and row.get("bytes") is not None
                and int(remote_size) != int(row["bytes"])) or \
                (remote_sha and row.get("sha256") and remote_sha != row["sha256"]):
            mismatches.append({
                "path": path, "manifest_bytes": row.get("bytes"),
                "remote_bytes": remote_size, "manifest_sha256": row.get("sha256"),
                "remote_sha256": remote_sha,
            })
    remote_paths = set(remote_shards)
    return {
        "repo": repo,
        "revision": pinned,
        "repository_items": len(entries),
        "files": len(files),
        "remote_shards": len(remote_shards),
        "remote_clean_shards": sum(1 for path in remote_by_path if path.startswith("clean/")),
        "manifest_shards": len(shard_rows),
        "manifest_published_shards": len(published_rows),
        "manifest_documents_all": sum(int(row.get("document_count", 0)) for row in shard_rows),
        "manifest_documents_published": sum(int(row.get("document_count", 0)) for row in published_rows),
        "manifest_bytes_all": sum(int(row.get("bytes", 0)) for row in shard_rows),
        "manifest_bytes_published": sum(int(row.get("bytes", 0)) for row in published_rows),
        "estimated_tokens_published": sum(int(row.get("estimated_token_count", 0)) for row in published_rows),
        "exact_tokens_published": sum(int(row.get("exact_token_count", 0)) for row in published_rows
                                      if row.get("exact_token_count") is not None),
        "domain_distribution": dict(domains), "source_distribution": dict(sources),
        "language_distribution": dict(languages), "license_distribution": dict(licenses),
        "missing_manifest_paths": sorted(set(manifest_paths) - remote_paths),
        "orphan_remote_shards": sorted(remote_paths - set(manifest_paths)),
        "remote_metadata_mismatches": mismatches,
        "duplicate_manifest_paths": [path for path, count in Counter(
            row.get("remote_path") or f"shards/{row.get('path', '')}" for row in shard_rows
        ).items() if count > 1],
        "manifest_fingerprint": (manifest or {}).get("corpus_fingerprint", ""),
        "state_fingerprint": (state or {}).get("corpus_fingerprint", ""),
        "source": "huggingface_hub_api",
    }


def write_audit(audit: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
