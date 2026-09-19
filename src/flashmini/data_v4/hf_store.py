"""Hugging Face storage wrapper (v4). Token is never logged or persisted."""

from __future__ import annotations

import os
from pathlib import Path


def load_token() -> str | None:
    """Prefer HF_TOKEN; fall back to .env aliases. Never prints or persists."""
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
           or os.environ.get("HUGGING_FACE_API_KEY"))
    if tok:
        os.environ["HF_TOKEN"] = tok
        return tok
    for candidate in (Path(".env"),
                      Path(__file__).resolve().parents[3] / ".env"):
        try:
            if candidate.is_file():
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    for key in ("HF_TOKEN", "HUGGING_FACE_API_KEY",
                                "HUGGINGFACE_HUB_TOKEN"):
                        if line.startswith(f"{key}="):
                            value = line.split("=", 1)[1].strip().strip("\"'")
                            if value:
                                os.environ["HF_TOKEN"] = value
                                return value
        except OSError:
            continue
    return None


def whoami() -> dict:
    from huggingface_hub import HfApi
    token = load_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set (and not found in .env)")
    api = HfApi(token=token)
    info = api.whoami()
    if isinstance(info, dict):
        return info
    return {"name": getattr(info, "name", str(info))}


def repo_id_for(kind: str, hf_user: str) -> str:
    if kind == "data":
        return f"{hf_user}/flashmini-data-v1"
    if kind == "eval":
        return f"{hf_user}/flashmini-eval-v1"
    raise ValueError(f"unknown repo kind: {kind}")


def ensure_repo(repo_id: str, *, private: bool = False) -> str:
    """Create-or-reuse after verifying ownership. Returns repo_id."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    token = load_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set (and not found in .env)")
    api = HfApi(token=token)
    me = whoami()
    owner = repo_id.split("/")[0]
    if owner != me.get("name"):
        raise ValueError(f"refusing to write to repo owned by {owner!r}")
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except HfHubHTTPError as exc:
        # 409/existing is fine; anything else re-raises.
        if "409" not in str(exc) and "already exists" not in str(exc).lower():
            raise
    return repo_id


def upload_file(repo_id: str, local_path: Path, path_in_repo: str,
                *, commit_message: str) -> str:
    from huggingface_hub import HfApi
    token = load_token()
    api = HfApi(token=token)
    return api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def remote_head_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=load_token())
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        for branch in getattr(refs, "branches", []):
            if getattr(branch, "name", "") == "main":
                return getattr(branch, "target_commit", "") or ""
    except Exception:
        pass
    return ""
