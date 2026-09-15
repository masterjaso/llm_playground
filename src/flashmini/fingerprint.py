"""Machine-readable execution fingerprint for the v3 PoC.

The fingerprint is a stable, hashed canonical record of the exact runtime
environment and provenance that produced an official run. It is recorded in
run metadata and checkpoints, and enforced on exact v3 resume and A/B/C
comparison. A materially different fingerprint must fail closed.

The canonical hash is computed over a sorted-key JSON rendering of the
fingerprint values, so the same environment always yields the same digest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    """Run a command and return stripped stdout, or '' on failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return (proc.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_commit(repo_root: Path) -> str:
    return _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])


def _git_dirty(repo_root: Path) -> bool:
    """True when the working tree has any uncommitted change."""
    out = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return bool(out)


def _nvidia_driver_version() -> str:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else ""


def _file_hash(path: Path) -> str | None:
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return None


def _source_hash(repo_root: Path) -> dict[str, str]:
    """Hash every FlashMini source file relative to the package root."""
    source_root = repo_root / "src" / "flashmini"
    files: dict[str, str] = {}
    for path in sorted(source_root.rglob("*.py")):
        files[str(path.relative_to(source_root))] = _file_hash(path) or ""
    return files


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch_version": None, "cuda_available": False,
                           "cuda_version": None, "device_count": 0, "devices": []}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": [props.major, props.minor],
                })
    except (ImportError, AttributeError, RuntimeError) as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    return info


def collect_fingerprint(repo_root: Path, *, config_sha256: str | None = None,
                       data_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Collect the full execution fingerprint for a run."""
    repo_root = Path(repo_root)
    source_files = _source_hash(repo_root)
    source_sha = hashlib.sha256(
        json.dumps(source_files, sort_keys=True).encode()
    ).hexdigest()
    fingerprint: dict[str, Any] = {
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "source_sha256": source_sha,
        "source_files": source_files,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyproject_sha256": _file_hash(repo_root / "pyproject.toml"),
        "uv_lock_sha256": _file_hash(repo_root / "uv.lock"),
        "requirements_sha256": _file_hash(repo_root / "requirements.txt"),
        "nvidia_driver_version": _nvidia_driver_version(),
        "torch": _torch_info(),
    }
    fingerprint["environment_fingerprint_sha256"] = environment_fingerprint_sha256(fingerprint)
    fingerprint["fingerprint_sha256"] = fingerprint_sha(fingerprint)
    return fingerprint


def fingerprint_sha(fingerprint: dict[str, Any]) -> str:
    """Stable canonical SHA-256 over the fingerprint (excluding its own hash)."""
    payload = {k: v for k, v in fingerprint.items() if k != "fingerprint_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# The treatment-neutral shared execution environment. These fields must be
# identical across A/B/C for a valid comparison: they describe the runtime
# environment and provenance, not the treatment-specific model config.
_ENVIRONMENT_FIELDS = (
    "git_commit",
    "git_dirty",
    "source_sha256",
    "data_manifest_sha256",
    "python_version",
    "platform",
    "pyproject_sha256",
    "uv_lock_sha256",
    "requirements_sha256",
    "nvidia_driver_version",
    "torch",
)


def environment_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Return the treatment-neutral shared execution environment record.

    This excludes the treatment-specific ``config_sha256``, the treatment
    name, the per-file ``source_files`` map, and the full ``fingerprint_sha256``.
    It captures only the shared execution provenance that must be identical
    across A/B/C for a valid comparison: git commit, clean/dirty, source hash,
    data manifest hash, Python version, PyTorch/CUDA version, NVIDIA driver,
    GPU identities and capabilities, platform, and dependency lock hashes.
    """
    return {field: fingerprint.get(field) for field in _ENVIRONMENT_FIELDS}


def environment_fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    """Stable canonical SHA-256 over the treatment-neutral environment record."""
    payload = environment_fingerprint(fingerprint)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _materially_different(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return the list of top-level keys whose values differ materially.

    ``git_dirty`` is compared as a boolean; ``source_files`` is compared via its
    aggregate ``source_sha256``. All other keys are compared by exact value.
    """
    diffs: list[str] = []
    keys = set(a) | set(b)
    for key in keys:
        if key == "fingerprint_sha256":
            continue
        if key == "source_files":
            if a.get("source_sha256") != b.get("source_sha256"):
                diffs.append(key)
            continue
        if a.get(key) != b.get(key):
            diffs.append(key)
    return diffs


def enforce_fingerprint_match(current: dict[str, Any], recorded: dict[str, Any],
                              *, allow_dirty: bool = False) -> None:
    """Fail closed if the current fingerprint materially differs from recorded.

    A dirty working tree is always a material difference for official runs.
    """
    if not allow_dirty and current.get("git_dirty") is True:
        raise ValueError("official run requires a clean working tree; git tree is dirty")
    diffs = _materially_different(current, recorded)
    if diffs:
        raise ValueError(
            "execution fingerprint mismatch on resume/comparison: "
            + ", ".join(sorted(diffs))
        )


def enforce_clean_tree(current: dict[str, Any]) -> None:
    """Fail closed if the working tree is dirty at launch."""
    if current.get("git_dirty") is True:
        raise ValueError("official launch requires a clean working tree; git tree is dirty")
