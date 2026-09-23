"""Full-state, replacement-safe checkpoints for the production trajectory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import torch

from .observability import atomic_write_json
from .production import CHECKPOINT_SCHEMA_VERSION, model_config_sha256


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise OSError(f"copy checksum mismatch: {source} -> {destination}")
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def checkpoint_identity(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_full_checkpoint(
    destination: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    exact_tokens: int,
    config: Any,
    identity: dict[str, Any],
    data_cursor: dict[str, Any],
    scheduler_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    metrics_state: dict[str, Any] | None = None,
    session_lineage: dict[str, Any] | None = None,
    parent_checkpoint_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and verify a complete checkpoint directory.

    The candidate is built beside the destination, read back, checksummed, and
    only then atomically promoted.  The previous destination remains intact if
    any write/read/checksum step fails.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if int(step) < 0 or int(exact_tokens) < 0:
        raise ValueError("checkpoint counters must be non-negative")
    candidate = Path(tempfile.mkdtemp(prefix=f".{destination.name}.candidate-", dir=destination.parent))
    payload_path = candidate / "state.pt"
    manifest_path = candidate / "manifest.json"
    try:
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": int(step), "exact_tokens": int(exact_tokens),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "data_cursor": dict(data_cursor),
            "metrics_state": metrics_state,
            "session_lineage": session_lineage,
            "identity": dict(identity), "extra": dict(extra or {}),
        }
        torch.save(payload, payload_path)
        with payload_path.open("rb") as handle:
            os.fsync(handle.fileno())
        loaded = torch.load(payload_path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "data_cursor", "identity", "step", "exact_tokens",
                    "optimizer_state_dict", "checkpoint_schema_version"}
        if not required.issubset(loaded):
            raise ValueError("checkpoint read-back is missing required full-state fields")
        if loaded["step"] != int(step) or loaded["exact_tokens"] != int(exact_tokens):
            raise ValueError("checkpoint read-back counter mismatch")
        if model_config_sha256(config) != identity.get("model_config_sha256"):
            raise ValueError("checkpoint identity does not match model config")
        payload_sha = sha256_file(payload_path)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": int(step), "exact_tokens": int(exact_tokens),
            "payload": "state.pt", "payload_sha256": payload_sha,
            "payload_bytes": payload_path.stat().st_size,
            "identity": dict(identity),
            "data_cursor": dict(data_cursor),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "session_lineage": dict(session_lineage or {}),
            "extra": dict(extra or {}),
        }
        manifest["checkpoint_sha256"] = checkpoint_identity(manifest)
        atomic_write_json(manifest_path, manifest)
        # A second read verifies both the manifest and the payload before any
        # old recovery point can be retired.
        checked = json.loads(manifest_path.read_text())
        if checked["payload_sha256"] != sha256_file(payload_path):
            raise ValueError("checkpoint payload checksum changed during promotion")
        os.replace(candidate, destination)
        _fsync_directory(destination.parent)
        return manifest
    except Exception:
        shutil.rmtree(candidate, ignore_errors=True)
        raise


def verify_checkpoint(path: Path | str, *, expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    manifest_path, payload_path = path / "manifest.json", path / "state.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version mismatch")
    if manifest.get("payload_sha256") != sha256_file(payload_path):
        raise ValueError("checkpoint payload checksum mismatch")
    recorded_identity = manifest.get("checkpoint_sha256")
    canonical_manifest = {key: value for key, value in manifest.items() if key != "checkpoint_sha256"}
    if recorded_identity and recorded_identity != checkpoint_identity(canonical_manifest):
        raise ValueError("checkpoint manifest checksum mismatch")
    if expected_identity:
        for key, value in expected_identity.items():
            if manifest.get("identity", {}).get(key) != value:
                raise ValueError(f"checkpoint identity mismatch for {key}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "data_cursor", "identity", "step", "exact_tokens"):
        if key not in payload:
            raise ValueError(f"checkpoint payload missing {key}")
    if payload.get("step") != manifest.get("step") or payload.get("exact_tokens") != manifest.get("exact_tokens"):
        raise ValueError("checkpoint payload and manifest counters disagree")
    if payload.get("identity") != manifest.get("identity"):
        raise ValueError("checkpoint payload and manifest identities disagree")
    return manifest


class RemoteCheckpointBackend(Protocol):
    def publish(self, local_checkpoint: Path, manifest: dict[str, Any]) -> dict[str, Any]: ...
    def verify(self, remote_checkpoint: Path, manifest: dict[str, Any]) -> None: ...
    def retire(self, previous: Path | None, *, keep: Path) -> None: ...


class FilesystemRemoteBackend:
    """Provider-neutral durable backend used by tests and mounted object stores."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest = self.root / "LATEST.json"

    def publish(self, local_checkpoint: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        name = f"checkpoint_step_{int(manifest['step'])}"
        target = self.root / name
        if target.exists():
            existing = verify_checkpoint(target)
            if existing.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"):
                return {"path": str(target), "checkpoint_sha256": manifest["checkpoint_sha256"]}
            raise ValueError(f"remote checkpoint step already exists with a different identity: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self.root))
        try:
            for source in local_checkpoint.iterdir():
                if source.is_file():
                    _copy_verified(source, temporary / source.name)
            os.replace(temporary, target)
            _fsync_directory(self.root)
            self.verify(target, manifest)
            atomic_write_json(self.latest, {"checkpoint": name, "checkpoint_sha256": manifest["checkpoint_sha256"]})
            return {"path": str(target), "checkpoint_sha256": manifest["checkpoint_sha256"]}
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def verify(self, remote_checkpoint: Path, manifest: dict[str, Any]) -> None:
        checked = verify_checkpoint(remote_checkpoint)
        if checked.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise ValueError("remote checkpoint manifest identity mismatch")

    def retire(self, previous: Path | None, *, keep: Path) -> None:
        if previous and previous.exists() and previous.resolve() != keep.resolve():
            shutil.rmtree(previous)

    def latest_path(self) -> Path | None:
        if not self.latest.is_file():
            return None
        value = json.loads(self.latest.read_text()).get("checkpoint")
        return self.root / value if value else None


class DurableCheckpointManager:
    """Local candidate -> remote verify -> pointer -> bounded retention."""

    def __init__(self, local_root: Path | str, remote: RemoteCheckpointBackend | None = None) -> None:
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.remote = remote

    def latest_local(self) -> Path | None:
        candidates = sorted(self.local_root.glob("checkpoint_step_*"), key=lambda p: p.name)
        return candidates[-1] if candidates else None

    def save_and_sync(self, *, step: int, **kwargs: Any) -> dict[str, Any]:
        destination = self.local_root / f"checkpoint_step_{int(step)}"
        previous_local = self.latest_local()
        manifest = save_full_checkpoint(destination, step=step, **kwargs)
        verify_checkpoint(destination)
        remote_result = None
        previous_remote = None
        if self.remote is not None:
            previous_remote = None
            latest_path = getattr(self.remote, "latest_path", None)
            if callable(latest_path):
                previous_remote = latest_path()
            remote_result = self.remote.publish(destination, manifest)
            # ``publish`` is required to verify before it returns.  Only now is
            # it safe to retire the previous recovery point.
            self.remote.retire(previous_remote, keep=Path(remote_result["path"]))
        for old in self.local_root.glob("checkpoint_step_*"):
            if old != destination and old.is_dir():
                shutil.rmtree(old)
        return {"manifest": manifest, "local_path": str(destination), "remote": remote_result,
                "previous_local": str(previous_local) if previous_local else None}


__all__ = [
    "DurableCheckpointManager", "FilesystemRemoteBackend", "RemoteCheckpointBackend",
    "checkpoint_identity", "save_full_checkpoint", "sha256_file", "verify_checkpoint",
]
