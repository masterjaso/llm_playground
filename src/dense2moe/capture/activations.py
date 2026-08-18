"""Deterministic, resumable binary activation capture.

Only MLP inputs are stored.  Values are written to safetensors shards; JSON
metadata contains hashes and shapes but never the activation payload itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ..provenance import current_git_commit
from ..state import atomic_write_json


def activation_partition_deterministic(num_items: int, partitions: int, seed: int = 0) -> list[list[int]]:
    if num_items < 0 or partitions <= 0:
        raise ValueError("num_items must be non-negative and partitions positive")
    order = sorted(range(num_items), key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest())
    result: list[list[int]] = [[] for _ in range(partitions)]
    for offset, item in enumerate(order):
        result[offset % partitions].append(item)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(raw: str | Path, metadata_path: Path) -> Path:
    """Resolve both run-relative and workspace-relative artifact locators."""

    value = str(raw)
    if os.sep != "\\" and "\\" in value:
        value = value.replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    local = metadata_path.parent / candidate
    if local.exists():
        return local
    workspace = Path.cwd() / candidate
    if workspace.exists():
        return workspace
    return local


def _write_shard(array: Any, path: Path, *, layer: int, shard_index: int, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        from safetensors.numpy import save_file  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy and safetensors are required for binary activation capture") from exc
    values = np.asarray(array)
    if values.ndim < 2:
        values = values.reshape(1, -1)
    if not np.isfinite(values).all():
        raise ValueError("activation shard contains NaN or Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        save_file({"mlp_input": values}, str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "layer": layer,
        "shard_index": shard_index,
        "format": "safetensors",
        "tensor": "mlp_input",
        "count": int(values.shape[0]),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "path": str(path),
        "sha256": _sha256(path),
        "metadata": dict(metadata or {}),
    }


def _valid_existing(metadata_path: Path) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        from safetensors import safe_open  # type: ignore
        if isinstance(payload.get("shards"), list):
            for shard in payload["shards"]:
                tensor_path = _resolve_manifest_path(str(shard["path"]), metadata_path)
                if not tensor_path.exists() or shard.get("sha256") != _sha256(tensor_path):
                    return None
                with safe_open(str(tensor_path), framework="numpy") as handle:
                    try:
                        shape = list(handle.get_slice("mlp_input").get_shape())
                    except (KeyError, RuntimeError):
                        return None
                if shape != list(shard.get("shape", [])):
                    return None
            return {**payload, "resumed": True}
        tensor_path = _resolve_manifest_path(str(payload["path"]), metadata_path)
        if payload.get("format") != "safetensors" or not tensor_path.exists() or payload.get("sha256") != _sha256(tensor_path):
            return None
        with safe_open(str(tensor_path), framework="numpy") as handle:
            try:
                shape = list(handle.get_slice("mlp_input").get_shape())
            except (KeyError, RuntimeError):
                return None
        if shape != list(payload.get("shape", [])):
            return None
        return {**payload, "path": str(tensor_path), "resumed": True}
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, json.JSONDecodeError):
        return None


def capture_activation_shards(
    activations: Iterable[Any],
    destination: str | Path,
    *,
    layer: int,
    shard_tokens: int = 8192,
    dtype: str = "float32",
    resume: bool = False,
    split: str | None = None,
    manifest_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture an iterable into atomic binary shards and a manifest."""

    if layer < 0 or shard_tokens <= 0:
        raise ValueError("layer must be non-negative and shard_tokens positive")
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for activation capture") from exc
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / (manifest_name or f"layer-{layer:04d}.json")
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("manifest_name must use the .json suffix")
    shard_stem = manifest_path.stem
    if resume:
        previous = _valid_existing(manifest_path)
        same_split = split is None or previous is not None and previous.get("split") == split
        expected_dataset = (metadata or {}).get("dataset_hash")
        same_dataset = not expected_dataset or previous is not None and previous.get("dataset_hash") == expected_dataset
        if previous is not None and same_split and same_dataset:
            return {**previous, "status": "CAPTURE_RESUMED"}
        if manifest_path.exists():
            return {
                "status": "CAPTURE_BLOCKED",
                "layer": layer,
                "count": 0,
                "code_commit": current_git_commit(),
                "message": "resume refused to overwrite an invalid or different split/dataset activation manifest",
            }
    pending: Any | None = None
    artifact_commit = current_git_commit()
    shard_metadata = dict(metadata or {})
    shard_metadata["code_commit"] = artifact_commit
    shards: list[dict[str, Any]] = []
    shard_index = 0
    for item in activations:
        values = np.asarray(item, dtype=dtype)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("activation items must be rank-1 or rank-2")
        pending = values if pending is None else np.concatenate((pending, values), axis=0)
        while pending.shape[0] >= shard_tokens:
            chunk = pending[:shard_tokens]
            pending = pending[shard_tokens:]
            tensor_path = dst / f"{shard_stem}-shard-{shard_index:05d}.safetensors"
            shard = _write_shard(chunk, tensor_path, layer=layer, shard_index=shard_index, metadata=shard_metadata)
            shards.append(shard)
            shard_index += 1
    if pending is not None and pending.shape[0] > 0:
        merged = pending
        tensor_path = dst / f"{shard_stem}-shard-{shard_index:05d}.safetensors"
        shards.append(_write_shard(merged, tensor_path, layer=layer, shard_index=shard_index, metadata=shard_metadata))
    if not shards:
        return {"status": "CAPTURE_BLOCKED", "layer": layer, "count": 0, "code_commit": artifact_commit, "message": "activation iterable is empty; no binary artifact was created"}
    manifest = {
        "schema_version": 2,
        "status": "CAPTURE_COMPLETE",
        "layer": layer,
        "dtype": dtype,
        "count": sum(int(item["count"]) for item in shards),
        "shard_tokens": shard_tokens,
        "shards": shards,
        "split": split,
        "metadata": dict(metadata or {}),
        "capture_kind": (metadata or {}).get("capture_kind", "mlp_input"),
        "hook_path": (metadata or {}).get("hook_path"),
        "source_snapshot": (metadata or {}).get("source_snapshot"),
        "tokenizer_revision": (metadata or {}).get("tokenizer_revision"),
        "dataset_hash": (metadata or {}).get("dataset_hash", ""),
        "source_revision": (metadata or {}).get("source_revision"),
        "code_commit": artifact_commit,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


class _StreamingShardWriter:
    """Bounded per-layer writer used by a single-forward fan-out capture."""

    def __init__(
        self,
        destination: Path,
        *,
        layer: int,
        shard_tokens: int,
        dtype: str,
        split: str | None,
        manifest_name: str,
        resume: bool,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        self.destination = destination
        self.layer = layer
        self.shard_tokens = shard_tokens
        self.dtype = dtype
        self.split = split
        self.manifest_path = destination / manifest_name
        self.metadata = dict(metadata or {})
        self.artifact_commit = current_git_commit()
        self.shards: list[dict[str, Any]] = []
        self.pending: Any | None = None
        self.shard_index = 0
        self.resumed_manifest: dict[str, Any] | None = None
        if resume:
            previous = _valid_existing(self.manifest_path)
            same_split = previous is not None and (split is None or previous.get("split") == split)
            expected_dataset = self.metadata.get("dataset_hash")
            same_dataset = previous is not None and (
                not expected_dataset or previous.get("dataset_hash") == expected_dataset
            )
            if previous is not None and same_split and same_dataset:
                self.resumed_manifest = previous
            elif self.manifest_path.exists():
                raise ValueError(
                    "resume refused to overwrite an invalid or different split/dataset activation manifest"
                )
        destination.mkdir(parents=True, exist_ok=True)

    @property
    def complete(self) -> bool:
        return self.resumed_manifest is not None

    def append(self, values: Any) -> None:
        if self.complete:
            return
        import numpy as np  # type: ignore

        array = np.asarray(values, dtype=self.dtype)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("activation items must be rank-1 or rank-2")
        if not np.isfinite(array).all():
            raise ValueError("activation shard contains NaN or Inf")
        self.pending = array if self.pending is None else np.concatenate((self.pending, array), axis=0)
        while self.pending.shape[0] >= self.shard_tokens:
            chunk = self.pending[: self.shard_tokens]
            self.pending = self.pending[self.shard_tokens :]
            path = self.destination / f"{self.manifest_path.stem}-shard-{self.shard_index:05d}.safetensors"
            self.shards.append(
                _write_shard(
                    chunk,
                    path,
                    layer=self.layer,
                    shard_index=self.shard_index,
                    metadata={**self.metadata, "split": self.split},
                )
            )
            self.shard_index += 1

    def finish(self) -> dict[str, Any]:
        previous = self.resumed_manifest
        if previous is not None:
            return {**previous, "status": "CAPTURE_RESUMED"}
        if self.pending is not None and self.pending.shape[0] > 0:
            path = self.destination / f"{self.manifest_path.stem}-shard-{self.shard_index:05d}.safetensors"
            self.shards.append(
                _write_shard(
                    self.pending,
                    path,
                    layer=self.layer,
                    shard_index=self.shard_index,
                    metadata={**self.metadata, "split": self.split},
                )
            )
            self.pending = None
        if not self.shards:
            return {
                "status": "CAPTURE_BLOCKED",
                "layer": self.layer,
                "count": 0,
                "code_commit": self.artifact_commit,
                "message": "activation iterable is empty; no binary artifact was created",
            }
        manifest = {
            "schema_version": 2,
            "status": "CAPTURE_COMPLETE",
            "layer": self.layer,
            "dtype": self.dtype,
            "count": sum(int(item["count"]) for item in self.shards),
            "shard_tokens": self.shard_tokens,
            "shards": self.shards,
            "split": self.split,
            "metadata": dict(self.metadata),
            "capture_kind": self.metadata.get("capture_kind", "mlp_input"),
            "hook_path": self.metadata.get("hook_path"),
            "source_snapshot": self.metadata.get("source_snapshot"),
            "tokenizer_revision": self.metadata.get("tokenizer_revision"),
            "dataset_hash": self.metadata.get("dataset_hash", ""),
            "source_revision": self.metadata.get("source_revision"),
            "code_commit": self.artifact_commit,
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest


def capture_multi_layer_activation_shards(
    batches: Iterable[Mapping[int, Any]],
    destination: str | Path,
    *,
    layers: Iterable[int],
    shard_tokens: int = 8192,
    dtype: str = "float32",
    resume: bool = False,
    split: str | None = None,
    manifest_names: Mapping[int, str] | None = None,
    metadata: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Write all selected layer streams from one forward-pass iterator.

    The iterator yields a mapping for each input microbatch.  Each layer owns
    an independent bounded pending buffer and shard manifest, so no complete
    corpus activation tensor is retained in memory and one slow layer cannot
    overwrite another layer's receipt.
    """

    selected = sorted({int(layer) for layer in layers})
    if not selected:
        raise ValueError("layers must contain at least one layer")
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    root = Path(destination)
    names = manifest_names or {}
    details = metadata or {}
    writers: dict[int, _StreamingShardWriter] = {}
    for layer in selected:
        writers[layer] = _StreamingShardWriter(
            root,
            layer=layer,
            shard_tokens=shard_tokens,
            dtype=dtype,
            split=split,
            manifest_name=str(names.get(layer, f"layer-{layer:04d}-{split or 'capture'}.json")),
            resume=resume,
            metadata=details.get(layer),
        )
    seen = False
    for batch in batches:
        seen = True
        for layer, writer in writers.items():
            if layer not in batch:
                raise ValueError(f"fan-out batch is missing requested layer {layer}")
            writer.append(batch[layer])
    results = {layer: writer.finish() for layer, writer in writers.items()}
    if not seen:
        return results
    return results


def capture_activations(
    activations: Iterable[Any],
    destination: str | Path,
    *,
    layer: int,
    dtype: str = "float32",
    shard_tokens: int = 8192,
    resume: bool = False,
    split: str | None = None,
    manifest_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper around :func:`capture_activation_shards`."""

    return capture_activation_shards(
        activations,
        destination,
        layer=layer,
        shard_tokens=shard_tokens,
        dtype=dtype,
        resume=resume,
        split=split,
        manifest_name=manifest_name,
        metadata=metadata,
    )


def iter_activation_shards(manifest_path: str | Path, *, expected_split: str | None = None) -> Iterator[Any]:
    """Yield binary MLP-input tensors from a validated capture manifest."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_split = str(manifest.get("split", ""))
    split_aliases = {
        "train": {"train", "FIT-TRAIN"},
        "FIT-TRAIN": {"train", "FIT-TRAIN"},
        "holdout": {"holdout", "FIT-DEV"},
        "FIT-DEV": {"holdout", "FIT-DEV"},
    }
    if expected_split is not None and actual_split not in split_aliases.get(expected_split, {expected_split}):
        raise ValueError(f"activation manifest split mismatch: expected {expected_split!r}, got {actual_split!r}")
    try:
        import numpy as np  # type: ignore
        from safetensors import safe_open  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy and safetensors are required") from exc
    for shard in manifest.get("shards", []):
        path = _resolve_manifest_path(str(shard["path"]), Path(manifest_path))
        if _sha256(path) != shard.get("sha256"):
            raise ValueError(f"activation shard hash mismatch: {path}")
        # The NumPy safetensors backend cannot decode BF16 on the Windows
        # runtime used for the streamed corpus.  Fall back to the PyTorch
        # backend and convert only the yielded view to float32; the durable
        # artifact remains BF16 and its hash is still checked above.
        input_name = str(shard.get("input_tensor") or manifest.get("input_tensor") or "mlp_input")
        try:
            with safe_open(str(path), framework="numpy") as handle:
                yield np.asarray(handle.get_tensor(input_name))
        except (TypeError, ValueError, RuntimeError):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                yield handle.get_tensor(input_name).float().numpy()
