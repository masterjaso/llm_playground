"""Exact layer-major Qwen3.5 teacher replay.

The ordinary teacher capture path keeps the complete text model resident.  On
Windows that makes a 27B checkpoint contend for VRAM, RAM, and page-file
commit.  This module intentionally has a different residency contract:

* the embedding table is loaded only while stage zero is materialised;
* one native :class:`Qwen3_5DecoderLayer` is loaded at a time;
* every corpus shard is replayed through that layer before it is unloaded; and
* hidden states and selected MLP inputs are durable BF16 safetensors.

No transformer operation is reimplemented here.  The installed Transformers
Qwen3.5 classes and mask/rotary helpers are instantiated directly from the
pinned source config.  The source checkpoint is read through the index and is
never modified.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..provenance import current_git_commit
from ..state import atomic_write_json
from .teacher import (
    TeacherCaptureBlocked,
    TokenizedExample,
    fixed_split_metadata,
    load_pinned_tokenizer,
    resolve_corpus_records,
    snapshot_tokenizer_hashes,
    tokenize_corpus_records,
)

STREAMING_SCHEMA_VERSION = 1
DEFAULT_SHARD_TOKENS = 2048
DEFAULT_DEVICE = "cuda:1"
REPRESENTATIVE_LAYERS = (0, 16, 32, 48, 63)
_TEXT_PREFIX = "model.language_model."
_LAYER_PREFIX = "model.language_model.layers."


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _tensor_hash(tensor: Any) -> str:
    """Hash tensor values without converting BF16 through unsupported NumPy."""

    import torch  # type: ignore

    value = tensor.detach().to("cpu").contiguous()
    if value.dtype in (torch.bfloat16, torch.float16):
        raw = value.view(torch.uint16).numpy().tobytes()
    elif value.dtype == torch.float32:
        raw = value.numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(raw)
    return digest.hexdigest()


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _atomic_save_tensors(tensors: Mapping[str, Any], destination: str | Path) -> str:
    """Publish one safetensors file atomically and return its SHA-256."""

    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("safetensors and torch are required for streaming replay") from exc
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        save_file(dict(tensors), str(temporary_path))
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return _sha256(target)


def _load_tensors(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    try:
        from safetensors.torch import load_file  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("safetensors and torch are required for streaming replay") from exc
    return dict(load_file(str(path), device=device))


@dataclass(frozen=True)
class SourceTensorReceipt:
    source_name: str
    source_shard: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_shard": self.source_shard,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


class IndexedSource:
    """Exact index-driven source tensor access."""

    def __init__(self, source_snapshot: str | Path) -> None:
        self.root = Path(source_snapshot)
        self.index_path = self.root / "model.safetensors.index.json"
        if not self.root.is_dir() or not self.index_path.is_file():
            raise TeacherCaptureBlocked(
                "STREAM_SOURCE_INDEX_MISSING",
                "streaming teacher requires a local model.safetensors.index.json",
                source_snapshot=str(self.root),
            )
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
            if not isinstance(weight_map, Mapping):
                raise TypeError("weight_map is not an object")
            self.weight_map = {str(name): str(shard) for name, shard in weight_map.items()}
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise TeacherCaptureBlocked("STREAM_SOURCE_INDEX_INVALID", f"source index is invalid: {exc}") from exc
        self.index_hash = _sha256(self.index_path)

    def names_for_prefix(self, prefix: str) -> dict[str, str]:
        return {name: shard for name, shard in self.weight_map.items() if name.startswith(prefix)}

    def _read(self, source_name: str, *, expected_shape: Sequence[int] | None = None) -> tuple[Any, SourceTensorReceipt]:
        try:
            import torch  # type: ignore
            from safetensors import safe_open  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("safetensors and torch are required for streaming replay") from exc
        shard_name = self.weight_map.get(source_name)
        if shard_name is None:
            raise TeacherCaptureBlocked(
                "STREAM_SOURCE_TENSOR_MISSING",
                f"source index has no tensor named {source_name}",
                source_name=source_name,
            )
        shard_path = self.root / shard_name
        if not shard_path.is_file():
            raise TeacherCaptureBlocked("STREAM_SOURCE_SHARD_MISSING", f"source shard is missing: {shard_path}")
        try:
            # Windows section-backed mmap can reserve commit for the whole
            # shard and has already produced WinError 1455/access violations
            # in this project.  safetensors' pread backend reads only the
            # requested tensor and is the required native Windows path.
            open_kwargs = {"backend": "pread"} if os.name == "nt" else {}
            with safe_open(str(shard_path), framework="pt", device="cpu", **open_kwargs) as handle:
                tensor = handle.get_tensor(source_name)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise TeacherCaptureBlocked(
                "STREAM_SOURCE_TENSOR_READ_FAILED",
                f"could not read {source_name} from {shard_path}: {exc}",
                source_name=source_name,
                source_shard=str(shard_path),
            ) from exc
        if expected_shape is not None and tuple(tensor.shape) != tuple(int(v) for v in expected_shape):
            raise TeacherCaptureBlocked(
                "STREAM_SOURCE_SHAPE_MISMATCH",
                f"source tensor {source_name} has shape {tuple(tensor.shape)}, expected {tuple(expected_shape)}",
                source_name=source_name,
            )
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TeacherCaptureBlocked(
                "STREAM_SOURCE_DTYPE_UNSUPPORTED",
                f"source tensor {source_name} has unsupported dtype {tensor.dtype}",
                source_name=source_name,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise TeacherCaptureBlocked("STREAM_SOURCE_NONFINITE", f"source tensor {source_name} contains NaN or Inf")
        receipt = SourceTensorReceipt(
            source_name=source_name,
            source_shard=shard_name,
            shape=tuple(int(v) for v in tensor.shape),
            dtype=_dtype_name(tensor.dtype),
            sha256=_tensor_hash(tensor),
        )
        return tensor, receipt

    def read_exact(self, source_name: str, *, expected_shape: Sequence[int] | None = None) -> tuple[Any, SourceTensorReceipt]:
        return self._read(source_name, expected_shape=expected_shape)


def _make_text_config(source_snapshot: Path, attention_implementation: str) -> Any:
    try:
        from transformers import AutoConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise TeacherCaptureBlocked("STREAM_TRANSFORMERS_UNAVAILABLE", "Transformers is required for streaming replay") from exc
    try:
        config = AutoConfig.from_pretrained(str(source_snapshot), local_files_only=True, trust_remote_code=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TeacherCaptureBlocked("STREAM_CONFIG_LOAD_FAILED", f"pinned source config could not be loaded: {exc}") from exc
    text_config = getattr(config, "text_config", config)
    # PreTrainedModel chooses SDPA by default in the installed runtime.  Set
    # the internal value explicitly so a standalone decoder layer follows the
    # same dispatch path as Qwen3_5ForCausalLM.from_pretrained.
    text_config._attn_implementation_internal = attention_implementation
    return text_config


class TargetedLayerLoader:
    """Construct a native one-layer module from only its indexed tensors."""

    def __init__(self, source_snapshot: str | Path, *, compute_dtype: str = "bfloat16", attention_implementation: str = "sdpa") -> None:
        try:
            import torch  # type: ignore
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer,  # type: ignore
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise TeacherCaptureBlocked("STREAM_RUNTIME_UNAVAILABLE", "torch, Transformers, and safetensors are required") from exc
        self.torch = torch
        self.decoder_layer_type = Qwen3_5DecoderLayer
        self.source = IndexedSource(source_snapshot)
        self.text_config = _make_text_config(self.source.root, attention_implementation)
        normalized = str(compute_dtype).lower().replace("-", "")
        aliases = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}
        if normalized not in aliases:
            raise TeacherCaptureBlocked("STREAM_DTYPE_UNSUPPORTED", f"streaming compute dtype must be BF16/FP16, got {compute_dtype}")
        self.compute_dtype = aliases[normalized]
        self.attention_implementation = attention_implementation

    def load_embedding(self, device: Any) -> tuple[Any, dict[str, Any]]:
        import torch  # type: ignore
        from torch import nn  # type: ignore

        name = f"{_TEXT_PREFIX}embed_tokens.weight"
        tensor, receipt = self.source.read_exact(name)
        expected = (int(self.text_config.vocab_size), int(self.text_config.hidden_size))
        if tuple(tensor.shape) != expected:
            raise TeacherCaptureBlocked("STREAM_EMBEDDING_SHAPE_MISMATCH", f"embedding shape {tuple(tensor.shape)} != {expected}")
        embedding = nn.Embedding(expected[0], expected[1], getattr(self.text_config, "pad_token_id", None))
        embedding = embedding.to(device=device, dtype=self.compute_dtype)
        with torch.no_grad():
            embedding.weight.copy_(tensor.to(device=device, dtype=self.compute_dtype))
        del tensor
        return embedding.eval(), {"tensor": receipt.as_dict(), "source_index_hash": self.source.index_hash}

    def load_layer(self, layer: int, device: Any) -> tuple[Any, dict[str, Any]]:
        import torch  # type: ignore

        if layer < 0 or layer >= int(self.text_config.num_hidden_layers):
            raise ValueError(f"layer out of range: {layer}")
        native = self.decoder_layer_type(self.text_config, layer)
        native = native.to(device="cpu", dtype=self.compute_dtype)
        expected_state = native.state_dict()
        prefix = f"{_LAYER_PREFIX}{layer}."
        names = self.source.names_for_prefix(prefix)
        expected_names = set(expected_state)
        source_short = {name[len(prefix) :]: name for name in names if name.startswith(prefix)}
        if set(source_short) != expected_names:
            missing = sorted(expected_names - set(source_short))
            unexpected = sorted(set(source_short) - expected_names)
            raise TeacherCaptureBlocked(
                "STREAM_LAYER_INVENTORY_MISMATCH",
                f"layer {layer} tensor inventory mismatch; missing={missing}, unexpected={unexpected}",
                layer=layer,
            )
        state: dict[str, Any] = {}
        receipts: list[dict[str, Any]] = []
        for short_name in sorted(expected_names):
            source_name = source_short[short_name]
            expected_shape = tuple(int(v) for v in expected_state[short_name].shape)
            tensor, tensor_receipt = self.source.read_exact(source_name, expected_shape=expected_shape)
            if tensor.dtype != torch.bfloat16:
                raise TeacherCaptureBlocked(
                    "STREAM_LAYER_DTYPE_MISMATCH",
                    f"layer {layer} tensor {source_name} is {tensor.dtype}, expected BF16 source weights",
                    layer=layer,
                    source_name=source_name,
                )
            state[short_name] = tensor
            receipts.append(tensor_receipt.as_dict())
        missing_keys, unexpected_keys = native.load_state_dict(state, strict=True)
        if missing_keys or unexpected_keys:
            raise TeacherCaptureBlocked(
                "STREAM_LAYER_LOAD_STATE_MISMATCH",
                f"native layer load mismatch: missing={missing_keys}, unexpected={unexpected_keys}",
                layer=layer,
            )
        native = native.to(device=device, dtype=self.compute_dtype).eval()
        del state
        gc.collect()
        receipt = {
            "layer": layer,
            "source_index_hash": self.source.index_hash,
            "source_tensor_count": len(receipts),
            "source_tensor_hashes": receipts,
            "source_bytes": sum(
                int(math.prod(item["shape"])) * (2 if item["dtype"] in {"bfloat16", "float16"} else 4)
                for item in receipts
            ),
            "compute_dtype": _dtype_name(self.compute_dtype),
            "attention_implementation": self.attention_implementation,
        }
        return native, receipt


def _group_examples(examples: Sequence[TokenizedExample], shard_tokens: int) -> list[list[TokenizedExample]]:
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    groups: list[list[TokenizedExample]] = []
    current: list[TokenizedExample] = []
    current_tokens = 0
    for item in examples:
        count = len(item.input_ids)
        if current and current_tokens + count > shard_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += count
    if current:
        groups.append(current)
    return groups


def _batch_token_tensors(group: Sequence[TokenizedExample], torch: Any, device: Any, pad_token_id: int = 0) -> tuple[Any, Any, Any]:
    max_len = max(len(item.input_ids) for item in group)
    input_ids = torch.full((len(group), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(group), max_len), dtype=torch.long, device=device)
    for row, item in enumerate(group):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(len(group), -1)
    return input_ids, attention_mask, position_ids


def _batch_hidden_tensors(group: Sequence[Mapping[str, Any]], torch: Any, device: Any) -> tuple[Any, Any, Any, list[int]]:
    hidden = torch.stack([item["hidden_states"] for item in group], dim=0).to(device=device)
    attention_mask = torch.stack([item["attention_mask"] for item in group], dim=0).to(device=device)
    lengths = [int(v) for item in group for v in item["lengths"].reshape(-1).tolist()]
    max_len = int(hidden.shape[1])
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(hidden.shape[0], -1)
    return hidden, attention_mask, position_ids, lengths


def _stage_root(root: Path, stage: int) -> Path:
    parent = root / ("hidden-A" if stage % 2 == 0 else "hidden-B")
    return parent / f"stage-{stage:04d}"


def _stage_manifest_path(root: Path, stage: int) -> Path:
    return _stage_root(root, stage) / "manifest.json"


def _valid_stage_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "STAGE_COMPLETE" or not isinstance(payload.get("shards"), list):
            return None
        for item in payload["shards"]:
            shard_path = path.parent / str(item["path"])
            if not shard_path.is_file() or _sha256(shard_path) != item.get("sha256"):
                return None
        return payload
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_hidden_shard(root: Path, stage: int, shard_id: int, tensors: Mapping[str, Any], records: Sequence[TokenizedExample], *, split: str, dataset_hash: str, source_revision: str) -> dict[str, Any]:
    stage_dir = _stage_root(root, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"shard-{shard_id:05d}.safetensors"
    digest = _atomic_save_tensors(tensors, path)
    lengths = [len(item.input_ids) for item in records]
    return {
        "shard_id": shard_id,
        "path": path.name,
        "sha256": digest,
        "count": int(sum(lengths)),
        "example_count": len(records),
        "bytes": path.stat().st_size,
        "shape": list(tensors["hidden_states"].shape),
        "dtype": _dtype_name(tensors["hidden_states"].dtype),
        "records": [
            {
                "example_id": item.example_id,
                "split": item.split,
                "source_record_index": item.source_record_index,
                "chunk_index": item.chunk_index,
                "length": len(item.input_ids),
                "content_sha256": item.content_sha256,
                "offset": int(sum(lengths[:index])),
            }
            for index, item in enumerate(records)
        ],
        "dataset_hash": dataset_hash,
        "source_revision": source_revision,
        "stage": stage,
    }


def _write_stage_manifest(root: Path, stage: int, shards: Sequence[Mapping[str, Any]], *, split: str, dataset_hash: str, source_revision: str, tokenizer_hash: str, code_commit: str, stage_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "schema_version": STREAMING_SCHEMA_VERSION,
        "status": "STAGE_COMPLETE",
        "stage": stage,
        "split": split,
        "shards": list(shards),
        "count": sum(int(item.get("count", 0)) for item in shards),
        "dataset_hash": dataset_hash,
        "tokenizer_hash": tokenizer_hash,
        "source_revision": source_revision,
        "code_commit": code_commit,
        "metadata": dict(stage_metadata or {}),
    }
    atomic_write_json(_stage_manifest_path(root, stage), manifest)
    return manifest


def _load_progress(path: Path, *, layer: int, split: str, dataset_hash: str, source_revision: str) -> set[int]:
    """Return only shard IDs backed by a matching in-progress receipt."""

    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    if (
        payload.get("status") != "IN_PROGRESS"
        or int(payload.get("layer", -1)) != layer
        or payload.get("split") != split
        or payload.get("dataset_hash") != dataset_hash
        or payload.get("source_revision") != source_revision
    ):
        return set()
    try:
        return {int(value) for value in payload.get("completed_shard_ids", [])}
    except (TypeError, ValueError):
        return set()


def _valid_capture_manifest(path: Path, *, dataset_hash: str, split: str, layer: int, expected_count: int | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "CAPTURE_COMPLETE"
            or int(payload.get("layer", -1)) != layer
            or payload.get("split") != split
            or payload.get("dataset_hash") != dataset_hash
            or not isinstance(payload.get("shards"), list)
        ):
            return None
        if expected_count is not None and int(payload.get("count", -1)) != expected_count:
            return None
        for item in payload["shards"]:
            shard_path = _resolve_capture_path(path, str(item["path"]))
            if not shard_path.is_file() or _sha256(shard_path) != item.get("sha256"):
                return None
        return payload
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _resolve_capture_path(manifest_path: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    # Stream manifests are rooted at run/capture; retain compatibility with
    # both POSIX and Windows-style relative locators.
    value = raw.replace("\\", "/") if os.sep != "\\" else raw
    candidate = Path(value)
    local = manifest_path.parent / candidate
    if local.exists():
        return local
    return manifest_path.parent.parent / candidate


def _read_stage(root: Path, stage: int, *, torch: Any, device: Any) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = _valid_stage_manifest(_stage_manifest_path(root, stage))
    if manifest is None:
        raise TeacherCaptureBlocked("STREAM_STAGE_INVALID", f"rolling hidden stage {stage} is incomplete or corrupt")
    for item in manifest["shards"]:
        path = _stage_root(root, stage) / str(item["path"])
        tensors = _load_tensors(path, device="cpu")
        yield item, tensors


def _run_native_layer(executor: TargetedLayerLoader, layer_module: Any, hidden: Any, attention_mask: Any, position_ids: Any, *, capture_mlp_input: bool) -> tuple[Any, Any | None]:
    import torch  # type: ignore
    from transformers.models.qwen3_5.modeling_qwen3_5 import (  # type: ignore
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    if hasattr(executor, "get_position_embeddings"):
        position_embeddings = executor.get_position_embeddings(hidden, position_ids)
    else:
        position_embeddings = executor.rotary_emb(hidden, position_ids) if hasattr(executor, "rotary_emb") else None
    if position_embeddings is None:
        raise RuntimeError("streaming executor rotary embedding was not initialized")
    block_type = str(layer_module.block_type)
    if block_type == "full_attention":
        mask = create_causal_mask(executor.text_config, hidden, attention_mask, None, position_ids=position_ids)
    else:
        mask = create_recurrent_attention_mask(executor.text_config, hidden, attention_mask)
    captured: list[Any] = []
    hook = None
    if capture_mlp_input:
        def _pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RuntimeError("native Qwen MLP hook received no input")
            captured.append(args[0].detach().clone())

        hook = layer_module.mlp.register_forward_pre_hook(_pre_hook)
    try:
        with torch.inference_mode():
            output = layer_module(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
    finally:
        if hook is not None:
            hook.remove()
    if not torch.isfinite(output).all():
        raise TeacherCaptureBlocked("STREAM_NONFINITE_OUTPUT", f"layer {getattr(layer_module, 'layer_idx', '?')} emitted NaN or Inf")
    value = captured[0] if captured else None
    return output, value


class StreamingTeacherExecutor(TargetedLayerLoader):
    """Native one-layer executor with exact Qwen rotary/mask construction."""

    def __init__(self, source_snapshot: str | Path, *, device: str = DEFAULT_DEVICE, compute_dtype: str = "bfloat16", attention_implementation: str = "sdpa") -> None:
        super().__init__(source_snapshot, compute_dtype=compute_dtype, attention_implementation=attention_implementation)
        import torch  # type: ignore
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,  # type: ignore
        )

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TeacherCaptureBlocked("STREAM_CUDA_UNAVAILABLE", f"layer streaming requires the requested CUDA device, got {device}")
        if self.device.type == "cuda" and self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise TeacherCaptureBlocked("STREAM_DEVICE_UNAVAILABLE", f"CUDA device does not exist: {device}")
        self.position_device = self.device
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.text_config).to(self.position_device)
        self._position_cache: dict[tuple[int, int], tuple[Any, Any]] = {}

    def set_device(self, device: str | Any) -> None:
        """Move the small rotary helper before a scheduled layer replay.

        The default production schedule remains one layer at a time on the
        5060 Ti.  A per-layer schedule is also supported for reproducing a
        previously validated native reference whose later layers executed on
        CPU under the old offload map.
        """

        import torch  # type: ignore

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise TeacherCaptureBlocked("STREAM_DEVICE_UNAVAILABLE", f"CUDA device does not exist: {self.device}")

    def get_position_embeddings(self, hidden: Any, position_ids: Any) -> tuple[Any, Any]:
        """Compute native rotary values once on the initial model device.

        Qwen3.5's text model constructs one rotary pair before its layer loop;
        it does not recompute RoPE on each offloaded layer.  Keeping that
        contract matters when a reference schedule moves later layers to CPU.
        """

        key = (int(hidden.shape[0]), int(hidden.shape[1]))
        pair = self._position_cache.get(key)
        if pair is None:
            import torch  # type: ignore

            rotary_hidden = hidden if hidden.device == self.position_device else torch.empty(hidden.shape, dtype=self.compute_dtype, device=self.position_device)
            rotary_positions = position_ids.to(device=self.position_device)
            pair = tuple(value.detach() for value in self.rotary_emb(rotary_hidden, rotary_positions))
            self._position_cache[key] = pair
            del rotary_hidden, rotary_positions
        return pair[0].to(device=hidden.device), pair[1].to(device=hidden.device)

    def materialize_embeddings(self, groups: Sequence[Sequence[TokenizedExample]], *, root: Path, split: str, dataset_hash: str, tokenizer_hash: str, source_revision: str, pad_token_id: int) -> dict[str, Any]:
        import torch  # type: ignore

        existing = _valid_stage_manifest(_stage_manifest_path(root, 0))
        expected_count = sum(len(item.input_ids) for group in groups for item in group)
        if existing is not None and existing.get("dataset_hash") == dataset_hash and existing.get("split") == split and int(existing.get("count", -1)) == expected_count:
            return existing
        embedding, receipt = self.load_embedding(self.device)
        shards: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            for shard_id, group in enumerate(groups):
                input_ids, attention_mask, _position_ids = _batch_token_tensors(group, torch, self.device, pad_token_id)
                with torch.inference_mode():
                    hidden = embedding(input_ids)
                hidden_cpu = hidden.detach().to("cpu")
                tensors = {
                    "hidden_states": hidden_cpu,
                    "input_ids": input_ids.detach().to("cpu"),
                    "attention_mask": attention_mask.detach().to("cpu"),
                    "lengths": torch.tensor([len(item.input_ids) for item in group], dtype=torch.int64),
                    "source_record_index": torch.tensor([item.source_record_index for item in group], dtype=torch.int64),
                }
                shards.append(_write_hidden_shard(root, 0, shard_id, tensors, group, split=split, dataset_hash=dataset_hash, source_revision=source_revision))
                del input_ids, attention_mask, hidden, hidden_cpu, tensors
        finally:
            del embedding
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return _write_stage_manifest(
            root,
            0,
            shards,
            split=split,
            dataset_hash=dataset_hash,
            source_revision=source_revision,
            tokenizer_hash=tokenizer_hash,
            code_commit=current_git_commit(),
            stage_metadata={
                "kind": "token_embedding_hidden_states",
                "weight_receipt": receipt,
                "example_count": len([item for group in groups for item in group]),
                "token_count": expected_count,
                "shard_tokens": max((sum(len(item.input_ids) for item in group) for group in groups), default=0),
                "tokens_per_second": sum(len(item.input_ids) for group in groups for item in group) / max(time.perf_counter() - started, 1e-9),
            },
        )

    def replay_layer(self, layer: int, *, root: Path, split: str, dataset_hash: str, tokenizer_hash: str, source_revision: str, selected_layer: bool, records_by_shard: Mapping[int, Sequence[TokenizedExample]], progress_path: Path, shard_tokens: int = DEFAULT_SHARD_TOKENS, expected_count: int | None = None) -> dict[str, Any]:
        import torch  # type: ignore

        input_stage = layer
        output_stage = layer + 1
        existing = _valid_stage_manifest(_stage_manifest_path(root, output_stage))
        if existing is not None and existing.get("dataset_hash") == dataset_hash and existing.get("split") == split and (expected_count is None or int(existing.get("count", -1)) == expected_count):
            if not selected_layer:
                return existing
            capture_raw = existing.get("capture_manifest")
            capture_path = Path(str(capture_raw)) if capture_raw else root.parent.parent / f"layer-{layer:04d}-{split}.json"
            capture = _valid_capture_manifest(capture_path, dataset_hash=dataset_hash, split=split, layer=layer, expected_count=expected_count)
            if capture is not None:
                return existing
            raise TeacherCaptureBlocked(
                "STREAM_CAPTURE_MANIFEST_MISSING",
                f"layer {layer} hidden stage is complete but its selected activation manifest is missing or invalid",
                layer=layer,
            )
        input_manifest = _valid_stage_manifest(_stage_manifest_path(root, input_stage))
        if input_manifest is None:
            raise TeacherCaptureBlocked("STREAM_INPUT_STAGE_MISSING", f"layer {layer} has no validated input stage")
        loaded_at = time.perf_counter()
        layer_module, layer_receipt = self.load_layer(layer, self.device)
        load_seconds = time.perf_counter() - loaded_at
        output_shards: list[dict[str, Any]] = []
        capture_shards: list[dict[str, Any]] = []
        completed: set[int] = set()
        resume_ids = _load_progress(progress_path, layer=layer, split=split, dataset_hash=dataset_hash, source_revision=source_revision)
        output_dir = _stage_root(root, output_stage)
        capture_dir = root / f"layer-{layer:04d}"
        # A crash can leave published output shards without a stage manifest.
        # Reuse only files named by a matching progress receipt and re-hash
        # them before skipping computation.
        for shard_meta in input_manifest["shards"]:
            shard_id = int(shard_meta["shard_id"])
            if shard_id not in resume_ids:
                continue
            output_path = output_dir / f"shard-{shard_id:05d}.safetensors"
            try:
                output_tensors = _load_tensors(output_path, device="cpu")
                if set(output_tensors) != {"hidden_states", "input_ids", "attention_mask", "lengths", "source_record_index"}:
                    continue
                if int(output_tensors["hidden_states"].shape[0]) != int(shard_meta["shape"][0]):
                    continue
                output_meta = {
                    "shard_id": shard_id,
                    "path": output_path.name,
                    "sha256": _sha256(output_path),
                    "count": int(shard_meta["count"]),
                    "example_count": int(shard_meta.get("example_count", len(records_by_shard[shard_id]))),
                    "bytes": output_path.stat().st_size,
                    "shape": list(output_tensors["hidden_states"].shape),
                    "dtype": _dtype_name(output_tensors["hidden_states"].dtype),
                    "records": shard_meta.get("records", []),
                    "dataset_hash": dataset_hash,
                    "source_revision": source_revision,
                    "stage": output_stage,
                }
                if selected_layer:
                    capture_path = capture_dir / f"shard-{shard_id:05d}.safetensors"
                    capture_tensors = _load_tensors(capture_path, device="cpu")
                    if "mlp_input" not in capture_tensors or int(capture_tensors["mlp_input"].shape[0]) != int(shard_meta["count"]):
                        continue
                    capture_shards.append({
                        "shard_id": shard_id,
                        "path": os.path.relpath(capture_path, root.parent.parent).replace("\\", "/"),
                        "sha256": _sha256(capture_path),
                        "count": int(capture_tensors["mlp_input"].shape[0]),
                        "bytes": capture_path.stat().st_size,
                        "shape": list(capture_tensors["mlp_input"].shape),
                        "dtype": _dtype_name(capture_tensors["mlp_input"].dtype),
                        "records": shard_meta.get("records", []),
                    })
                output_shards.append(output_meta)
                completed.add(shard_id)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                continue
        compute_start = time.perf_counter()
        read_seconds = 0.0
        write_seconds = 0.0
        hidden_bytes_read = 0
        hidden_bytes_written = 0
        activation_bytes_written = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for shard_meta in input_manifest["shards"]:
                shard_id = int(shard_meta["shard_id"])
                if shard_id in completed:
                    continue
                read_started = time.perf_counter()
                tensors = _load_tensors(_stage_root(root, input_stage) / str(shard_meta["path"]), device="cpu")
                read_seconds += time.perf_counter() - read_started
                hidden_bytes_read += int((_stage_root(root, input_stage) / str(shard_meta["path"])).stat().st_size)
                hidden = tensors["hidden_states"].to(device=self.device, dtype=self.compute_dtype)
                attention_mask = tensors["attention_mask"].to(device=self.device)
                max_len = int(hidden.shape[1])
                position_ids = torch.arange(max_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(hidden.shape[0], -1)
                output, mlp_input = _run_native_layer(self, layer_module, hidden, attention_mask, position_ids, capture_mlp_input=selected_layer)
                output_cpu = output.detach().to("cpu")
                out_tensors = {
                    "hidden_states": output_cpu,
                    "input_ids": tensors["input_ids"],
                    "attention_mask": tensors["attention_mask"],
                    "lengths": tensors["lengths"],
                    "source_record_index": tensors["source_record_index"],
                }
                group = list(records_by_shard[shard_id])
                write_started = time.perf_counter()
                output_meta = _write_hidden_shard(root, output_stage, shard_id, out_tensors, group, split=split, dataset_hash=dataset_hash, source_revision=source_revision)
                write_seconds += time.perf_counter() - write_started
                hidden_bytes_written += int(output_meta.get("bytes", 0))
                output_shards.append(output_meta)
                if selected_layer and mlp_input is not None:
                    lengths = [int(value) for value in tensors["lengths"].reshape(-1).tolist()]
                    valid_rows = torch.cat([mlp_input[row, :length] for row, length in enumerate(lengths)], dim=0).to("cpu")
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    capture_path = capture_dir / f"shard-{shard_id:05d}.safetensors"
                    capture_base = root.parent.parent
                    write_started = time.perf_counter()
                    capture_digest = _atomic_save_tensors({"mlp_input": valid_rows}, capture_path)
                    write_seconds += time.perf_counter() - write_started
                    activation_bytes_written += capture_path.stat().st_size
                    capture_shards.append({
                        "shard_id": shard_id,
                        "path": os.path.relpath(str(capture_path), str(capture_base)).replace("\\", "/"),
                        "sha256": capture_digest,
                        "count": int(valid_rows.shape[0]),
                        "bytes": capture_path.stat().st_size,
                        "shape": list(valid_rows.shape),
                        "dtype": _dtype_name(valid_rows.dtype),
                        "records": shard_meta.get("records", []),
                    })
                    del valid_rows
                completed.add(shard_id)
                atomic_write_json(progress_path, {
                    "schema_version": STREAMING_SCHEMA_VERSION,
                    "status": "IN_PROGRESS",
                    "split": split,
                    "layer": layer,
                    "input_stage": input_stage,
                    "next_stage": output_stage,
                    "completed_shard_ids": sorted(completed),
                    "output_shard_hashes": {str(item["shard_id"]): item["sha256"] for item in output_shards},
                    "capture_shard_hashes": {str(item["shard_id"]): item["sha256"] for item in capture_shards},
                    "dataset_hash": dataset_hash,
                    "source_revision": source_revision,
                    "code_commit": current_git_commit(),
                })
                del hidden, attention_mask, position_ids, output, output_cpu, out_tensors, tensors
        finally:
            del layer_module
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        output_shards.sort(key=lambda item: int(item["shard_id"]))
        capture_shards.sort(key=lambda item: int(item["shard_id"]))
        if selected_layer:
            capture_manifest = {
                "schema_version": 2,
                "status": "CAPTURE_COMPLETE",
                "capture_kind": "streaming_teacher_mlp_input",
                "diagnostic_only": False,
                "quality_gate_eligible": True,
                "layer": layer,
                "split": split,
                "dtype": "bfloat16",
                "count": sum(int(item["count"]) for item in capture_shards),
                "shard_tokens": shard_tokens,
                "shards": capture_shards,
                "dataset_hash": dataset_hash,
                "source_revision": source_revision,
                "source_snapshot": str(self.source.root),
                "tokenizer_hash": tokenizer_hash,
                "input_stage": input_stage,
                "output_stage": output_stage,
                "layer_receipt": layer_receipt,
                "code_commit": current_git_commit(),
            }
            capture_base = root.parent.parent
            capture_path = capture_base / f"layer-{layer:04d}-{split}.json"
            atomic_write_json(capture_path, capture_manifest)
        stage_manifest = _write_stage_manifest(
            root,
            output_stage,
            output_shards,
            split=split,
            dataset_hash=dataset_hash,
            source_revision=source_revision,
            tokenizer_hash=tokenizer_hash,
            code_commit=current_git_commit(),
            stage_metadata={
                "kind": "native_qwen3_5_decoder_layer",
                "layer": layer,
                "layer_receipt": layer_receipt,
                "load_seconds": load_seconds,
                "compute_seconds": time.perf_counter() - compute_start,
                "read_seconds": read_seconds,
                "write_seconds": write_seconds,
                "hidden_bytes_read": hidden_bytes_read,
                "hidden_bytes_written": hidden_bytes_written,
                "activation_bytes_written": activation_bytes_written,
                "source_weight_bytes": int(layer_receipt.get("source_bytes", 0)),
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
                "tokens_per_second": sum(int(item.get("count", 0)) for item in output_shards) / max(time.perf_counter() - compute_start, 1e-9),
            },
        )
        if selected_layer:
            stage_manifest["capture_manifest"] = str(capture_path)
            atomic_write_json(_stage_manifest_path(root, output_stage), stage_manifest)
        atomic_write_json(progress_path, {
            "schema_version": STREAMING_SCHEMA_VERSION,
            "status": "LAYER_COMPLETE",
            "split": split,
            "layer": layer,
            "input_stage": input_stage,
            "next_stage": output_stage,
            "completed_shard_ids": sorted(completed),
            "dataset_hash": dataset_hash,
            "source_revision": source_revision,
            "layer_receipt": layer_receipt,
            "code_commit": current_git_commit(),
        })
        # The next stage has been fully hashed and published; the old stage is
        # now disposable.  This is the A/B rolling disk bound.
        old_stage = _stage_root(root, input_stage)
        if old_stage != _stage_root(root, output_stage) and old_stage.exists():
            shutil.rmtree(old_stage)
        return stage_manifest


def _load_fixed_examples(dataset_manifest: str | Path, source_snapshot: str | Path, split: str, *, max_examples: int | None = None) -> tuple[list[TokenizedExample], dict[str, Any]]:
    manifest_path = Path(dataset_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = resolve_corpus_records(manifest_path, split)
    tokenizer = load_pinned_tokenizer(source_snapshot, str(payload.get("tokenizer_revision") or payload.get("source_revision") or ""))
    examples = tokenize_corpus_records(records, tokenizer, split=split, sequence_length=int(payload.get("sequence_length", 2048)))
    if max_examples is not None:
        examples = examples[:max_examples]
    tokenizer_hashes = snapshot_tokenizer_hashes(source_snapshot)
    metadata = fixed_split_metadata(records, split=split, tokenizer_revision=str(payload.get("tokenizer_revision", "")), tokenizer_hashes=tokenizer_hashes)
    metadata["dataset_hash"] = str(payload.get("dataset_hash", ""))
    metadata["source_revision"] = str(payload.get("source_revision", ""))
    metadata["tokenizer_hash"] = _json_hash(tokenizer_hashes)
    metadata["example_count"] = len(examples)
    metadata["token_count"] = sum(len(item.input_ids) for item in examples)
    return examples, metadata


def stream_teacher_split(source_snapshot: str | Path, dataset_manifest: str | Path, run_dir: str | Path, *, split: str, layers: Sequence[int] = REPRESENTATIVE_LAYERS, device: str = DEFAULT_DEVICE, layer_devices: Mapping[int, str] | None = None, compute_dtype: str = "bfloat16", shard_tokens: int = DEFAULT_SHARD_TOKENS, max_examples: int | None = None, attention_implementation: str = "sdpa") -> dict[str, Any]:
    """Replay one fixed split layer-major and publish resumable evidence."""

    run = Path(run_dir)
    root = run / "capture" / "streaming" / split
    root.mkdir(parents=True, exist_ok=True)
    examples, metadata = _load_fixed_examples(dataset_manifest, source_snapshot, split, max_examples=max_examples)
    if not examples:
        raise TeacherCaptureBlocked("STREAM_EMPTY_SPLIT", f"no tokenized examples were found for {split}")
    selected = sorted({int(layer) for layer in layers})
    if not selected or min(selected) < 0:
        raise ValueError("layers must contain at least one non-negative layer")
    max_layer = max(selected)
    expected_count = sum(len(item.input_ids) for item in examples)

    def _complete_capture(layer: int) -> dict[str, Any] | None:
        return _valid_capture_manifest(
            root.parent.parent / f"layer-{layer:04d}-{split}.json",
            dataset_hash=str(metadata["dataset_hash"]),
            split=split,
            layer=layer,
            expected_count=expected_count,
        )

    final_stage = _valid_stage_manifest(_stage_manifest_path(root, max_layer + 1))
    if final_stage is not None and int(final_stage.get("count", -1)) == expected_count and all(_complete_capture(layer) is not None for layer in selected):
        previous = run / "metrics" / f"streaming-{split}.json"
        if previous.is_file():
            try:
                cached = json.loads(previous.read_text(encoding="utf-8"))
                if cached.get("status") == "STREAMING_CAPTURE_COMPLETE" and cached.get("dataset_hash") == metadata["dataset_hash"] and int(cached.get("tokens", -1)) == expected_count:
                    return cached
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return {
            "status": "STREAMING_CAPTURE_COMPLETE",
            "schema_version": STREAMING_SCHEMA_VERSION,
            "split": split,
            "layers": selected,
            "replayed_layers": max_layer + 1,
            "examples": len(examples),
            "tokens": expected_count,
            "shard_tokens": shard_tokens,
            "dataset_hash": metadata["dataset_hash"],
            "tokenizer_hash": metadata["tokenizer_hash"],
            "source_revision": metadata["source_revision"],
            "source_snapshot": str(Path(source_snapshot)),
            "device": device,
            "compute_dtype": compute_dtype,
            "attention_implementation": attention_implementation,
            "final_stage": final_stage,
            "code_commit": current_git_commit(),
            "resumed_complete": True,
        }

    current_stage = 0
    for stage in range(max_layer, -1, -1):
        candidate = _valid_stage_manifest(_stage_manifest_path(root, stage))
        if candidate is not None and int(candidate.get("count", -1)) == expected_count:
            current_stage = stage
            break
    # A selected capture is durable evidence, not merely an optional sidecar.
    # If its layer has already been retired from the rolling stages, a missing
    # sidecar cannot be regenerated without changing the corpus computation.
    for layer in selected:
        if layer < current_stage and _complete_capture(layer) is None:
            raise TeacherCaptureBlocked(
                "STREAM_CAPTURE_MANIFEST_MISSING",
                f"selected layer {layer} is behind the current rolling stage but has no validated activation manifest",
                layer=layer,
            )
    executor = StreamingTeacherExecutor(source_snapshot, device=device, compute_dtype=compute_dtype, attention_implementation=attention_implementation)
    groups = _group_examples(examples, shard_tokens)
    pad_token_id = int(getattr(executor.text_config, "pad_token_id", None) or 0)
    stage0 = (
        executor.materialize_embeddings(groups, root=root, split=split, dataset_hash=str(metadata["dataset_hash"]), tokenizer_hash=str(metadata["tokenizer_hash"]), source_revision=str(metadata["source_revision"]), pad_token_id=pad_token_id)
        if current_stage == 0
        else {"status": "STAGE_REUSED", "stage": 0, "resume_from_stage": current_stage, "count": expected_count}
    )
    records_by_shard = {index: group for index, group in enumerate(groups)}
    layer_reports: list[dict[str, Any]] = []
    for layer in range(current_stage, max_layer + 1):
        if layer_devices and layer in layer_devices:
            executor.set_device(layer_devices[layer])
        report = executor.replay_layer(
            layer,
            root=root,
            split=split,
            dataset_hash=str(metadata["dataset_hash"]),
            tokenizer_hash=str(metadata["tokenizer_hash"]),
            source_revision=str(metadata["source_revision"]),
            selected_layer=layer in selected,
            records_by_shard=records_by_shard,
            progress_path=root / "progress.json",
            shard_tokens=shard_tokens,
            expected_count=expected_count,
        )
        layer_reports.append({"layer": layer, "stage": layer + 1, "capture_manifest": report.get("capture_manifest"), "metadata": report.get("metadata", {})})
    final_stage = _valid_stage_manifest(_stage_manifest_path(root, max_layer + 1))
    result = {
        "status": "STREAMING_CAPTURE_COMPLETE",
        "schema_version": STREAMING_SCHEMA_VERSION,
        "split": split,
        "layers": selected,
        "replayed_layers": max_layer + 1,
        "examples": len(examples),
        "tokens": expected_count,
        "shard_tokens": shard_tokens,
        "dataset_hash": metadata["dataset_hash"],
        "tokenizer_hash": metadata["tokenizer_hash"],
        "source_revision": metadata["source_revision"],
        "source_snapshot": str(Path(source_snapshot)),
        "device": device,
        "layer_devices": {str(key): value for key, value in (layer_devices or {}).items()},
        "compute_dtype": compute_dtype,
        "attention_implementation": attention_implementation,
        "stage_zero": stage0,
        "layer_reports": layer_reports,
        "final_stage": final_stage,
        "code_commit": current_git_commit(),
    }
    atomic_write_json(run / "metrics" / f"streaming-{split}.json", result)
    return result


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_SHARD_TOKENS",
    "REPRESENTATIVE_LAYERS",
    "IndexedSource",
    "SourceTensorReceipt",
    "StreamingTeacherExecutor",
    "TargetedLayerLoader",
    "stream_teacher_split",
]
