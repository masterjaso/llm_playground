"""Native text-to-teacher activation capture.

This module is deliberately conservative about model loading.  It only opens
an already materialized local snapshot, always disables remote code, and
returns a structured blocker when the tokenizer, model runtime, or source
records cannot be verified.  In particular, a missing 27B snapshot never
causes synthetic activations to be emitted.

Only the tensor entering a verified dense MLP is persisted.  Teacher MLP
outputs are held briefly for the hook-equivalence check and are never written
to an activation shard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None  # type: ignore[assignment]

from ..provenance import current_git_commit
from .activations import capture_activation_shards

PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
REPRESENTATIVE_LAYERS = (0, 16, 32, 48, 63)
_MLP_PATH_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp$")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
)


class TeacherCaptureBlocked(RuntimeError):
    """A truthful, resumable external/resource or validation blocker."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "BLOCKED",
            "blocker_code": self.code,
            "message": self.message,
            **self.details,
        }


@dataclass(frozen=True)
class HookVerification:
    """Evidence that a discovered module is the dense SwiGLU MLP target."""

    status: str
    layer: int
    module_path: str
    normalized_mse: float | None
    cosine: float | None
    threshold: float
    formula: str
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizedExample:
    """One fixed corpus example or sequence-length chunk."""

    example_id: str
    split: str
    input_ids: tuple[int, ...]
    source_record_index: int
    chunk_index: int
    domain: str
    content_sha256: str


def is_pinned_source_revision(revision: str | None) -> bool:
    """Return whether ``revision`` is an immutable 40-character commit SHA."""

    return bool(revision and PINNED_REVISION_RE.fullmatch(str(revision).strip()))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def snapshot_tokenizer_hashes(source_snapshot: str | Path) -> dict[str, str]:
    """Hash tokenizer files from a local source snapshot.

    Missing optional tokenizer files are omitted.  The caller decides whether
    an empty inventory is acceptable after the tokenizer has loaded.
    """

    root = Path(source_snapshot)
    hashes: dict[str, str] = {}
    for name in _TOKENIZER_FILES:
        path = root / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    return hashes


def load_pinned_tokenizer(source_snapshot: str | Path, revision: str, *, use_fast: bool = True) -> Any:
    """Load a tokenizer from an existing pinned snapshot only.

    ``local_files_only=True`` and ``trust_remote_code=False`` are intentional
    hard requirements.  The function raises :class:`TeacherCaptureBlocked`
    rather than falling back to a different tokenizer or a whitespace count.
    """

    snapshot = Path(source_snapshot)
    if not snapshot.is_dir():
        raise TeacherCaptureBlocked(
            "SOURCE_SNAPSHOT_MISSING",
            f"pinned teacher snapshot is unavailable: {snapshot}",
            source_snapshot=str(snapshot),
            source_revision=revision,
        )
    if not is_pinned_source_revision(revision):
        raise TeacherCaptureBlocked(
            "SOURCE_REVISION_UNPINNED",
            "native teacher capture requires a 40-character immutable source revision",
            source_snapshot=str(snapshot),
            source_revision=revision,
        )
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise TeacherCaptureBlocked(
            "TOKENIZER_RUNTIME_UNAVAILABLE",
            "Transformers is not installed; no tokenizer or activation was created",
            source_snapshot=str(snapshot),
            source_revision=revision,
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(snapshot),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=use_fast,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TeacherCaptureBlocked(
            "TOKENIZER_LOAD_FAILED",
            f"pinned local tokenizer could not be loaded: {exc}",
            source_snapshot=str(snapshot),
            source_revision=revision,
        ) from exc


def _torch_dtype(torch: Any, value: str | None) -> Any:
    if value is None or value.lower() in {"", "auto", "none"}:
        return None
    normalized = value.lower().replace("-", "")
    aliases = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "fp32": "float32"}
    name = aliases.get(normalized)
    if name is None or not hasattr(torch, name):
        raise TeacherCaptureBlocked("TEACHER_DTYPE_UNSUPPORTED", f"unsupported teacher dtype: {value}")
    return getattr(torch, name)


def _available_host_memory() -> int | None:
    """Best-effort available RAM for a safe CPU teacher-load preflight."""

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        return None


def _snapshot_weight_bytes(snapshot: Path) -> int | None:
    """Sum unique safetensor files referenced by the local index."""

    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = set(payload.get("weight_map", {}).values())
        return sum((snapshot / str(name)).stat().st_size for name in names)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_native_teacher(
    source_snapshot: str | Path,
    revision: str,
    *,
    device_map: str | Mapping[str, Any] = "auto",
    device: str | None = None,
    compute_dtype: str | None = "bfloat16",
    max_memory: Mapping[Any, Any] | None = None,
    offload_folder: str | Path | None = None,
) -> Any:
    """Load the native Transformers teacher without network or remote code."""

    snapshot = Path(source_snapshot)
    if not snapshot.is_dir():
        raise TeacherCaptureBlocked(
            "SOURCE_SNAPSHOT_MISSING",
            f"pinned teacher snapshot is unavailable: {snapshot}",
            source_snapshot=str(snapshot),
            source_revision=revision,
        )
    if not is_pinned_source_revision(revision):
        raise TeacherCaptureBlocked(
            "SOURCE_REVISION_UNPINNED",
            "native teacher capture requires a 40-character immutable source revision",
            source_snapshot=str(snapshot),
            source_revision=revision,
        )
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:
        raise TeacherCaptureBlocked(
            "TEACHER_RUNTIME_UNAVAILABLE",
            "PyTorch and Transformers are required for native teacher capture; no activation was created",
            source_snapshot=str(snapshot),
            source_revision=revision,
        ) from exc
    estimated_bytes = _snapshot_weight_bytes(snapshot)
    # The bundled runtime in this workspace is CPU-only.  Loading a 27B
    # safetensor snapshot into CPU memory when its files exceed available RAM
    # would turn a resumable scientific blocker into an OOM kill.  Refuse
    # before model construction and preserve the exact resource evidence.
    cpu_only = not bool(torch.cuda.is_available()) and (
        isinstance(device_map, Mapping) or str(device_map).lower() in {"", "auto", "cpu", "none"}
    )
    available_bytes = _available_host_memory() if cpu_only else None
    if cpu_only and estimated_bytes is not None and available_bytes is not None and estimated_bytes > int(available_bytes * 0.90):
        raise TeacherCaptureBlocked(
            "TEACHER_RESOURCE_INSUFFICIENT",
            "native teacher snapshot exceeds safe available host memory for the installed CPU-only runtime",
            source_snapshot=str(snapshot),
            source_revision=revision,
            estimated_weight_bytes=estimated_bytes,
            available_host_memory_bytes=available_bytes,
            device_map=device_map,
            remediation="install a CUDA-enabled PyTorch build or provide a multi-device/offload runtime, then resume capture",
        )
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    dtype = _torch_dtype(torch, compute_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if isinstance(device_map, Mapping):
        kwargs["device_map"] = dict(device_map)
    elif device_map and device_map.lower() not in {"none", "cpu"}:
        kwargs["device_map"] = device_map
    if max_memory:
        kwargs["max_memory"] = dict(max_memory)
    if offload_folder is not None:
        kwargs["offload_folder"] = str(offload_folder)
    # low_cpu_mem_usage is useful for the 27B snapshot but requires accelerate.
    try:
        import accelerate  # type: ignore  # noqa: F401

        kwargs["low_cpu_mem_usage"] = True
    except ImportError:
        pass
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
        except (OSError, RuntimeError, TypeError, ValueError) as causal_exc:
            # Qwen3.5 snapshots may advertise the multimodal conditional-
            # generation class rather than a plain causal-LM class.  Keep the
            # same pinned/local/no-remote-code kwargs for the fallback.
            try:
                from transformers import AutoModelForConditionalGeneration  # type: ignore

                model = AutoModelForConditionalGeneration.from_pretrained(str(snapshot), **kwargs)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                raise causal_exc
        loaded_model: Any = model
        loaded_model.eval()
        if device and not kwargs.get("device_map"):
            loaded_model.to(device)
        return loaded_model
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TeacherCaptureBlocked(
            "TEACHER_LOAD_FAILED",
            f"native teacher could not be loaded from the pinned snapshot: {exc}",
            source_snapshot=str(snapshot),
            source_revision=revision,
            device_map=device_map,
        ) from exc


def _iter_named_modules(model: Any) -> list[tuple[str, Any]]:
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        return [(str(name), module) for name, module in named_modules()]
    raise TeacherCaptureBlocked(
        "TEACHER_MODULES_UNAVAILABLE",
        "loaded teacher does not expose named modules for explicit MLP hook discovery",
    )


def _discover_mlp_records(model: Any, expected_layer_count: int | None = None) -> dict[int, tuple[str, Any]]:
    records: dict[int, tuple[str, Any]] = {}
    for name, module in _iter_named_modules(model):
        match = _MLP_PATH_RE.search(name)
        if match is None:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            continue
        layer = int(match.group(1))
        if layer in records and records[layer][0] != name:
            raise TeacherCaptureBlocked(
                "MLP_HOOK_AMBIGUOUS",
                f"multiple verified MLP modules were found for layer {layer}",
                layer=layer,
                paths=[records[layer][0], name],
            )
        records[layer] = (name, module)
    if not records:
        raise TeacherCaptureBlocked(
            "MLP_HOOK_NOT_FOUND",
            "no module path matching layers[N].mlp with gate_proj/up_proj/down_proj was found",
        )
    if expected_layer_count is not None:
        expected = set(range(expected_layer_count))
        observed = set(records)
        if observed != expected:
            raise TeacherCaptureBlocked(
                "MLP_LAYER_SET_MISMATCH",
                "verified MLP module set is not the expected contiguous transformer layer set",
                expected_layers=sorted(expected),
                observed_layers=sorted(observed),
            )
    return records


def discover_mlp_modules(model: Any, expected_layer_count: int | None = None) -> dict[int, Any]:
    """Discover verified MLP modules by their actual named-module paths."""

    return {layer: module for layer, (_path, module) in _discover_mlp_records(model, expected_layer_count).items()}


def discover_mlp_paths(model: Any, expected_layer_count: int | None = None) -> dict[int, str]:
    """Return the exact paths used for hook registration."""

    return {layer: path for layer, (path, _module) in _discover_mlp_records(model, expected_layer_count).items()}


def _model_layer_count(model: Any) -> int | None:
    config = getattr(model, "config", None)
    direct = config.get("num_hidden_layers") if isinstance(config, Mapping) else getattr(config, "num_hidden_layers", None)
    if direct is not None:
        return int(direct)
    nested = config.get("text_config") if isinstance(config, Mapping) else getattr(config, "text_config", None)
    value = nested.get("num_hidden_layers") if isinstance(nested, Mapping) else getattr(nested, "num_hidden_layers", None)
    return int(value) if value is not None else None


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("module returned an empty tuple/list")
        return value[0]
    return value


def _manual_swiglu(module: Any, inputs: Any) -> Any:
    try:
        from torch.nn import functional  # type: ignore
    except ImportError as exc:
        raise TeacherCaptureBlocked("TEACHER_RUNTIME_UNAVAILABLE", "PyTorch is required for MLP verification") from exc
    gate = module.gate_proj(inputs)
    up = module.up_proj(inputs)
    activation = getattr(module, "act_fn", None)
    activated = activation(gate) if callable(activation) else functional.silu(gate)
    return module.down_proj(activated * up)


def verify_mlp_reconstruction(
    module: Any,
    mlp_input: Any,
    teacher_output: Any,
    *,
    layer: int = 0,
    module_path: str = "",
    threshold: float = 1e-7,
) -> HookVerification:
    """Compare explicit gate/up/down reconstruction with the teacher MLP."""

    formula = "down_proj(silu(gate_proj(x)) * up_proj(x))"
    try:
        import torch  # type: ignore

        predicted = _manual_swiglu(module, mlp_input)
        target = _first_tensor(teacher_output)
        predicted_f = predicted.float()
        target_f = target.float()
        if tuple(predicted_f.shape) != tuple(target_f.shape):
            raise ValueError(f"reconstruction shape {tuple(predicted_f.shape)} != teacher shape {tuple(target_f.shape)}")
        difference = predicted_f - target_f
        denominator = torch.mean(target_f * target_f).clamp_min(torch.finfo(torch.float32).eps)
        normalized_mse = float(torch.mean(difference * difference).div(denominator).detach().cpu())
        flattened_predicted = predicted_f.reshape(-1)
        flattened_target = target_f.reshape(-1)
        cosine = float(
            torch.dot(flattened_predicted, flattened_target)
            / (torch.linalg.vector_norm(flattened_predicted) * torch.linalg.vector_norm(flattened_target)).clamp_min(torch.finfo(torch.float32).eps)
        )
        status = "HOOK_VERIFIED" if normalized_mse <= threshold else "HOOK_BLOCKED"
        return HookVerification(status, layer, module_path, normalized_mse, cosine, threshold, formula, None if status == "HOOK_VERIFIED" else "MLP reconstruction exceeds the numerical equivalence gate")
    except TeacherCaptureBlocked:
        raise
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        return HookVerification("HOOK_BLOCKED", layer, module_path, None, None, threshold, formula, str(exc))


def validate_mlp_hook(*args: Any, **kwargs: Any) -> HookVerification:
    """Compatibility alias for callers that use the validation terminology."""

    return verify_mlp_reconstruction(*args, **kwargs)


def _read_source_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise TeacherCaptureBlocked("CORPUS_SOURCE_MISSING", f"corpus source record file is missing: {path}", source_file=str(path))
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        record = dict(value)
                        record["__dense2moe_source_record_index"] = line_number
                        records.append(record)
            return records
        if path.suffix.lower() == ".tsv":
            lines = path.read_text(encoding="utf-8").splitlines()
            headers = lines[0].split("\t") if lines else []
            return [{**dict(zip(headers, row.split("\t"))), "__dense2moe_source_record_index": row_index} for row_index, row in enumerate(lines[1:])]
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TeacherCaptureBlocked("CORPUS_SOURCE_INVALID", f"unable to read corpus source records: {exc}", source_file=str(path)) from exc
    if isinstance(value, Mapping):
        value = value.get("examples", value.get("records", []))
    if not isinstance(value, list):
        raise TeacherCaptureBlocked("CORPUS_SOURCE_INVALID", "corpus source must contain a list of records", source_file=str(path))
    return [{**dict(item), "__dense2moe_source_record_index": index} for index, item in enumerate(value) if isinstance(item, Mapping)]


def _record_text(record: Mapping[str, Any]) -> str:
    for key in ("text", "content", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        answer = record.get("answer")
        return f"{question}\n{answer}" if isinstance(answer, str) and answer else question
    messages = record.get("messages", record.get("conversations"))
    if isinstance(messages, list):
        chunks: list[str] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role", message.get("from", ""))
                content = message.get("content", message.get("value", ""))
                if content:
                    chunks.append(f"{role}: {content}" if role else str(content))
            elif message:
                chunks.append(str(message))
        if chunks:
            return "\n".join(chunks)
    return ""


def _normalized_content_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = " ".join(normalized.split())
    return _sha256_text(normalized)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_corpus_records(dataset_manifest: str | Path, split: str) -> list[dict[str, Any]]:
    """Reopen and hash-check the immutable source records for one split."""

    manifest_path = Path(dataset_manifest)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TeacherCaptureBlocked("DATASET_MANIFEST_INVALID", f"cannot read dataset manifest: {exc}", manifest=str(manifest_path)) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get(split), list):
        raise TeacherCaptureBlocked("DATASET_SPLIT_MISSING", f"dataset manifest has no fixed {split} split", split=split)
    source = payload.get("source")
    recorded_base = payload.get("resolvability", {}).get("base_dir") if isinstance(payload.get("resolvability"), Mapping) else None

    def resolve_locator(raw: str | Path) -> Path:
        """Resolve a manifest locator from either the run or acquisition root.

        ``prepare-data`` records the source path relative to the acquisition
        manifest (for example ``data/public_v2/corpus.jsonl``), while the
        resulting run manifest lives under ``runs/.../capture``.  Try the
        manifest directory first for self-contained artifacts, then the
        recorded acquisition root and current workspace so a run can reopen
        an intentionally external, hash-verified corpus.
        """

        candidate = Path(str(raw))
        if candidate.is_absolute():
            return candidate
        roots = [manifest_path.parent]
        if recorded_base:
            roots.append(Path(str(recorded_base)))
        roots.append(Path.cwd())
        for root in roots:
            resolved = root / candidate
            if resolved.exists():
                return resolved
        return roots[0] / candidate

    source_path = None
    if isinstance(source, Mapping) and source.get("path"):
        source_path = resolve_locator(str(source["path"]))
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    source_hash_cache: dict[Path, str] = {}
    resolved: list[dict[str, Any]] = []
    for ordinal, item_raw in enumerate(payload[split]):
        if not isinstance(item_raw, Mapping):
            raise TeacherCaptureBlocked("DATASET_RECORD_INVALID", f"fixed {split} record {ordinal} is not an object", split=split)
        item = dict(item_raw)
        expected_hash = str(item.get("content_sha256", item.get("text_sha256", "")))
        candidate_path = source_path
        if item.get("source_file"):
            candidate_path = resolve_locator(str(item["source_file"]))
        text = _record_text(item)
        source_index_raw = item.get("source_record_index", item.get("record_index"))
        source_index: int | None = int(source_index_raw) if source_index_raw is not None else None
        source_record: Mapping[str, Any] | None = None
        if candidate_path is not None and candidate_path not in source_cache:
            source_cache[candidate_path] = _read_source_records(candidate_path)
        records = source_cache.get(candidate_path, []) if candidate_path is not None else []
        if source_index is not None:
            indexed = next((record for record in records if int(record.get("__dense2moe_source_record_index", -1)) == source_index), None)
            if indexed is None and 0 <= source_index < len(records):
                indexed = records[source_index]
            if indexed is None:
                raise TeacherCaptureBlocked("CORPUS_RECORD_UNRESOLVABLE", f"source record index is out of range for {candidate_path}", split=split, example_id=item.get("id"), source_record_index=source_index)
            source_record = indexed
            expected_record_id = item.get("source_record_id", item.get("record_id"))
            actual_record_id = source_record.get("source_record_id", source_record.get("record_id", source_record.get("id")))
            if expected_record_id is not None and actual_record_id is not None and str(expected_record_id) != str(actual_record_id):
                raise TeacherCaptureBlocked("CORPUS_RECORD_UNRESOLVABLE", "fixed source_record_id does not match the reopened source record", split=split, example_id=item.get("id"), source_record_id=expected_record_id)
        elif not text and expected_hash:
            matches = [(index, record) for index, record in enumerate(records) if _sha256_text(_record_text(record)) == expected_hash]
            if len(matches) != 1:
                raise TeacherCaptureBlocked("CORPUS_RECORD_UNRESOLVABLE", f"content hash does not resolve uniquely in {candidate_path}", split=split, example_id=item.get("id"), matches=len(matches))
            source_index, source_record = matches[0]
        if source_record is not None:
            text = _record_text(source_record)
        actual_hash = _sha256_text(text)
        if not text.strip() or not expected_hash or actual_hash != expected_hash:
            raise TeacherCaptureBlocked("CORPUS_HASH_MISMATCH", f"fixed {split} record text failed its content hash check", split=split, example_id=item.get("id"), expected_hash=expected_hash, actual_hash=actual_hash)
        expected_normalized = str(item.get("normalized_content_sha256", ""))
        if expected_normalized and _normalized_content_hash(text) != expected_normalized:
            raise TeacherCaptureBlocked("CORPUS_HASH_MISMATCH", f"fixed {split} record failed its normalized content hash check", split=split, example_id=item.get("id"))
        expected_source_file_hash = str(item.get("source_file_sha256", ""))
        if expected_source_file_hash and candidate_path is not None:
            if candidate_path not in source_hash_cache:
                source_hash_cache[candidate_path] = _sha256_path(candidate_path)
            if source_hash_cache[candidate_path] != expected_source_file_hash:
                raise TeacherCaptureBlocked("CORPUS_SOURCE_HASH_MISMATCH", "reopened source file does not match the manifest hash", split=split, example_id=item.get("id"), source_file=str(candidate_path))
        resolved.append({**item, "text": text, "content_sha256": actual_hash, "source_file": str(candidate_path) if candidate_path is not None else None, "source_record_index": source_index, "domain": str(item.get("domain", "general"))})
    return resolved


def _token_ids(encoded: Any) -> list[int]:
    value = encoded["input_ids"] if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", encoded)
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise TeacherCaptureBlocked("TOKENIZER_BATCH_SHAPE_INVALID", "tokenizer returned multiple sequences for one corpus record")
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TeacherCaptureBlocked("TOKENIZER_OUTPUT_INVALID", "tokenizer did not return a one-dimensional input_ids sequence")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TeacherCaptureBlocked("TOKENIZER_OUTPUT_INVALID", "tokenizer input_ids contains a non-integer value") from exc


def tokenize_corpus_records(records: Sequence[Mapping[str, Any]], tokenizer: Any, *, split: str, sequence_length: int) -> list[TokenizedExample]:
    """Tokenize fixed records exactly and chunk without changing token order."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    result: list[TokenizedExample] = []
    for record in records:
        text = str(record["text"])
        try:
            add_special_tokens = bool(record.get("add_special_tokens", True))
            encode = getattr(tokenizer, "encode", None)
            encoded = encode(text, add_special_tokens=add_special_tokens) if callable(encode) else tokenizer(text, add_special_tokens=add_special_tokens, truncation=False, return_attention_mask=False)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise TeacherCaptureBlocked("TOKENIZER_FAILED", f"tokenizer failed for fixed example {record.get('id')}: {exc}", split=split, example_id=record.get("id")) from exc
        ids = _token_ids(encoded)
        expected_tokens = int(record.get("token_count", len(ids)))
        if expected_tokens != len(ids):
            raise TeacherCaptureBlocked("TOKEN_COUNT_MISMATCH", f"tokenizer produced {len(ids)} tokens but manifest records {expected_tokens}", split=split, example_id=record.get("id"), expected_tokens=expected_tokens, actual_tokens=len(ids))
        for chunk_index, start in enumerate(range(0, len(ids), sequence_length)):
            chunk = tuple(ids[start : start + sequence_length])
            if chunk:
                source_index = record.get("source_record_index")
                result.append(TokenizedExample(str(record.get("id", record["content_sha256"])), split, chunk, int(source_index) if source_index is not None else -1, chunk_index, str(record.get("domain", "general")), str(record["content_sha256"])))
    if not result:
        raise TeacherCaptureBlocked("TOKENIZED_SPLIT_EMPTY", f"fixed {split} split produced no tokens", split=split)
    return result


def fixed_split_metadata(records: Sequence[Mapping[str, Any]], *, split: str, tokenizer_revision: str, tokenizer_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Return immutable split identity fields copied into every shard."""

    ids = [str(record.get("id", "")) for record in records]
    contents = [str(record.get("content_sha256", record.get("text_sha256", ""))) for record in records]
    return {
        "split": split,
        "split_record_count": len(records),
        "split_ids_sha256": _sha256_bytes("\n".join(ids).encode("utf-8")),
        "split_content_sha256": _sha256_bytes("\n".join(contents).encode("utf-8")),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_file_hashes": dict(tokenizer_hashes),
        "bos_eos_handling": "per-record manifest add_special_tokens flag",
    }


def _model_input_device(model: Any, torch: Any) -> Any:
    candidate = getattr(model, "device", None)
    if candidate is not None and str(candidate) != "meta":
        return candidate
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        embedding = embeddings()
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _forward_model(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        try:
            return model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            return model(input_ids=input_ids)


def _batch_inputs(batch: Sequence[TokenizedExample], torch: Any, device: Any) -> tuple[Any, Any]:
    width = max(len(item.input_ids) for item in batch)
    input_ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    for row, item in enumerate(batch):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _capture_forward(
    model: Any,
    layers: Mapping[int, Any],
    batch: Sequence[TokenizedExample],
    *,
    collect_outputs: bool = False,
) -> tuple[dict[int, Any], dict[int, Any]]:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise TeacherCaptureBlocked("TEACHER_RUNTIME_UNAVAILABLE", "PyTorch is required for native teacher execution") from exc
    device = _model_input_device(model, torch)
    input_ids, attention_mask = _batch_inputs(batch, torch, device)
    captured: dict[int, list[Any]] = {layer: [] for layer in layers}
    outputs: dict[int, list[Any]] = {layer: [] for layer in layers}
    handles: list[Any] = []
    valid_rows = attention_mask.detach().cpu().bool()
    for layer, module in layers.items():
        def pre_hook(_module: Any, args: tuple[Any, ...], layer_id: int = layer) -> None:
            if not args:
                raise TeacherCaptureBlocked("MLP_HOOK_INPUT_MISSING", f"verified MLP layer {layer_id} received no positional input", layer=layer_id)
            captured[layer_id].append(args[0].detach())

        handles.append(module.register_forward_pre_hook(pre_hook))
        if collect_outputs:
            def output_hook(_module: Any, _args: tuple[Any, ...], output: Any, layer_id: int = layer) -> None:
                outputs[layer_id].append(_first_tensor(output).detach())

            handles.append(module.register_forward_hook(output_hook))
    try:
        with torch.inference_mode():
            _forward_model(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    filtered: dict[int, Any] = {}
    filtered_outputs: dict[int, Any] = {}
    for layer in layers:
        if len(captured[layer]) != 1:
            raise TeacherCaptureBlocked("MLP_HOOK_CALL_COUNT_INVALID", f"layer {layer} hook fired {len(captured[layer])} times for one teacher forward", layer=layer)
        value = captured[layer][0]
        if value.ndim == 3:
            value = value[valid_rows.to(value.device)]
        filtered[layer] = value
        if collect_outputs:
            if len(outputs[layer]) != 1:
                raise TeacherCaptureBlocked("MLP_OUTPUT_CALL_COUNT_INVALID", f"layer {layer} output hook fired {len(outputs[layer])} times for one teacher forward", layer=layer)
            output = outputs[layer][0]
            if output.ndim == 3:
                output = output[valid_rows.to(output.device)]
            filtered_outputs[layer] = output
    return filtered, filtered_outputs


def _batches(items: Sequence[TokenizedExample], microbatch: int) -> Iterator[Sequence[TokenizedExample]]:
    for start in range(0, len(items), microbatch):
        yield items[start : start + microbatch]


def _resource_metrics(start: float, token_count: int, model: Any) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - start, 1e-9)
    peak_rss: int | None
    if resource is not None:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    else:
        try:
            import psutil  # type: ignore

            peak_rss = int(psutil.Process().memory_info().rss)
        except ImportError:
            peak_rss = None
    result: dict[str, Any] = {
        "tokens": token_count,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "peak_host_rss_bytes": peak_rss,
    }
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            result["peak_gpu_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    del model
    return result


def _blocked_from_exception(exc: TeacherCaptureBlocked, *, manifest: str | Path, layers: Sequence[int]) -> dict[str, Any]:
    return {**exc.as_dict(), "dataset_manifest": str(manifest), "layers": list(layers), "resumable": True, "code_commit": current_git_commit()}


def capture_text_teacher_activations(
    dataset_manifest: str | Path,
    source_snapshot: str | Path,
    destination: str | Path,
    *,
    layers: Sequence[int] = REPRESENTATIVE_LAYERS,
    source_revision: str = "",
    split: str | Sequence[str] = "both",
    shard_tokens: int = 8192,
    dtype: str = "float16",
    microbatch: int = 1,
    device_map: str | Mapping[str, Any] = "auto",
    device: str | None = None,
    compute_dtype: str | None = "bfloat16",
    max_memory: Mapping[Any, Any] | None = None,
    offload_folder: str | Path | None = None,
    resume: bool = False,
    sequence_length: int | None = None,
    hook_threshold: float = 1e-7,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Capture fixed corpus text through a native teacher into split shards."""

    selected_layers = sorted({int(layer) for layer in layers})
    if not selected_layers or any(layer < 0 for layer in selected_layers):
        raise ValueError("layers must contain at least one non-negative layer")
    if microbatch <= 0 or shard_tokens <= 0:
        raise ValueError("microbatch and shard_tokens must be positive")
    if not is_pinned_source_revision(source_revision):
        return _blocked_from_exception(TeacherCaptureBlocked("SOURCE_REVISION_UNPINNED", "native teacher capture requires a 40-character immutable source revision", source_revision=source_revision), manifest=dataset_manifest, layers=selected_layers)
    if isinstance(split, str):
        splits: tuple[str, ...] = ("train", "holdout") if split == "both" else (split,)
    else:
        splits = tuple(str(item) for item in split)
    if not splits or any(item not in {"train", "holdout"} for item in splits):
        raise ValueError("split must be train, holdout, both, or a sequence of train/holdout")
    try:
        manifest_payload = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, Mapping) or manifest_payload.get("status") != "CALIBRATION_READY":
            raise TeacherCaptureBlocked("DATASET_MANIFEST_NOT_READY", "native teacher capture requires a CALIBRATION_READY dataset manifest")
        fixed_records = {item: resolve_corpus_records(dataset_manifest, item) for item in splits}
        if "train" in fixed_records and "holdout" in fixed_records:
            train_ids = {str(item.get("id")) for item in fixed_records["train"]}
            holdout_ids = {str(item.get("id")) for item in fixed_records["holdout"]}
            train_hashes = {str(item["content_sha256"]) for item in fixed_records["train"]}
            holdout_hashes = {str(item["content_sha256"]) for item in fixed_records["holdout"]}
            if train_ids & holdout_ids or train_hashes & holdout_hashes:
                raise TeacherCaptureBlocked("FIXED_SPLIT_OVERLAP", "train and holdout records are not disjoint by stable ID and content hash")
        snapshot = Path(source_snapshot)
        if tokenizer is None:
            tokenizer = load_pinned_tokenizer(snapshot, source_revision)
        tokenizer_hashes = snapshot_tokenizer_hashes(snapshot)
        manifest_sequence_length = int(sequence_length or manifest_payload.get("sequence_length", 2048))
        tokenized = {item: tokenize_corpus_records(fixed_records[item], tokenizer, split=item, sequence_length=manifest_sequence_length) for item in splits}
        if model is None:
            model = load_native_teacher(snapshot, source_revision, device_map=device_map, device=device, compute_dtype=compute_dtype, max_memory=max_memory, offload_folder=offload_folder)
        records = _discover_mlp_records(model, _model_layer_count(model))
        missing = [layer for layer in selected_layers if layer not in records]
        if missing:
            raise TeacherCaptureBlocked("MLP_LAYER_NOT_FOUND", f"requested layers are not present in the verified teacher module set: {missing}", requested_layers=selected_layers, discovered_layers=sorted(records))
        module_map = {layer: records[layer][1] for layer in selected_layers}
        module_paths = {layer: records[layer][0] for layer in selected_layers}
        verification: list[dict[str, Any]] = []
        first_example = next(iter(tokenized.values()))[0]
        for layer in selected_layers:
            captured, outputs = _capture_forward(model, {layer: module_map[layer]}, [first_example], collect_outputs=True)
            result = verify_mlp_reconstruction(module_map[layer], captured[layer], outputs[layer], layer=layer, module_path=module_paths[layer], threshold=hook_threshold)
            verification.append(result.as_dict())
            if result.status != "HOOK_VERIFIED":
                raise TeacherCaptureBlocked("MLP_HOOK_VERIFICATION_FAILED", result.message or "MLP hook verification failed", layer=layer, verification=result.as_dict())
        layer_results: list[dict[str, Any]] = []
        started = time.perf_counter()
        for split_name in splits:
            split_meta = fixed_split_metadata(fixed_records[split_name], split=split_name, tokenizer_revision=str(manifest_payload.get("tokenizer_revision") or source_revision), tokenizer_hashes=tokenizer_hashes)
            split_token_count = sum(len(item.input_ids) for item in tokenized[split_name])
            for layer in selected_layers:
                def activation_stream(layer_id: int = layer, split_id: str = split_name) -> Iterator[Any]:
                    for batch in _batches(tokenized[split_id], microbatch):
                        captured, _ = _capture_forward(model, {layer_id: module_map[layer_id]}, batch)
                        yield captured[layer_id].detach().to("cpu").float().numpy()

                metadata = {
                    **split_meta,
                    "dataset_hash": manifest_payload.get("dataset_hash", ""),
                    "source_revision": source_revision,
                    "source_snapshot": str(Path(source_snapshot)),
                    "hook_path": module_paths[layer],
                    "hook_verification": verification[selected_layers.index(layer)],
                    "sequence_length": manifest_sequence_length,
                    "microbatch": microbatch,
                    "device_map": device_map,
                    "capture_kind": "native_teacher_mlp_input",
                    "split_token_count": split_token_count,
                }
                captured_manifest = capture_activation_shards(
                    activation_stream(),
                    destination,
                    layer=layer,
                    split=split_name,
                    manifest_name=f"layer-{layer:04d}-{split_name}.json",
                    shard_tokens=shard_tokens,
                    dtype=dtype,
                    resume=resume,
                    metadata=metadata,
                )
                layer_results.append(captured_manifest)
        total_tokens = sum(sum(len(item.input_ids) for item in tokenized[name]) for name in splits)
        return {
            "status": "CAPTURE_COMPLETE" if all(item.get("status") in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED"} for item in layer_results) else "BLOCKED",
            "layers": layer_results,
            "dataset_manifest": str(dataset_manifest),
            "source_snapshot": str(source_snapshot),
            "source_revision": source_revision,
            "module_paths": module_paths,
            "hook_verification": verification,
            "split_names": list(splits),
            "resource_metrics": _resource_metrics(started, total_tokens, model),
            "resumable": True,
            "code_commit": current_git_commit(),
        }
    except TeacherCaptureBlocked as exc:
        return _blocked_from_exception(exc, manifest=dataset_manifest, layers=selected_layers)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError) as exc:
        blocked = TeacherCaptureBlocked("TEACHER_CAPTURE_FAILED", f"native teacher capture failed before a validated result was produced: {exc}")
        return _blocked_from_exception(blocked, manifest=dataset_manifest, layers=selected_layers)


capture_native_teacher_activations = capture_text_teacher_activations
capture_native_teacher = capture_text_teacher_activations
capture_teacher_activations = capture_text_teacher_activations


__all__ = [
    "REPRESENTATIVE_LAYERS",
    "HookVerification",
    "TeacherCaptureBlocked",
    "TokenizedExample",
    "capture_native_teacher",
    "capture_native_teacher_activations",
    "capture_teacher_activations",
    "capture_text_teacher_activations",
    "discover_mlp_modules",
    "discover_mlp_paths",
    "fixed_split_metadata",
    "is_pinned_source_revision",
    "load_native_teacher",
    "load_pinned_tokenizer",
    "resolve_corpus_records",
    "snapshot_tokenizer_hashes",
    "tokenize_corpus_records",
    "validate_mlp_hook",
    "verify_mlp_reconstruction",
]
