"""Pinned source-model inspection without remote-code execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TEXT_INCLUDE = ("model.", "transformer.", "layers.", "embed", "lm_head", "norm")
TEXT_EXCLUDE = ("vision", "visual", "image", "video", "audio", "mtp", "nextn", "projector")


def filter_text_tensor_names(names: Iterable[str]) -> list[str]:
    """Return deterministic text-backbone names while excluding multimodal heads."""

    kept: list[str] = []
    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in TEXT_EXCLUDE):
            continue
        if any(lowered.startswith(prefix) or prefix in lowered for prefix in TEXT_INCLUDE):
            kept.append(name)
    return sorted(set(kept))


def text_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize multimodal configs to the nested text-backbone mapping."""

    nested = config.get("text_config")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(config)


def verify_qwen_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return observed geometry without treating prompt values as facts."""

    text = text_config(config)
    fields = {
        "hidden_size": text.get("hidden_size"),
        "dense_intermediate_size": text.get("intermediate_size", text.get("dense_intermediate_size")),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "model_type": text.get("model_type", config.get("model_type")),
        "layer_types": text.get("layer_types"),
        "full_attention_interval": text.get("full_attention_interval"),
    }
    fields["shape_verified"] = all(fields[key] is not None for key in ("hidden_size", "dense_intermediate_size", "num_hidden_layers"))
    return fields


def is_pinned_revision(revision: str | None) -> bool:
    if not revision:
        return False
    value = revision.strip()
    if len(value) >= 7 and all(char in "0123456789abcdefABCDEF" for char in value):
        return True
    return value.startswith(("refs/", "commit:"))


@dataclass(frozen=True)
class SourceManifest:
    model: str
    revision: str
    revision_pinned: bool
    config: dict[str, Any]
    tensor_names: list[str]
    text_tensor_names: list[str]
    files: list[dict[str, Any]]
    license: str
    remote_code: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not root.exists():
        return files
    # A continuation run may point at an immutable snapshot whose parent run
    # already recorded a complete inventory.  Reusing that evidence avoids a
    # second multi-gigabyte hash pass while preserving the original hashes.
    cached = root.parent / "source-manifest.json"
    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            cached_files = payload.get("files", []) if isinstance(payload, Mapping) else []
            if isinstance(cached_files, list) and cached_files:
                return [dict(item) for item in cached_files if isinstance(item, Mapping)]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files.append({"path": str(path.relative_to(root)), "size": size, "sha256": digest.hexdigest()})
    return files


def inspect_local_source(root: str | Path, *, model: str = "local", revision: str = "unknown") -> SourceManifest:
    source = Path(root)
    config: dict[str, Any] = {}
    config_path = source / "config.json"
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                config = dict(loaded)
        except json.JSONDecodeError:
            config = {"error": "invalid config.json"}
    tensor_names: list[str] = []
    for path in sorted(source.glob("*.safetensors")):
        try:
            from safetensors import safe_open  # type: ignore

            with safe_open(str(path), framework="numpy") as handle:
                tensor_names.extend(handle.keys())
        except (OSError, RuntimeError, ValueError, KeyError):
            # A malformed or unavailable optional reader is evidence, not a
            # reason to invent an inventory.
            continue
    text_names = filter_text_tensor_names(tensor_names)
    license_name = "unknown"
    for candidate in (source / "LICENSE", source / "LICENSE.txt", source / "LICENSE.md"):
        if candidate.exists():
            license_name = candidate.name
            break
    return SourceManifest(model, revision, is_pinned_revision(revision), config, sorted(set(tensor_names)), text_names, _file_inventory(source), license_name)


def inspect_hub_source(model: str, revision: str = "main") -> SourceManifest:
    """Inspect Hub metadata when available; never enables trust_remote_code."""

    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        try:
            info = api.model_info(model, revision=revision, files_metadata=True)
        except TypeError:
            info = api.model_info(model, revision=revision)
        siblings = getattr(info, "siblings", []) or []
        files = [{"path": getattr(item, "rfilename", ""), "size": getattr(item, "size", None)} for item in siblings]
        config = {}
        resolved_revision = str(getattr(info, "sha", "") or revision)
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            # Download only the small JSON config; no model code is imported or
            # executed, so `trust_remote_code` is deliberately not involved.
            config_path = hf_hub_download(repo_id=model, filename="config.json", revision=revision)
            loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                config = dict(loaded)
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            config = {"error": str(exc)}
        card = getattr(info, "cardData", None)
        if isinstance(card, Mapping):
            config["card_data"] = dict(card)
        license_name = str(config.get("license", "unknown"))
        if license_name == "unknown" and isinstance(card, Mapping):
            license_name = str(card.get("license", "unknown"))
        return SourceManifest(model, resolved_revision, is_pinned_revision(resolved_revision), config, [], [], files, license_name)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return SourceManifest(model, revision, is_pinned_revision(revision), {"error": str(exc)}, [], [], [], "unknown")
