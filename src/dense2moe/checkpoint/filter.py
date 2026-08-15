"""Text-backbone tensor selection and immutable checkpoint copying."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..discovery.source import filter_text_tensor_names


def text_tensor_filter(names: Iterable[str]) -> list[str]:
    return filter_text_tensor_names(names)


def filter_text_checkpoint(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create a manifest for text tensors and copy only metadata/config files.

    Copying multi-gigabyte safetensors is intentionally delegated to the
    optional safetensors writer.  This function is safe to run in discovery and
    produces an explicit `ready_for_extraction` flag rather than pretending a
    copy succeeded.
    """

    src, dst = Path(source), Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    files = sorted(src.glob("*.safetensors")) if src.exists() else []
    for path in files:
        try:
            from .safetensors import SafetensorsSliceReader

            with SafetensorsSliceReader(path) as reader:
                names.extend(reader.keys())
        except (OSError, RuntimeError, ValueError, KeyError):
            continue
    selected = text_tensor_filter(names)
    manifest = {
        "source": str(src),
        "destination": str(dst),
        "source_files": [str(path) for path in files],
        "tensor_names": sorted(set(names)),
        "text_tensor_names": selected,
        "ready_for_extraction": bool(selected),
        "remote_code": False,
    }
    (dst / "text-filter-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        source_file = src / name
        if source_file.exists() and not (dst / name).exists():
            (dst / name).write_bytes(source_file.read_bytes())
    return manifest


def extract_text_checkpoint(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Materialize text-only safetensor shards one source shard at a time.

    Completed source files are never modified, and an existing destination
    shard is reused so an interrupted extraction can resume safely.
    """

    src, dst = Path(source), Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    source_files = sorted(src.glob("*.safetensors"))
    if not source_files:
        return filter_text_checkpoint(src, dst)
    writer: tuple[str, Any] | None = None
    # Qwen checkpoints commonly contain BF16 tensors, which NumPy cannot
    # materialize on all versions. Prefer the torch writer when available and
    # retain NumPy as a CPU-only fallback for ordinary dtypes.
    try:
        from safetensors.torch import save_file as save_torch  # type: ignore

        writer = ("pt", save_torch)
    except ImportError:
        try:
            from safetensors.numpy import save_file as save_numpy  # type: ignore

            writer = ("np", save_numpy)
        except ImportError:
            writer = None
    if writer is None:
        return filter_text_checkpoint(src, dst)
    weight_map: dict[str, str] = {}
    selected_names: list[str] = []
    for source_file in source_files:
        shard_name = source_file.name
        destination_file = dst / shard_name
        if destination_file.exists():
            try:
                from .safetensors import SafetensorsSliceReader

                with SafetensorsSliceReader(destination_file) as reader:
                    names = reader.keys()
                selected_names.extend(names)
                weight_map.update({name: shard_name for name in names})
                continue
            except (OSError, RuntimeError, ValueError, KeyError):
                destination_file.unlink()
        try:
            from safetensors import safe_open  # type: ignore

            with safe_open(str(source_file), framework=writer[0]) as handle:
                names = text_tensor_filter(handle.keys())
                if not names:
                    continue
                tensors = {name: handle.get_tensor(name) for name in names}
            writer[1](tensors, str(destination_file))
            selected_names.extend(names)
            weight_map.update({name: shard_name for name in names})
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            continue
    manifest = {
        "source": str(src),
        "destination": str(dst),
        "source_files": [str(path) for path in source_files],
        "text_tensor_names": sorted(set(selected_names)),
        "weight_map": weight_map,
        "ready_for_extraction": bool(weight_map),
        "materialized": bool(weight_map),
        "remote_code": False,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dst / "text-filter-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "merges.txt", "vocab.json"):
        source_file = src / name
        if source_file.exists() and not (dst / name).exists():
            (dst / name).write_bytes(source_file.read_bytes())
    return manifest
