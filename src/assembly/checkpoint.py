"""Safe assembly of layer artifacts into a manifest-backed checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dense2moe.state import atomic_write_json


def target_state_dict_inventory(*, num_layers: int, hidden_size: int, routed_experts: int, expert_intermediate_size: int, shared_intermediate_size: int) -> list[str]:
    if min(num_layers, hidden_size, routed_experts, expert_intermediate_size, shared_intermediate_size) <= 0:
        raise ValueError("target dimensions must be positive")
    keys: list[str] = ["model.embed_tokens.weight"]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        keys.extend([f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"])
        keys.extend([f"{prefix}.mlp.router.weight", f"{prefix}.mlp.shared_expert.gate.weight", f"{prefix}.mlp.shared_expert.w1.weight", f"{prefix}.mlp.shared_expert.w2.weight"])
        for expert in range(routed_experts):
            keys.extend([f"{prefix}.mlp.experts.{expert}.w1.weight", f"{prefix}.mlp.experts.{expert}.w2.weight"])
    keys.extend(["model.norm.weight", "lm_head.weight"])
    return keys


def assemble_checkpoint(layer_paths: Iterable[str | Path], destination: str | Path, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    layers: list[dict[str, Any]] = []
    for path in sorted((Path(item) for item in layer_paths), key=lambda item: item.name):
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        layers.append({"path": str(path), "sha256": digest, "size": path.stat().st_size})
    manifest_metadata = dict(metadata or {})
    expected_layers = manifest_metadata.get("expected_layers")
    complete = bool(layers)
    if expected_layers is not None:
        try:
            expected = int(expected_layers)
        except (TypeError, ValueError):
            expected = -1
        complete = complete and expected > 0 and len(layers) == expected
    manifest = {"format": "dense2moe-manifest-v1", "layers": layers, "metadata": manifest_metadata, "complete": complete}
    atomic_write_json(dst / "manifest.json", manifest)
    return manifest
