"""Full Qwen3.5 dense-to-MoE replacement boundary.

The installed Transformers implementation owns attention, Gated DeltaNet,
normalization, embeddings, and the language head.  This module deliberately
reuses that model and replaces only decoder-layer ``mlp`` objects, so a
conversion cannot silently substitute a tiny surrogate backbone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # Optional dependency path for the control-plane installation.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

from ..checkpoint.layer import validate_layer_checkpoint
from ..config import TopologyContract, active_topology_contract
from .torch_moe import TorchQwen35SwiGLUMoE


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the Qwen3.5 full-model path")
    return torch


def _tensor_digest(value: Any) -> str:
    _require_torch()
    tensor = value.detach().to(device="cpu").contiguous()
    payload = tensor.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _non_ffn_inventory(model: Any) -> dict[str, dict[str, Any]]:
    """Hash every tensor outside decoder-layer MLP ownership."""

    inventory: dict[str, dict[str, Any]] = {}
    for name, value in model.state_dict().items():
        if ".mlp." in name or name.startswith("mlp."):
            continue
        inventory[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_digest(value),
        }
    return inventory


def _text_backbone(model: Any) -> Any:
    """Find the Qwen text model without assuming multimodal wrapper depth."""

    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
        model,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise TypeError("model does not expose a Qwen text backbone with decoder layers")


def _layer_mlp(layer: Any) -> Any:
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        raise TypeError("decoder layer does not expose a dense SwiGLU mlp")
    return mlp


def replace_qwen35_ffns(
    model: Any,
    *,
    topology: str | TopologyContract,
    strict_layer_count: int = 64,
) -> dict[str, Any]:
    """Replace all Qwen3.5 dense SwiGLUs while preserving the backbone.

    The operation is in-place and returns a receipt-like dictionary.  No
    attention or Gated DeltaNet object is reconstructed.  A second conversion
    is rejected rather than stacking incompatible MoE blocks.
    """

    runtime = _require_torch()
    contract = topology if isinstance(topology, TopologyContract) else active_topology_contract(str(topology))
    text_model = _text_backbone(model)
    layers = list(text_model.layers)
    if strict_layer_count > 0 and len(layers) != strict_layer_count:
        raise ValueError(f"strict Qwen layer count mismatch: expected {strict_layer_count}, got {len(layers)}")
    before_non_ffn = _non_ffn_inventory(model)
    replaced: list[str] = []
    for index, layer in enumerate(layers):
        dense_mlp = _layer_mlp(layer)
        if isinstance(dense_mlp, TorchQwen35SwiGLUMoE):
            raise TypeError(f"layer {index} is already a dense2moe block")
        gate = dense_mlp.gate_proj.weight
        up = dense_mlp.up_proj.weight
        down = dense_mlp.down_proj.weight
        converted = TorchQwen35SwiGLUMoE.from_dense(
            gate,
            up,
            down,
            routed_experts=contract.routed_experts,
            shared_intermediate_size=contract.shared_intermediate_size,
            top_k=contract.top_k,
            routing_mode="independent_positive",
            dtype=gate.dtype,
            device=gate.device,
        )
        converted.train(dense_mlp.training)
        layer.mlp = converted
        replaced.append(f"layers.{index}.mlp")
    after_non_ffn = _non_ffn_inventory(model)
    if before_non_ffn != after_non_ffn:
        mismatches = sorted(set(before_non_ffn) ^ set(after_non_ffn))
        mismatches.extend(
            name for name in sorted(set(before_non_ffn) & set(after_non_ffn)) if before_non_ffn[name] != after_non_ffn[name]
        )
        raise ValueError(f"non-FFN tensor preservation failed: {mismatches[:8]}")
    return {
        "schema_version": 1,
        "receipt_type": "dense2moe-qwen35-full-replacement",
        "status": "FULL_MODEL_FFNS_REPLACED",
        "topology": contract.topology_id,
        "profile": contract.profile_name,
        "replaced_layers": replaced,
        "replaced_layer_count": len(replaced),
        "strict_layer_count": strict_layer_count,
        "non_ffn_inventory": before_non_ffn,
        "non_ffn_preserved": True,
        "attention_and_gated_deltanet_reused": True,
        "sparse_dispatch_required": True,
        "source_model_class": type(model).__name__,
        "torch_version": runtime.__version__,
    }


def _resolve_manifest_path(manifest_path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return manifest_path.parent / candidate


def apply_layer_checkpoints(
    wrapper: Qwen35DenseToMoE,
    manifest_path: str | Path,
    *,
    expected_profile: str,
    strict_layer_count: int = 64,
) -> dict[str, Any]:
    """Apply validated trained layer tensors to a converted full model.

    A queue or metadata-only manifest is not sufficient.  Every layer must be
    ``TRAINED_VALIDATED`` with a real, hashed safetensors artifact and a green
    quality gate before its tensors are loaded into the corresponding Qwen
    decoder MLP.  The source model remains the owner of all non-FFN tensors.
    """

    runtime = _require_torch()
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise ValueError("full64 checkpoint manifest must be complete before assembly")
    entries = payload.get("layers")
    if not isinstance(entries, list) or len(entries) != strict_layer_count:
        raise ValueError(f"full64 checkpoint manifest must contain exactly {strict_layer_count} layers")
    text_model = _text_backbone(wrapper.model)
    if len(text_model.layers) != strict_layer_count:
        raise ValueError("converted Qwen model layer count does not match the full64 manifest")

    try:
        from safetensors.torch import load_file  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for full-model checkpoint assembly") from exc

    applied: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for item in sorted(entries, key=lambda value: int(value.get("layer", -1)) if isinstance(value, Mapping) else -1):
        if not isinstance(item, Mapping):
            raise TypeError("full64 layer manifest entries must be objects")
        layer = int(item.get("layer", -1))
        if layer in seen_layers or layer < 0 or layer >= strict_layer_count:
            raise ValueError(f"invalid or duplicate full64 layer {layer}")
        seen_layers.add(layer)
        if str(item.get("profile", expected_profile)) != expected_profile:
            raise ValueError(f"layer {layer} profile does not match {expected_profile}")
        metadata_value = item.get("metadata")
        if not metadata_value:
            raise ValueError(f"layer {layer} metadata path is required")
        metadata_path = _resolve_manifest_path(manifest, str(metadata_value))
        valid, errors, checkpoint = validate_layer_checkpoint(
            metadata_path,
            expected_profile=expected_profile,
            expected_layer=layer,
            require_quality=True,
        )
        if not valid or checkpoint is None:
            raise ValueError(f"layer {layer} checkpoint validation failed: {errors[:8]}")
        if checkpoint.quality_gate.get("overall") != "green":
            raise ValueError(f"layer {layer} is not product-green: {checkpoint.quality_gate.get('overall')!r}")
        tensor_path = _resolve_manifest_path(metadata_path, str(checkpoint.tensor_file or ""))
        raw_state = load_file(str(tensor_path), device="cpu")
        prefix = f"model.layers.{layer}."
        block_state: dict[str, Any] = {}
        for name, value in raw_state.items():
            key = str(name)
            key = key.removeprefix(prefix).removeprefix("mlp.")
            block_state[key] = value
        expected_keys = set(text_model.layers[layer].mlp.state_dict())
        if set(block_state) != expected_keys:
            raise ValueError(
                f"layer {layer} tensor namespace mismatch: "
                f"missing={sorted(expected_keys - set(block_state))[:8]}, "
                f"unexpected={sorted(set(block_state) - expected_keys)[:8]}"
            )
        text_model.layers[layer].mlp.load_state_dict(block_state, strict=True)
        applied.append(
            {
                "layer": layer,
                "metadata": str(metadata_path),
                "tensor_file": str(tensor_path),
                "tensor_sha256": checkpoint.tensor_sha256,
                "quality_gate": checkpoint.quality_gate,
                "code_commit": checkpoint.code_commit,
            }
        )
    if seen_layers != set(range(strict_layer_count)):
        raise ValueError("full64 manifest does not cover every decoder layer")
    receipt = {
        "schema_version": 1,
        "receipt_type": "dense2moe-qwen35-layer-checkpoint-application",
        "status": "FULL64_LAYER_CHECKPOINTS_APPLIED",
        "profile": expected_profile,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "applied_layer_count": len(applied),
        "layers": applied,
        "dense_fallback_used": False,
        "source_backbone_preserved": True,
        "torch_version": runtime.__version__,
    }
    wrapper.receipt["layer_checkpoint_application"] = receipt
    return receipt


class Qwen35DenseToMoE(nn.Module if nn is not None else object):
    """Reloadable wrapper around a converted Hugging Face Qwen3.5 model."""

    architecture = "qwen3_5_dense_to_moe_v1"

    def __init__(self, model: Any, *, receipt: Mapping[str, Any] | None = None) -> None:
        _require_torch()
        super().__init__()
        self.model = model
        self.receipt = dict(receipt or {})

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        topology: str | TopologyContract,
        revision: str | None = None,
        local_files_only: bool = True,
        strict_layer_count: int = 64,
        **kwargs: Any,
    ) -> Qwen35DenseToMoE:
        _require_torch()
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for Qwen3.5 full-model loading") from exc
        dense = AutoModelForCausalLM.from_pretrained(
            str(source),
            revision=revision,
            local_files_only=local_files_only,
            **kwargs,
        )
        receipt = replace_qwen35_ffns(dense, topology=topology, strict_layer_count=strict_layer_count)
        return cls(dense, receipt=receipt)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def sparse_dispatch_receipt(self, inputs: Tensor, *, layer: int = 0) -> dict[str, Any]:
        """Run one layer with router telemetry for a blocking dispatch proof."""

        runtime = _require_torch()
        text_model = _text_backbone(self.model)
        if layer < 0 or layer >= len(text_model.layers):
            raise IndexError(layer)
        block = text_model.layers[layer].mlp
        if not isinstance(block, TorchQwen35SwiGLUMoE):
            raise TypeError("selected layer is not a dense2moe block")
        with runtime.inference_mode():
            _output, info = block(inputs, return_router=True)
        return {
            "dispatch_mode": info["dispatch_mode"],
            "dense_fallback_used": bool(info["dense_fallback_used"]),
            "dispatch_token_count": int(info["dispatch_token_count"]),
            "selected_dispatches": int(info["selected_dispatches"]),
            "expert_token_counts": list(info["expert_token_counts"]),
            "active_intermediate_width": int(info["active_intermediate_width"]),
            "dense_intermediate_width": int(info["dense_intermediate_width"]),
            "estimated_ffn_reduction": float(info["estimated_ffn_reduction"]),
            "runtime": "pytorch-sparse-token-dispatch",
        }

    def save_pretrained(self, destination: str | Path, **kwargs: Any) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, **kwargs)
        metadata = {
            "architecture": self.architecture,
            "receipt": self.receipt,
            "state_dict_inventory": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_digest(value)}
                for name, value in self.state_dict().items()
            },
        }
        (destination / "dense2moe-receipt.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


__all__ = ["Qwen35DenseToMoE", "apply_layer_checkpoints", "replace_qwen35_ffns"]
