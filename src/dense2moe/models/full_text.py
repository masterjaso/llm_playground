"""Tiny Qwen-style text target used to prove full-model serialization.

The real Qwen 3.5 checkpoint is multimodal and too large for a unit-test
fixture.  This module mirrors the text-side ownership boundary: embeddings,
attention-shaped residual blocks, norms, and the language head remain ordinary
PyTorch modules while exactly one dense MLP can be replaced by the trainable
MoE block.  The spike is intentionally small, strict, and reloadable from
disk without retaining the original Python object.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional dependency path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

# Keep the package importable in the dependency-light control-plane
# environment.  The concrete module operations still fail closed through
# ``_require_torch`` when a caller tries to instantiate or run the spike.
_ModuleBase = nn.Module if nn is not None else object

from .torch_moe import TorchQwen35SwiGLUMoE


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the full-model spike")
    return torch


@dataclass(frozen=True)
class TinyQwen35TextConfig:
    vocab_size: int = 97
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    routed_experts: int = 4
    shared_intermediate_size: int = 16
    top_k: int = 2
    moe_layer: int = 0


class _TinyDenseMLP(_ModuleBase):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyTextBlock(_ModuleBase):
    def __init__(self, config: TinyQwen35TextConfig, *, is_moe: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp: Any
        if is_moe:
            expert_width = (config.intermediate_size - config.shared_intermediate_size) // config.routed_experts
            self.mlp = TorchQwen35SwiGLUMoE(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                routed_experts=config.routed_experts,
                expert_intermediate_size=expert_width,
                shared_intermediate_size=config.shared_intermediate_size,
                top_k=config.top_k,
                learnable_scales=True,
            )
        else:
            self.mlp = _TinyDenseMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyQwen35TextMoE(_ModuleBase):
    """Minimal reloadable text model with one replaceable MLP."""

    architecture = "qwen3_5_text_tiny_moe_v1"

    def __init__(self, config: TinyQwen35TextConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or TinyQwen35TextConfig()
        if not 0 <= self.config.moe_layer < self.config.num_hidden_layers:
            raise ValueError("moe_layer must identify one configured layer")
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [_TinyTextBlock(self.config, is_moe=index == self.config.moe_layer) for index in range(self.config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))

    def generate(self, input_ids: Tensor, *, max_new_tokens: int = 4) -> Tensor:
        runtime = _require_torch()
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        with runtime.no_grad():
            output = input_ids.clone()
            for _ in range(max_new_tokens):
                logits = self(output)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                output = runtime.cat((output, next_token), dim=1)
            return output

    def save_pretrained(self, destination: str | Path) -> Path:
        runtime = _require_torch()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required for full-model serialization") from exc
        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(state, str(destination / "model.safetensors"))
        config = {
            "architectures": [self.architecture],
            "model_type": "qwen3_5_text",
            "tiny_config": asdict(self.config),
            "state_dict_inventory": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()},
            "torch_version": runtime.__version__,
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_pretrained(cls, source: str | Path, *, strict: bool = True) -> TinyQwen35TextMoE:
        _require_torch()
        source = Path(source)
        config_payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config = TinyQwen35TextConfig(**dict(config_payload["tiny_config"]))
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required for full-model reload") from exc
        state = load_file(str(source / "model.safetensors"), device="cpu")
        expected = set(config_payload.get("state_dict_inventory", {}))
        if strict and set(state) != expected:
            raise ValueError(f"strict tensor inventory mismatch: missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}")
        model = cls(config)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise ValueError(f"strict state-dict mismatch: missing={missing}, unexpected={unexpected}")
        return model


def run_full_model_spike(destination: str | Path, *, seed: int = 17) -> dict[str, Any]:
    """Save, destroy, strictly reload, and compare logits/generation."""

    runtime = _require_torch()
    runtime.manual_seed(seed)
    model = TinyQwen35TextMoE()
    model.eval()
    input_ids = runtime.randint(0, model.config.vocab_size, (2, 5))
    with runtime.inference_mode():
        before_logits = model(input_ids)
        before_generation = model.generate(input_ids, max_new_tokens=4)
    path = model.save_pretrained(destination)
    inventory = set(model.state_dict())
    del model
    gc.collect()
    reloaded = TinyQwen35TextMoE.from_pretrained(path, strict=True)
    reloaded.eval()
    with runtime.inference_mode():
        after_logits = reloaded(input_ids)
        after_generation = reloaded.generate(input_ids, max_new_tokens=4)
    max_logit_delta = float((before_logits - after_logits).abs().max().item())
    generation_equal = bool(runtime.equal(before_generation, after_generation))
    result = {
        "status": "FULL_MODEL_RELOAD_GREEN" if max_logit_delta <= 1e-6 and generation_equal else "FULL_MODEL_RELOAD_RED",
        "architecture": TinyQwen35TextMoE.architecture,
        "path": str(path),
        "strict_tensor_inventory": True,
        "tensor_count": len(inventory),
        "max_logit_delta": max_logit_delta,
        "generation_equal": generation_equal,
        "seed": seed,
        "config": asdict(reloaded.config),
    }
    (Path(destination) / "full-model-spike.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = ["TinyQwen35TextConfig", "TinyQwen35TextMoE", "run_full_model_spike"]
