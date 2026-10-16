"""Hugging Face adapter for the native FlashMini v2/v3 model.

The native model remains the source of truth for parameter names, initialization,
and forward math.  This module supplies the small Transformers surface needed by
HF trainers and ``AutoModelForCausalLM``.  Native FlashMini has no KV cache and
its causal attention does not consume an arbitrary attention mask, so this
adapter handles contiguous right padding by evaluating each non-padded row at
its true length.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Keep these imports absolute.  Transformers' custom-object exporter scans
# relative imports as sibling files; ``flashmini.models`` is a package rather
# than a ``models.py`` module.  The Desktop/HF environment installs FlashMini
# alongside the exported adapter, so the package import is available there.
from flashmini.config import FlashMiniConfig as NativeFlashMiniConfig
from flashmini.models import FlashMiniModel
from flashmini.models.ple import PLE, PLEV3
from flashmini.models.tiny import CausalAttention

_NATIVE_FIELDS = (
    "vocab_size",
    "d_model",
    "num_layers",
    "num_heads",
    "head_dim",
    "gdn_per_attention",
    "max_seq_len",
    "dropout",
    "use_hyperconnection",
    "use_ple",
    "moe",
    "gdn",
    "ple",
    "kvc",
    "attention_layers",
    "architecture_version",
    "experiment_mode",
    "hc_count",
    "hc_lowrank",
    "rope_theta",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise TypeError(f"expected a mapping or dataclass, got {type(value).__name__}")


def _native_values(native: NativeFlashMiniConfig) -> dict[str, Any]:
    """Return every normalized native field, including v2 compatibility fields."""
    values = dataclasses.asdict(native)
    return {key: copy.deepcopy(values[key]) for key in _NATIVE_FIELDS if key in values}


def _native_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a native checkpoint envelope or a bare native state dict."""
    path = Path(path)
    if path.is_dir():
        state_path = path / "state.pt"
        if not state_path.is_file():
            candidates = sorted(path.glob("*.pt"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"native checkpoint directory {path} must contain state.pt or one .pt file"
                )
            state_path = candidates[0]
        path = state_path
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"native checkpoint {path} must contain a mapping")
    return dict(payload)


def _config_from_any(config: Any) -> FlashMiniConfig:
    if isinstance(config, FlashMiniConfig):
        return config
    if isinstance(config, NativeFlashMiniConfig):
        return FlashMiniConfig.from_native_config(config)
    if isinstance(config, Mapping):
        return FlashMiniConfig.from_dict(dict(config))
    raise TypeError("config must be a FlashMiniConfig, native FlashMiniConfig, or mapping")


def _config_mismatch(saved: Any, current: Any, path: tuple[str, ...] = ()) -> str | None:
    """Return the first semantic config difference.

    Native checkpoint loading permits changing only PLE storage placement.  The
    adapter applies the same rule after normalizing both configs through the
    native dataclass, so an explicit HF config cannot silently reinterpret a
    checkpoint's PLE hash, KVC, or architecture settings.
    """
    if path == ("ple", "offload"):
        return None
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        if set(saved) != set(current):
            missing = sorted(set(saved) ^ set(current))
            return ".".join(path + (str(missing[0]),)) if missing else ".".join(path)
        for key in sorted(saved):
            mismatch = _config_mismatch(saved[key], current[key], path + (str(key),))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(saved) != len(current):
            return ".".join(path)
        for index, (saved_value, current_value) in enumerate(zip(saved, current)):
            mismatch = _config_mismatch(saved_value, current_value, path + (str(index),))
            if mismatch is not None:
                return mismatch
        return None
    return None if saved == current else ".".join(path)


class FlashMiniConfig(PretrainedConfig):
    """Transformers configuration backed by :class:`flashmini.config.FlashMiniConfig`."""

    model_type = "flashmini"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        head_dim: int = 64,
        gdn_per_attention: int = 3,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_hyperconnection: bool = True,
        use_ple: bool = False,
        moe: Mapping[str, Any] | None = None,
        gdn: Mapping[str, Any] | None = None,
        ple: Mapping[str, Any] | None = None,
        kvc: Mapping[str, Any] | None = None,
        attention_layers: list[int] | None = None,
        architecture_version: int = 2,
        experiment_mode: str = "screening",
        hc_count: int = 4,
        hc_lowrank: int | None = None,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("is_encoder_decoder", False)
        kwargs.setdefault("architectures", ["FlashMiniForCausalLM"])
        super().__init__(**kwargs)

        native = NativeFlashMiniConfig.from_dict(
            {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "gdn_per_attention": gdn_per_attention,
                "max_seq_len": max_seq_len,
                "dropout": dropout,
                "use_hyperconnection": use_hyperconnection,
                "use_ple": use_ple,
                "moe": _as_dict(moe) if moe is not None else {},
                "gdn": _as_dict(gdn) if gdn is not None else {},
                "ple": _as_dict(ple) if ple is not None else {},
                "kvc": _as_dict(kvc) if kvc is not None else {},
                "attention_layers": list(attention_layers) if attention_layers is not None else None,
                "architecture_version": architecture_version,
                "experiment_mode": experiment_mode,
                "hc_count": hc_count,
                "hc_lowrank": hc_lowrank,
                "rope_theta": rope_theta,
            }
        )
        # Store normalized plain values so config.json can reconstruct the exact
        # native dataclass without serializing dataclass instances.
        for key, value in _native_values(native).items():
            setattr(self, key, value)

        self.hidden_size = native.d_model
        self.num_hidden_layers = native.num_layers
        self.num_attention_heads = native.num_heads
        self.max_position_embeddings = native.max_seq_len
        self.intermediate_size = native.moe.expert_intermediate
        self.tie_word_embeddings = True
        self.use_cache = False
        self.gradient_checkpointing = False
        self._flashmini_native_config = _native_values(native)

    @classmethod
    def from_native_config(cls, config: NativeFlashMiniConfig) -> FlashMiniConfig:
        return cls(**_native_values(config))

    def to_native_config(self) -> NativeFlashMiniConfig:
        values = {
            key: copy.deepcopy(getattr(self, key))
            for key in _NATIVE_FIELDS
            if hasattr(self, key)
        }
        return NativeFlashMiniConfig.from_dict(values)

    @property
    def native_config(self) -> NativeFlashMiniConfig:
        """Build a native config on demand, preserving the HF config as owner."""
        return self.to_native_config()


@dataclasses.dataclass
class FlashMiniCausalLMOutput(CausalLMOutputWithPast):
    """Causal LM output with the native logical MoE auxiliary loss."""

    router_aux_loss: torch.Tensor | None = None


def _ensure_auto_map(config: PretrainedConfig) -> None:
    """Make custom code discoverable by ``save_pretrained`` and Auto classes."""
    module = FlashMiniConfig.__module__.split(".")[-1]
    current = dict(getattr(config, "auto_map", None) or {})
    current.setdefault("AutoConfig", f"{module}.FlashMiniConfig")
    current.setdefault("AutoModelForCausalLM", f"{module}.FlashMiniForCausalLM")
    config.auto_map = current


def _reject_sparse_ple(config: FlashMiniConfig) -> None:
    native = config.to_native_config()
    if native.use_ple and native.ple.sparse:
        raise ValueError(
            "HF training uses dense AdamW; sparse PLE gradients are unsupported. "
            "Set ple.sparse=false (and use offload='gpu' for a CUDA run), or use the native optimizer."
        )


def _validate_right_padding(attention_mask: torch.Tensor, batch: int, steps: int) -> list[int]:
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, steps):
        raise ValueError("FlashMini HF adapter supports only a rank-two attention_mask matching input_ids")
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
    else:
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1 values")
        mask = attention_mask != 0
    rows = mask.detach().to("cpu").tolist()
    lengths: list[int] = []
    for row in rows:
        seen_zero = False
        length = 0
        for value in row:
            if value:
                if seen_zero:
                    raise ValueError(
                        "FlashMini HF adapter supports contiguous right padding only; "
                        "left or interior padding is unsupported"
                    )
                length += 1
            else:
                seen_zero = True
        if length == 0:
            raise ValueError("attention_mask rows must contain at least one unmasked token")
        lengths.append(length)
    return lengths


def _aggregate_router_aux(stats_rows: list[dict[str, Any]], *, device: torch.device) -> torch.Tensor:
    """Reconstruct one native logical-batch aux value from additive router stats."""
    counts: dict[int, torch.Tensor] = {}
    probabilities: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    slot_counts: dict[int, int] = {}
    num_experts: int | None = None

    for stats in stats_rows:
        exp_rows = stats.get("router_exp_counts", [])
        prob_rows = stats.get("router_prob_sum", [])
        token_rows = stats.get("router_token_count", [])
        slot_rows = stats.get("router_slot_count", [])
        if not (len(exp_rows) == len(prob_rows) == len(token_rows) == len(slot_rows)):
            raise RuntimeError("native router statistics have inconsistent layer counts")
        for layer, (exp, prob, tokens, slots) in enumerate(
            zip(exp_rows, prob_rows, token_rows, slot_rows)
        ):
            exp = exp.to(device)
            prob = prob.to(device)
            num_experts = int(exp.numel())
            counts[layer] = counts.get(layer, torch.zeros_like(exp)) + exp
            probabilities[layer] = probabilities.get(layer, torch.zeros_like(prob)) + prob
            token_counts[layer] = token_counts.get(layer, 0) + int(tokens.item())
            slot_counts[layer] = slot_counts.get(layer, 0) + int(slots.item())

    if not counts:
        return torch.zeros((), device=device)
    assert num_experts is not None
    per_layer = [
        num_experts
        * ((counts[layer] / slot_counts[layer]) * (probabilities[layer] / token_counts[layer])).sum()
        for layer in sorted(counts)
    ]
    return torch.stack(per_layer).mean()


def _replace_runtime_buffer(
    module: nn.Module, name: str, value: torch.Tensor, device: torch.device
) -> None:
    """Restore a non-persistent derived buffer after meta-device HF loading."""
    module._buffers[name] = value.to(device=device)


def _rebuild_native_runtime_buffers(model: FlashMiniModel) -> None:
    """Recreate causal masks and PLE hash metadata omitted from state_dicts.

    Transformers may instantiate custom models on ``meta`` during
    ``from_pretrained``.  Native FlashMini intentionally marks these buffers
    non-persistent because they derive entirely from config; rebuilding them
    after weights are materialized avoids leaving zero or meta hash tables.
    """
    for module in model.modules():
        if isinstance(module, CausalAttention):
            device = module.qkv.weight.device
            dtype = module.qkv.weight.dtype
            if module.architecture_version == 2:
                mask = torch.tril(
                    torch.ones(
                        module.causal_mask.shape[-1],
                        module.causal_mask.shape[-1],
                        device=device,
                        dtype=dtype,
                    )
                )
                mask = mask.view(1, 1, module.causal_mask.shape[-1], module.causal_mask.shape[-1])
            else:
                mask = torch.empty(0, device=device, dtype=dtype)
            _replace_runtime_buffer(module, "causal_mask", mask, device)
        elif isinstance(module, PLEV3):
            device = module.key_proj.weight.device
            sizes = list(module.sizes)
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            _replace_runtime_buffer(
                module, "head_vocab_sizes", torch.tensor(sizes, dtype=torch.long), device
            )
            _replace_runtime_buffer(
                module, "head_offsets", torch.tensor(offsets, dtype=torch.long), device
            )
            _replace_runtime_buffer(
                module,
                "layer_multipliers",
                module._build_multipliers(
                    module.config.vocab_size, module.ngram, module.config.hash_seed
                ),
                device,
            )
        elif isinstance(module, PLE):
            device = module.out_proj.weight.device
            sizes = list(module.sizes)
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            bases = [2 + ((1009 + h * 9176) % (size - 2)) for h, size in enumerate(sizes)]
            orders = [2 + h % (module.config.ngram - 1) for h in range(module.num_heads)]
            _replace_runtime_buffer(module, "sizes_tensor", torch.tensor(sizes), device)
            _replace_runtime_buffer(module, "offsets", torch.tensor(offsets), device)
            _replace_runtime_buffer(module, "bases", torch.tensor(bases), device)
            _replace_runtime_buffer(module, "orders", torch.tensor(orders), device)


class FlashMiniForCausalLM(PreTrainedModel):
    """HF ``PreTrainedModel`` wrapper around the unchanged native model."""

    config_class = FlashMiniConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    # Transformers 5.x uses a mapping here; older releases iterate it as a
    # sequence, for which mapping keys have the same intended behavior.
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "model.head.weight": "model.embed.weight"
    }
    # The adapter computes a token-mean CE after trimming right padding.  HF
    # Trainer should not inject a separate num_items_in_batch argument.
    accepts_loss_kwargs = False

    def __init__(self, config: FlashMiniConfig) -> None:
        super().__init__(config)
        _reject_sparse_ple(config)
        self.model = FlashMiniModel(config.to_native_config())
        self.tie_weights()
        # Registration is intentionally explicit at construction time because
        # callers may instantiate a config and save immediately afterwards.
        _ensure_auto_map(self.config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed = value
        self.model.head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.model.head = new_embeddings

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = True,
    ) -> FlashMiniForCausalLM:
        # Keep the mapping available to Transformers 5.x's loading and safe
        # tensor saving code.  The fallback covers older Transformers releases
        # whose tie_weights method has no keyword arguments.
        self.all_tied_weights_keys = {"model.head.weight": "model.embed.weight"}
        try:
            super().tie_weights(missing_keys=missing_keys, recompute_mapping=False)
        except TypeError:
            self.model.head.weight = self.model.embed.weight
        return self

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> nn.Embedding:
        raise NotImplementedError(
            "FlashMini token and output projections are tied and must be resized in the native model config"
        )

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "FlashMini HF adapter does not implement gradient checkpointing; "
            "load or train with use_gradient_checkpointing=False"
        )

    def _native_logits_and_stats(
        self, input_ids: torch.Tensor, lengths: list[int]
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        _, steps = input_ids.shape
        native_device = next(self.model.parameters()).device
        input_ids = input_ids.to(native_device)
        if max(lengths) > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {max(lengths)} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        if all(length == steps for length in lengths):
            native = self.model(input_ids)
            return native["logits"], [native["stats"]]

        rows: list[torch.Tensor] = []
        stats_rows: list[dict[str, Any]] = []
        for row, length in enumerate(lengths):
            native = self.model(input_ids[row : row + 1, :length])
            rows.append(F.pad(native["logits"], (0, 0, 0, steps - length))[0])
            stats_rows.append(native["stats"])
        return torch.stack(rows, dim=0), stats_rows

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        return_dict: bool | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> FlashMiniCausalLMOutput | tuple[torch.Tensor, ...]:
        if inputs_embeds is not None:
            raise ValueError("FlashMini HF adapter accepts input_ids, not inputs_embeds")
        if input_ids is None:
            raise ValueError("input_ids is required")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-two torch.int64 tensor")
        if labels is not None and (labels.ndim != 2 or tuple(labels.shape) != tuple(input_ids.shape)):
            raise ValueError("labels must have the same rank and shape as input_ids")
        if labels is not None and labels.dtype != torch.long:
            labels = labels.to(torch.long)
        if past_key_values is not None:
            raise ValueError("FlashMini has no KV-cache implementation; past_key_values is unsupported")
        if use_cache:
            raise ValueError("FlashMini has no KV-cache implementation; use_cache must be false")
        if output_attentions or output_hidden_states:
            raise ValueError("FlashMini HF adapter does not expose attention or hidden-state collections")
        if position_ids is not None:
            if position_ids.ndim != 2 or tuple(position_ids.shape) != tuple(input_ids.shape):
                raise ValueError("position_ids must match input_ids when supplied")
            expected = torch.arange(input_ids.shape[1], device=position_ids.device).expand_as(position_ids)
            if not torch.equal(position_ids, expected):
                raise ValueError("FlashMini accepts only implicit contiguous position_ids")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported FlashMini forward arguments: {unsupported}")

        batch, steps = input_ids.shape
        if attention_mask is None:
            lengths = [steps] * batch
            mask = torch.ones((batch, steps), dtype=torch.bool, device=input_ids.device)
        else:
            lengths = _validate_right_padding(attention_mask, batch, steps)
            mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        logits, stats_rows = self._native_logits_and_stats(input_ids, lengths)
        aux = _aggregate_router_aux(stats_rows, device=logits.device)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].clone()
            shifted_labels.masked_fill_(~mask.to(logits.device)[:, 1:], -100)
            if int((shifted_labels != -100).sum().item()) == 0:
                ce = shifted_logits.sum() * 0.0
            else:
                ce = F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                )
            coefficient = float(self.config.moe.get("aux_loss_coef", 0.01))
            loss = ce + coefficient * aux

        if return_dict is None:
            return_dict = bool(self.config.use_return_dict)
        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,))
        return FlashMiniCausalLMOutput(loss=loss, logits=logits, router_aux_loss=aux)

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any) -> None:
        _ensure_auto_map(self.config)
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> FlashMiniForCausalLM:
        model = super().from_pretrained(*args, **kwargs)
        _rebuild_native_runtime_buffers(model.model)
        return model

    def load_native_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True):
        """Load a native ``FlashMiniModel.state_dict()`` into the wrapped model."""
        state = dict(state_dict)
        if not state:
            raise ValueError("native state_dict is empty")
        native_keys = set(self.model.state_dict())
        if not set(state).issubset(native_keys):
            if all(key.startswith("model.") for key in state):
                state = {key.removeprefix("model."): value for key, value in state.items()}
            elif all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
        return self.model.load_state_dict(state, strict=strict)

    @classmethod
    def from_native_checkpoint(
        cls,
        path: str | Path,
        *,
        config: FlashMiniConfig | NativeFlashMiniConfig | Mapping[str, Any] | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> FlashMiniForCausalLM:
        payload = _native_payload(path, map_location=map_location)
        saved_config = payload.get("config")
        saved_native: NativeFlashMiniConfig | None = None
        if saved_config is not None:
            if not isinstance(saved_config, Mapping):
                raise TypeError("native checkpoint config must be a mapping")
            saved_native = NativeFlashMiniConfig.from_dict(dict(saved_config))

        saved_version = payload.get("architecture_version")
        if saved_version is not None:
            try:
                saved_version = int(saved_version)
            except (TypeError, ValueError) as exc:
                raise ValueError("native checkpoint has invalid architecture_version") from exc
            if saved_native is not None and saved_version != saved_native.architecture_version:
                raise ValueError(
                    "native checkpoint config architecture mismatch: "
                    f"envelope={saved_version}, config={saved_native.architecture_version}"
                )

        config_value = config if config is not None else saved_config
        if config_value is None:
            raise ValueError("native checkpoint has no config; pass config explicitly")
        hf_config = _config_from_any(config_value)
        requested_native = hf_config.to_native_config()
        if saved_version is not None and saved_version != requested_native.architecture_version:
            raise ValueError(
                "native checkpoint architecture mismatch: "
                f"checkpoint={saved_version}, config={requested_native.architecture_version}"
            )
        if saved_native is not None:
            # Compare the same version-aware canonical serialization used by
            # native checkpoints.  v3 intentionally omits fields derived from
            # ``ngram_vocab_size_base``/``embed_dim`` and v2 omits legacy
            # compatibility fields; comparing fully expanded dataclasses would
            # reject a checkpoint against its own reconstructed config.
            mismatch = _config_mismatch(saved_native.to_dict(), requested_native.to_dict())
            if mismatch is not None:
                raise ValueError(
                    f"native checkpoint config mismatch at {mismatch}; only ple.offload may differ"
                )

        model = cls(hf_config)
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if state is None:
            # A bare state dict is accepted when a config was supplied.
            state = payload
        model.load_native_state_dict(state, strict=strict)
        return model

    def save_native_checkpoint(
        self,
        path: str | Path,
        *,
        step: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Export weights in the native checkpoint envelope without optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        native = self.model.config
        payload = {
            "step": int(step),
            "model_state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer_state_dict": None,
            "config": native.to_dict(),
            "architecture_version": int(native.architecture_version),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)


def export_native_checkpoint(
    model: FlashMiniForCausalLM,
    path: str | Path,
    *,
    step: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Functional alias for :meth:`FlashMiniForCausalLM.save_native_checkpoint`."""
    model.save_native_checkpoint(path, step=step, extra=extra)


def register_flashmini_auto_classes() -> None:
    """Register local Auto mappings for this Python process."""
    try:
        AutoConfig.register(FlashMiniConfig.model_type, FlashMiniConfig, exist_ok=True)
    except TypeError:  # Transformers releases before ``exist_ok``.
        try:
            AutoConfig.register(FlashMiniConfig.model_type, FlashMiniConfig)
        except ValueError:
            pass
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(FlashMiniConfig, FlashMiniForCausalLM, exist_ok=True)
    except TypeError:
        try:
            AutoModelForCausalLM.register(FlashMiniConfig, FlashMiniForCausalLM)
        except ValueError:
            pass
    except ValueError:
        pass


FlashMiniConfig.register_for_auto_class("AutoConfig")
FlashMiniForCausalLM.register_for_auto_class("AutoModelForCausalLM")
register_flashmini_auto_classes()


__all__ = [
    "FlashMiniCausalLMOutput",
    "FlashMiniConfig",
    "FlashMiniForCausalLM",
    "export_native_checkpoint",
    "register_flashmini_auto_classes",
]
