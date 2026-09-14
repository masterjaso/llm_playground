"""FlashMini configuration dataclasses.

All model dimensions are generated from an explicit parameter-budget calculator
(see accounting.py) rather than guessed. This module holds the frozen config
schema used across every A/B/C and scale run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class MoEConfig:
    """Sparse top-k MoE configuration."""

    num_experts: int = 16
    top_k: int = 2
    shared_experts: int = 1
    expert_intermediate: int = 512  # per-expert FFN hidden dim
    aux_loss_coef: float = 0.01
    capacity_factor: float = 1.0  # 1.0 = no capacity limit (token choice)


@dataclass
class GatedDeltaNetConfig:
    """GatedDeltaNet linear-attention layer configuration."""

    d_state: int = 64  # recurrent state dimension
    d_ffn: int = 512  # unused v2 compatibility field; omitted from v3 serialization
    chunk_size: int = 64  # chunked parallel recurrence block size
    use_short_conv: bool = True
    short_conv_kernel: int = 4
    # v2 kept a residual shortcut inside the mixer.  v3 moves residual
    # ownership to the block's gated-residual path.  ``None`` is resolved by
    # FlashMiniConfig so direct v2 callers keep their historical behavior.
    residual_in_mixer: bool | None = None


@dataclass
class PLEConfig:
    """PLE / Engram n-gram associative memory configuration."""

    enabled: bool = False
    ngram: int = 3  # n-gram order
    vocab_size: int = 32768
    d_model: int = 256
    num_heads: int = 4
    head_dim: int = 64
    # Total hash heads across orders 2..ngram, each with a separate table.
    table_size: int | None = None
    injection_layer: int = 1
    eos_id: int | None = None
    gate_init: float = 0.1
    sparse: bool = True
    # CPU keeps both table and sparse optimizer state off the accelerator.
    offload: str = "gpu"
    # Reserved for future caching; only 1.0 (no cache policy) is supported.
    working_set_fraction: float = 1.0
    # v3 follows Qwen's heads-per-order layout: one group for every order
    # rather than alternating orders across a single head list.  The fields
    # below are ignored by the v2 implementation and are omitted from v2
    # checkpoint config serialization.
    heads_per_ngram: int = 8
    embed_dim: int | None = None
    ngram_vocab_size_base: int | None = None
    hash_seed: int = 1234
    conv_kernel_size: int = 4
    conv_dilation: int | None = None
    architecture_version: int = 2

    def __post_init__(self) -> None:
        if self.ngram < 2 or self.num_heads < self.ngram - 1:
            raise ValueError("PLE needs at least one hash head per n-gram order (2..ngram)")
        if self.head_dim <= 0 or self.vocab_size <= 0 or self.d_model <= 0:
            raise ValueError("PLE dimensions must be positive")
        if self.table_size is not None and self.table_size < 3:
            raise ValueError("PLE table_size must be at least 3")
        if self.offload not in ("cpu", "gpu"):
            raise ValueError("PLE supports cpu/gpu storage; NVMe offload is not implemented")
        if self.working_set_fraction != 1.0:
            raise ValueError("PLE caching is not implemented; working_set_fraction must be 1.0")
        if not 0 < self.gate_init <= 1 or self.injection_layer < 0:
            raise ValueError("PLE gate_init must be in (0,1] and injection_layer nonnegative")
        if self.heads_per_ngram <= 0:
            raise ValueError("PLE heads_per_ngram must be positive")
        if self.embed_dim is not None and self.embed_dim <= 0:
            raise ValueError("PLE embed_dim must be positive")
        if self.ngram_vocab_size_base is not None and self.ngram_vocab_size_base < 3:
            raise ValueError("PLE ngram_vocab_size_base must be at least 3")
        if self.conv_kernel_size <= 0:
            raise ValueError("PLE conv_kernel_size must be positive")
        if self.conv_dilation is not None and self.conv_dilation <= 0:
            raise ValueError("PLE conv_dilation must be positive")
        if self.architecture_version not in (2, 3):
            raise ValueError("PLE architecture_version must be 2 or 3")


@dataclass
class FlashMiniConfig:
    """Top-level FlashMini model configuration."""

    vocab_size: int = 32768
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    head_dim: int = 64
    # 3 GatedDeltaNet : 1 full attention, repeated through decoder
    gdn_per_attention: int = 3
    max_seq_len: int = 2048
    dropout: float = 0.0
    use_hyperconnection: bool = True
    use_ple: bool = False
    moe: MoEConfig = field(default_factory=MoEConfig)
    gdn: GatedDeltaNetConfig = field(default_factory=GatedDeltaNetConfig)
    ple: PLEConfig = field(default_factory=PLEConfig)
    # Which layers are full attention (indices into layer list)
    attention_layers: list[int] | None = None
    architecture_version: int = 2
    # Screening permits cheap toy probes.  Decisive runs are checked against
    # the frozen-corpus contract by the training/data harness.
    experiment_mode: str = "screening"
    # Reduced v3 gated-residual dimensions.  ``hc_lowrank`` defaults to a
    # quarter of d_model when omitted, while official configs pin it.
    hc_count: int = 4
    hc_lowrank: int | None = None
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.architecture_version not in (2, 3):
            raise ValueError("architecture_version must be 2 or 3")
        if self.experiment_mode not in ("screening", "decisive"):
            raise ValueError("experiment_mode must be screening or decisive")
        if self.hc_count <= 0:
            raise ValueError("hc_count must be positive")
        if self.hc_lowrank is not None and self.hc_lowrank <= 0:
            raise ValueError("hc_lowrank must be positive")
        if self.architecture_version == 3:
            if self.head_dim % 2 or self.rope_theta <= 1:
                raise ValueError("v3 rotary attention requires even head_dim and rope_theta > 1")
            if self.gdn.d_ffn != 512:
                raise ValueError("gdn.d_ffn is a legacy unused field; omit it for v3")
            if self.moe.capacity_factor != 1.0:
                raise ValueError("MoE capacity limiting is unsupported; capacity_factor must be 1")
            if self.ple.gate_init != 0.1:
                raise ValueError("ple.gate_init is a legacy scalar-scale field; omit it for v3")
            if not self.use_hyperconnection:
                raise ValueError("v3 requires four-stream gated residual connections")
            if self.hc_count != 4:
                raise ValueError("v3 currently requires four residual streams")
            if self.hc_lowrank is None:
                self.hc_lowrank = max(1, self.d_model // 4)
            # The v3 block owns the residual around GDN.  Resolve the field so
            # it is explicit in manifests/checkpoints, but keep v2 omitted.
            if self.gdn.residual_in_mixer is None:
                self.gdn.residual_in_mixer = False
            elif self.gdn.residual_in_mixer:
                raise ValueError("v3 GDN residual_in_mixer must be false")
            self.ple.architecture_version = 3
            self.ple.hc_count = self.hc_count
            if self.ple.conv_dilation is None:
                self.ple.conv_dilation = self.ple.ngram
            self.ple.num_heads = (self.ple.ngram - 1) * self.ple.heads_per_ngram
            if self.ple.embed_dim is None:
                self.ple.embed_dim = self.ple.num_heads * self.ple.head_dim
            if self.ple.embed_dim % self.ple.num_heads:
                raise ValueError("PLE embed_dim must divide across heads_per_ngram and orders")
            self.ple.head_dim = self.ple.embed_dim // self.ple.num_heads
            if self.ple.ngram_vocab_size_base is None:
                self.ple.ngram_vocab_size_base = self.ple.table_size or self.vocab_size
        else:
            if self.gdn.residual_in_mixer is None:
                self.gdn.residual_in_mixer = True
            self.ple.architecture_version = 2
        if self.attention_layers is None:
            if self.gdn_per_attention <= 0:
                # Control variant: all layers are full attention (no GDN)
                self.attention_layers = list(range(self.num_layers))
            else:
                # 3 GDN : 1 attention pattern
                self.attention_layers = list(
                    range(self.gdn_per_attention, self.num_layers, self.gdn_per_attention + 1)
                )
        self.ple.enabled = self.use_ple
        self.ple.d_model = self.d_model
        self.ple.vocab_size = self.vocab_size
        self.ple.__post_init__()
        if self.use_ple and self.ple.injection_layer >= self.num_layers:
            if self.num_layers == 1 and self.ple.injection_layer == 1:
                self.ple.injection_layer = 0
            else:
                raise ValueError("PLE injection_layer must address an existing block")

    def is_attention_layer(self, idx: int) -> bool:
        return idx in self.attention_layers

    def to_dict(self) -> dict:
        values = dataclasses.asdict(self)
        if self.architecture_version == 3:
            values["gdn"].pop("d_ffn")
            for key in ("num_heads", "head_dim", "table_size", "gate_init", "working_set_fraction"):
                values["ple"].pop(key)
        if self.architecture_version == 2:
            # Keep historical v2 checkpoint envelopes loadable when they were
            # written before v3-only schema fields existed.
            values.pop("experiment_mode", None)
            values.pop("hc_count", None)
            values.pop("hc_lowrank", None)
            values.pop("rope_theta", None)
            values["gdn"].pop("residual_in_mixer", None)
            for key in (
                "heads_per_ngram",
                "embed_dim",
                "ngram_vocab_size_base",
                "hash_seed",
                "conv_kernel_size",
                "conv_dilation",
                "architecture_version",
            ):
                values["ple"].pop(key, None)
        return values

    @classmethod
    def from_dict(cls, d: dict) -> FlashMiniConfig:
        d = dict(d)
        d["moe"] = MoEConfig(**d.get("moe", {}))
        d["gdn"] = GatedDeltaNetConfig(**d.get("gdn", {}))
        d["ple"] = PLEConfig(**d.get("ple", {}))
        return cls(**d)
