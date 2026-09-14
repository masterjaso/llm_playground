"""FlashMini configuration dataclasses.

All model dimensions are generated from an explicit parameter-budget calculator
(see accounting.py) rather than guessed. This module holds the frozen config
schema used across every A/B/C and scale run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


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
    d_ffn: int = 512  # internal gating MLP hidden dim
    chunk_size: int = 64  # chunked parallel recurrence block size
    use_short_conv: bool = True
    short_conv_kernel: int = 4


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
    table_size: Optional[int] = None
    injection_layer: int = 1
    eos_id: Optional[int] = None
    gate_init: float = 0.1
    sparse: bool = True
    # CPU keeps both table and sparse optimizer state off the accelerator.
    offload: str = "gpu"
    # Reserved for future caching; only 1.0 (no cache policy) is supported.
    working_set_fraction: float = 1.0

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
    attention_layers: Optional[list[int]] = None
    architecture_version: int = 2

    def __post_init__(self) -> None:
        if self.architecture_version != 2:
            raise ValueError("Legacy FlashMini checkpoints used noncausal mixing; start a fresh v2 run")
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
        if self.use_ple and self.ple.injection_layer >= self.num_layers:
            if self.num_layers == 1 and self.ple.injection_layer == 1:
                self.ple.injection_layer = 0
            else:
                raise ValueError("PLE injection_layer must address an existing block")

    def is_attention_layer(self, idx: int) -> bool:
        return idx in self.attention_layers

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FlashMiniConfig":
        d = dict(d)
        d["moe"] = MoEConfig(**d.get("moe", {}))
        d["gdn"] = GatedDeltaNetConfig(**d.get("gdn", {}))
        d["ple"] = PLEConfig(**d.get("ple", {}))
        return cls(**d)
