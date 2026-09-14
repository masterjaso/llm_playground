"""Parameter and FLOP accounting for FlashMini models.

Provides:
- count_parameters: total / core / embedding-head / PLE / active-per-token counts.
- estimate_flops_per_token: forward FLOPs/token estimate.
- ParameterBudget: explicit budget calculator that derives model dimensions from a
  target main/core parameter count (so dimensions are never guessed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .config import FlashMiniConfig


@dataclass
class ParameterCounts:
    total: int
    core: int  # backbone (attention + MoE + GDN + norms), excluding embeddings/head/PLE
    embedding_head: int
    ple: int
    active_per_token: int
    num_experts: int
    experts_per_token: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "core": self.core,
            "embedding_head": self.embedding_head,
            "ple": self.ple,
            "active_per_token": self.active_per_token,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
        }


def count_parameters(model: nn.Module, config: FlashMiniConfig) -> ParameterCounts:
    """Count parameters by category.

    Uses module name prefixes to classify: 'embed'/'head' for embeddings,
    'ple' for PLE, everything else is core.
    """
    total = 0
    core = 0
    embedding_head = 0
    ple = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if name.startswith("embed") or name.startswith("head"):
            embedding_head += n
        elif name.startswith("ple"):
            ple += n
        else:
            core += n

    # The tied LM head uses its whole matrix, once. PLE retrieves one row/head;
    # its remaining dense projections/norms/gate participate on every token.
    active_ple = ple
    if getattr(model, "ple", None) is not None:
        table = model.ple.value_embed.weight
        active_ple = ple - table.numel() + config.ple.num_heads * config.ple.head_dim
    active = embedding_head + active_ple
    # Count active core params by inspecting module structure.
    active_core = _count_active_core(model, config)
    active += active_core

    return ParameterCounts(
        total=total,
        core=core,
        embedding_head=embedding_head,
        ple=ple,
        active_per_token=active,
        num_experts=config.moe.num_experts,
        experts_per_token=config.moe.top_k + config.moe.shared_experts,
    )


def _count_active_core(model: nn.Module, config: FlashMiniConfig) -> int:
    """Count core parameters active per token (MoE experts prorated by top_k)."""
    active = 0
    for name, module in model.named_modules():
        if name.endswith("moe"):
            # MoE: router + shared experts always active; routed experts prorated
            for pname, p in module.named_parameters():
                if "router" in pname or "shared" in pname:
                    active += p.numel()
                elif "experts" in pname:
                    frac = config.moe.top_k / config.moe.num_experts
                    active += int(p.numel() * frac)
        elif name.endswith("ple"):
            continue  # counted separately
        else:
            # Non-MoE, non-PLE leaf params are always active
            if not any(c in name for c in ("moe", "ple")) and name not in ("embed", "head"):
                for p in module.parameters(recurse=False):
                    active += p.numel()
    return active


def estimate_flops_per_token(config: FlashMiniConfig) -> float:
    """Forward matrix-arithmetic estimate at max_seq_len, 2 FLOPs/MAC.

    Includes dense attention score/value matmuls (even masked positions), chunked
    delta recurrence, active expert matmuls and both PLE projections. Excludes
    softmax/norms/activations, hashing, transfers, and backward/optimizer work.
    This is analytical accounting, not measured hardware work or throughput.
    """
    d = config.d_model
    v = config.vocab_size
    n_layers = config.num_layers
    n_attn = len(config.attention_layers)
    n_gdn = n_layers - n_attn

    # Embedding + head
    flops = 2.0 * v * d  # embedding lookup is ~free; head projection 2*v*d

    width = config.num_heads * config.head_dim
    attn_flops = 8.0 * d * width + 4.0 * config.max_seq_len * width
    flops += n_attn * attn_flops

    ds = config.gdn.d_state
    chunk = config.gdn.chunk_size
    # q/k/v/gate/out + beta/decay; short convolution; triangular chunk solve.
    gdn_flops = 10.0 * d * ds + 4.0 * d
    if config.gdn.use_short_conv:
        gdn_flops += 2.0 * d * ds * config.gdn.short_conv_kernel
    gdn_flops += 6.0 * ds * ds + 7.0 * chunk * ds
    flops += n_gdn * gdn_flops

    # MoE: active experts only
    moe_flops = 0.0
    for _ in range(n_layers):
        active_experts = config.moe.top_k + config.moe.shared_experts
        moe_flops += 2.0 * d * config.moe.expert_intermediate * 2 * active_experts
    flops += moe_flops
    flops += n_layers * 2.0 * d * config.moe.num_experts
    if config.use_ple:
        flops += 4.0 * d * config.ple.num_heads * config.ple.head_dim

    return flops


@dataclass
class ParameterBudget:
    """Explicit budget calculator: derive dims from a target main/core param count."""

    target_core: int
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    shared_experts: int
    expert_intermediate: int
    d_state: int
    gdn_per_attention: int
    use_ple: bool
    ple_ngram: int
    ple_heads: int
    ple_head_dim: int
    _config: FlashMiniConfig | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: FlashMiniConfig) -> "ParameterBudget":
        return cls(
            target_core=0,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_experts=config.moe.num_experts,
            top_k=config.moe.top_k,
            shared_experts=config.moe.shared_experts,
            expert_intermediate=config.moe.expert_intermediate,
            d_state=config.gdn.d_state,
            gdn_per_attention=config.gdn_per_attention,
            use_ple=config.use_ple,
            ple_ngram=config.ple.ngram,
            ple_heads=config.ple.num_heads,
            ple_head_dim=config.ple.head_dim,
            _config=config,
        )

    def core_params(self) -> int:
        """Count the actual module shapes on meta without allocating weight storage."""
        return self._counts().core

    def _counts(self) -> ParameterCounts:
        from .config import GatedDeltaNetConfig, MoEConfig, PLEConfig
        from .models import FlashMiniModel

        config = self._config or FlashMiniConfig(
            vocab_size=self.vocab_size, d_model=self.d_model, num_layers=self.num_layers,
            num_heads=self.num_heads, head_dim=self.head_dim,
            gdn_per_attention=self.gdn_per_attention, use_ple=self.use_ple,
            moe=MoEConfig(num_experts=self.num_experts, top_k=self.top_k,
                          shared_experts=self.shared_experts,
                          expert_intermediate=self.expert_intermediate),
            gdn=GatedDeltaNetConfig(d_state=self.d_state),
            ple=PLEConfig(ngram=self.ple_ngram, num_heads=self.ple_heads,
                          head_dim=self.ple_head_dim))
        with torch.device("meta"):
            return count_parameters(FlashMiniModel(config), config)

    def gdn_ffn_hidden(self) -> int:
        return max(64, self.d_model * 2)

    def ple_params(self) -> int:
        return self._counts().ple

    def total_params(self) -> int:
        emb = self.vocab_size * self.d_model
        return self.core_params() + emb + self.ple_params()

    def report(self) -> dict[str, Any]:
        return {
            "target_core": self.target_core,
            "estimated_core": self.core_params(),
            "estimated_embedding_head": self.vocab_size * self.d_model,
            "estimated_ple": self.ple_params(),
            "estimated_total": self.total_params(),
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "shared_experts": self.shared_experts,
            "expert_intermediate": self.expert_intermediate,
            "d_state": self.d_state,
            "gdn_per_attention": self.gdn_per_attention,
        }
