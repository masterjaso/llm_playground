"""FlashMini full model — 3 GatedDeltaNet : 1 full attention, MoE, optional PLE.

Structure per block group:
  - 3 GatedDeltaNet layers (with MoE + HyperConnection)
  - 1 full causal attention layer (with MoE + HyperConnection)
repeated through the decoder.

PLE is injected at a configurable early block, after contextual mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FlashMiniConfig
from .gated_delta_net import GatedDeltaNet
from .hyperconnection import HyperConnection
from .moe import MoE
from .ple import PLE


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, 3 * num_heads * head_dim, bias=False)
        self.out = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.out(out)


class Block(nn.Module):
    """One decoder block: attention-or-GDN + MoE + HyperConnection."""

    def __init__(self, config: FlashMiniConfig, is_attention: bool):
        super().__init__()
        d = config.d_model
        self.norm1 = nn.LayerNorm(d)
        if is_attention:
            self.mixer = CausalAttention(d, config.num_heads, config.head_dim, config.max_seq_len, config.dropout)
        else:
            self.mixer = GatedDeltaNet(d, config.gdn)
        self.norm2 = nn.LayerNorm(d)
        self.moe = MoE(d, config.moe)
        self.hyper = HyperConnection(d) if config.use_hyperconnection else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        stats: dict = {}
        if self.hyper is not None:
            h = self.hyper(x, self.mixer(self.norm1(x)))
        else:
            h = x + self.mixer(self.norm1(x))
        moe_out, moe_stats = self.moe(self.norm2(h))
        stats.update(moe_stats)
        if self.hyper is not None:
            h = self.hyper(h, moe_out)
        else:
            h = h + moe_out
        return h, stats


class FlashMiniModel(nn.Module):
    def __init__(self, config: FlashMiniConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        # Tied default unit-variance embeddings make initial logits excessively
        # large; use the conventional small LM embedding initialization.
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(config, config.is_attention_layer(i)) for i in range(config.num_layers)]
        )
        self.norm_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie embeddings and head
        self.head.weight = self.embed.weight
        # Construct PLE last so matched seeds preserve shared backbone initialization.
        self.ple = PLE(config.ple) if config.use_ple else None

    def parallelize(self, devices: list[torch.device]) -> FlashMiniModel:
        """Shard consecutive blocks; keep tied embedding/head on the first device.

        This is sequential model parallelism, not replicated data parallelism.
        Place before optimizer construction; CPU PLE rows remain on the host.
        """
        if not devices or len(devices) > len(self.blocks):
            raise ValueError("Need between one and num_layers devices")
        self.embed.to(devices[0])
        self.head.to(devices[0])
        self.norm_f.to(devices[0])
        for i, block in enumerate(self.blocks):
            device = devices[min(i * len(devices) // len(self.blocks), len(devices) - 1)]
            block.to(device)
            if self.ple is not None and i == self.config.ple.injection_layer:
                self.ple.to(device)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        ple_enabled: bool | None = None,
    ) -> dict:
        """Forward pass.

        Returns dict with 'logits' (or 'loss'), and 'stats' (routing/ple stats).
        """
        x = self.embed(input_ids)
        stats: dict = {}
        for i, block in enumerate(self.blocks):
            x = x.to(block.norm1.weight.device)
            if self.ple is not None and i == self.config.ple.injection_layer:
                enabled = True if ple_enabled is None else ple_enabled
                ple_out = self.ple.forward_with_ablation(input_ids.to(x.device), enabled, x)
                stats["ple_active"] = float(enabled)
                stats["ple_scale"] = self.ple.scale.detach().tanh().to(self.embed.weight.device)
                stats["ple_norm_ratio"] = (ple_out.detach().float().square().mean().sqrt()
                                           / x.detach().float().square().mean().sqrt().clamp_min(1e-8)).to(self.embed.weight.device)
                x = x + ple_out
            x, block_stats = block(x)
            for k, v in block_stats.items():
                stats.setdefault(k, []).append(v.to(self.embed.weight.device) if isinstance(v, torch.Tensor) else v)

        x = self.norm_f(x.to(self.embed.weight.device))
        logits = self.head(x)

        result: dict = {"logits": logits, "stats": stats}
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            result["loss"] = loss
        return result
