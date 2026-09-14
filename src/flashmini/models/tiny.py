"""FlashMini full model — 3 GatedDeltaNet : 1 full attention, MoE, optional PLE.

Structure per block group:
  - 3 GatedDeltaNet layers (with MoE + HyperConnection)
  - 1 full causal attention layer (with MoE + HyperConnection)
repeated through the decoder.

PLE is injected at a configurable early block, after contextual mixing.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from ..config import FlashMiniConfig
from .gated_delta_net import GatedDeltaNet
from .hyperconnection import GatedResidual, HyperConnection
from .moe import MoE
from .ple import PLE, PLEV3


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, max_seq_len: int, dropout: float = 0.0,
                 *, architecture_version=2, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.architecture_version = architecture_version
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(d_model, 3 * num_heads * head_dim, bias=False)
        self.out = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            (torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
             if architecture_version == 2 else torch.empty(0)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.architecture_version >= 3:
            frequencies = self.rope_theta ** (-torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim)
            angles = torch.arange(T, device=x.device).float()[:, None] * frequencies[None, :]
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
            def rotate(t):
                first, second = t.chunk(2, dim=-1)
                return torch.cat((-second, first), dim=-1)
            q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0)
            return self.out(out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim))
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
        self.architecture_version = config.architecture_version
        self.norm1 = nn.LayerNorm(d) if config.architecture_version == 2 else nn.Identity()
        if is_attention:
            self.mixer = CausalAttention(d, config.num_heads, config.head_dim, config.max_seq_len, config.dropout,
                architecture_version=config.architecture_version, rope_theta=config.rope_theta)
        else:
            self.mixer = GatedDeltaNet(
                d,
                config.gdn,
                residual_in_mixer=(config.architecture_version < 3),
            )
        self.norm2 = nn.LayerNorm(d) if config.architecture_version == 2 else nn.Identity()
        self.moe = MoE(d, config.moe)
        if config.architecture_version >= 3:
            self.hyper = None
            self.attn_hyper_connection = GatedResidual(
                d,
                hc_count=config.hc_count,
                hc_lowrank=config.hc_lowrank,
            )
            self.mlp_hyper_connection = GatedResidual(
                d,
                hc_count=config.hc_count,
                hc_lowrank=config.hc_lowrank,
            )
        else:
            self.hyper = HyperConnection(d) if config.use_hyperconnection else None
            self.attn_hyper_connection = None
            self.mlp_hyper_connection = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        stats: dict = {}
        if self.architecture_version >= 3:
            if self.attn_hyper_connection is None or self.mlp_hyper_connection is None:
                raise RuntimeError("v3 block is missing gated residual connections")
            mixed, residual, injection = self.attn_hyper_connection.read_with_injection(x)
            mixer_out = self.mixer(self.norm1(mixed))
            h = self.attn_hyper_connection.write(residual, mixer_out, injection)
            mixed, residual, injection = self.mlp_hyper_connection.read_with_injection(h)
            moe_out, moe_stats = self.moe(self.norm2(mixed))
            stats.update(moe_stats)
            h = self.mlp_hyper_connection.write(residual, moe_out, injection)
            return h, stats
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
        self.norm_f = nn.LayerNorm(config.d_model) if config.architecture_version == 2 else nn.Identity()
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie embeddings and head
        self.head.weight = self.embed.weight
        if config.architecture_version >= 3:
            self.final_hyper_connection = GatedResidual(
                config.d_model,
                hc_count=config.hc_count,
                hc_lowrank=config.hc_lowrank,
                use_combine=False,
            )
        else:
            self.final_hyper_connection = None
        if config.architecture_version >= 3:
            self._initialize_v3(self)
        # Construct PLE last so matched seeds preserve shared backbone initialization.
        if config.use_ple:
            if config.architecture_version >= 3:
                # PLE allocation must not shift the shared dropout RNG trajectory.
                rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
                with torch.random.fork_rng(devices=rng_devices):
                    self.ple = PLEV3(config.ple)
            else:
                self.ple = PLE(config.ple)
        else:
            self.ple = None
        if config.architecture_version >= 3 and self.ple is not None:
            self._initialize_v3(self.ple, prefix="ple.")
            nn.init.zeros_(self.ple.conv1d.weight)

    @staticmethod
    def _initialize_v3(root, prefix=""):
        # Name-keyed streams keep A/B common parameters as well as B/C identical,
        # even when construction consumes a different number of random values.
        seen = set()
        for name, module in root.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                if module.weight.is_meta:
                    continue
                if id(module.weight) in seen:
                    continue
                seen.add(id(module.weight))
                digest = hashlib.sha256(f"{torch.initial_seed()}:{prefix}{name}".encode()).digest()
                generator = torch.Generator(device=module.weight.device).manual_seed(
                    int.from_bytes(digest[:8], "little") % (2**63 - 1))
                nn.init.normal_(module.weight, std=0.02, generator=generator)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    nn.init.zeros_(bias)

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
        if self.final_hyper_connection is not None:
            self.final_hyper_connection.to(devices[0])
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
        if self.config.architecture_version >= 3:
            x = x.repeat(1, 1, self.config.hc_count)
        stats: dict = {}
        for i, block in enumerate(self.blocks):
            x = x.to(next(block.parameters()).device)
            if self.ple is not None and i == self.config.ple.injection_layer:
                enabled = True if ple_enabled is None else ple_enabled
                ple_out = self.ple.forward_with_ablation(input_ids.to(x.device), enabled, x)
                stats["ple_active"] = float(enabled)
                ple_scale = getattr(self.ple, "scale", None)
                if ple_scale is None:
                    ple_scale = torch.ones(1, device=self.embed.weight.device, dtype=x.dtype)
                else:
                    ple_scale = ple_scale.detach().tanh().to(self.embed.weight.device)
                stats["ple_scale"] = ple_scale
                stats["ple_norm_ratio"] = (ple_out.detach().float().square().mean().sqrt()
                                           / x.detach().float().square().mean().sqrt().clamp_min(1e-8)).to(self.embed.weight.device)
                x = x + ple_out
            x, block_stats = block(x)
            for k, v in block_stats.items():
                stats.setdefault(k, []).append(v.to(self.embed.weight.device) if isinstance(v, torch.Tensor) else v)

        if self.final_hyper_connection is not None:
            x = self.final_hyper_connection(x.to(self.embed.weight.device))
        x = self.norm_f(x.to(self.embed.weight.device))
        logits = self.head(x)

        result: dict = {"logits": logits, "stats": stats}
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            result["loss"] = loss
        return result
