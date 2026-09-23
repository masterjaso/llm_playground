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

from ..config import FlashMiniConfig, KVConfig
from ..kvc import KVQuantizer
from .gated_delta_net import GatedDeltaNet
from .hyperconnection import GatedResidual, HyperConnection
from .moe import MoE
from .ple import PLE, PLEV3


class CausalAttention(nn.Module):
    """v3 rotary causal attention with optional KVC source/reuse roles.

    Plain mode: standard QKV projection, RoPE, SDPA — identical to C.

    KVC source: computes K3/V3, applies RoPE to K3, fake-quantizes BOTH K3 and V3
    to the shared low-bit bank, and attends Q3 against the quantized bank. The
    returned bank (dequantized) is the single low-bit tensor used downstream.

    KVC reuse: projects only its Q slice (K and V slices remain allocated but
    inactive, matching C's parameter layout and initialization); the attention
    attends Q7 against the EXTERNAL bank produced by the source layer. This
    preserves matched initialization with C: the full QKV weight tensor is
    present and initialized identically, but only the first 1/3 slice is used.
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int, max_seq_len: int, dropout: float = 0.0,
                 *, architecture_version=2, rope_theta=10000.0, kvc_role: str | None = None,
                 quantizer: KVQuantizer | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.architecture_version = architecture_version
        self.rope_theta = rope_theta
        self.kvc_role = kvc_role  # None, 'source', or 'reuse'
        self.quantizer = quantizer
        self.qkv = nn.Linear(d_model, 3 * num_heads * head_dim, bias=False)
        self.out = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            (torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
             if architecture_version == 2 else torch.empty(0)),
            persistent=False,
        )

    def _rope(self, t: torch.Tensor, T: int, device: torch.device, dtype) -> torch.Tensor:
        """Apply rotary positional embedding to a (B, H, T, D) tensor."""
        frequencies = self.rope_theta ** (-torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
        angles = torch.arange(T, device=device).float()[:, None] * frequencies[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        first, second = t.chunk(2, dim=-1)
        return t * cos + torch.cat((-second, first), dim=-1) * sin

    def forward(self, x: torch.Tensor, *, kvc_bank: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Returns the attention output; in source mode, also the KVC bank."""
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.kvc_role == "reuse":
            # Reuse: Q from own slice, K/V from the external source bank.
            q_rope = self._rope(q, T, x.device, q.dtype)
            bank_k, bank_v = kvc_bank
            out = F.scaled_dot_product_attention(q_rope, bank_k, bank_v, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0)
            return self.out(out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim))

        if self.kvc_role == "source":
            # Source: RoPE on Q and K, fake-quant K and V, attend Q against bank.
            q_rope = self._rope(q, T, x.device, q.dtype)
            k_rope = self._rope(k, T, x.device, k.dtype)
            bank_k = self.quantizer.quantize_dequantize(k_rope)
            bank_v = self.quantizer.quantize_dequantize(v)
            out = F.scaled_dot_product_attention(q_rope, bank_k, bank_v, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0)
            result = self.out(out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim))
            return result, (bank_k, bank_v)

        # Plain v3 path (unchanged from C)
        if self.architecture_version >= 3:
            q, k = self._rope(q, T, x.device, q.dtype), self._rope(k, T, x.device, k.dtype)
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
    """One decoder block: attention-or-GDN + MoE + HyperConnection.

    For KVC, the first attention block (source) also computes and owns the
    shared low-bit bank; the reuse block (second attention block) receives that
    bank as a runtime argument.  The KVC role is resolved from config.kvc_role(i)
    so the model can be constructed as ``Block(config, i)`` without an extra
    flag.  Under KVC, the reuse block's K/V slices are INACTIVE but still
    allocated (same parameter layout / initialization as C).
    """

    def __init__(self, config: FlashMiniConfig, is_attention: bool, kvc_role: str | None = None):
        super().__init__()
        d = config.d_model
        self.architecture_version = config.architecture_version
        self.norm1 = nn.LayerNorm(d) if config.architecture_version == 2 else nn.Identity()
        if is_attention:
            quantizer = None
            if config.kvc.enabled and kvc_role == "source":
                quantizer = KVQuantizer(
                    kv_bits=config.kvc.kv_bits,
                    quant_format=config.kvc.quant_format,
                    scale_format=config.kvc.scale_format,
                    scale_group_size=config.kvc.scale_group_size,
                )
            self.mixer = CausalAttention(
                d, config.num_heads, config.head_dim, config.max_seq_len, config.dropout,
                architecture_version=config.architecture_version, rope_theta=config.rope_theta,
                kvc_role=kvc_role, quantizer=quantizer,
            )
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

    def forward(self, x: torch.Tensor, *, kvc_bank=None) -> tuple[torch.Tensor, dict]:
        stats: dict = {}
        if self.architecture_version >= 3:
            if self.attn_hyper_connection is None or self.mlp_hyper_connection is None:
                raise RuntimeError("v3 block is missing gated residual connections")
            mixed, residual, injection = self.attn_hyper_connection.read_with_injection(x)
            role = getattr(self.mixer, "kvc_role", None)
            if role == "source":
                mixer_out, bank = self.mixer(self.norm1(mixed))
                stats["kvc_bank"] = bank
            elif role == "reuse":
                if kvc_bank is None:
                    raise RuntimeError("KVC reuse block requires the source kvc_bank")
                mixer_out = self.mixer(self.norm1(mixed), kvc_bank=kvc_bank)
            else:
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
            [Block(config, config.is_attention_layer(i), kvc_role=config.kvc_role(i)) for i in range(config.num_layers)]
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

    def parallelize(self, devices: list[torch.device], stage_split: int | None = None) -> FlashMiniModel:
        """Shard consecutive blocks; keep tied embedding/head on the first device.

        This is sequential model parallelism, not replicated data parallelism.
        Place before optimizer construction; CPU PLE rows remain on the host.

        ``stage_split`` pins the pipeline boundary explicitly: blocks
        ``[0, stage_split)`` go to ``devices[0]`` and ``[stage_split, num_layers)``
        to ``devices[1]``.  The default keeps the historical even split.  Only
        device placement changes; parameter identity, shape and names do not, so
        existing checkpoints load unchanged.
        """
        if not devices or len(devices) > len(self.blocks):
            raise ValueError("Need between one and num_layers devices")
        if stage_split is None:
            placement = [
                devices[min(i * len(devices) // len(self.blocks), len(devices) - 1)]
                for i in range(len(self.blocks))
            ]
        else:
            if len(devices) != 2:
                raise ValueError("stage_split requires exactly two model-parallel devices")
            if not 0 < stage_split < len(self.blocks):
                raise ValueError(
                    "stage_split must leave at least one block on each device"
                )
            placement = [devices[0]] * stage_split + [devices[1]] * (len(self.blocks) - stage_split)
        self.embed.to(devices[0])
        self.head.to(devices[0])
        self.norm_f.to(devices[0])
        if self.final_hyper_connection is not None:
            self.final_hyper_connection.to(devices[0])
        for i, block in enumerate(self.blocks):
            block.to(placement[i])
            if self.ple is not None and i == self.config.ple.injection_layer:
                self.ple.to(placement[i])
        self.block_devices = list(placement)
        # First block index owned by a device other than the first block's.
        self.stage_split = next(
            (i for i, device in enumerate(placement) if device != placement[0]),
            len(placement),
        )
        return self

    def _ple_injection(self, x, input_ids, ple_enabled, stats):
        """Inject PLE at its configured layer and record its statistics."""
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
        return x + ple_out

    def _run_blocks(self, x, start, end, *, input_ids=None, kvc_bank=None, ple_enabled=None,
                    stats=None, stats_device=None):
        """Run blocks ``[start, end)`` in order, carrying the KVC bank.

        With ``stats_device=None`` every statistic stays on the device that
        produced it, which is what the overlapped pipeline wants: moving each
        one across the stage boundary is a synchronization cost with no benefit,
        because aggregation happens on device tensors anyway.
        """
        stats = {} if stats is None else stats
        for i in range(start, end):
            block = self.blocks[i]
            x = x.to(next(block.parameters()).device)
            if self.ple is not None and i == self.config.ple.injection_layer:
                x = self._ple_injection(x, input_ids, ple_enabled, stats)
            x, block_stats = block(x, kvc_bank=kvc_bank)
            new_bank = block_stats.get("kvc_bank")
            if new_bank is not None:
                kvc_bank = new_bank  # pass through to the reuse block
            for k, v in block_stats.items():
                if k == "kvc_bank":
                    continue
                if stats_device is not None and isinstance(v, torch.Tensor):
                    v = v.to(stats_device)
                stats.setdefault(k, []).append(v)
            # Carrying the shared K/V bank to the next block's device preserves
            # the autograd graph (the bank is an active compute node, not an
            # inert constant). A stage boundary transfers it exactly once, in
            # the pipeline executor, not here.
            if kvc_bank is not None and i + 1 < end:
                nxt = next(self.blocks[i + 1].parameters()).device
                if kvc_bank[0].device != nxt:
                    kvc_bank = (kvc_bank[0].to(nxt), kvc_bank[1].to(nxt))
        return x, kvc_bank, stats

    def pipeline_stage0(self, input_ids, *, stage_end=None, ple_enabled=None,
                        stats_device=None) -> dict:
        """First pipeline stage: embedding, PLE injection, blocks ``[0, stage_end)``.

        Returns the hidden activation, the KVC bank when the boundary sits after
        the source layer, and this stage's routing/PLE statistics.  Everything
        needed by the next stage is present in the mapping; nothing is
        transferred implicitly.
        """
        stage_end = len(self.blocks) if stage_end is None else stage_end
        x = self.embed(input_ids)
        if self.config.architecture_version >= 3:
            x = x.repeat(1, 1, self.config.hc_count)
        x, kvc_bank, stats = self._run_blocks(
            x, 0, stage_end, input_ids=input_ids, ple_enabled=ple_enabled,
            stats={}, stats_device=stats_device,
        )
        return {"hidden": x, "kvc_bank": kvc_bank, "stats": stats}

    def pipeline_stage1(self, hidden, *, stage_start, stage_end=None, kvc_bank=None,
                        input_ids=None, ple_enabled=None, stats=None,
                        stats_device=None) -> dict:
        """Middle pipeline stage: blocks ``[stage_start, stage_end)``."""
        stage_end = len(self.blocks) if stage_end is None else stage_end
        x, kvc_bank, stats = self._run_blocks(
            hidden, stage_start, stage_end, input_ids=input_ids, kvc_bank=kvc_bank,
            ple_enabled=ple_enabled, stats=stats, stats_device=stats_device,
        )
        return {"hidden": x, "kvc_bank": kvc_bank, "stats": stats}

    def pipeline_output(self, hidden) -> dict:
        """Output stage: final residual combination, normalization, tied head."""
        x = hidden.to(self.embed.weight.device)
        if self.final_hyper_connection is not None:
            x = self.final_hyper_connection(x)
        x = self.norm_f(x)
        return {"logits": self.head(x)}

    @staticmethod
    def pipeline_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over a flattened logit/label pair (mean over tokens)."""
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        ple_enabled: bool | None = None,
    ) -> dict:
        """Forward pass.

        Returns dict with 'logits' (or 'loss'), and 'stats' (routing/ple stats).
        """
        stats_device = self.embed.weight.device
        staged = self.pipeline_stage0(
            input_ids, ple_enabled=ple_enabled, stats_device=stats_device
        )
        result: dict = {
            "logits": self.pipeline_output(staged["hidden"])["logits"],
            "stats": staged["stats"],
        }
        if labels is not None:
            result["loss"] = self.pipeline_loss(result["logits"], labels)
        return result

