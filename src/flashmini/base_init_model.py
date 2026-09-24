"""FlashMini v4 production modules and exact meta-model.

The complete 50B model can be constructed on ``meta`` without allocating its
weights.  A scaled configuration using the same class implementations exercises
forward/backward semantics.  Learned tensor names and shapes are authoritative;
runtime-only physical expert fusion is deliberately not represented in names.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base_init_config import FlashMini50BConfig
from .kvc import KVQuantizer


class GroupedRMSNormV4(nn.Module):
    def __init__(self, width: int, groups: int, group_width: int, eps: float = 1e-6):
        super().__init__()
        if width != groups * group_width:
            raise ValueError("grouped RMSNorm dimensions must multiply")
        self.groups, self.group_width, self.eps = groups, group_width, eps
        self.offset = nn.Parameter(torch.zeros(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        grouped = value.reshape(*shape[:-1], self.groups, self.group_width).float()
        grouped = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        return (grouped * (1.0 + self.offset.float()).reshape(self.groups, self.group_width)).to(value.dtype).reshape(shape)


class HyperConnectionV4(nn.Module):
    """Qwen-style four-stream read/write HC with persistent width hc_count*d."""

    def __init__(self, d_model: int, lowrank: int, *, final: bool = False):
        super().__init__()
        self.d_model, self.lowrank, self.final = d_model, lowrank, final
        self.streams = 4
        self.width = self.streams * d_model
        self.norm_offset = nn.Parameter(torch.zeros(self.width))
        self.read_down = nn.Linear(self.width, lowrank, bias=False)
        self.read_up = nn.Linear(lowrank, self.width, bias=False)
        self.write_gate = None if final else nn.Linear(self.width, self.streams, bias=False)

    def normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.width:
            raise ValueError(f"expected HC width {self.width}, got {hidden.shape[-1]}")
        streams = hidden.float().reshape(*hidden.shape[:-1], self.streams, self.d_model)
        normalized = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + 1e-6)
        normalized = normalized * (1.0 + self.norm_offset.float()).reshape(self.streams, self.d_model)
        return normalized.to(hidden.dtype).reshape_as(hidden)

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.streams, self.d_model)
        weights = torch.sigmoid(self.read_up(F.silu(self.read_down(normalized) / self.streams)))
        weights = weights.reshape(*weights.shape[:-1], self.streams, self.d_model)
        return (weights * streams).mean(-2)

    def read_write(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.write_gate is None:
            raise RuntimeError("final HC is read-only")
        normalized = self.normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.streams, self.d_model)
        weights = torch.sigmoid(self.read_up(F.silu(self.read_down(normalized) / self.streams)))
        weights = weights.reshape(*weights.shape[:-1], self.streams, self.d_model)
        mixed = (weights * streams).mean(-2)
        gates = 2.0 * torch.sigmoid(self.write_gate(normalized) / self.streams)
        return mixed, hidden, gates

    def write(self, hidden: torch.Tensor, output: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        return hidden + (output.unsqueeze(-2) * gates.unsqueeze(-1)).reshape_as(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.read(hidden)


class AttentionV4(nn.Module):
    """GQA attention with Qwen-style output gating and optional KVC reuse."""

    def __init__(self, config: FlashMini50BConfig, role: str | None = None):
        super().__init__()
        attention = config.section("attention")
        d = config.d_model
        self.role = role
        self.query_heads = attention["query_heads"]
        self.kv_heads = attention["kv_heads"]
        self.head_dim = attention["head_dim"]
        self.rotary_dim = attention["rotary_dimensions"]
        self.rope_theta = attention["rope_theta"]
        self.q_proj = nn.Linear(d, self.query_heads * self.head_dim * 2, bias=False)
        self.q_norm = GroupedRMSNormV4(self.head_dim, 1, self.head_dim)
        self.o_proj = nn.Linear(self.query_heads * self.head_dim, d, bias=False)
        if role != "reuse":
            self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
            self.k_norm = GroupedRMSNormV4(self.head_dim, 1, self.head_dim)
        else:
            self.k_proj = self.v_proj = self.k_norm = None
        self.quantizer = KVQuantizer() if role == "source" else None

    def _rope(self, value: torch.Tensor) -> torch.Tensor:
        rotary = value[..., : self.rotary_dim]
        passed = value[..., self.rotary_dim:]
        frequency = self.rope_theta ** (-torch.arange(0, self.rotary_dim, 2, device=value.device, dtype=torch.float32) / self.rotary_dim)
        position = torch.arange(value.shape[-2], device=value.device, dtype=torch.float32)
        angle = position[:, None] * frequency[None, :]
        cos, sin = angle.cos()[None, None], angle.sin()[None, None]
        first, second = rotary.float().chunk(2, dim=-1)
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(value.dtype)
        return torch.cat((rotated, passed), dim=-1)

    def forward(self, hidden: torch.Tensor, bank: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, steps, _ = hidden.shape
        query_and_gate = self.q_proj(hidden).reshape(batch, steps, self.query_heads, self.head_dim * 2)
        query, gate = query_and_gate.chunk(2, dim=-1)
        query = self._rope(self.q_norm(query).transpose(1, 2))
        if self.role == "reuse":
            if bank is None:
                raise RuntimeError("KVC reuse attention requires a source bank")
            key, value = bank
        else:
            key = self._rope(self.k_norm(self.k_proj(hidden).reshape(batch, steps, self.kv_heads, self.head_dim)).transpose(1, 2))
            value = self.v_proj(hidden).reshape(batch, steps, self.kv_heads, self.head_dim).transpose(1, 2)
            if self.role == "source":
                bank = self.quantizer.quantize_dequantize(key), self.quantizer.quantize_dequantize(value)
                key, value = bank
        repeats = self.query_heads // self.kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, steps, -1)
        attended = attended * torch.sigmoid(gate).reshape(batch, steps, -1)
        output = self.o_proj(attended)
        return (output, bank) if bank is not None else output


class GatedDeltaNetV4(nn.Module):
    """Production 16-K-head / 32-V-head GDN with stable fp32 recurrence."""

    def __init__(self, config: FlashMini50BConfig, chunk_size: int = 64):
        super().__init__()
        gdn = config.section("gdn")
        d = config.d_model
        self.chunk_size = chunk_size
        self.num_k_heads = gdn["key_query_heads"]
        self.num_v_heads = gdn["value_heads"]
        self.head_k = gdn["key_head_dim"]
        self.head_v = gdn["value_head_dim"]
        self.key_dim = self.num_k_heads * self.head_k
        self.value_dim = self.num_v_heads * self.head_v
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkvz = nn.Linear(d, self.key_dim * 2 + self.value_dim * 2, bias=False)
        self.in_proj_ba = nn.Linear(d, self.num_v_heads * 2, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, gdn["short_conv_kernel"], groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm_offset = nn.Parameter(torch.zeros(self.head_v))
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def _recurrent(self, query, key, value, beta, decay):
        # Stable token-wise reference recurrence. Production kernels may fuse or
        # chunk this recurrence without changing logical heads or parameters.
        batch, steps = query.shape[:2]
        query = F.normalize(query.float(), dim=-1)
        key = F.normalize(key.float(), dim=-1)
        value, beta, decay = value.float(), beta.float(), decay.float()
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        state = query.new_zeros(batch, self.num_v_heads, self.head_k, self.head_v)
        outputs = []
        for step in range(steps):
            previous = torch.einsum("bhk,bhkv->bhv", key[:, step], state)
            update = value[:, step] - decay[:, step].unsqueeze(-1) * previous
            outer = torch.einsum("bhk,bhv->bhkv", key[:, step], update)
            state = decay[:, step].reshape(-1, self.num_v_heads, 1, 1) * state + beta[:, step].reshape(-1, self.num_v_heads, 1, 1) * outer
            outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, step], state))
        return torch.stack(outputs, dim=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = hidden.shape
        mixed = self.in_proj_qkvz(hidden).reshape(batch, steps, self.num_k_heads, 2 * self.head_k + 2 * self.head_v * self.num_v_heads // self.num_k_heads)
        query, key, value, z = torch.split(mixed, [self.head_k, self.head_k, 2 * self.head_v, 2 * self.head_v], dim=-1)
        ba = self.in_proj_ba(hidden).reshape(batch, steps, self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)
        beta_raw, a_raw = torch.split(ba, [self.num_v_heads // self.num_k_heads] * 2, dim=-1)
        beta = beta_raw.reshape(batch, steps, self.num_v_heads).sigmoid()
        a = a_raw.reshape(batch, steps, self.num_v_heads)
        conv_input = torch.cat((query.reshape(batch, steps, self.key_dim), key.reshape(batch, steps, self.key_dim), value.reshape(batch, steps, self.value_dim)), dim=-1)
        conv = F.conv1d(F.pad(conv_input.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0)), self.conv1d.weight, groups=self.conv_dim)[..., :steps].transpose(1, 2)
        query = (query.reshape(batch, steps, self.key_dim) + conv[..., :self.key_dim]).reshape(batch, steps, self.num_k_heads, self.head_k)
        key = (key.reshape(batch, steps, self.key_dim) + conv[..., self.key_dim:2 * self.key_dim]).reshape(batch, steps, self.num_k_heads, self.head_k)
        value = (value.reshape(batch, steps, self.value_dim) + conv[..., 2 * self.key_dim:]).reshape(batch, steps, self.num_v_heads, self.head_v)
        z = z.reshape(batch, steps, self.num_v_heads, self.head_v)
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        recurrent = self._recurrent(query, key, value, beta, decay)
        normalized = recurrent.float() * torch.rsqrt(recurrent.float().square().mean(-1, keepdim=True) + 1e-6)
        normalized = normalized * (1.0 + self.norm_offset.float()) * F.silu(z.float())
        return self.out_proj(normalized.to(hidden.dtype).reshape(batch, steps, self.value_dim))


class SwiGLUExpertV4(nn.Module):
    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class MoEV4(nn.Module):
    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        moe = config.section("moe")
        d, intermediate = config.d_model, moe["expert_intermediate"]
        self.num_experts, self.top_k = moe["routed_experts"], moe["top_k"]
        self.router = nn.Linear(d, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpertV4(d, intermediate) for _ in range(self.num_experts)])
        self.shared_expert = SwiGLUExpertV4(d, intermediate)
        self.shared_gate = nn.Linear(d, 1, bias=False)

    def forward(self, hidden: torch.Tensor):
        batch, steps, width = hidden.shape
        flat = hidden.reshape(-1, width)
        probabilities = torch.softmax(self.router(flat).float(), dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(hidden.dtype)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            slots = (indices == expert_index).nonzero(as_tuple=False)
            if slots.numel() == 0:
                continue
            token_ids, rank_ids = slots[:, 0], slots[:, 1]
            output.index_add_(0, token_ids, expert(flat[token_ids]) * weights[token_ids, rank_ids, None])
        output = output + self.shared_gate(flat).sigmoid() * self.shared_expert(flat)
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
        stats = {
            "router_aux_loss": self.num_experts * (counts / counts.numel() * probabilities.mean(0)).sum(),
            "expert_load": counts / counts.numel(),
            "router_token_count": torch.tensor(float(flat.shape[0])),
        }
        return output.reshape(batch, steps, width), stats


class PLEV4(nn.Module):
    """One PLE injection with independently addressable hash-head tables."""

    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        ple = config.section("ple")
        self.config, self.rows = config, list(ple["hash_head_rows"])
        self.tables = nn.ModuleList([nn.Embedding(rows, ple["head_dim"]) for rows in self.rows])
        hc_width = config.section("hyperconnections")["persistent_residual_width"]
        self.key_proj = nn.Linear(ple["aggregate_width"], hc_width, bias=False)
        self.value_proj = nn.Linear(ple["aggregate_width"], config.d_model, bias=False)
        self.key_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.query_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.conv_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.conv1d = nn.Conv1d(hc_width, hc_width, 4, groups=hc_width, bias=False)
        self.multipliers = nn.Parameter(torch.ones(3, dtype=torch.int64), requires_grad=False)

    def _keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("PLE v4 input_ids must be rank-two int64")
        batch, steps = input_ids.shape
        previous = F.pad(input_ids, (1, 0))
        row_tensor = torch.tensor(self.rows, device=input_ids.device, dtype=torch.long)
        bigram = (previous[:, 1:] * 1_000_003 + input_ids).unsqueeze(-1) % row_tensor[:8]
        prev2 = F.pad(input_ids, (2, 0))[:, :-2]
        trigram = (prev2 * 1_000_003 * 1_000_033 + previous[:, 1:] * 1_000_033 + input_ids).unsqueeze(-1) % row_tensor[8:]
        return torch.cat((bigram, trigram), dim=-1)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return torch.zeros_like(hidden)
        keys = self._keys(input_ids)
        embeddings = torch.cat([table(keys[..., index]) for index, table in enumerate(self.tables)], dim=-1)
        key, query = self.key_norm(self.key_proj(embeddings)), self.query_norm(hidden)
        gate = (key.reshape(*key.shape[:-1], 4, -1) * query.reshape(*query.shape[:-1], 4, -1)).sum(-1, keepdim=True) / math.sqrt(self.config.d_model)
        gated = torch.sigmoid(gate) * self.value_proj(embeddings).unsqueeze(-2)
        width = self.conv1d.in_channels
        conv_input = self.conv_norm(gated.reshape(*gated.shape[:-2], width))
        conv = F.conv1d(F.pad(conv_input.transpose(1, 2), (3, 0)), self.conv1d.weight, groups=width)[..., :input_ids.shape[1]].transpose(1, 2)
        return gated.reshape(*gated.shape[:-2], width) + F.silu(conv)

    def forward_with_ablation(self, input_ids: torch.Tensor, hidden: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        return self.forward(input_ids, hidden, enabled)


class BlockV4(nn.Module):
    def __init__(self, config: FlashMini50BConfig, layer_index: int, *, force_attention: bool = False):
        super().__init__()
        self.layer_index = layer_index
        role = config.kvc_role(layer_index)
        is_attention = force_attention or config.is_attention_layer(layer_index)
        self.mixer = AttentionV4(config, role) if is_attention else GatedDeltaNetV4(config)
        self.moe = MoEV4(config)
        lowrank = config.section("hyperconnections")["lowrank"]
        self.mixer_hc = HyperConnectionV4(config.d_model, lowrank)
        self.moe_hc = HyperConnectionV4(config.d_model, lowrank)
        self.kvc_role = role

    def forward(self, hidden: torch.Tensor, bank=None):
        mixed, residual, gate = self.mixer_hc.read_write(hidden)
        result = self.mixer(mixed, bank) if self.kvc_role == "reuse" else self.mixer(mixed)
        if isinstance(result, tuple):
            result, bank = result
        hidden = self.mixer_hc.write(residual, result, gate)
        mixed, residual, gate = self.moe_hc.read_write(hidden)
        moe, stats = self.moe(mixed)
        return self.moe_hc.write(residual, moe, gate), bank, stats


class LMHeadV4(nn.Module):
    """Untied LM head with explicit [d_model, vocab] parameter orientation."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight


class MTPFusionV4(nn.Module):
    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        d = config.d_model
        width = config.section("hyperconnections")["persistent_residual_width"]
        self.hidden_norm = GroupedRMSNormV4(width, 4, d)
        self.embedding_norm = GroupedRMSNormV4(d, 1, d)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        self.fc_embedding = nn.Linear(d, d, bias=False)

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        stream_hidden = self.fc_hidden(self.hidden_norm(hidden).reshape(*hidden.shape[:-1], 4, self.fc_hidden.in_features))
        stream_embedding = self.fc_embedding(self.embedding_norm(embedding))
        return (stream_hidden + stream_embedding.unsqueeze(-2)).reshape_as(hidden)


class MTPV4(nn.Module):
    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        self.fusion = MTPFusionV4(config)
        self.block = BlockV4(config, -1, force_attention=True)
        self.block.kvc_role = None
        self.block.mixer.role = None
        self.final_hc = HyperConnectionV4(config.d_model, config.section("hyperconnections")["lowrank"], final=True)

    def forward(self, backbone_hidden: torch.Tensor, shifted_embedding: torch.Tensor):
        state = self.fusion(backbone_hidden, shifted_embedding)
        state, _, _ = self.block(state)
        return state

    def collapse(self, state: torch.Tensor) -> torch.Tensor:
        return self.final_hc(state)


class FlashMini50BBaseInit(nn.Module):
    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.ple = PLEV4(config)
        self.blocks = nn.ModuleList([BlockV4(config, index) for index in range(config.num_layers)])
        self.final_hc = HyperConnectionV4(config.d_model, config.section("hyperconnections")["lowrank"], final=True)
        # Store the head in the frozen [d_model, vocab] orientation. Forward
        # uses an explicit matmul; this is not a transposed alias of the input
        # embedding and therefore remains independently checkpointable.
        self.lm_head = LMHeadV4(config.d_model, config.vocab_size)
        self.mtp = MTPV4(config)
        self._initialize_non_meta_parameters()

    @staticmethod
    def _name_seed(name: str, chunk_index: int, global_seed: int = 500_277) -> int:
        payload = f"flashmini-init-v1\0{global_seed}\0{name}\0{chunk_index}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little") % (2**63 - 1)

    def _initialize_non_meta_parameters(self) -> None:
        """Initialize materialized parameters by the InitSpec name/chunk law."""
        chunk_elements = 1_048_576
        for name, parameter in self.named_parameters():
            if parameter.is_meta or not parameter.is_floating_point():
                continue
            flat = parameter.reshape(-1)
            with torch.no_grad():
                for chunk_index, start in enumerate(range(0, flat.numel(), chunk_elements)):
                    stop = min(flat.numel(), start + chunk_elements)
                    generator = torch.Generator(device="cpu").manual_seed(
                        self._name_seed(name, chunk_index)
                    )
                    if name.endswith("A_log"):
                        values = torch.rand(stop - start, generator=generator, dtype=torch.float32)
                        values = (0.01 + values * 15.99).log()
                    elif name.endswith("dt_bias"):
                        values = torch.ones(stop - start, dtype=torch.float32)
                    elif name.endswith(("norm_offset", ".conv1d.weight", ".key_norm.offset", ".query_norm.offset", ".conv_norm.offset")):
                        values = torch.zeros(stop - start, dtype=torch.float32)
                    else:
                        values = torch.randn(stop - start, generator=generator, dtype=torch.float32) * 0.02
                    flat[start:stop].copy_(values.to(parameter.dtype))

    def backbone(self, input_ids: torch.Tensor, *, ple_enabled: bool = True):
        hidden = self.embed_tokens(input_ids).repeat(1, 1, 4)
        banks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        stats: list[dict[str, Any]] = []
        for index, block in enumerate(self.blocks):
            if index == self.config.ple_injection_layer:
                hidden = hidden + self.ple.forward_with_ablation(input_ids, hidden, ple_enabled)
            source = self.config.kvc_source_for(index)
            result, bank, layer_stats = block(hidden, banks.get(source) if source is not None else None)
            hidden = result
            if bank is not None and block.kvc_role == "source":
                banks[index] = bank
            stats.append(layer_stats)
        return hidden, stats

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, *, mtp_window: int = 0, ple_enabled: bool = True):
        if not 0 <= mtp_window <= 4:
            raise ValueError("mtp_window must be in 0..4")
        backbone_hidden, stats = self.backbone(input_ids, ple_enabled=ple_enabled)
        collapsed = self.final_hc(backbone_hidden)
        logits = self.lm_head(collapsed)
        result: dict[str, Any] = {"logits": logits, "stats": stats, "backbone_hidden": backbone_hidden}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if mtp_window and labels is not None:
            mtp_logits = []
            state = backbone_hidden
            for depth in range(1, mtp_window + 1):
                padding = labels[:, :1].expand(-1, depth)
                shifted_ids = torch.cat((padding, labels[:, :-depth]), dim=1)
                shifted = self.embed_tokens(shifted_ids)
                state = self.mtp(state, shifted)
                mtp_logits.append(self.lm_head(self.mtp.collapse(state)))
            result["mtp_logits"] = mtp_logits
            mtp_losses = []
            for depth, value in enumerate(mtp_logits, start=2):
                target = F.pad(labels[:, depth:], (0, depth), value=-100)
                mtp_losses.append(F.cross_entropy(value.reshape(-1, value.shape[-1]), target.reshape(-1), ignore_index=-100))
            result["mtp_loss"] = torch.stack(mtp_losses).mean()
        return result


def surrogate_config(vocab_size: int = 64, d_model: int = 32, num_layers: int = 12, rows: int = 17) -> FlashMini50BConfig:
    """Return a schema-shaped tiny config for semantic tests, not a production config."""
    from copy import deepcopy
    from flashmini.base_init_config import MODEL_ID
    production = FlashMini50BConfig(__import__("flashmini.base_init_config", fromlist=["load_config"]).load_config().raw)
    raw = deepcopy(production.raw)
    raw["tokenizer"]["vocab_size"] = vocab_size
    raw["architecture"].update(d_model=d_model, decoder_layers=num_layers, input_embedding_shape=[vocab_size, d_model], lm_head_shape=[d_model, vocab_size])
    width = 4 * d_model
    head_dim = max(4, d_model // 8)
    q_heads, kv_heads = 4, 2
    raw["attention"].update(query_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, query_width=q_heads * head_dim, kv_projection_width=kv_heads * head_dim, rotary_dimensions=head_dim // 2, rope_theta=10_000)
    raw["gdn"].update(key_query_heads=4, key_head_dim=head_dim, value_heads=8, value_head_dim=head_dim)
    raw["moe"].update(routed_experts=4, top_k=2, expert_intermediate=4 * d_model)
    raw["hyperconnections"].update(persistent_residual_width=width, lowrank=max(4, d_model // 4))
    raw["ple"].update(ngram_vocab_size_base=rows, hash_head_rows=[19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83], aggregate_width=16 * max(1, d_model // 16), head_dim=max(1, d_model // 16))
    raw["mtp"]["moe"] = deepcopy(raw["moe"])
    raw["architecture_contract"] = {"status": "surrogate", "source": MODEL_ID, "changes_require": "test_only", "training_started": False}
    return _SurrogateConfig(raw)


class _SurrogateConfig(FlashMini50BConfig):
    def __init__(self, raw):
        self._raw = raw
        self._surrogate = True

    def _validate(self):
        return None

    def section(self, name):
        import copy
        return copy.deepcopy(self._raw[name])

    @property
    def hyperconnections(self):
        return self._raw["hyperconnections"]

    @property
    def ple_injection_layer(self):
        return self._raw["ple"]["injection_layer_zero_based"]

    def is_attention_layer(self, index):
        return index in self._raw["mixer"]["attention_layers_zero_based"]

    def kvc_role(self, index):
        for source, reuse in self._raw["kvc"]["pairs_zero_based"]:
            if index == source: return "source"
            if index == reuse: return "reuse"
        return None

    def kvc_source_for(self, index):
        for source, reuse in self._raw["kvc"]["pairs_zero_based"]:
            if index == reuse: return source
        return None

    @property
    def vocab_size(self): return self._raw["tokenizer"]["vocab_size"]
    @property
    def d_model(self): return self._raw["architecture"]["d_model"]
    @property
    def num_layers(self): return self._raw["architecture"]["decoder_layers"]


__all__ = [
    "AttentionV4", "BlockV4", "FlashMini50BBaseInit", "GatedDeltaNetV4", "GroupedRMSNormV4",
    "HyperConnectionV4", "MTPV4", "MoEV4", "PLEV4", "SwiGLUExpertV4", "surrogate_config",
]
