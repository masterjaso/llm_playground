"""FlashMini v4 production modules and exact meta-model.

The complete 50B model can be constructed on ``meta`` without allocating its
weights.  A scaled configuration using the same class implementations exercises
forward/backward semantics.  Learned tensor names and shapes are authoritative;
runtime-only physical expert fusion is deliberately not represented in names.

PLE hash-head tables are held by :class:`~flashmini.v4_ple_store.PLETableStore`
(host memory, sparse row transfer) rather than as ``nn.Parameter`` objects; their
logical names ``ple.tables.<h>.weight`` are reported by
:meth:`FlashMini50BBaseInit.named_logical_tensors`.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from . import v4_init
from .base_init_config import FlashMini50BConfig
from .kvc import KVQuantizer
from .v4_ple_store import PLETableStore

MTP_MAX_WINDOW = 4
MTP_MAX_RECURSIVE_STEPS = 3


def mtp_recursive_steps(mtp_window: int) -> int:
    """0 = disabled, 1 = base only, 2..4 = base + (window - 1) MTP tokens."""
    if isinstance(mtp_window, bool) or not isinstance(mtp_window, int) or not 0 <= mtp_window <= MTP_MAX_WINDOW:
        raise ValueError(f"mtp_window must be an integer in 0..{MTP_MAX_WINDOW}")
    return max(0, mtp_window - 1)


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

    def _read_weights(self, normalized: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.read_up(F.silu(self.read_down(normalized) / self.streams)))
        return weights.reshape(*weights.shape[:-1], self.streams, self.d_model)

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.streams, self.d_model)
        return (self._read_weights(normalized) * streams).mean(-2)

    def read_write(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.write_gate is None:
            raise RuntimeError("final HC is read-only")
        normalized = self.normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.streams, self.d_model)
        mixed = (self._read_weights(normalized) * streams).mean(-2)
        gates = 2.0 * torch.sigmoid(self.write_gate(normalized) / self.streams)
        return mixed, hidden, gates

    def write(self, hidden: torch.Tensor, output: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        return hidden + (output.unsqueeze(-2) * gates.unsqueeze(-1)).reshape_as(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.read(hidden)


class AttentionV4(nn.Module):
    """GQA attention with Qwen-style output gating and optional KVC reuse.

    ``q_proj`` physically stores, per query head, ``[query(256) | gate(256)]``;
    the two halves are separate logical operators for the optimizer.
    """

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


# -- Gated DeltaNet recurrence (Qwen3-Next semantics) --------------------------------

def gdn_log_decay(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Per-token log transition ``g = -exp(A_log) * softplus(a + dt_bias) <= 0`` (fp32)."""
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def _l2norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    """Token-wise reference recurrence; inputs (B, T, H, D), g/beta (B, T, H).

    ``S_t = exp(g_t) S_{t-1};  S_t += k_t (beta_t (v_t - k_t^T S_t))^T;  o_t = S_t^T q_t / sqrt(d_k)``
    with q and k L2-normalized, identical to Qwen3-Next ``torch_recurrent_gated_delta_rule``.
    """
    query, key = _l2norm(query.float()), _l2norm(key.float())
    query, key, value, beta, g = (x.transpose(1, 2).float() for x in (query, key, value, beta, g))
    batch, heads, steps, key_dim = key.shape
    query = query * key_dim ** -0.5
    state = query.new_zeros(batch, heads, key_dim, value.shape[-1]) if initial_state is None else initial_state.float()
    outputs = []
    for step in range(steps):
        state = state * g[:, :, step].exp()[..., None, None]
        memory = (state * key[:, :, step, :, None]).sum(-2)
        delta = (value[:, :, step] - memory) * beta[:, :, step, None]
        state = state + key[:, :, step, :, None] * delta[:, :, None, :]
        outputs.append((state * query[:, :, step, :, None]).sum(-2))
    output = torch.stack(outputs, dim=2) if outputs else value.new_zeros(batch, heads, 0, value.shape[-1])
    return output.transpose(1, 2), state


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int = 64, initial_state=None):
    """Chunked WY-form recurrence with an fp32 unit-lower-triangular solve.

    Mathematically identical to :func:`recurrent_gated_delta_rule`; this is the
    Qwen3-Next ``torch_chunk_gated_delta_rule`` with its row loop replaced by
    ``torch.linalg.solve_triangular``.  All exponents are <= 0 by construction.
    """
    query, key = _l2norm(query.float()), _l2norm(key.float())
    query, key, value, beta, g = (x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g))
    batch, heads, steps, key_dim = key.shape
    value_dim = value.shape[-1]
    pad = (chunk_size - steps % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    chunks = (steps + pad) // chunk_size
    query = query * key_dim ** -0.5
    k_beta = key * beta[..., None]
    v_beta = value * beta[..., None]
    query, key, k_beta, v_beta = (x.reshape(batch, heads, chunks, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta))
    g = g.reshape(batch, heads, chunks, chunk_size).cumsum(-1)
    lower = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=g.device).tril()
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(~lower, -math.inf).exp()
    strict = torch.tril(k_beta @ key.transpose(-1, -2) * decay, diagonal=-1)
    system = strict + torch.eye(chunk_size, dtype=strict.dtype, device=strict.device)
    solved = torch.linalg.solve_triangular(system, torch.cat((v_beta, k_beta * g.exp()[..., None]), dim=-1), upper=False, unitriangular=True)
    u, w = solved.split((value_dim, key_dim), dim=-1)
    state = query.new_zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state.float()
    outputs = []
    for index in range(chunks):
        q_i, k_i, g_i = query[:, :, index], key[:, :, index], g[:, :, index]
        attention = q_i @ k_i.transpose(-1, -2) * decay[:, :, index]
        v_new = u[:, :, index] - w[:, :, index] @ state
        outputs.append((q_i * g_i[..., None].exp()) @ state + attention @ v_new)
        state = state * g_i[:, :, -1, None, None].exp() + (k_i * (g_i[:, :, -1:] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
    output = torch.cat(outputs, dim=2)[:, :, :steps]
    return output.transpose(1, 2), state


class GatedDeltaNetV4(nn.Module):
    """Production 16-K-head / 32-V-head GDN with fp32 recurrence.

    ``in_proj_qkvz`` physically stores, per key head, ``[q(128) k(128) v(256) z(256)]``
    and ``in_proj_ba`` stores ``[beta(2) a(2)]``; each is a separate logical operator.
    The depthwise short convolution is applied additively (``x + conv(x)``) on the
    q/k/v channels, the reviewed FlashMini v4 formulation.
    """

    def __init__(self, config: FlashMini50BConfig, chunk_size: int = 64, kernel: str = "chunked"):
        super().__init__()
        gdn = config.section("gdn")
        d = config.d_model
        if kernel not in {"chunked", "recurrent"}:
            raise ValueError("GDN kernel must be 'chunked' or 'recurrent'")
        self.chunk_size, self.kernel = chunk_size, kernel
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

    def _project(self, hidden: torch.Tensor):
        batch, steps, _ = hidden.shape
        ratio = self.num_v_heads // self.num_k_heads
        mixed = self.in_proj_qkvz(hidden).reshape(batch, steps, self.num_k_heads, 2 * self.head_k + 2 * self.head_v * ratio)
        query, key, value, z = torch.split(mixed, [self.head_k, self.head_k, ratio * self.head_v, ratio * self.head_v], dim=-1)
        ba = self.in_proj_ba(hidden).reshape(batch, steps, self.num_k_heads, 2 * ratio)
        beta_raw, a_raw = torch.split(ba, [ratio, ratio], dim=-1)
        beta = beta_raw.reshape(batch, steps, self.num_v_heads).float().sigmoid()
        a = a_raw.reshape(batch, steps, self.num_v_heads)
        conv_input = torch.cat((query.reshape(batch, steps, self.key_dim), key.reshape(batch, steps, self.key_dim), value.reshape(batch, steps, self.value_dim)), dim=-1)
        conv = F.conv1d(F.pad(conv_input.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0)), self.conv1d.weight, groups=self.conv_dim)[..., :steps].transpose(1, 2)
        mixed_qkv = conv_input + conv
        query = mixed_qkv[..., :self.key_dim].reshape(batch, steps, self.num_k_heads, self.head_k)
        key = mixed_qkv[..., self.key_dim:2 * self.key_dim].reshape(batch, steps, self.num_k_heads, self.head_k)
        value = mixed_qkv[..., 2 * self.key_dim:].reshape(batch, steps, self.num_v_heads, self.head_v)
        z = z.reshape(batch, steps, self.num_v_heads, self.head_v)
        g = gdn_log_decay(self.A_log, a, self.dt_bias)
        query = query.repeat_interleave(ratio, dim=2)
        key = key.repeat_interleave(ratio, dim=2)
        return query, key, value, z, g, beta

    def decay_factors(self, hidden: torch.Tensor) -> torch.Tensor:
        """Multiplicative recurrent decay ``exp(g)`` in (0, 1] per token and value head."""
        return self._project(hidden)[4].exp()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = hidden.shape
        query, key, value, z, g, beta = self._project(hidden)
        context = torch.autocast(device_type=hidden.device.type, enabled=False) if hidden.device.type in ("cuda", "cpu") else contextlib.nullcontext()
        with context:
            if self.kernel == "recurrent":
                recurrent, _ = recurrent_gated_delta_rule(query, key, value, g, beta)
            else:
                recurrent, _ = chunk_gated_delta_rule(query, key, value, g, beta, self.chunk_size)
            normalized = recurrent * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + 1e-6)
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
    """80 routed Top-6 SwiGLU experts plus one sigmoid-gated shared expert.

    Returns routing statistics, not a balancing loss: the loss must be finalized
    over the logical/global batch by :class:`flashmini.v4_balance.RouterBalance`.
    """

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
                if torch.is_grad_enabled():
                    output = output + 0.0 * sum(p.sum() for p in expert.parameters()).to(output.dtype)
                continue
            token_ids, rank_ids = slots[:, 0], slots[:, 1]
            output = output.index_add(0, token_ids, expert(flat[token_ids]) * weights[token_ids, rank_ids, None])
        output = output + self.shared_gate(flat).sigmoid() * self.shared_expert(flat)
        stats = {
            "expert_counts": torch.bincount(indices.reshape(-1), minlength=self.num_experts).to(torch.float64).detach(),
            "router_prob_sum": probabilities.sum(0),
            "tokens": int(flat.shape[0]),
            "top_k": self.top_k,
        }
        return output.reshape(batch, steps, width), stats


class PLEV4(nn.Module):
    """One PLE injection with sixteen independently addressable host hash-head tables.

    Hash preimage per position ``t`` of a segment: ``(x_t, x_{t-1}, x_{t-2})`` where
    history positions before the segment start (sequence start or the token after
    an EOS) are replaced by the EOS sentinel.  Bigram heads hash ``(x_{t-1}, x_t)``;
    trigram heads hash ``(x_{t-2}, x_{t-1}, x_t)``; head ``h`` takes the hash modulo
    its prime row count.  The depthwise convolution is masked to the same segment.
    """

    def __init__(self, config: FlashMini50BConfig, *, store: PLETableStore | None = None):
        super().__init__()
        ple = config.section("ple")
        self.config = config
        self.rows = list(ple["hash_head_rows"])
        self.heads_per_order = ple["heads_per_order"]
        self.head_dim = ple["head_dim"]
        hash_spec = ple["hash"]
        self.eos_id = int(config.section("tokenizer")["special_token_ids"]["eos"])
        self.multipliers = self.build_multipliers(config.vocab_size, int(hash_spec["seed"]))
        self.store = store if store is not None else PLETableStore(self.rows, self.head_dim)
        hc_width = config.section("hyperconnections")["persistent_residual_width"]
        self.key_proj = nn.Linear(ple["aggregate_width"], hc_width, bias=False)
        self.value_proj = nn.Linear(ple["aggregate_width"], config.d_model, bias=False)
        self.key_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.query_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.conv_norm = GroupedRMSNormV4(hc_width, 4, config.d_model)
        self.conv_kernel = ple["convolution"]["kernel"]
        self.conv1d = nn.Conv1d(hc_width, hc_width, self.conv_kernel, groups=hc_width, bias=False)

    @staticmethod
    def _splitmix64(value: int) -> int:
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    @classmethod
    def build_multipliers(cls, vocab_size: int, seed: int) -> tuple[int, int, int]:
        """Odd multipliers for (x_t, x_{t-1}, x_{t-2}); products fit signed int64."""
        half = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
        return tuple(2 * (cls._splitmix64((seed + 0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1)) % half) + 1 for index in range(3))

    def segment_history(self, input_ids: torch.Tensor, shift: int) -> torch.Tensor:
        """``x_{t-shift}`` within the EOS-delimited segment, else the EOS sentinel."""
        if shift == 0:
            return input_ids
        batch, steps = input_ids.shape
        positions = torch.arange(steps, device=input_ids.device)
        eos_positions = torch.where(input_ids == self.eos_id, positions, -1)
        previous_eos = torch.cat((eos_positions.new_full((batch, 1), -1), eos_positions.cummax(dim=1).values[:, :-1]), dim=1)
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source = (positions - shift).clamp_min(0).unsqueeze(0).expand(batch, -1)
        shifted = input_ids.gather(1, source)
        return torch.where(position_in_segment >= shift, shifted, torch.full_like(shifted, self.eos_id))

    def ngram_components(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hash preimage ``(x_t, x_{t-1}, x_{t-2})`` per position, shape (B, T, 3)."""
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("PLE v4 input_ids must be rank-two int64")
        return torch.stack([self.segment_history(input_ids, shift) for shift in range(3)], dim=-1)

    def keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        components = self.ngram_components(input_ids)
        m0, m1, m2 = self.multipliers
        bigram = torch.bitwise_xor(components[..., 0] * m0, components[..., 1] * m1)
        trigram = torch.bitwise_xor(bigram, components[..., 2] * m2)
        rows = torch.tensor(self.rows, device=input_ids.device, dtype=torch.long)
        per_order = self.heads_per_order
        return torch.cat((torch.remainder(bigram.unsqueeze(-1), rows[:per_order]),
                          torch.remainder(trigram.unsqueeze(-1), rows[per_order:])), dim=-1)

    def segment_conv(self, values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Causal depthwise convolution whose taps never cross an EOS boundary."""
        channels, steps, kernel = values.shape[-1], values.shape[1], self.conv_kernel
        windows = F.pad(values.transpose(1, 2), (kernel - 1, 0)).unfold(2, kernel, 1)  # (B, C, T, K) oldest first
        segment = F.pad((input_ids[:, :-1] == self.eos_id).long().cumsum(1), (1, 0))
        tap_segment = F.pad(segment, (kernel - 1, 0), value=-1).unfold(1, kernel, 1)  # (B, T, K)
        valid = (tap_segment == segment.unsqueeze(-1)).to(values.dtype).unsqueeze(1)
        return (windows * valid * self.conv1d.weight[:, 0, :].view(1, channels, 1, kernel)).sum(-1).transpose(1, 2)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return torch.zeros_like(hidden)
        embeddings = self.store.lookup(self.keys(input_ids), device=hidden.device, dtype=self.key_proj.weight.dtype,
                                       requires_grad=torch.is_grad_enabled() and self.training)
        key, query = self.key_norm(self.key_proj(embeddings)), self.query_norm(hidden)
        gate = (key.reshape(*key.shape[:-1], 4, -1) * query.reshape(*query.shape[:-1], 4, -1)).sum(-1, keepdim=True) / math.sqrt(self.config.d_model)
        gated = (torch.sigmoid(gate) * self.value_proj(embeddings).unsqueeze(-2)).reshape(*hidden.shape)
        return gated + F.silu(self.segment_conv(self.conv_norm(gated), input_ids))

    def forward_with_ablation(self, input_ids: torch.Tensor, hidden: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        return self.forward(input_ids, hidden, enabled)


class BlockV4(nn.Module):
    def __init__(self, config: FlashMini50BConfig, layer_index: int, *, force_attention: bool = False):
        super().__init__()
        self.layer_index = layer_index
        role = None if force_attention else config.kvc_role(layer_index)
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
    """One full-attention MTP layer applied recursively (at most three steps)."""

    def __init__(self, config: FlashMini50BConfig):
        super().__init__()
        self.fusion = MTPFusionV4(config)
        self.block = BlockV4(config, -1, force_attention=True)
        self.final_hc = HyperConnectionV4(config.d_model, config.section("hyperconnections")["lowrank"], final=True)

    def forward(self, previous_state: torch.Tensor, teacher_embedding: torch.Tensor):
        state, _, stats = self.block(self.fusion(previous_state, teacher_embedding))
        return state, stats

    def collapse(self, state: torch.Tensor) -> torch.Tensor:
        return self.final_hc(state)


def mtp_teacher_and_targets(labels: torch.Tensor, steps: int, pad_id: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Teacher-forcing inputs and targets for recursive MTP depths ``1..steps``.

    With ``labels[t] = x_{t+1}``, depth ``k`` at position ``t`` embeds the ground
    truth ``x_{t+k} = labels[t+k-1]`` and predicts ``x_{t+k+1} = labels[t+k]``.
    Positions whose teacher or target lies past the window (or is ignored, -100)
    get ``pad_id`` as teacher and ``-100`` as target.
    """
    if steps > MTP_MAX_RECURSIVE_STEPS:
        raise ValueError(f"at most {MTP_MAX_RECURSIVE_STEPS} recursive MTP steps")
    batch, length = labels.shape
    result = []
    for depth in range(1, steps + 1):
        teacher = F.pad(labels[:, depth - 1:], (0, depth - 1), value=-100)
        target = F.pad(labels[:, depth:], (0, depth), value=-100)
        invalid = (teacher == -100) | (target == -100)
        result.append((teacher.masked_fill(invalid, pad_id), target.masked_fill(invalid, -100)))
    return result


class FlashMini50BBaseInit(nn.Module):
    def __init__(self, config: FlashMini50BConfig, *, dtype: torch.dtype | None = None, gdn_kernel: str = "chunked",
                 ple_store: PLETableStore | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.ple = PLEV4(config, store=ple_store)
        self.blocks = nn.ModuleList([BlockV4(config, index) for index in range(config.num_layers)])
        for block in self.blocks:
            if isinstance(block.mixer, GatedDeltaNetV4):
                block.mixer.kernel = gdn_kernel
        self.final_hc = HyperConnectionV4(config.d_model, config.section("hyperconnections")["lowrank"], final=True)
        # Store the head in the frozen [d_model, vocab] orientation. Forward
        # uses an explicit matmul; this is not a transposed alias of the input
        # embedding and therefore remains independently checkpointable.
        self.lm_head = LMHeadV4(config.d_model, config.vocab_size)
        self.mtp = MTPV4(config)
        self.pad_id = int(config.section("tokenizer")["special_token_ids"]["pad"])
        self.activation_checkpointing = False
        if dtype is not None:
            self.to(dtype)
        self.initialize()

    def named_logical_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Every learned tensor of the checkpoint: parameters plus host PLE tables."""
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter
        yield from self.ple.store.named_tables()

    def initialize(self) -> None:
        """Fill every non-meta learned tensor with the shared v4 InitSpec law."""
        for name, tensor in self.named_logical_tensors():
            if tensor.is_meta or not tensor.is_floating_point():
                continue
            v4_init.fill_(tensor, name)

    def backbone(self, input_ids: torch.Tensor, *, ple_enabled: bool = True):
        hidden = self.embed_tokens(input_ids).repeat(1, 1, 4)
        banks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        stats: list[dict[str, Any]] = []
        for index, block in enumerate(self.blocks):
            if index == self.config.ple_injection_layer:
                hidden = hidden + self.ple.forward_with_ablation(input_ids, hidden, ple_enabled)
            source = self.config.kvc_source_for(index)
            bank_in = banks.get(source) if source is not None else None
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden, bank, layer_stats = torch.utils.checkpoint.checkpoint(block, hidden, bank_in, use_reentrant=False)
            else:
                hidden, bank, layer_stats = block(hidden, bank_in)
            if bank is not None and block.kvc_role == "source":
                banks[index] = bank
            stats.append({"layer": f"blocks.{index}", **layer_stats})
        return hidden, stats

    @staticmethod
    def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), ignore_index=-100, reduction="sum")
        return total, int((target != -100).sum())

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, *, mtp_window: int = 0,
                ple_enabled: bool = True, routing_only: bool = False):
        steps = mtp_recursive_steps(mtp_window)
        backbone_hidden, stats = self.backbone(input_ids, ple_enabled=ple_enabled)
        if routing_only:
            # Routing statistics of every MoE invocation (backbone and MTP depths), no logits or losses.
            state = backbone_hidden
            pairs = mtp_teacher_and_targets(labels, steps, self.pad_id) if steps else []
            for depth, (teacher, _) in enumerate(pairs, start=1):
                state, layer_stats = self.mtp(state, self.embed_tokens(teacher))
                stats.append({"layer": "mtp.block", "depth": depth, **layer_stats})
            return {"stats": stats}
        logits = self.lm_head(self.final_hc(backbone_hidden))
        result: dict[str, Any] = {"logits": logits, "stats": stats, "backbone_hidden": backbone_hidden, "mtp_steps": steps}
        if labels is not None:
            loss_sum, count = self._cross_entropy(logits, labels)
            result.update(loss_sum=loss_sum, loss_count=count, loss=loss_sum / max(count, 1))
        if steps:
            if labels is None:
                raise ValueError("training-time MTP requires labels for ground-truth teacher tokens")
            state = backbone_hidden
            mtp_logits, sums, counts = [], [], []
            for depth, (teacher, target) in enumerate(mtp_teacher_and_targets(labels, steps, self.pad_id), start=1):
                state, layer_stats = self.mtp(state, self.embed_tokens(teacher))
                stats.append({"layer": "mtp.block", "depth": depth, **layer_stats})
                depth_logits = self.lm_head(self.mtp.collapse(state))
                mtp_logits.append(depth_logits)
                depth_sum, depth_count = self._cross_entropy(depth_logits, target)
                sums.append(depth_sum)
                counts.append(depth_count)
            result.update(
                mtp_logits=mtp_logits, mtp_loss_sums=sums, mtp_loss_counts=counts,
                mtp_losses=[total / max(count, 1) for total, count in zip(sums, counts)],
            )
            result["mtp_loss"] = torch.stack(result["mtp_losses"]).mean()
        return result


def surrogate_config(vocab_size: int = 64, d_model: int = 32, num_layers: int = 12, rows: int = 17) -> FlashMini50BConfig:
    """Return a schema-shaped tiny config for semantic tests, not a production config."""
    from copy import deepcopy

    from flashmini.base_init_config import MODEL_ID, load_config

    raw = deepcopy(load_config().raw)
    raw["tokenizer"]["vocab_size"] = vocab_size
    raw["tokenizer"]["special_token_ids"] = {"eos": 0, "pad": 1}
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
    """Unvalidated scaled config; same code paths, test-only geometry."""

    def __init__(self, raw):
        self._raw = raw
        self._surrogate = True

    def _validate(self):
        return None


__all__ = [
    "AttentionV4", "BlockV4", "FlashMini50BBaseInit", "GatedDeltaNetV4", "GroupedRMSNormV4",
    "HyperConnectionV4", "MTPV4", "MTP_MAX_RECURSIVE_STEPS", "MoEV4", "PLEV4", "SwiGLUExpertV4",
    "chunk_gated_delta_rule", "gdn_log_decay", "mtp_recursive_steps", "mtp_teacher_and_targets",
    "recurrent_gated_delta_rule", "surrogate_config",
]
