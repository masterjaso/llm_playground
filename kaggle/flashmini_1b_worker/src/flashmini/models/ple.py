"""Causal multi-head n-gram memory with contextual gating and CPU row lookup.

Each hash head has a separate prime-sized table. Only selected rows travel to
the compute device; CPU storage never requires a full-table accelerator copy.
This is a small experimental conditional-memory module, not a Qwen replica.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PLEConfig


def table_sizes(config: PLEConfig) -> list[int]:
    """Distinct prime capacities at or above the requested per-head bucket budget."""
    candidate = config.table_size or config.vocab_size
    sizes = []
    while len(sizes) < config.num_heads:
        if candidate >= 2 and all(candidate % p for p in range(2, math.isqrt(candidate) + 1)):
            sizes.append(candidate)
        candidate += 1
    return sizes


class _RowEmbedding(nn.Embedding):
    def __init__(self, rows: int, dim: int, *, offload: str, sparse: bool):
        super().__init__(rows, dim, sparse=sparse)
        self.offload = offload

    def _apply(self, fn, recurse=True):
        if self.offload != "cpu":
            return super()._apply(fn, recurse=recurse)
        # Probe only an empty tensor. Applying fn to the weight first would
        # transiently allocate the entire table on CUDA and defeat offloading.
        probe = fn(torch.empty(0, dtype=self.weight.dtype, device="cpu"))
        with torch.no_grad():
            self.weight.data = self.weight.data.to(device="cpu", dtype=probe.dtype)
            if self.weight.grad is not None:
                self.weight.grad = self.weight.grad.to(device="cpu", dtype=probe.dtype)
        return self


class PLE(nn.Module):
    def __init__(self, config: PLEConfig):
        super().__init__()
        config.__post_init__()
        self.config = config
        self.ngram = config.ngram
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.sizes = table_sizes(config)
        offsets = [0]
        for size in self.sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("sizes_tensor", torch.tensor(self.sizes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        # Each base is nonzero and distinct from that head's modulus.
        bases = [2 + ((1009 + h * 9176) % (size - 2)) for h, size in enumerate(self.sizes)]
        self.register_buffer("bases", torch.tensor(bases), persistent=False)
        orders = [2 + h % (config.ngram - 1) for h in range(config.num_heads)]
        self.register_buffer("orders", torch.tensor(orders), persistent=False)
        self.value_embed = _RowEmbedding(sum(self.sizes), config.head_dim,
                                        offload=config.offload, sparse=config.sparse)
        width = config.num_heads * config.head_dim
        self.out_proj = nn.Linear(width, config.d_model, bias=False)
        self.key_proj = nn.Linear(width, config.d_model, bias=False)
        self.norm = nn.RMSNorm(config.d_model)
        self.key_norm = nn.RMSNorm(config.d_model)
        self.query_norm = nn.RMSNorm(config.d_model)
        # Nonzero initial scale allows table/key gradients on the first step.
        self.scale = nn.Parameter(torch.tensor(config.gate_init))
        nn.init.normal_(self.value_embed.weight, std=0.02)

    def _ngram_keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Global row IDs, shape (batch, time, heads), using causal suffixes only."""
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("PLE input_ids must be a rank-two int64 tensor")
        b, t = input_ids.shape
        # Zero is a separate boundary sentinel, including for actual token ID 0.
        padded = F.pad(input_ids + 1, (self.ngram - 1, 0))
        windows = padded.unfold(1, self.ngram, 1)
        if self.config.eos_id is not None:
            positions = torch.arange(t, device=input_ids.device)
            boundaries = torch.where(input_ids == self.config.eos_id, positions, -1)
            last_boundary = boundaries.cummax(dim=1).values
            window_positions = positions[:, None] + torch.arange(1 - self.ngram, 1,
                                                                  device=input_ids.device)
            windows = windows.masked_fill(window_positions[None] < last_boundary[:, :, None], 0)
        keys = torch.zeros(b, t, self.num_heads, device=input_ids.device, dtype=torch.long)
        for j in range(self.ngram):
            token = windows[:, :, j, None]
            updated = (keys * self.bases + token) % self.sizes_tensor
            keys = torch.where(j >= self.ngram - self.orders, updated, keys)
        return keys + self.offsets

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        keys = self._ngram_keys(input_ids)
        table_device = self.value_embed.weight.device
        # Repeated phrases share both their transfer and their sparse gradient row.
        unique, inverse = keys.reshape(-1).unique(return_inverse=True)
        values = self.value_embed(unique.to(table_device))
        target = self.out_proj.weight.device
        if table_device.type == "cpu" and target.type == "cuda":
            values = values.pin_memory().to(target, non_blocking=True)
        else:
            values = values.to(target)
        memory = values[inverse.to(target)].reshape(*input_ids.shape, -1)
        value = self.norm(self.out_proj(memory))
        query = hidden if hidden is not None else torch.zeros_like(value)
        key = self.key_norm(self.key_proj(memory))
        gate_logits = (self.query_norm(query).float() * key.float()).sum(-1, keepdim=True)
        gate = torch.sigmoid(gate_logits / math.sqrt(self.config.d_model)).to(value.dtype)
        return self.scale.tanh() * gate * value

    def forward_with_ablation(self, input_ids: torch.Tensor, enabled: bool,
                              hidden: torch.Tensor | None = None) -> torch.Tensor:
        if not enabled:
            if hidden is not None:
                return torch.zeros_like(hidden)
            return torch.zeros(*input_ids.shape, self.config.d_model,
                               device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        return self(input_ids, hidden)


class PLEV3(nn.Module):
    """Reduced Qwen4-Exp PLE with grouped heads and causal local convolution.

    v3 uses ``heads_per_ngram`` heads for every order (all bigram heads first,
    then all trigram heads), XOR hashing with deterministic odd multipliers,
    and a depthwise kernel-4 convolution dilated by the n-gram order.  The
    table capacity is intentionally reduced for FlashMini, while the tensor
    and reset semantics follow the released implementation.
    """

    def __init__(self, config: PLEConfig):
        super().__init__()
        config.__post_init__()
        if config.architecture_version != 3:
            raise ValueError("PLEV3 requires PLEConfig architecture_version=3")
        self.config = config
        self.ngram = config.ngram
        self.heads_per_ngram = config.heads_per_ngram
        self.num_heads = (self.ngram - 1) * self.heads_per_ngram
        self.hc_count = getattr(config, "hc_count", 4)
        if self.hc_count != 4:
            raise ValueError("PLEV3 currently requires four residual streams")
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim or self.num_heads * self.head_dim
        if self.embed_dim <= 0 or self.embed_dim % self.num_heads:
            raise ValueError("PLEV3 embed_dim must be divisible by total n-gram heads")
        self.eos_id = config.eos_id
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_dilation = config.conv_dilation or self.ngram

        candidate = config.ngram_vocab_size_base or config.table_size or config.vocab_size
        self.sizes = table_sizes(
            PLEConfig(
                ngram=config.ngram,
                vocab_size=config.vocab_size,
                num_heads=self.num_heads,
                table_size=candidate,
            )
        )
        offsets = [0]
        for size in self.sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("head_vocab_sizes", torch.tensor(self.sizes, dtype=torch.long), persistent=False)
        self.register_buffer("head_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer(
            "layer_multipliers",
            self._build_multipliers(config.vocab_size, self.ngram, config.hash_seed),
            persistent=False,
        )

        self.value_embed = _RowEmbedding(
            sum(self.sizes),
            self.embed_dim // self.num_heads,
            offload=config.offload,
            sparse=config.sparse,
        )
        hc_hidden_size = self.hc_count * config.d_model
        self.key_proj = nn.Linear(self.embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(self.embed_dim, config.d_model, bias=False)
        self.norm_key = _GroupedRMSNorm(hc_hidden_size, self.hc_count, config.d_model)
        self.norm_query = _GroupedRMSNorm(hc_hidden_size, self.hc_count, config.d_model)
        self.norm_conv = _GroupedRMSNorm(hc_hidden_size, self.hc_count, config.d_model)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=hc_hidden_size,
            dilation=self.conv_dilation,
            bias=False,
        )
        # Qwen initializes the PLE depthwise convolution to zero; its direct
        # gated lookup remains active while the local-context path learns.
        nn.init.zeros_(self.conv1d.weight)

    @staticmethod
    def _splitmix64(value: int) -> int:
        value &= (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return (value ^ (value >> 31)) & ((1 << 64) - 1)

    @classmethod
    def _build_multipliers(cls, vocab_size: int, ngram: int, seed: int) -> torch.Tensor:
        # Keep products in signed int64 while retaining odd, deterministic
        # multipliers as in the reference hash construction.
        max_long = (1 << 63) - 1
        bound = max(1, max_long // max(vocab_size, 1))
        half = max(1, bound // 2)
        values = []
        for index in range(ngram):
            # FlashMini has a single PLE layer: upstream ple_layer_index=0.
            mixed = cls._splitmix64((seed + 0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1))
            values.append(2 * (mixed % half) + 1)
        return torch.tensor(values, dtype=torch.long)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        if self.eos_id is None:
            return F.pad(token_ids[:, :-shift], (shift, 0), value=0)
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_id))

    def _ngram_keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("PLE input_ids must be a rank-two int64 tensor")
        if input_ids.numel() == 0:
            return input_ids.new_empty((*input_ids.shape, self.num_heads))
        context_len = self.ngram - 1
        sentinel = self.eos_id if self.eos_id is not None else 0
        previous = input_ids.new_full((input_ids.shape[0], context_len), sentinel)
        history = torch.cat([previous, input_ids], dim=1)
        shifted = [self._shift_right_ignore_eos(history, shift) for shift in range(self.ngram)]
        blocks = []
        for order in range(2, self.ngram + 1):
            start = (order - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(
                    mixed,
                    shifted[position] * self.layer_multipliers[position],
                )
            sizes = self.head_vocab_sizes[start:end]
            offsets = self.head_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1))
            blocks.append(ids + offsets.view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]

    def _lookup(self, keys: torch.Tensor, target: torch.device) -> torch.Tensor:
        # TPU tables stay resident on XLA.  Direct fixed-shape gathers avoid
        # ``unique(return_inverse=True)``'s data-dependent output and the
        # resulting recompilation/host-sync path.  CPU/CUDA keep the deduped
        # lookup used by the PoC to reduce transfer and sparse-gradient work.
        if self.value_embed.weight.device.type == "xla":
            values = self.value_embed(keys.to(self.value_embed.weight.device))
            return values.reshape(*keys.shape[:-1], self.embed_dim).to(target)
        unique, inverse = keys.reshape(-1).unique(return_inverse=True)
        values = self.value_embed(unique.to(self.value_embed.weight.device))
        if self.value_embed.weight.device.type == "cpu" and target.type == "cuda":
            values = values.pin_memory().to(target, non_blocking=True)
        else:
            values = values.to(target)
        return values[inverse.to(target)].reshape(*keys.shape[:-1], self.embed_dim)

    def _short_conv(self, values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply a static-shape depthwise causal convolution per EOS segment.

        The previous implementation converted EOS positions to Python lists
        and launched one convolution per row/segment.  That is correct on CPU
        but forces host synchronizations and dynamic output shapes on XLA.  A
        fixed receptive-field window plus a segment-id equality mask preserves
        the reset semantics while compiling as one graph.
        """

        if values.shape[1] == 0:
            return values
        channels, steps = values.shape[-1], values.shape[1]
        dilation = int(self.conv_dilation)
        kernel = int(self.conv_kernel_size)
        receptive = dilation * (kernel - 1) + 1
        # (B, C, T, K), oldest tap first, exactly matching left-padded Conv1d.
        padded = F.pad(values.transpose(1, 2), (receptive - 1, 0))
        windows = padded.unfold(2, receptive, 1)[..., ::dilation]
        if self.eos_id is None:
            valid = torch.ones((values.shape[0], 1, steps, kernel), device=values.device, dtype=values.dtype)
        else:
            # An EOS token belongs to the segment it terminates; only the
            # following position starts the next segment.
            eos_before = (input_ids[:, :-1] == self.eos_id).to(torch.long).cumsum(dim=1)
            segment_ids = F.pad(eos_before, (1, 0))
            segment_padded = F.pad(segment_ids.unsqueeze(1).to(values.dtype), (receptive - 1, 0))
            segment_windows = segment_padded.unfold(2, receptive, 1)[..., ::dilation]
            current = segment_ids[:, None, :, None].to(values.dtype)
            valid = (segment_windows == current).to(values.dtype)
            # Negative padded positions are intentionally invalid, even when
            # the segment id happens to match the first token.
            positions = torch.arange(steps, device=values.device)
            tap_positions = positions[:, None] - dilation * (kernel - 1 - torch.arange(kernel, device=values.device)[None, :])
            valid = valid * (tap_positions[None, None, :, :] >= 0).to(values.dtype)
        filtered = (windows * valid * self.conv1d.weight[:, 0, :].view(1, channels, 1, kernel)).sum(dim=-1)
        return F.silu(filtered.transpose(1, 2))

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        keys = self._ngram_keys(input_ids)
        target = self.key_proj.weight.device
        embeddings = self._lookup(keys, target)
        if hidden is None:
            hidden = torch.zeros(
                *input_ids.shape,
                self.hc_count * self.config.d_model,
                device=target,
                dtype=self.key_proj.weight.dtype,
            )
        elif hidden.shape[-1] == self.config.d_model:
            hidden = hidden.repeat(1, 1, self.hc_count)
        if hidden.shape[-1] != self.hc_count * self.config.d_model:
            raise ValueError("PLEV3 hidden state must contain four residual streams")
        key = self.norm_key(self.key_proj(embeddings))
        query = self.norm_query(hidden)
        value = self.value_proj(embeddings)
        gate = (key.reshape(*key.shape[:-1], self.hc_count, self.config.d_model)
                * query.reshape(*query.shape[:-1], self.hc_count, self.config.d_model)).sum(
                    dim=-1, keepdim=True
                ) / math.sqrt(self.config.d_model)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated = gated.reshape(*gated.shape[:-2], self.hc_count * self.config.d_model)
        conv_input = self.norm_conv(gated)
        return gated + self._short_conv(conv_input, input_ids)

    def forward_with_ablation(
        self,
        input_ids: torch.Tensor,
        enabled: bool,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if enabled:
            return self(input_ids, hidden)
        width = self.hc_count * self.config.d_model
        device = hidden.device if hidden is not None else self.key_proj.weight.device
        dtype = hidden.dtype if hidden is not None else self.key_proj.weight.dtype
        return torch.zeros(*input_ids.shape, width, device=device, dtype=dtype)


class _GroupedRMSNorm(nn.Module):
    """Per-stream RMS normalization used by v3 GR and PLE."""

    def __init__(self, width: int, groups: int, group_width: int, eps: float = 1e-6):
        super().__init__()
        if width != groups * group_width:
            raise ValueError("Grouped RMS norm dimensions do not multiply")
        self.groups = groups
        self.group_width = group_width
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        grouped = value.reshape(*shape[:-1], self.groups, self.group_width).float()
        grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        grouped = grouped * (1.0 + self.weight).reshape(self.groups, self.group_width)
        return grouped.to(value.dtype).reshape(shape)
