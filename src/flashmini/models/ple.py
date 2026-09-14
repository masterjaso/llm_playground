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
