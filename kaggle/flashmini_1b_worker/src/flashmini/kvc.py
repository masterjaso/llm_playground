"""KVC (cross-layer KV sharing + low-bit KV) for PoC_D.

KVC is ONE architectural feature with a single on/off switch.  When enabled,
the first global-attention layer ("source") produces the K/V bank, applies the
frozen low-bit fake-quantization to both tensors, and the dequantized bank is
consumed by BOTH the source layer's own attention and the second
global-attention layer ("reuse").  The reuse layer generates its own query
from its own hidden state and never reuses the source's query or its own
K/V projections (those slices remain allocated but inactive so that C and D
share an identical parameter layout and initialization).

The fake-quant is deterministic for a fixed input/state: fixed format
(E2M1 data, E4M3 per-group scales, scale_group_size groups), straight-through
gradient estimator, no stochastic rounding.  The same dequantized bank feeds
both attention sites, so the architecture trains against exactly the low-bit
numbers it will deploy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# E2M1 (OCP FP4): 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; max 6.
_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _e4m3_magnitudes() -> torch.Tensor:
    """All positive E4M3 (FP8) representable values, sorted (min 2**-10 .. 448)."""
    values = []
    # Subnormals: 2**-7 * (m/8), m = 1..7  -> step 2**-10, min 2**-10.
    for m in range(1, 8):
        values.append(2.0 ** -7 * (m / 8.0))
    # Normalized: 2**(e-7) * (1 + m/8), e = 0..14, m = 0..7; e = 15 is satfinite 448.
    for e in range(0, 15):
        base = 2.0 ** (e - 7)
        for m in range(0, 8):
            values.append(base * (1.0 + m / 8.0))
    values.append(448.0)  # e=15 satfinite
    return torch.tensor(sorted(values), dtype=torch.float32)


_E4M3_MAGNITUDES: torch.Tensor | None = None


class KVQuantizer:
    """Deterministic low-bit fake-quant (QAT) for K/V tensors.

    Layout: last dim is the channel dim; values are grouped along the
    batch/time/head/channel axes in flat chunks of ``scale_group_size``
    contiguous values, matching a packed-cache byte accounting where each
    scale covers that group of values.
    """

    def __init__(
        self,
        kv_bits: int = 4,
        quant_format: str = "e2m1",
        scale_format: str = "e4m3",
        scale_group_size: int = 16,
    ):
        if kv_bits != 4:
            raise ValueError("official PoC_D KVC requires kv_bits=4")
        if quant_format != "e2m1":
            raise ValueError("official PoC_D KVC requires quant_format=e2m1")
        if scale_format != "e4m3":
            raise ValueError("official PoC_D KVC requires scale_format=e4m3")
        if scale_group_size <= 0 or scale_group_size % 2:
            raise ValueError("scale_group_size must be a positive even integer for e2m1 packs")
        self.kv_bits = kv_bits
        self.quant_format = quant_format
        self.scale_format = scale_format
        self.scale_group_size = scale_group_size
        self._e4m3_table: torch.Tensor | None = None

    def _table(self, device: torch.device) -> torch.Tensor:
        global _E4M3_MAGNITUDES
        if _E4M3_MAGNITUDES is None:
            _E4M3_MAGNITUDES = _e4m3_magnitudes()
        if self._e4m3_table is None or self._e4m3_table.device != device:
            self._e4m3_table = _E4M3_MAGNITUDES.to(device)
        return self._e4m3_table

    @staticmethod
    def _nearest_magnitude(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        # x >= 0 float tensor. searchsorted gives the first index with table >= x.
        idx = torch.searchsorted(table, x)
        idx = idx.clamp(0, table.numel() - 1)
        lower = table.index_select(0, (idx - 1).clamp(0, table.numel() - 1))
        upper = table.index_select(0, idx)
        return torch.where((x - lower).abs() <= (upper - x).abs(), lower, upper)

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize x to e2m1 data + e4m3 per-group scales, return the dequantized
        tensor in x's original dtype.

        Deterministic for a fixed input/state; differentiable via straight-through:
        the forward is the quantized/dequantized value and the backward passes
        the upstream gradient unmodified (no stochastic rounding).
        """
        if not torch.isfinite(x.detach().float()).all():
            raise FloatingPointError("KVC bank contains non-finite values")
        orig_dtype = x.dtype
        device = x.device
        table = self._table(device)
        e2m1 = _E2M1_MAGNITUDES.to(device)
        shape = x.shape
        flat = x.detach().to(torch.float32).reshape(-1, 1)  # (N, 1), fp32 compute
        g = self.scale_group_size
        n = flat.numel()
        if n % g:
            # Pad the last partial group so byte accounting holds for the cache;
            # training tensors from a fixed logical batch are group-aligned.
            pad = g - n % g
            flat = F.pad(flat, (0, 0, 0, pad))
        grouped = flat.reshape(-1, g)
        # (G_groups, g)
        amax = grouped.abs().amax(dim=1)  # (G,)
        scale = self._nearest_magnitude(amax / 6.0, table)  # (G,)
        scale = scale.clamp_min(table[0]).unsqueeze(-1)  # (G, 1)
        q = grouped / scale  # (G, g); |q| <= amax/scale ~= 6 by construction
        q_rep = self._nearest_magnitude(q.abs().reshape(-1), e2m1).reshape(q.shape)
        q_deq = torch.sign(q) * q_rep
        out = (q_deq * scale).reshape(shape).to(orig_dtype)
        return _FakeQuantFn.apply(x, out)


class _FakeQuantFn(torch.autograd.Function):
    """Straight-through: forward = precomputed dequant, backward = identity."""

    @staticmethod
    def forward(ctx, x, deq):
        return deq

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def kvc_bank_bytes(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    kv_bits: int = 4,
    scale_group_size: int = 16,
) -> dict[str, float | int]:
    """Byte accounting for the PACKED shared K/V bank (one shared bank serving
    both attention layers, so the 2x cross-layer sharing is intrinsic).

    Data: kv_bits per value (e2m1).  Scales: e4m3 = 8 bits per
    scale_group_size values.  Metadata: 8 bytes of per-tensor header (bit-width,
    group size, counts) per K tensor and V tensor.
    """
    values = 2 * batch * seq_len * num_heads * head_dim  # K + V
    raw_bytes = values * 2  # BF16 raw
    data_bytes = values * kv_bits / 8.0
    scale_values = (values + scale_group_size - 1) // scale_group_size
    scale_bytes = scale_values * 8 / 8.0
    metadata_bytes = 2 * 8
    packed_bytes = data_bytes + scale_bytes + metadata_bytes
    eff_bits = 8.0 * packed_bytes / values
    return {
        "raw_bf16_bytes": float(raw_bytes),
        "packed_bytes": float(packed_bytes),
        "data_bytes": float(data_bytes),
        "scale_bytes": float(scale_bytes),
        "metadata_bytes": float(metadata_bytes),
        "effective_bits_per_value": float(eff_bits),
        "compression_ratio": float(raw_bytes / packed_bytes),
    }
