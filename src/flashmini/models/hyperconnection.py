"""HyperConnection / Gated Residual (Qwen3-Flash-Next style).

Instead of a plain residual `x + f(x)`, HyperConnection learns per-layer gating
coefficients that mix the residual stream and the layer output:

    x' = alpha * x + beta * f(x)

with alpha, beta learned scalars (optionally per-channel). This is the
"Gated Residual / HyperConnection" behavior required for Flash variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    def __init__(self, d_model: int, per_channel: bool = False):
        super().__init__()
        if per_channel:
            self.alpha = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.ones(d_model))
        else:
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta * f