"""HyperConnection / Gated Residual implementations.

``HyperConnection`` is the scalar v2 implementation.  v3 adds
``GatedResidual``: a reduced-scale implementation of the released Qwen
four-stream dynamic read/write mechanism.  Keeping the classes separate is
intentional: checkpoints from v2 must retain their original semantics.

The v2 module learns per-layer gating coefficients that mix the residual stream
and the layer output:

    x' = alpha * x + beta * f(x)

with alpha, beta learned scalars (optionally per-channel). This is the
"Gated Residual / HyperConnection" behavior required for Flash variants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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


class GatedResidual(nn.Module):
    """Reduced Qwen-style four-stream gated residual connection.

    The released Qwen4-Exp implementation carries ``hc_count`` persistent
    residual streams.  A low-rank MLP produces per-stream/channel read gates;
    a separate per-stream injection projection produces dynamic write gates.
    This module exposes those two operations explicitly so the containing
    decoder block remains the sole owner of residual application.

    ``hidden`` is flattened as ``(..., hc_count * d_model)``.  ``read``
    returns the mixed ``(..., d_model)`` view consumed by a mixer or MoE;
    ``write`` injects a block output back into the persistent streams.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hc_count: int = 4,
        hc_lowrank: int | None = None,
        eps: float = 1e-6,
        use_combine: bool = True,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("GatedResidual d_model must be positive")
        if hc_count != 4:
            raise ValueError("v3 GatedResidual currently requires four streams")
        if hc_lowrank is None:
            hc_lowrank = max(1, d_model // 4)
        if hc_lowrank <= 0:
            raise ValueError("GatedResidual hc_lowrank must be positive")
        self.d_model = d_model
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.eps = eps
        self.use_combine = use_combine
        hc_hidden_size = hc_count * d_model

        # Qwen's group RMS norm has one weight per flattened stream/channel,
        # but normalizes each stream independently.
        self.norm_weight = nn.Parameter(torch.zeros(hc_hidden_size))
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, hc_count, bias=False) if use_combine else None
        )

    @property
    def hidden_size(self) -> int:
        return self.hc_count * self.d_model

    def _normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} hyper-connection features, "
                f"got {hidden.shape[-1]}"
            )
        streams = hidden.reshape(*hidden.shape[:-1], self.hc_count, self.d_model)
        # Match the reference's float32 normalization under half/bfloat16.
        normalized = streams.float() * torch.rsqrt(
            streams.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized * (1.0 + self.norm_weight).reshape(
            self.hc_count, self.d_model
        )
        return normalized.to(hidden.dtype).reshape_as(hidden)

    def _read_weights(self, normalized: torch.Tensor) -> torch.Tensor:
        weights = F.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        weights = torch.sigmoid(self.input_mix_weight_up(weights))
        return weights.reshape(*weights.shape[:-1], self.hc_count, self.d_model)

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read persistent streams into one mixer-width representation."""

        normalized = self._normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.hc_count, self.d_model)
        weights = self._read_weights(normalized)
        return (weights * streams).mean(dim=-2)

    def read_with_injection(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return read output, untouched streams, and dynamic write gates."""

        normalized = self._normalize(hidden)
        streams = normalized.reshape(*normalized.shape[:-1], self.hc_count, self.d_model)
        weights = self._read_weights(normalized)
        mixed = (weights * streams).mean(dim=-2)
        if self.block_inject_weight is None:
            raise RuntimeError("read_with_injection requires use_combine=True")
        # The reference scales logits by stream count and maps to (0, 2).
        injection = 2.0 * torch.sigmoid(
            self.block_inject_weight(normalized) / self.hc_count
        )
        return mixed, hidden, injection

    def write(
        self,
        hidden: torch.Tensor,
        output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        """Inject a mixer/MoE output into all persistent residual streams."""

        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected {self.hidden_size} hyper-connection features, "
                f"got {hidden.shape[-1]}"
            )
        if output.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected output width {self.d_model}, got {output.shape[-1]}"
            )
        if injection.shape[-1] != self.hc_count:
            raise ValueError(
                f"Expected {self.hc_count} injection gates, got {injection.shape[-1]}"
            )
        injected = output.unsqueeze(-2) * injection.unsqueeze(-1)
        return hidden + injected.reshape_as(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Read streams, optionally returning state needed for a block write."""

        if self.use_combine:
            return self.read_with_injection(hidden)
        return self.read(hidden)


# Name used by the upstream implementation and by architecture-focused tests.
HyperConnectionV3 = GatedResidual
