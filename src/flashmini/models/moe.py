"""Sparse top-k Mixture-of-Experts with optional shared expert.

Implements token-choice routing with an auxiliary load-balancing loss and
router entropy/load statistics for the measurement harness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoEConfig


def topk_router(
    x: torch.Tensor,
    router: nn.Linear,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens to top-k experts.

    Returns (expert_indices, expert_weights, router_logits, aux_loss).
    x: (B*T, d_model)
    """
    logits = router(x)  # (N, num_experts)
    weights, indices = torch.topk(logits, top_k, dim=-1)  # (N, top_k)
    weights = F.softmax(weights, dim=-1)

    # Aux load-balancing loss (switch-style): fraction routed * fraction prob
    probs = F.softmax(logits, dim=-1)
    frac_routed = torch.zeros(num_experts, device=x.device)
    frac_routed.scatter_add_(0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=torch.float32))
    frac_routed = frac_routed / indices.numel()
    frac_prob = probs.mean(dim=0)
    aux_loss = num_experts * (frac_routed * frac_prob).sum()
    return indices, weights, logits, aux_loss


class MoE(nn.Module):
    """Token-choice MoE with shared expert."""

    def __init__(self, d_model: int, config: MoEConfig):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.router = nn.Linear(d_model, config.num_experts, bias=False)
        # experts: (num_experts, d_model, expert_intermediate) for gate, then up
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, config.expert_intermediate, bias=False),
                    nn.GELU(),
                    nn.Linear(config.expert_intermediate, d_model, bias=False),
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, config.expert_intermediate, bias=False),
                    nn.GELU(),
                    nn.Linear(config.expert_intermediate, d_model, bias=False),
                )
                for _ in range(config.shared_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        indices, weights, logits, aux_loss = topk_router(
            flat, self.router, self.config.num_experts, self.config.top_k
        )

        out = torch.zeros_like(flat)
        # Dispatch tokens to experts
        for e in range(self.config.num_experts):
            mask = indices == e  # (N, top_k)
            if mask.any():
                # tokens that routed to expert e (may appear multiple times)
                token_ids, slot = mask.nonzero(as_tuple=True)
                expert_out = self.experts[e](flat[token_ids])
                out.index_add_(0, token_ids, expert_out * weights[token_ids, slot].unsqueeze(-1))

        # Shared experts
        for se in self.shared_experts:
            out = out + se(flat)

        out = out.reshape(B, T, C)

        # Statistics for harness
        stats = {
            "router_aux_loss": aux_loss,
            "router_entropy": _router_entropy(logits),
            "expert_load": _expert_load(indices, self.config.num_experts),
        }
        return out, stats


def _router_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()


def _expert_load(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.zeros(num_experts, device=indices.device)
    counts.scatter_add_(0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=torch.float32))
    return counts / indices.numel()