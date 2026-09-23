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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    """Route tokens to top-k experts.

    Returns (expert_indices, expert_weights, router_logits, aux_loss, aux_stats).
    aux_loss is the frozen C switch-style scalar (byte-identical, so pipeline
    disabled or whole-batch reproduces C exactly). aux_stats carries the additive
    sufficient statistics (exp_counts, prob_sum, token_count, slot_count) used to
    recompute the logical-batch aux exactly in the microbatch pipeline.
    x: (B*T, d_model)
    """
    logits = router(x)  # (N, num_experts)
    weights, indices = torch.topk(logits, top_k, dim=-1)  # (N, top_k)
    weights = F.softmax(weights, dim=-1)

    # Aux load-balancing loss (switch-style): fraction routed * fraction prob.
    # Scalar, IDENTICAL to the frozen C reference so the D trajectory is unchanged
    # whenever the pipeline is disabled or runs whole-batch.
    probs = F.softmax(logits, dim=-1)
    counts = torch.zeros(num_experts, device=x.device, dtype=torch.float32)
    counts.scatter_add_(0, indices.reshape(-1),
                        torch.ones(indices.numel(), dtype=torch.float32, device=x.device))
    # Two distinct denominators (frozen C): routed fraction uses votes (N*top_k),
    # probability fraction uses tokens (N).  Both are additive across
    # microbatches so the logical-batch aux is exactly recomputable.
    slot_count = indices.numel()  # N * top_k -> denominator of frac_routed
    token_count = x.shape[0]      # N         -> denominator of frac_prob
    frac_routed = counts / slot_count
    frac_prob = probs.mean(dim=0)  # == prob_sum / N
    aux_loss = (num_experts * (frac_routed * frac_prob).sum())
    aux_stats = {"exp_counts": counts, "prob_sum": probs.sum(dim=0),
                 "token_count": token_count, "slot_count": slot_count}
    return indices, weights, logits, aux_loss, aux_stats


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
        indices, weights, logits, aux_loss, aux_stats = topk_router(
            flat, self.router, self.config.num_experts, self.config.top_k
        )

        out = torch.zeros_like(flat)
        # Dispatch tokens to experts.
        #
        # `mask.any()` + `mask.nonzero()` forced two host round-trips per expert
        # per layer (32 device synchronizations per MoE layer per microbatch),
        # which made the training step CPU-bound rather than GPU-bound.  The
        # stable-argsort form below needs exactly one small host transfer (the
        # per-expert counts) per layer, because the per-expert slice bounds must
        # exist as host integers to size the expert calls.
        #
        # The math is unchanged: a stable argsort of the flattened (token, slot)
        # index visits pairs in the same (token, slot) lexicographic order as
        # `mask.nonzero()`, so `index_add_` accumulates each expert's
        # contributions in the identical order and the result is bit-identical.
        num_experts = self.config.num_experts
        top_k = self.config.top_k
        flat_indices = indices.reshape(-1)
        order = torch.argsort(flat_indices, stable=True)
        sorted_indices = flat_indices[order]
        boundaries = torch.searchsorted(
            sorted_indices,
            torch.arange(num_experts + 1, device=x.device, dtype=flat_indices.dtype),
        )
        counts = (boundaries[1:] - boundaries[:-1]).tolist()
        start = 0
        for e, count in enumerate(counts):
            if count == 0:
                continue
            rows = order[start:start + count]
            start += count
            token_ids = torch.div(rows, top_k, rounding_mode="floor")
            slot = rows - token_ids * top_k
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
        # Sufficient statistics for exact logical-batch aux aggregation
        # across pipeline microbatches (sums are additive).
        stats["router_exp_counts"] = aux_stats["exp_counts"]
        stats["router_prob_sum"] = aux_stats["prob_sum"]
        stats["router_token_count"] = torch.as_tensor(aux_stats["token_count"], dtype=torch.float32, device=x.device)
        stats["router_slot_count"] = torch.as_tensor(aux_stats["slot_count"], dtype=torch.float32, device=x.device)
        return out, stats


def _router_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()


def _expert_load(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.zeros(num_experts, device=indices.device)
    counts.scatter_add_(0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=torch.float32))
    return counts / indices.numel()