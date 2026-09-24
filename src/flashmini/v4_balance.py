"""Global/logical-batch MoE load balancing (Qwen global-batch LBL).

Per logical MoE layer ``l`` (48 backbone layers plus the MTP layer, whose three
recursive invocations form one token population):

``f_i = c_i / (T * k)``  (fraction of routed assignments; ``sum_i f_i = 1``)
``P_i = (1/T) sum_t p_i(x_t)``  (mean router probability, differentiable)
``L_l = E * sum_i f_i P_i``  (equals 1 for perfectly uniform routing)

``f`` is computed over the logical batch: expert counts are all-reduced across
data-parallel ranks for every microbatch.  Two gradient-accumulation modes:

* ``exact_prepass`` - a no-grad routing pass over all microbatches of the step
  fixes the exact logical-batch ``f`` before any loss is formed; the resulting
  gradient equals that of the unpartitioned logical batch.
* ``ga_buffer`` - Qwen's buffer: microbatch ``m`` uses the counts accumulated over
  microbatches ``1..m`` (exact for the final microbatch and for ``GA = 1``).

The microbatch objective ``sum_l w_m L_{l,m} / num_layers`` uses the token weight
``w_m = T_{m,rank} * world / T_logical`` so rank-summed, microbatch-summed
gradients equal those of ``E * sum_i f_i P_i`` over the whole logical batch.
The reported ``router_aux_logical`` is always the exact logical-batch value.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

MODES = ("exact_prepass", "ga_buffer")


def logical_layer_key(stat: dict[str, Any]) -> str:
    return stat["layer"]


class RouterBalance:
    def __init__(self, num_experts: int, top_k: int, *, mode: str, group: Any = None, device: torch.device | str = "cpu"):
        if mode not in MODES:
            raise ValueError(f"router balance mode must be one of {MODES}")
        self.num_experts, self.top_k, self.mode, self.group = num_experts, top_k, mode, group
        self.device = torch.device(device)
        self.world = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
        self.begin_step()

    def begin_step(self) -> None:
        self.counts: dict[str, torch.Tensor] = {}
        self.tokens: dict[str, float] = {}
        self.prob_sums: dict[str, torch.Tensor] = {}
        self.prepass_counts: dict[str, torch.Tensor] | None = None
        self.prepass_tokens: dict[str, float] | None = None
        self.logical_tokens: float | None = None

    def _all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(value, group=self.group)
        return value

    @staticmethod
    def merge_layers(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Combine per-invocation stats into logical layers (MTP depths share one layer)."""
        merged: dict[str, dict[str, Any]] = {}
        for stat in stats:
            key = logical_layer_key(stat)
            if key not in merged:
                merged[key] = {"expert_counts": stat["expert_counts"].clone(), "router_prob_sum": stat["router_prob_sum"], "tokens": stat["tokens"]}
            else:
                item = merged[key]
                item["expert_counts"] = item["expert_counts"] + stat["expert_counts"]
                item["router_prob_sum"] = item["router_prob_sum"] + stat["router_prob_sum"]
                item["tokens"] += stat["tokens"]
        return merged

    def set_logical_tokens(self, local_tokens_per_step: int) -> None:
        """Total tokens per logical layer invocation over all ranks and microbatches."""
        value = torch.tensor([float(local_tokens_per_step)], dtype=torch.float64, device=self.device)
        self.logical_tokens = float(self._all_reduce(value))

    def add_prepass(self, stats: list[dict[str, Any]]) -> None:
        if self.prepass_counts is None:
            self.prepass_counts, self.prepass_tokens = {}, {}
        for key, item in self.merge_layers(stats).items():
            counts = self._all_reduce(item["expert_counts"].to(self.device, torch.float64).clone())
            tokens = float(self._all_reduce(torch.tensor([float(item["tokens"])], dtype=torch.float64, device=self.device)))
            self.prepass_counts[key] = self.prepass_counts.get(key, 0) + counts
            self.prepass_tokens[key] = self.prepass_tokens.get(key, 0.0) + tokens

    def microbatch_loss(self, stats: list[dict[str, Any]], positions: int) -> torch.Tensor:
        """Differentiable balancing contribution of one microbatch on this rank.

        ``positions`` is the number of backbone token positions in the microbatch;
        a logical layer with ``n`` invocations (MTP depths) routes ``n * positions``.
        Summing the returned value over ranks and microbatches gives the exact
        logical-batch objective ``mean_l E * sum_i f_i P_i``.
        """
        if self.logical_tokens is None:
            raise RuntimeError("set_logical_tokens must be called before microbatch_loss")
        if self.mode == "exact_prepass" and self.prepass_counts is None:
            raise RuntimeError("exact_prepass mode requires add_prepass for every microbatch first")
        merged = self.merge_layers(stats)
        losses = []
        for key in sorted(merged):
            item = merged[key]
            counts = self._all_reduce(item["expert_counts"].to(self.device, torch.float64).clone())
            tokens = float(self._all_reduce(torch.tensor([float(item["tokens"])], dtype=torch.float64, device=self.device)))
            self.counts[key] = self.counts.get(key, 0) + counts
            self.tokens[key] = self.tokens.get(key, 0.0) + tokens
            prob_sum = item["router_prob_sum"].float()
            self.prob_sums[key] = self.prob_sums.get(key, 0) + prob_sum.detach().double().to(self.device)
            if self.mode == "exact_prepass":
                fraction = self.prepass_counts[key] / (self.prepass_tokens[key] * self.top_k)
            else:
                fraction = self.counts[key] / (self.tokens[key] * self.top_k)
            layer_tokens = self.logical_tokens * item["tokens"] / positions
            losses.append(self.num_experts * (fraction.to(prob_sum.device, torch.float32) * prob_sum).sum() / layer_tokens)
        return torch.stack(losses).mean() if losses else torch.zeros(())

    def finalize(self) -> dict[str, Any]:
        """Exact logical-batch balance metrics (independent of partitioning)."""
        layers = {}
        for key in sorted(self.counts):
            counts, tokens = self.counts[key], self.tokens[key]
            probs = self._all_reduce(self.prob_sums[key].clone()) / tokens
            fraction = counts / (tokens * self.top_k)
            entropy = -float((fraction.clamp_min(1e-12) * fraction.clamp_min(1e-12).log()).sum())
            layers[key] = {
                "aux": float(self.num_experts * (fraction * probs).sum()),
                "load_fraction": fraction.tolist(),
                "load_entropy": entropy,
                "load_entropy_normalized": entropy / math.log(self.num_experts),
                "max_over_mean_load": float(fraction.max() * self.num_experts),
                "min_over_mean_load": float(fraction.min() * self.num_experts),
                "tokens": tokens,
            }
        aux = sum(item["aux"] for item in layers.values()) / max(len(layers), 1)
        return {"router_aux_logical": aux, "layers": layers, "mode": self.mode}


__all__ = ["MODES", "RouterBalance"]
