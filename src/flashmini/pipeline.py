"""GPipe-style microbatch pipeline for PoC_D.

A logical batch (default 16 sequences) is ONE optimizer update.  It is executed
as ``len(microbatches)`` microbatches (default 4), each run independently through
the staged, model-parallel model.  The objective and every gradient are built so
the microbatch pipeline is EXACTLY equivalent to one full-batch update:

* CE: the sum over microbatches of ``ce_mb / m`` equals the full-batch
  ``F.cross_entropy`` mean (each microbatch has the same token count), and its
  gradient equals the full-batch ``cross_entropy`` gradient exactly — every token
  contributes ``1 / total_tokens`` of its token-NLL gradient.
* aux: per-layer sufficient statistics (``exp_counts``, ``prob_sum``,
  ``token_count``, ``slot_count``) are additive over microbatches and are
  reconstructed to the frozen-C per-layer scalar, then averaged over layers.
  The reconstruction stays differentiable through the router, so the router
  auxiliary gradient is preserved byte-equivalent to the full-batch path.
* exactly one gradient clip (independent shared / ple_dense / ple_sparse groups)
  and one optimizer step per logical batch.

With a single microbatch this degenerates exactly to ``train_step``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .optim import clip_gradient_groups
from .training import _as_float, _finite_tensor


class _StatsAccumulator:
    """Accumulate per-microbatch, per-layer sufficient statistics in place."""

    def __init__(self, num_experts: int, device: torch.device):
        self.num_experts = num_experts
        self.device = device
        self.exp_counts: dict[int, torch.Tensor] = {}
        self.prob_sum: dict[int, torch.Tensor] = {}
        self.token_count: dict[int, int] = {}
        self.slot_count: dict[int, int] = {}
        self.expert_load_sum: dict[int, torch.Tensor] = {}

    def add_layer_stats(self, stats: dict[str, Any]) -> None:
        for key in (
            "router_exp_counts",
            "router_prob_sum",
            "router_token_count",
            "router_slot_count",
        ):
            values = stats.get(key)
            if not values:
                continue
            for layer, value in enumerate(values):
                if key == "router_exp_counts":
                    base = self.exp_counts.get(
                        layer,
                        torch.zeros((self.num_experts,), dtype=value.dtype, device=value.device),
                    )
                    self.exp_counts[layer] = base + value
                    # Deterministic expert load (counts / slots) for metrics.
                    base_load = self.expert_load_sum.get(
                        layer,
                        torch.zeros((self.num_experts,), dtype=value.dtype, device=value.device),
                    )
                    self.expert_load_sum[layer] = base_load + value
                elif key == "router_prob_sum":
                    prev = self.prob_sum.get(layer)
                    self.prob_sum[layer] = (prev + value) if prev is not None else value
                elif key == "router_token_count":
                    self.token_count[layer] = self.token_count.get(layer, 0) + int(value.item())
                elif key == "router_slot_count":
                    self.slot_count[layer] = self.slot_count.get(layer, 0) + int(value.item())


def _logical_aux(
    num_experts: int,
    exp_counts: dict[int, torch.Tensor],
    prob_sum: dict[int, torch.Tensor],
    token_count: dict[int, int],
    slot_count: dict[int, int],
) -> torch.Tensor:
    """Reconstruct the frozen-C logical-batch aux from accumulated stats.

    Returns the mean over MoE layers of the per-layer scalar
    ``num_experts * sum_e( (counts_e / slot) * (prob_e / token) )``.
    The probability term stays differentiable through the router.
    """
    if not exp_counts:
        return torch.zeros((), device=next(iter(prob_sum.values())).device)
    per_layer: list[torch.Tensor] = []
    for layer in sorted(exp_counts):
        frac_routed = exp_counts[layer] / slot_count[layer]      # routing data (detached)
        prob = prob_sum[layer] / token_count[layer]              # differentiable
        per_layer.append((num_experts * (frac_routed * prob).sum()))
    return torch.stack(per_layer).mean()


def pipeline_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    aux_loss_coef: float | None = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> dict:
    """Run one optimizer update over a logical batch split into microbatches.

    ``microbatches`` is a list of ``(input_ids, labels)`` pairs; together they
    form one logical batch that produces a single optimizer update.  The result
    is metric-for-metric identical to a full-batch :func:`train_step` run on the
    same logical batch.
    """
    m = len(microbatches)
    if not microbatches:
        raise ValueError("pipeline_train_step requires at least one microbatch")
    config = getattr(model, "config", None)
    arch = getattr(config, "architecture_version", 2)
    moe_cfg = getattr(config, "moe", None) if config is not None else None
    num_experts = getattr(moe_cfg, "num_experts", 0)
    if aux_loss_coef is None:
        aux_loss_coef = moe_cfg.aux_loss_coef if (arch >= 3 and moe_cfg is not None) else 0.01

    input_device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logical_ce = torch.zeros((), device=input_device, requires_grad=True)
    logical_ce_val = 0.0
    acc = _StatsAccumulator(num_experts, input_device)
    entropy_sum = 0.0
    entropy_tokens = 0
    ple_scale_value = None
    ple_norm_ratio_sum = 0.0

    for input_ids, labels in microbatches:
        input_ids = input_ids.to(input_device)
        labels = labels.to(input_device)
        if use_amp and input_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
        else:
            out = model(input_ids, labels=labels)

        ce_mb = out["loss"]
        if not _finite_tensor(ce_mb):
            raise FloatingPointError("microbatch CE is non-finite")
        contrib = ce_mb / m  # sum over microbatches -> full-batch CE mean
        logical_ce = logical_ce + contrib
        logical_ce_val += _as_float(contrib, "microbatch CE contribution")

        stats = out.get("stats") or {}
        acc.add_layer_stats(stats)
        if stats.get("router_entropy"):
            for value in stats["router_entropy"]:
                entropy_sum += _as_float(value, "router entropy")
            entropy_tokens += sum(int(value.item()) for value in stats.get("router_token_count") or [])
        if "ple_scale" in stats:
            ple_scale_value = stats["ple_scale"]
        if "ple_norm_ratio" in stats:
            ple_norm_ratio_sum += _as_float(stats["ple_norm_ratio"], "ple_norm_ratio")

    logical_aux = _logical_aux(
        num_experts, acc.exp_counts, acc.prob_sum, acc.token_count, acc.slot_count
    )
    if not _finite_tensor(logical_aux):
        raise FloatingPointError("router auxiliary loss is non-finite")
    total_loss = logical_ce + aux_loss_coef * logical_aux
    if not _finite_tensor(total_loss):
        raise FloatingPointError("total loss is non-finite before backward")

    total_loss.backward()

    clipping: dict[str, Any] = {}
    if arch >= 3:
        clipping = clip_gradient_groups(model, grad_clip)
        grad_norm = (
            sum(clipping[f"grad_norm_{group}_preclip"] ** 2
                for group in ("shared", "ple_dense", "ple_sparse")) ** 0.5
        )
    else:
        from .optim import clip_gradients
        grad_norm = clip_gradients(model, grad_clip)
    grad_norm_value = _as_float(grad_norm, "gradient norm")
    optimizer.step()

    metrics: dict[str, Any] = {
        "loss": logical_ce_val,
        "total_loss": _as_float(total_loss, "total loss"),
        "grad_norm": grad_norm_value,
        "router_aux_loss": _as_float(logical_aux, "router auxiliary loss"),
        **clipping,
    }
    if entropy_tokens > 0:
        metrics["router_entropy"] = entropy_sum / entropy_tokens
    if acc.expert_load_sum:
        load_tensors = [v / acc.slot_count[layer] for layer, v in acc.expert_load_sum.items()]
        stacked = torch.stack(load_tensors).mean(0)
        load_mean = float(stacked.mean().item())
        metrics["expert_load_max"] = float(stacked.max().item())
        metrics["expert_load_mean"] = load_mean
        metrics["expert_load_ratio"] = float(stacked.max().item() / (load_mean + 1e-9))
        metrics["expert_load_dist"] = [float(x) for x in stacked.detach().cpu().tolist()]
    if ple_scale_value is not None:
        metrics["ple_scale"] = _as_float(ple_scale_value, "ple_scale")
    if m > 0:
        metrics["ple_norm_ratio"] = ple_norm_ratio_sum / m
    return metrics
