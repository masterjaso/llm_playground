"""Dense backbone AdamW and sparse, device-local PLE Adam updates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class _OptimizerPair:
    """One training-loop interface; each optimizer retains its state on its device."""

    def __init__(self, dense, sparse):
        self.optimizers = [dense, sparse]
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return {"format": "dense-sparse-v1", "optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        if state.get("format") != "dense-sparse-v1" or len(state["optimizers"]) != 2:
            raise ValueError("Checkpoint optimizer does not match dense/sparse PLE training")
        for opt, saved in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(saved)
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]


def build_optimizer(model: nn.Module, lr: float, ple_lr_multiplier: float = 1.0):
    """SparseAdam updates touched rows without weight decay; moments stay by the table.

    SparseAdam moments are dense CPU arrays for CPU tables. This saves GPU memory,
    not total host RAM; the table plus two moments needs roughly 3x its weight bytes.
    """
    if lr <= 0 or ple_lr_multiplier <= 0:
        raise ValueError("Learning rates must be positive")
    sparse = [m.weight for m in model.modules() if isinstance(m, nn.Embedding) and m.sparse]
    sparse_ids = {id(p) for p in sparse}
    decay, no_decay, dense_tables = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in sparse_ids:
            continue
        if name.endswith("value_embed.weight"):
            dense_tables.append(p)
        elif p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": ps, "lr": rate, "weight_decay": wd, "name": name}
              for ps, rate, wd, name in [(decay, lr, 0.1, "backbone_decay"),
                                         (no_decay, lr, 0.0, "norms_and_gates"),
                                         (dense_tables, lr * ple_lr_multiplier, 0.0, "ple_table")]
              if ps]
    dense = torch.optim.AdamW(groups, lr=lr)
    recipe = {"dense_family": "AdamW", "table_family": "SparseAdam" if getattr(
        getattr(model, "config", None), "ple", None) is not None and model.config.ple.sparse else "AdamW",
        "base_lr": lr, "ple_lr_multiplier": ple_lr_multiplier,
        "dense_weight_decay": 0.1, "table_weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8}
    if not sparse:
        dense._flashmini_recipe = recipe
        return dense
    sparse_opt = torch.optim.SparseAdam([{"params": sparse, "lr": lr * ple_lr_multiplier,
                                          "name": "ple_table"}])
    pair = _OptimizerPair(dense, sparse_opt)
    pair._flashmini_recipe = recipe
    return pair


def clip_gradients(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Global L2 clipping across CPU/CUDA and sparse rows, rejecting nonfinite values."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    grads = []
    norms_by_device = {}
    for p in model.parameters():
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            p.grad = p.grad.coalesce()
            g = p.grad.values()
        else:
            g = p.grad
        norms_by_device.setdefault(g.device, []).append(g.detach().float().norm())
        grads.append(g)
    squares = 0.0
    for norms in norms_by_device.values():
        aggregate = torch.stack(norms).square().sum()
        if not torch.isfinite(aggregate).item():
            raise FloatingPointError("Nonfinite gradient; optimizer step refused")
        squares += float(aggregate)
    norm = squares ** 0.5
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    with torch.no_grad():
        for g in grads:
            g.mul_(coefficient)
    return torch.tensor(norm)


_GRADIENT_GROUPS = ("shared", "ple_dense", "ple_sparse")


def gradient_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Return the v3 clipping groups for a FlashMini model.

    The PLE module is identified by its module path rather than by a fixed
    parameter list so adding another dense PLE parameter cannot accidentally
    place it in the shared clipping pool. Sparse embedding weights are kept in
    their own group, even when they live on the CPU for an offloaded model.
    """
    ple_ids: set[int] = set()
    for module_name, module in model.named_modules():
        path = module_name.split(".") if module_name else []
        if (path and path[-1] == "ple") or module.__class__.__name__.lower() == "ple":
            ple_ids.update(id(parameter) for parameter in module.parameters())

    sparse_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.sparse
    }
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in _GRADIENT_GROUPS}
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if id(parameter) in sparse_ids:
            group = "ple_sparse"
        elif id(parameter) in ple_ids:
            group = "ple_dense"
        else:
            group = "shared"
        groups[group].append(parameter)
    return groups


def _group_gradient_values(parameters: Iterable[nn.Parameter]) -> list[torch.Tensor]:
    """Collect gradient values without converting sparse tensors to dense."""
    values: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            # Coalescing preserves the sparse layout and is required before
            # values can be scaled safely when duplicate row indices exist.
            parameter.grad = gradient.coalesce()
            values.append(parameter.grad.values())
        else:
            values.append(gradient)
    return values


def _gradient_stats(values: Iterable[torch.Tensor], max_norm: float) -> tuple[float, float, bool]:
    """Return one group's (preclip norm, coefficient, clipped) tuple."""
    values = list(values)
    if not values:
        return 0.0, 1.0, False
    norms_by_device: dict[torch.device, list[torch.Tensor]] = {}
    for value in values:
        norms_by_device.setdefault(value.device, []).append(value.detach().float().norm())
    squares = 0.0
    for norms in norms_by_device.values():
        aggregate = torch.stack(norms).square().sum()
        if not torch.isfinite(aggregate).item():
            raise FloatingPointError("Nonfinite gradient; optimizer step refused")
        squares += float(aggregate)
    norm = squares ** 0.5
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    return norm, coefficient, coefficient < 1.0


def _scale_gradient_values(values: Iterable[torch.Tensor], coefficient: float) -> None:
    """Scale dense or sparse value tensors in place without changing layout."""
    with torch.no_grad():
        for value in values:
            value.mul_(coefficient)


def clip_gradient_groups(model: nn.Module, max_norm: float) -> dict[str, Any]:
    """Clip shared, PLE-dense, and PLE-sparse gradients independently.

    Unlike :func:`clip_gradients`, a large PLE gradient cannot change the
    coefficient applied to shared backbone gradients. Sparse PLE gradients
    are coalesced and scaled through ``values()`` only, so the full embedding
    table is never materialized as a dense gradient.

    The result is JSON-friendly and contains one preclip norm, coefficient,
    and clipping flag for each group. A group with no gradient reports norm
    zero, coefficient one, and ``False``.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    metrics: dict[str, Any] = {}
    groups = gradient_parameter_groups(model)
    values_by_group: dict[str, list[torch.Tensor]] = {}
    stats_by_group: dict[str, tuple[float, float, bool]] = {}
    for group_name in _GRADIENT_GROUPS:
        values = _group_gradient_values(groups[group_name])
        values_by_group[group_name] = values
        stats_by_group[group_name] = _gradient_stats(values, max_norm)
    # Compute every group's norm before scaling any group. A nonfinite PLE
    # gradient therefore cannot leave the shared gradients partially clipped.
    for group_name in _GRADIENT_GROUPS:
        norm, coefficient, clipped = stats_by_group[group_name]
        _scale_gradient_values(values_by_group[group_name], coefficient)
        metrics[f"grad_norm_{group_name}_preclip"] = norm
        metrics[f"grad_clip_coefficient_{group_name}"] = coefficient
        metrics[f"grad_clipped_{group_name}"] = clipped
    return metrics
