"""Dense backbone AdamW and sparse, device-local PLE Adam updates."""

from __future__ import annotations

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
    if not sparse:
        return dense
    sparse_opt = torch.optim.SparseAdam([{"params": sparse, "lr": lr * ple_lr_multiplier,
                                          "name": "ple_table"}])
    return _OptimizerPair(dense, sparse_opt)


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
