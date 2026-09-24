"""Canonical v4 optimizer stack: logical-slice Muon, AdamW, and row-sparse PLE Adam.

Muon (Moonshot scaling): per tensor ``buf = mu * buf + g``; the Nesterov update
``u = g + mu * buf`` is split into its logical slices; each slice is orthogonalized
independently by 8 quintic Newton-Schulz iterations and scaled by
``0.2 * sqrt(max(rows, columns))`` of that slice; then decoupled weight decay
``p *= 1 - lr * wd`` and ``p -= lr * update``.

Distributed Muon: for FSDP2 (DTensor ``Shard(0)``) parameters the owner rank
``index % world`` gathers the full momentum-corrected gradient, orthogonalizes
every slice, and scatters the update shards back.  Replicated (single process or
replicated data parallel) parameters are updated locally with identical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .base_init_optimizer import ADAMW, MUON, PLE_ADAM, OptimizerTaxonomy, ParameterClass
from .v4_ple_store import PLESparseAdam

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

try:  # DTensor is present in every supported torch build; keep import local-safe.
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover
    DTensor = ()  # type: ignore[assignment]


def newton_schulz(matrix: torch.Tensor, steps: int = 8, eps: float = 1e-14, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    a, b, c = NS_COEFFICIENTS
    x = matrix.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class LogicalMuon(torch.optim.Optimizer):
    def __init__(self, named: list[tuple[str, torch.nn.Parameter, ParameterClass]], *, lr: float, weight_decay: float,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 8, eps: float = 1e-14,
                 ns_dtype: torch.dtype = torch.float32, group: Any = None):
        for name, _, cls in named:
            if cls.family != MUON:
                raise ValueError(f"{name} is {cls.family}, not Muon")
        super().__init__([{"params": [p for _, p, _ in named], "lr": lr, "weight_decay": weight_decay}],
                         {"lr": lr, "weight_decay": weight_decay})
        self.names = [name for name, _, _ in named]
        self.classes = [cls for _, _, cls in named]
        self.momentum, self.nesterov, self.ns_steps, self.eps, self.ns_dtype = momentum, nesterov, ns_steps, eps, ns_dtype
        self.group = group
        self.world = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if self.world > 1 else 0

    def _orthogonalized(self, full: torch.Tensor, cls: ParameterClass) -> torch.Tensor:
        update = torch.empty_like(full, dtype=torch.float32)
        for item in cls.slices:
            rows = item.row_index(full.device)
            matrix = full.index_select(0, rows).float()
            scale = 0.2 * math.sqrt(max(item.rows, item.columns))
            update.index_copy_(0, rows, newton_schulz(matrix, self.ns_steps, self.eps, self.ns_dtype) * scale)
        return update

    def _sharded_update(self, index: int, source: torch.Tensor, cls: ParameterClass) -> torch.Tensor:
        owner = index % self.world
        total_rows = source.shape[0]
        chunk = -(-total_rows // self.world)
        local = _local(source).float()
        padded = F.pad(local, (0, 0, 0, chunk - local.shape[0]))
        parts = [torch.empty_like(padded) for _ in range(self.world)] if self.rank == owner else None
        dist.gather(padded, parts, dst=owner, group=self.group)
        if self.rank == owner:
            update = self._orthogonalized(torch.cat(parts)[:total_rows], cls)
            scatter = list(F.pad(update, (0, 0, 0, chunk * self.world - total_rows)).split(chunk))
        else:
            scatter = None
        received = torch.empty_like(padded)
        dist.scatter(received, scatter, src=owner, group=self.group)
        return received[: local.shape[0]]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, decay = group["lr"], group["weight_decay"]
            for index, (param, cls) in enumerate(zip(group["params"], self.classes)):
                grad = param.grad if param.grad is not None else torch.zeros_like(param)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buffer = state["momentum_buffer"]
                buffer.mul_(self.momentum).add_(grad)
                source = grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
                if isinstance(param, DTensor) and self.world > 1:
                    update = self._sharded_update(index, source, cls)
                else:
                    update = self._orthogonalized(_local(source), cls)
                local = _local(param)
                local.mul_(1 - lr * decay)
                local.add_(update.to(local.dtype), alpha=-lr)
        return None

    def state_dict(self):
        state = super().state_dict()
        state["flashmini_logical_slices"] = {name: [item.as_dict() for item in cls.slices] for name, cls in zip(self.names, self.classes)}
        return state

    def load_state_dict(self, state_dict):
        expected = {name: [item.as_dict() for item in cls.slices] for name, cls in zip(self.names, self.classes)}
        if state_dict.get("flashmini_logical_slices") != expected:
            raise ValueError("Muon checkpoint logical slices do not match the current model")
        state_dict = dict(state_dict)
        state_dict.pop("flashmini_logical_slices")
        super().load_state_dict(state_dict)


@dataclass
class OptimizerSettings:
    muon_lr: float
    muon_weight_decay: float
    muon_ns_dtype: str
    adamw_lr: float
    adamw_betas: tuple[float, float]
    adamw_eps: float
    adamw_weight_decay: dict[str, float]
    ple_lr: float
    ple_betas: tuple[float, float]
    ple_eps: float
    grad_clip: float

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "OptimizerSettings":
        muon, adamw, ple = raw["muon"], raw["adamw"], raw["ple"]
        decay = dict(adamw["weight_decay"])
        if set(decay) != {"embedding", "control_matrix", "no_decay"}:
            raise ValueError("adamw.weight_decay must set embedding, control_matrix and no_decay explicitly")
        if decay["no_decay"] != 0:
            raise ValueError("adamw.weight_decay.no_decay must be 0 (norms, scalars, controls)")
        if muon["ns_dtype"] not in {"float32", "bfloat16"}:
            raise ValueError("muon.ns_dtype must be float32 or bfloat16")
        return cls(float(muon["lr"]), float(muon["weight_decay"]), muon["ns_dtype"], float(adamw["lr"]),
                   tuple(float(v) for v in adamw["betas"]), float(adamw["eps"]), {k: float(v) for k, v in decay.items()},
                   float(ple["lr"]), tuple(float(v) for v in ple["betas"]), float(ple["eps"]), float(raw["grad_clip"]))


def classify_model(model, taxonomy: OptimizerTaxonomy) -> list[tuple[str, torch.nn.Parameter, ParameterClass]]:
    """Classify every trainable parameter; fail closed on unknown or mixed-family tensors."""
    result = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            raise ValueError(f"v4 has no frozen parameters; {name} has requires_grad=False")
        cls = taxonomy.classify(name, tuple(param.shape))
        if cls.family not in {MUON, ADAMW}:
            raise ValueError(f"{name}: dense parameter classified as {cls.family}")
        result.append((name, param, cls))
    for name in model.ple.store.names():
        if taxonomy.classify(name, (model.ple.store.rows[int(name.split('.')[2])], model.ple.store.head_dim)).family != PLE_ADAM:
            raise ValueError(f"{name}: PLE table not classified as PLE_Adam")
    return result


class OptimizerStack:
    """Muon + AdamW + PLE Adam with one global gradient clip and per-family norms."""

    def __init__(self, model, taxonomy: OptimizerTaxonomy, settings: OptimizerSettings, *, group: Any = None,
                 ple_group: Any = None):
        self.settings = settings
        classified = classify_model(model, taxonomy)
        muon = [item for item in classified if item[2].family == MUON]
        adamw = [item for item in classified if item[2].family == ADAMW]
        self.families = {MUON: [p for _, p, _ in muon], ADAMW: [p for _, p, _ in adamw]}
        self.muon = LogicalMuon(muon, lr=settings.muon_lr, weight_decay=settings.muon_weight_decay,
                                ns_dtype=getattr(torch, settings.muon_ns_dtype), group=group)
        groups = []
        for decay_class in ("embedding", "control_matrix", "no_decay"):
            params = [p for _, p, cls in adamw if cls.decay_class == decay_class]
            if params:
                groups.append({"params": params, "weight_decay": settings.adamw_weight_decay[decay_class], "decay_class": decay_class})
        self.adamw_names = [name for _, _, cls in adamw for name in [cls.name]]
        self.adamw = torch.optim.AdamW(groups, lr=settings.adamw_lr, betas=settings.adamw_betas, eps=settings.adamw_eps, foreach=False)
        self.ple = PLESparseAdam(model.ple.store, lr=settings.ple_lr, betas=settings.ple_betas, eps=settings.ple_eps, group=ple_group)
        self.base_lr = {"muon": settings.muon_lr, "adamw": settings.adamw_lr, "ple": settings.ple_lr}
        self.lr_multiplier = 1.0
        self.classified = classified

    def set_lr_multiplier(self, multiplier: float) -> None:
        self.lr_multiplier = float(multiplier)
        for group in self.muon.param_groups:
            group["lr"] = self.base_lr["muon"] * multiplier
        for group in self.adamw.param_groups:
            group["lr"] = self.base_lr["adamw"] * multiplier

    def current_lrs(self) -> dict[str, float]:
        return {name: value * self.lr_multiplier for name, value in self.base_lr.items()}

    @staticmethod
    def _squared_norm(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return torch.zeros((), dtype=torch.float64)
        norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0, foreach=False)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        return norm.detach().double().cpu().square()

    def zero_grad(self) -> None:
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)
        self.ple.store.zero_grad()

    def step(self) -> dict[str, Any]:
        """Reduce PLE grads, clip globally, fail on non-finite, then update all families."""
        reduced = self.ple.reduce_gradients()
        squared = {MUON: self._squared_norm(self.families[MUON]), ADAMW: self._squared_norm(self.families[ADAMW]),
                   PLE_ADAM: self.ple.global_squared_norm(reduced)}
        total = math.sqrt(float(sum(squared.values())))
        norms = {family: math.sqrt(float(value)) for family, value in squared.items()}
        if not math.isfinite(total):
            self.zero_grad()
            raise FloatingPointError(f"non-finite gradient norm {norms}; optimizer step refused")
        clip = self.settings.grad_clip
        coefficient = min(1.0, clip / (total + 1e-6)) if clip > 0 else 1.0
        if coefficient < 1.0:
            for params in self.families.values():
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(coefficient)
        self.muon.step()
        self.adamw.step()
        self.ple.step(reduced, lr=self.base_lr["ple"] * self.lr_multiplier, grad_scale=coefficient)
        self.zero_grad()
        return {"grad_norm_total": total, "grad_norm_by_family": norms, "clip_coefficient": coefficient,
                "ple_rows_updated": int(sum(rows.numel() for rows, _ in reduced.values()))}

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict(), "lr_multiplier": self.lr_multiplier,
                "adamw_names": self.adamw_names}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["adamw_names"] != self.adamw_names:
            raise ValueError("AdamW checkpoint parameter order does not match the current model")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.set_lr_multiplier(state["lr_multiplier"])


__all__ = ["LogicalMuon", "OptimizerSettings", "OptimizerStack", "classify_model", "newton_schulz"]
