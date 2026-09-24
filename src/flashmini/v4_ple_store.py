"""Host-resident PLE hash-head tables with sparse row transfer and sparse Adam.

Storage contract (``ple.training_storage = bf16_master_host_or_distributed_off_accelerator``):

* Weights: sixteen independent BF16 tensors ``ple.tables.<h>.weight`` held in
  host memory (process-local or node-shared ``/dev/shm``).  They are not
  ``nn.Parameter``s, so ``model.to(device)`` and FSDP never move or shard them.
* Lookup: per head, the unique touched rows of the current microbatch are
  gathered on host and copied to the accelerator as a small leaf tensor.  The
  accelerator working set is ``unique_rows x head_dim`` per head, never a table.
* Gradients: after backward the leaf gradients (touched rows only) are copied
  back to host and coalesced per head in sorted row order.
* Update ownership: head ``h`` is owned by global rank ``h % world_size``.  Every
  rank sends its coalesced rows for ``h`` to the owner, the owner applies the
  update and broadcasts the updated rows so every replica stays identical.
* Optimizer: Adam, weight decay 0, fp32 ``exp_avg``/``exp_avg_sq`` plus an int32
  per-row step count, allocated only for owned heads.  Untouched rows are never
  modified; a touched row receives exactly the Adam update of its own gradient
  sequence (bias correction uses that row's own step count).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

TABLE_DTYPE = torch.bfloat16


def table_name(head: int) -> str:
    return f"ple.tables.{head}.weight"


class PLETableStore:
    """Sixteen independently addressable host tables plus pending row gradients."""

    def __init__(self, rows: list[int], head_dim: int, *, device: torch.device | str | None = None,
                 dtype: torch.dtype = TABLE_DTYPE):
        self.rows = [int(value) for value in rows]
        self.head_dim = int(head_dim)
        self.dtype = dtype
        device = torch.empty(0).device if device is None else torch.device(device)
        self.tables = [torch.empty(count, self.head_dim, dtype=dtype, device=device) for count in self.rows]
        self._pending: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._grads: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.stats = PLEStats()

    @property
    def num_heads(self) -> int:
        return len(self.rows)

    @property
    def device(self) -> torch.device:
        return self.tables[0].device

    def names(self) -> list[str]:
        return [table_name(head) for head in range(self.num_heads)]

    def named_tables(self):
        for head, table in enumerate(self.tables):
            yield table_name(head), table

    def allocate(self, *, device: str = "cpu", backing: str = "process", shm_dir: str | None = None,
                 writer: bool = True) -> None:
        """Replace meta/placeholder tables with real host storage.

        ``backing="shm"`` maps each table to ``<shm_dir>/ple_head_<h>.bin`` so all
        ranks on a node share one copy; only ``writer`` ranks mutate it.
        """
        self.backing, self.writer = backing, writer
        tables = []
        for head, count in enumerate(self.rows):
            numel = count * self.head_dim
            if backing == "shm":
                if shm_dir is None:
                    raise ValueError("shm backing requires shm_dir")
                path = Path(shm_dir) / f"ple_head_{head:02d}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                tensor = torch.from_file(str(path), shared=True, size=numel, dtype=self.dtype).view(count, self.head_dim)
            elif backing == "process":
                tensor = torch.empty(count, self.head_dim, dtype=self.dtype, device=device)
            else:
                raise ValueError(f"unknown PLE backing {backing!r}")
            tables.append(tensor)
        self.tables = tables

    def lookup(self, keys: torch.Tensor, *, device: torch.device, dtype: torch.dtype, requires_grad: bool) -> torch.Tensor:
        """Gather rows for ``keys`` (B, T, heads) -> (B, T, heads * head_dim)."""
        if keys.shape[-1] != self.num_heads:
            raise ValueError(f"expected {self.num_heads} PLE heads, got {keys.shape[-1]}")
        outputs = []
        for head, table in enumerate(self.tables):
            flat = keys[..., head].reshape(-1)
            unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
            host_rows = unique.to(table.device)
            if table.device.type == "meta":
                rows = torch.empty(unique.numel(), self.head_dim, device="meta", dtype=dtype)
            else:
                gathered = table.index_select(0, host_rows)
                if gathered.device.type == "cpu" and torch.device(device).type == "cuda":
                    gathered = gathered.pin_memory()
                rows = gathered.to(device=device, dtype=dtype, non_blocking=True)
            self.stats.lookups += int(flat.numel())
            self.stats.unique_rows += int(unique.numel())
            self.stats.transfer_bytes += int(unique.numel()) * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()
            self.stats.max_working_set_rows = max(self.stats.max_working_set_rows, int(unique.numel()))
            if requires_grad:
                rows = rows.detach().requires_grad_(True)
                self._pending.append((head, host_rows.to("cpu"), rows))
            outputs.append(rows[inverse.to(rows.device)].reshape(*keys.shape[:-1], self.head_dim))
        return torch.cat(outputs, dim=-1)

    def collect_gradients(self) -> None:
        """Move touched-row gradients of completed backward passes to host."""
        for head, rows, leaf in self._pending:
            if leaf.grad is not None:
                self._grads.setdefault(head, []).append((rows, leaf.grad.detach().to("cpu", torch.float32)))
        self._pending.clear()

    def discard_pending(self) -> None:
        self._pending.clear()

    def coalesced_gradients(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per head: (sorted unique rows, summed fp32 grads) for this rank."""
        self.collect_gradients()
        result = {}
        for head, parts in sorted(self._grads.items()):
            ids = torch.cat([part[0] for part in parts])
            grads = torch.cat([part[1] for part in parts])
            result[head] = coalesce(ids, grads, self.head_dim)
        return result

    def zero_grad(self) -> None:
        self._pending.clear()
        self._grads.clear()

    def write_rows(self, head: int, rows: torch.Tensor, values: torch.Tensor) -> None:
        if getattr(self, "writer", True):
            self.tables[head].index_copy_(0, rows, values.to(self.dtype))


def coalesce(ids: torch.Tensor, grads: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    summed = torch.zeros(unique.numel(), head_dim, dtype=torch.float32)
    summed.index_add_(0, inverse, grads.to(torch.float32))
    return unique, summed


@dataclass
class PLEStats:
    lookups: int = 0
    unique_rows: int = 0
    transfer_bytes: int = 0
    max_working_set_rows: int = 0
    updated_rows: int = 0
    update_bytes: int = 0

    def snapshot_and_reset(self) -> dict[str, int]:
        values = dict(self.__dict__)
        for key in values:
            setattr(self, key, 0)
        return values


@dataclass
class PLEAdamState:
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor


@dataclass
class PLESparseAdam:
    """Row-sparse Adam (weight decay 0) over a :class:`PLETableStore`."""

    store: PLETableStore
    lr: float
    betas: tuple[float, float]
    eps: float
    group: Any = None
    state: dict[int, PLEAdamState] = field(default_factory=dict)

    def __post_init__(self):
        if not (self.lr > 0 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1 and self.eps > 0):
            raise ValueError("invalid PLE Adam hyperparameters")
        self.world = dist.get_world_size(self.group) if self._distributed else 1
        self.rank = dist.get_rank(self.group) if self._distributed else 0

    @property
    def _distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.group is not False

    def owner(self, head: int) -> int:
        return head % self.world

    def owned_heads(self) -> list[int]:
        return [head for head in range(self.store.num_heads) if self.owner(head) == self.rank]

    def _state_for(self, head: int) -> PLEAdamState:
        if head not in self.state:
            rows, dim = self.store.rows[head], self.store.head_dim
            self.state[head] = PLEAdamState(
                torch.zeros(rows, dim, dtype=torch.float32),
                torch.zeros(rows, dim, dtype=torch.float32),
                torch.zeros(rows, dtype=torch.int32),
            )
        return self.state[head]

    def reduce_gradients(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Return globally summed sparse grads for the heads this rank owns."""
        local = self.store.coalesced_gradients()
        if self.world == 1:
            return local
        heads = self.store.num_heads
        lengths = torch.tensor([local[h][0].numel() if h in local else 0 for h in range(heads)], dtype=torch.int64)
        all_lengths = [torch.zeros_like(lengths) for _ in range(self.world)]
        dist.all_gather(all_lengths, lengths, group=self.group)
        table = torch.stack(all_lengths)  # (world, heads)
        result = {}
        dim = self.store.head_dim
        for head in range(heads):
            owner = self.owner(head)
            width = int(table[:, head].max())
            if width == 0:
                continue
            ids = torch.full((width,), -1, dtype=torch.int64)
            grads = torch.zeros(width, dim, dtype=torch.float32)
            if head in local:
                count = local[head][0].numel()
                ids[:count], grads[:count] = local[head]
            gather_ids = [torch.empty_like(ids) for _ in range(self.world)] if self.rank == owner else None
            gather_grads = [torch.empty_like(grads) for _ in range(self.world)] if self.rank == owner else None
            dist.gather(ids, gather_ids, dst=owner, group=self.group)
            dist.gather(grads, gather_grads, dst=owner, group=self.group)
            if self.rank == owner:
                all_ids = torch.cat([part[: int(table[r, head])] for r, part in enumerate(gather_ids)])
                all_grads = torch.cat([part[: int(table[r, head])] for r, part in enumerate(gather_grads)])
                result[head] = coalesce(all_ids, all_grads, dim)
        return result

    @staticmethod
    def squared_norm(reduced: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for _, grads in reduced.values():
            total += grads.double().square().sum()
        return total

    def global_squared_norm(self, reduced) -> torch.Tensor:
        value = self.squared_norm(reduced)
        if self.world > 1:
            dist.all_reduce(value, group=self.group)
        return value

    def step(self, reduced: dict[int, tuple[torch.Tensor, torch.Tensor]], *, lr: float | None = None,
             grad_scale: float = 1.0) -> None:
        lr = self.lr if lr is None else lr
        beta1, beta2 = self.betas
        updates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for head, (rows, grads) in reduced.items():
            if self.owner(head) != self.rank:
                raise RuntimeError(f"rank {self.rank} does not own PLE head {head}")
            state = self._state_for(head)
            grads = grads * grad_scale
            step = state.step.index_select(0, rows) + 1
            exp_avg = state.exp_avg.index_select(0, rows).mul_(beta1).add_(grads, alpha=1 - beta1)
            exp_avg_sq = state.exp_avg_sq.index_select(0, rows).mul_(beta2).addcmul_(grads, grads, value=1 - beta2)
            bias1 = 1 - beta1 ** step.double()
            bias2 = 1 - beta2 ** step.double()
            denom = (exp_avg_sq / bias2.unsqueeze(-1).float()).sqrt_().add_(self.eps)
            weight = self.store.tables[head].index_select(0, rows).float()
            weight = weight - lr * (exp_avg / bias1.unsqueeze(-1).float()) / denom
            state.exp_avg.index_copy_(0, rows, exp_avg)
            state.exp_avg_sq.index_copy_(0, rows, exp_avg_sq)
            state.step.index_copy_(0, rows, step.to(torch.int32))
            updates[head] = (rows, weight.to(self.store.dtype))
        self._apply_updates(updates)
        self.store.zero_grad()

    def _apply_updates(self, updates: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        dim = self.store.head_dim
        for head in range(self.store.num_heads):
            owner = self.owner(head)
            if self.world == 1:
                if head in updates:
                    self._write(head, *updates[head])
                continue
            count = torch.tensor([updates[head][0].numel() if head in updates else 0], dtype=torch.int64)
            dist.broadcast(count, src=owner, group=self.group)
            n = int(count)
            if n == 0:
                continue
            if self.rank == owner:
                rows, values = updates[head]
            else:
                rows = torch.empty(n, dtype=torch.int64)
                values = torch.empty(n, dim, dtype=self.store.dtype)
            dist.broadcast(rows, src=owner, group=self.group)
            dist.broadcast(values, src=owner, group=self.group)
            self._write(head, rows, values)
        if self.world > 1 and getattr(self.store, "backing", "process") == "shm":
            dist.barrier(group=self.group)

    def _write(self, head: int, rows: torch.Tensor, values: torch.Tensor) -> None:
        self.store.write_rows(head, rows, values)
        self.store.stats.updated_rows += int(rows.numel())
        self.store.stats.update_bytes += int(values.numel()) * values.element_size()

    # -- checkpointing -------------------------------------------------------------
    def save(self, directory: Path | str) -> list[str]:
        """Owner ranks write ``ple/head_<h>.safetensors`` with table and Adam state."""
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for head in self.owned_heads():
            state = self._state_for(head)
            path = directory / f"head_{head:02d}.safetensors"
            save_file({
                table_name(head): self.store.tables[head].contiguous(),
                "exp_avg": state.exp_avg, "exp_avg_sq": state.exp_avg_sq, "step": state.step,
            }, str(path), metadata={"head": str(head), "rows": str(self.store.rows[head]), "head_dim": str(self.store.head_dim)})
            written.append(str(path))
        return written

    def load(self, directory: Path | str) -> None:
        from safetensors import safe_open

        directory = Path(directory)
        for head in range(self.store.num_heads):
            path = directory / f"head_{head:02d}.safetensors"
            with safe_open(str(path), framework="pt") as handle:
                table = handle.get_tensor(table_name(head))
                if getattr(self.store, "writer", True):
                    self.store.tables[head].copy_(table)
                if self.owner(head) == self.rank:
                    self.state[head] = PLEAdamState(handle.get_tensor("exp_avg"), handle.get_tensor("exp_avg_sq"),
                                                    handle.get_tensor("step"))
        if self._distributed and self.world > 1 and getattr(self.store, "backing", "process") == "shm":
            dist.barrier(group=self.group)


def host_memory_requirement(rows: list[int], head_dim: int, *, world_size: int, ranks_per_node: int,
                            backing: str) -> dict[str, int]:
    """Bytes needed per node for replicated BF16 tables and owned fp32 Adam state."""
    table_bytes = sum(rows) * head_dim * 2
    heads = len(rows)
    per_rank_state = [0] * world_size
    for head, count in enumerate(rows):
        per_rank_state[head % world_size] += count * head_dim * 8 + count * 4
    node_state = max(sum(per_rank_state[start:start + ranks_per_node]) for start in range(0, world_size, ranks_per_node))
    replicas = 1 if backing == "shm" else ranks_per_node
    return {"table_bytes": table_bytes, "table_replicas_per_node": replicas,
            "node_state_bytes": node_state, "node_total_bytes": table_bytes * replicas + node_state,
            "heads": heads}


def available_host_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - psutil is a declared dependency
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


__all__ = ["PLESparseAdam", "PLEStats", "PLETableStore", "available_host_bytes", "coalesce", "host_memory_requirement", "table_name"]
