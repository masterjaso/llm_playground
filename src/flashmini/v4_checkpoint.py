"""Full training-state checkpoint and restore for the v4 runner.

Layout of ``<root>/step_<N>/`` (written to ``step_<N>.tmp`` and renamed after
every rank has finished; ``COMPLETE.json`` is written last):

* ``model/rank_<r>.safetensors`` - rank 0 only when replicated; every rank's local
  shard when FSDP2-sharded (restore then requires the identical topology).
* ``optim/rank_<r>.pt`` - Muon momentum, AdamW moments/steps, LR multiplier.
* ``ple/head_<h>.safetensors`` - BF16 table rows plus row-sparse Adam state (owners).
* ``rng/rank_<r>.pt`` - Python, NumPy, torch CPU and CUDA generator states.
* ``trainer_state.json`` - counters, data cursor, curriculum phase, MTP phase,
  identity fingerprints, source commit, topology.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

COMPLETE = "COMPLETE.json"


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _is_sharded(model: torch.nn.Module) -> bool:
    return any(isinstance(p, DTensor) for p in model.parameters())


def _to_local_tree(value: Any) -> Any:
    if isinstance(value, DTensor):
        return value.to_local().detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_local_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_local_tree(item) for item in value)
    return value


def _optimizer_params(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def _restore_optimizer(optimizer: torch.optim.Optimizer, saved: dict[str, Any]) -> None:
    params = _optimizer_params(optimizer)
    for index, state in saved["state"].items():
        param = params[int(index)]
        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor) or key == "step":
                continue
            if isinstance(param, DTensor):
                local = param.to_local()
                if value.shape != local.shape:
                    raise ValueError(f"optimizer state {key} shape {tuple(value.shape)} != local shard {tuple(local.shape)}")
                state[key] = DTensor.from_local(value.to(local.device, local.dtype), param.device_mesh, param.placements,
                                                shape=param.shape, stride=param.stride())
            else:
                state[key] = value.to(param.device)
    optimizer.load_state_dict(saved)


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def step_directory(root: Path | str, step: int) -> Path:
    return Path(root) / f"step_{step:08d}"


def save(root: Path | str, step: int, *, model: torch.nn.Module, stack, trainer_state: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    rank, world = _rank_world()
    final = step_directory(root, step)
    tmp = final.with_name(final.name + ".tmp")
    if rank == 0:
        if tmp.exists():
            shutil.rmtree(tmp)
        for sub in ("model", "optim", "ple", "rng"):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
    _barrier()
    sharded = _is_sharded(model)
    if sharded or rank == 0:
        tensors = {name: value.contiguous() for name, value in _to_local_tree(model.state_dict()).items()}
        save_file(tensors, str(tmp / "model" / f"rank_{rank:05d}.safetensors"))
        torch.save(_to_local_tree(stack.state_dict()), tmp / "optim" / f"rank_{rank:05d}.pt")
    stack.ple.save(tmp / "ple")
    torch.save(rng_state(), tmp / "rng" / f"rank_{rank:05d}.pt")
    _barrier()
    if rank == 0:
        state = dict(trainer_state, checkpoint={"step": step, "world_size": world, "model_sharded": sharded})
        (tmp / "trainer_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        files = sorted(str(path.relative_to(tmp)) for path in tmp.rglob("*") if path.is_file())
        (tmp / COMPLETE).write_text(json.dumps({"step": step, "files": files}, indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        latest = Path(root) / "latest"
        latest_tmp = Path(root) / "latest.tmp"
        latest_tmp.write_text(final.name + "\n")
        os.replace(latest_tmp, latest)
    _barrier()
    return final


def latest(root: Path | str) -> Path | None:
    pointer = Path(root) / "latest"
    if not pointer.exists():
        return None
    path = Path(root) / pointer.read_text().strip()
    return path if (path / COMPLETE).exists() else None


def load(directory: Path | str, *, model: torch.nn.Module, stack, expect: dict[str, Any]) -> dict[str, Any]:
    """Restore every state component; ``expect`` identity fields must match exactly."""
    from safetensors.torch import load_file

    directory = Path(directory)
    if not (directory / COMPLETE).exists():
        raise FileNotFoundError(f"{directory}: incomplete checkpoint (no {COMPLETE})")
    state = json.loads((directory / "trainer_state.json").read_text())
    for key, value in expect.items():
        if state.get(key) != value:
            raise ValueError(f"checkpoint {key}={state.get(key)!r} does not match current {value!r}")
    rank, world = _rank_world()
    sharded = _is_sharded(model)
    if state["checkpoint"]["model_sharded"] != sharded or (sharded and state["checkpoint"]["world_size"] != world):
        raise ValueError("checkpoint sharding/topology differs from the current run; resharding restore is not supported")
    source_rank = rank if sharded else 0
    tensors = load_file(str(directory / "model" / f"rank_{source_rank:05d}.safetensors"))
    current = model.state_dict()
    if set(tensors) != set(current):
        raise ValueError(f"checkpoint model keys differ: missing={sorted(set(current) - set(tensors))[:5]} extra={sorted(set(tensors) - set(current))[:5]}")
    with torch.no_grad():
        for name, target in current.items():
            local = target.to_local() if isinstance(target, DTensor) else target
            if local.shape != tensors[name].shape:
                raise ValueError(f"{name}: checkpoint shape {tuple(tensors[name].shape)} != {tuple(local.shape)}")
            local.copy_(tensors[name])
    optim = torch.load(directory / "optim" / f"rank_{source_rank:05d}.pt", weights_only=False)
    if optim["adamw_names"] != stack.adamw_names:
        raise ValueError("AdamW checkpoint parameter order does not match the current model")
    _restore_optimizer(stack.muon, optim["muon"])
    _restore_optimizer(stack.adamw, optim["adamw"])
    stack.set_lr_multiplier(optim["lr_multiplier"])
    stack.ple.load(directory / "ple")
    rng_path = directory / "rng" / f"rank_{rank:05d}.pt"
    if rng_path.exists():
        set_rng_state(torch.load(rng_path, weights_only=False))
    return state


__all__ = ["COMPLETE", "latest", "load", "rng_state", "save", "set_rng_state", "step_directory"]
