"""Optional PyTorch/XLA backend used by the Kaggle TPU worker.

The package remains importable on CPU-only workstations.  XLA imports happen
only when a worker explicitly requests a TPU, which keeps local unit tests and
the workstation controller free of accelerator side effects.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TPUBackendConfig:
    world_size: int = 8
    sequence_length: int = 2048
    static_shapes: bool = True
    dtype: str = "bfloat16"
    cache_dir: str = ""
    compile_timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if int(self.world_size) != 8:
            raise ValueError("FlashMini production targets all eight TPU v5e devices")
        if int(self.sequence_length) <= 0 or not self.static_shapes:
            raise ValueError("TPU backend requires a positive static sequence shape")
        if self.dtype != "bfloat16":
            raise ValueError("production TPU policy is BF16")


@dataclass
class TPUProbeResult:
    available: bool
    world_size: int
    device: str | None
    torch_version: str
    torch_xla_version: str | None
    topology: str
    compile_seconds: float | None = None
    first_step_seconds: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SessionBudget:
    """Soft-stop a Kaggle VM before the platform's hard runtime ceiling."""

    hard_limit_seconds: float = 21_600.0
    reserve_seconds: float = 3_600.0
    started_at: float = 0.0

    def start(self, now: float | None = None) -> None:
        self.started_at = time.monotonic() if now is None else float(now)

    @property
    def soft_limit_seconds(self) -> float:
        return max(0.0, float(self.hard_limit_seconds) - float(self.reserve_seconds))

    def elapsed(self, now: float | None = None) -> float:
        if not self.started_at:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.started_at)

    def should_soft_stop(self, now: float | None = None) -> bool:
        return self.elapsed(now) >= self.soft_limit_seconds

    def seconds_remaining(self, now: float | None = None) -> float:
        return max(0.0, self.soft_limit_seconds - self.elapsed(now))


def _xla_modules():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        return xm, xr
    except ImportError:
        return None, None


def tpu_available() -> bool:
    xm, xr = _xla_modules()
    if xm is None:
        return False
    try:
        if hasattr(xr, "world_size"):
            return int(xr.world_size()) >= 1
        return bool(xm.xla_device())
    except Exception:  # noqa: BLE001 - optional runtime probing
        return False


def discover_topology() -> dict[str, Any]:
    xm, xr = _xla_modules()
    result = {"available": xm is not None, "world_size": 0, "device": None,
              "torch_version": torch.__version__, "torch_xla_version": None,
              "topology": "unavailable"}
    try:
        import torch_xla
        result["torch_xla_version"] = getattr(torch_xla, "__version__", None)
    except ImportError:
        return result
    try:
        if xr is not None and hasattr(xr, "world_size"):
            result["world_size"] = int(xr.world_size())
        elif xm is not None:
            result["world_size"] = 1
        result["device"] = str(xm.xla_device()) if xm is not None else None
        result["topology"] = "TPU-v5e-8" if result["world_size"] == 8 else f"xla-{result['world_size']}"
    except Exception as exc:  # noqa: BLE001 - optional runtime probing
        result["error"] = str(exc)
    return result


def validate_topology(*, expected_world_size: int = 8, allow_local_cpu: bool = False) -> dict[str, Any]:
    topology = discover_topology()
    if not topology["available"]:
        if allow_local_cpu:
            return {**topology, "validated": False, "reason": "torch_xla_unavailable"}
        raise RuntimeError("PyTorch/XLA is unavailable; run this worker on Kaggle TPU v5e-8")
    if int(topology.get("world_size", 0)) != int(expected_world_size):
        raise RuntimeError(
            f"expected {expected_world_size} XLA devices, got {topology.get('world_size')}"
        )
    return {**topology, "validated": True}


def static_shape_guard(input_ids: torch.Tensor, labels: torch.Tensor, *, sequence_length: int,
                       batch_size: int | None = None) -> None:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("TPU batches must have matching rank-two input/label shapes")
    if int(input_ids.shape[1]) != int(sequence_length):
        raise ValueError("dynamic sequence length is forbidden on the production TPU path")
    if batch_size is not None and int(input_ids.shape[0]) != int(batch_size):
        raise ValueError("dynamic batch size is forbidden on the production TPU path")


class XLACompileCache:
    """Initialize the supported persistent cache when the installed runtime exposes it."""

    def __init__(self, root: Path | str, *, identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.identity = dict(identity)
        self.identity_hash = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.path = self.root / self.identity_hash[:24]

    def initialize(self) -> dict[str, Any]:
        self.path.mkdir(parents=True, exist_ok=True)
        marker = self.path / "cache_identity.json"
        if marker.is_file():
            old = json.loads(marker.read_text())
            if old.get("identity_hash") != self.identity_hash:
                # A mismatched cache is a cache miss, never an unsafe reuse.
                self.path = self.root / f"{self.identity_hash[:24]}-miss"
                self.path.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"identity": self.identity, "identity_hash": self.identity_hash}, indent=2))
        _xm, xr = _xla_modules()
        initialized = False
        if xr is not None:
            initialize_cache = getattr(xr, "initialize_cache", None)
            if callable(initialize_cache):
                try:
                    initialize_cache(str(self.path), readonly=False)
                    initialized = True
                except TypeError:
                    try:
                        initialize_cache(str(self.path))
                        initialized = True
                    except Exception:  # noqa: BLE001 - cache is best effort
                        initialized = False
                except Exception:  # noqa: BLE001 - cache is best effort
                    initialized = False
        return {"path": str(self.path), "identity_hash": self.identity_hash, "initialized": initialized}


class TPUBackend:
    """Thin XLA execution adapter; model/data semantics stay in FlashMini."""

    def __init__(self, config: TPUBackendConfig | None = None) -> None:
        self.config = config or TPUBackendConfig()
        self.xm, self.xr = _xla_modules()
        self.device: torch.device | None = None
        self.step_count = 0
        self.compile_count = 0

    def initialize(self, *, cache_identity: dict[str, Any] | None = None) -> dict[str, Any]:
        topology = validate_topology(expected_world_size=self.config.world_size)
        self.device = self.xm.xla_device()
        cache = None
        if self.config.cache_dir and cache_identity:
            cache = XLACompileCache(self.config.cache_dir, identity=cache_identity).initialize()
        return {"topology": topology, "cache": cache, "device": str(self.device)}

    def to_device(self, value: Any) -> Any:
        if self.device is None:
            raise RuntimeError("TPUBackend.initialize must run before to_device")
        return value.to(self.device)

    def shard_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap the model in XLA FSDP; never fall back to replicated 1B state."""
        if self.xm is None or self.device is None:
            raise RuntimeError("TPUBackend.initialize must run before shard_model")
        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
        except ImportError as exc:
            raise RuntimeError("PyTorch/XLA FSDP is required for the 1B TPU state") from exc
        # The wrapper owns parameter and optimizer-state sharding across the
        # PJRT world.  A caller may still provide an explicit SPMD mesh when a
        # newer runtime exposes one; the model semantics and names remain
        # backend-neutral.
        return XlaFullyShardedDataParallel(model.to(self.device), sync_module_states=True)

    def optimizer_step(self, optimizer: torch.optim.Optimizer, *, barrier: bool = True) -> None:
        if self.xm is None:
            raise RuntimeError("PyTorch/XLA is unavailable")
        self.xm.optimizer_step(optimizer, barrier=barrier)
        self.step_count += 1

    def mark_step(self) -> None:
        if self.xm is not None:
            self.xm.mark_step()

    def rendezvous(self, tag: str) -> None:
        if self.xm is not None:
            self.xm.rendezvous(str(tag))

    def all_reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        if self.xm is None:
            return value
        reduce = getattr(self.xm, "REDUCE_MEAN", "mean")
        return self.xm.all_reduce(reduce, value)

    def run_step(self, fn: Callable[[], Any], *, status=None) -> tuple[Any, float]:
        """Execute one fixed-shape step and mark it for XLA compilation."""
        started = time.monotonic()
        if status is not None:
            status.update(phase="xla_compile", compile_elapsed_seconds=0.0, event="xla_compile_start")
        result = fn()
        self.mark_step()
        elapsed = time.monotonic() - started
        self.compile_count += 1
        if status is not None:
            status.update(phase="training", compile_elapsed_seconds=elapsed,
                          compile_count=self.compile_count, event="xla_compile_complete")
        return result, elapsed


__all__ = [
    "SessionBudget", "TPUBackend", "TPUBackendConfig", "TPUProbeResult", "XLACompileCache",
    "discover_topology", "static_shape_guard", "tpu_available", "validate_topology",
]
