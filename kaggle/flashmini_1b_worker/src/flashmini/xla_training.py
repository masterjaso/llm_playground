"""Minimal deterministic XLA training loop used by the Kaggle worker."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .observability import StatusLogger, Watchdog
from .production import checkpoint_boundary
from .tpu_backend import SessionBudget, TPUBackend, static_shape_guard


@dataclass(frozen=True)
class XLATrainingConfig:
    batch_size: int = 8
    sequence_length: int = 2048
    target_tokens: int = 100_000_000_000
    stop_after_tokens: int = 10_000_000_000
    checkpoint_interval_tokens: int = 25_000_000
    log_every_steps: int = 10
    grad_clip_norm: float = 1.0
    aux_loss_coef: float = 0.01

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("XLA training dimensions must be positive")
        if self.stop_after_tokens <= 0 or self.stop_after_tokens > self.target_tokens:
            raise ValueError("stop_after_tokens must lie on the declared trajectory")


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def train_xla(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Any,
    *,
    backend: TPUBackend,
    config: XLATrainingConfig,
    status: StatusLogger,
    watchdog: Watchdog | None = None,
    sampler: Any | None = None,
    scheduler: Any | None = None,
    checkpoint: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    on_boundary: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    session_budget: SessionBudget | None = None,
    initial_step: int = 0,
    initial_tokens: int = 0,
) -> dict[str, Any]:
    """Train one logical batch per optimizer update on all XLA workers.

    ``checkpoint`` receives a complete, serializable trainer state at each
    threshold or soft stop.  It is deliberately outside this loop so the
    checkpoint manager can perform local read-back and remote verification.
    """
    if backend.device is None:
        backend.initialize()
    device = backend.device
    model.to(device)
    stream_mode = hasattr(dataset, "next_batch")
    if not stream_mode:
        sampler = sampler or dataset.sampler_for(seed=0, consumed=initial_tokens)
    watchdog = watchdog or Watchdog()
    watchdog.set_phase("xla_compile")
    if session_budget is not None:
        session_budget.start()
    status.start()
    step = int(initial_step)
    tokens_seen = int(initial_tokens)
    started = time.monotonic()
    last_loss: float | None = None
    ema_loss: float | None = None
    checkpoint_rows: list[dict[str, Any]] = []
    first_step = True

    while tokens_seen < config.stop_after_tokens:
        if session_budget is not None and session_budget.should_soft_stop():
            break
        if should_stop and should_stop():
            break
        if stream_mode:
            input_array, label_array = dataset.next_batch(config.batch_size)
        else:
            indices = sampler.take(config.batch_size)
            if len(indices) != config.batch_size:
                raise RuntimeError("deterministic sampler ended before a complete logical batch")
            input_array, label_array = dataset.get_batch(indices)
        input_ids = torch.as_tensor(input_array, dtype=torch.long, device=device)
        labels = torch.as_tensor(label_array, dtype=torch.long, device=device)
        static_shape_guard(input_ids, labels, sequence_length=config.sequence_length,
                           batch_size=config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        out = model(input_ids, labels=labels)
        if not isinstance(out, dict) or "loss" not in out:
            raise ValueError("XLA model forward must return a loss mapping")
        loss = out["loss"]
        stats = out.get("stats") or {}
        aux = stats.get("router_aux_loss")
        if isinstance(aux, (list, tuple)):
            aux = torch.stack([item if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device)
                               for item in aux]).mean()
        elif not isinstance(aux, torch.Tensor):
            aux = torch.zeros((), device=device)
        total_loss = loss + config.aux_loss_coef * aux
        if not _finite(total_loss):
            raise FloatingPointError("non-finite XLA loss")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        if not _finite(torch.as_tensor(grad_norm, device=device)):
            raise FloatingPointError("non-finite XLA gradient norm")
        backend.optimizer_step(optimizer, barrier=True)
        backend.mark_step()
        if first_step:
            first_step = False
            backend.compile_count += 1
            watchdog.set_phase("training")
            status.update(phase="training", event="xla_compile_complete",
                          compile_count=backend.compile_count + 1)
        if scheduler is not None:
            scheduler.step()
        step += 1
        batch_tokens = int(input_ids.numel())
        previous_tokens = tokens_seen
        tokens_seen += batch_tokens
        last_loss = float(loss.detach().float().item())
        ema_loss = last_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * last_loss
        elapsed = max(time.monotonic() - started, 1e-9)
        recent_elapsed = max(time.monotonic() - step_started, 1e-9)
        recent_rate = batch_tokens / recent_elapsed
        session_rate = tokens_seen / elapsed
        data_metrics = getattr(dataset, "metrics", {}) or {}
        watchdog.observe(step=step, tokens=tokens_seen,
                         estimated_data_wait_seconds=data_metrics.get("data_wait_seconds", 0.0),
                         checkpoint_state="none")
        if step % max(1, config.log_every_steps) == 0:
            status.update(progress=True, event="train_status", phase="training", global_step=step,
                          global_exact_tokens=tokens_seen, foundation_tokens=tokens_seen,
                          recent_loss=last_loss, ema_loss=ema_loss, recent_total_loss=float(total_loss.detach().float().item()),
                          grad_norm=float(grad_norm.detach().float().item()),
                          router_aux_loss=float(aux.detach().float().item()),
                          tokens_per_sec_recent=recent_rate, tokens_per_sec_session=session_rate,
                          tokens_per_sec_lifetime=session_rate, checkpoint_state="none")
        boundary = checkpoint_boundary(previous_tokens, batch_tokens, interval=config.checkpoint_interval_tokens)
        if boundary is not None:
            row = {**boundary, "global_step": step, "loss": last_loss,
                   "ema_loss": ema_loss, "recent_tokens_per_sec": recent_rate,
                   "session_tokens_per_sec": session_rate,
                   "data_cursor": dataset.state() if stream_mode else (sampler.state() if hasattr(sampler, "state") else None)}
            if checkpoint:
                saved = checkpoint({**row, "exact_tokens": tokens_seen, "step": step,
                                    "sampler_state": row["data_cursor"]}) or {}
                row.update(saved)
            checkpoint_rows.append(row)
            on_boundary(row) if on_boundary else None
            status.update(progress=False, event="checkpoint", phase="training",
                          global_step=step, global_exact_tokens=tokens_seen,
                          checkpoint_state="verified", latest_checkpoint=row.get("latest_checkpoint"),
                          latest_checkpoint_sha256=row.get("checkpoint_sha256"))
        if (event := watchdog.poll()) is not None:
            status.emit(event["event"], **event)
            if event["event"] == "STALL_HARD":
                raise RuntimeError("XLA hard stall detected")

    if checkpoint and tokens_seen and (not checkpoint_rows or checkpoint_rows[-1]["actual_tokens_seen"] != tokens_seen):
        final_row = {
            "checkpoint_reason": "graceful_session_end",
            "checkpoint_threshold_tokens": None,
            "actual_tokens_seen": tokens_seen,
            "overshoot_tokens": 0,
            "global_step": step,
            "exact_tokens": tokens_seen,
            "loss": last_loss,
            "ema_loss": ema_loss,
            "sampler_state": dataset.state() if stream_mode else (sampler.state() if hasattr(sampler, "state") else None),
        }
        final_row.update(checkpoint(final_row) or {})
        checkpoint_rows.append(final_row)
    final = {"status": "complete" if tokens_seen >= config.stop_after_tokens else "paused",
             "step": step, "exact_tokens": tokens_seen, "loss": last_loss,
             "ema_loss": ema_loss, "tokens_per_sec": tokens_seen / max(time.monotonic() - started, 1e-9),
             "sampler_state": dataset.state() if stream_mode else (sampler.state() if hasattr(sampler, "state") else None),
             "checkpoints": checkpoint_rows}
    status.update(progress=True, event="train_complete", phase="complete" if final["status"] == "complete" else "paused",
                  global_step=step, global_exact_tokens=tokens_seen, recent_loss=last_loss,
                  ema_loss=ema_loss, checkpoint_state="verified")
    status.stop()
    return final


__all__ = ["XLATrainingConfig", "train_xla"]
