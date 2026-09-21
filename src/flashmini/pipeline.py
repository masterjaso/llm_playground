"""Microbatch execution engines for PoC_D.

Two engines exist.  They are metric-for-metric equivalent, but they are NOT the
same execution policy and are recorded separately so a mid-run change stays
auditable:

``serial_microbatch_v1`` — :func:`pipeline_train_step`
    Each microbatch runs the whole model in turn.  Despite the historical
    "gpipe" label there is no stage overlap: microbatch *i+1* cannot start on
    the first device until microbatch *i* has left the second device.

``overlapped_2gpu_v1`` — :func:`overlapped_pipeline_train_step`
    Explicit stages on explicit CUDA streams.  Device A runs embedding + PLE +
    blocks ``[0, split)`` for microbatch *i+1* while device B runs blocks
    ``[split, num_layers)`` for microbatch *i*; the output stage (final residual,
    normalization, tied head, CE) is scheduled on device A around the incoming
    stage-B results.  The activation and the shared KVC bank cross the boundary
    exactly once per microbatch.

Both engines hold the logical-batch objective fixed:

* CE: ``sum_mb(ce_mb / microbatch_count)`` equals the full-batch
  ``F.cross_entropy`` mean, and its gradient gives every token ``1/total_tokens``.
* aux: per-layer sufficient statistics (``exp_counts``, ``prob_sum``,
  ``token_count``, ``slot_count``) are additive over microbatches, so ONE
  logical router auxiliary loss is reconstructed and it stays differentiable
  through the router.  Microbatches never get independent objectives.
* exactly one gradient clip and one optimizer step per logical batch.

Neither engine accumulates gradients; ``pipeline_microbatch_size`` is an
execution detail, not gradient accumulation.

Synchronization
---------------
Nothing here reads a device scalar per microbatch or per layer.  Routing, PLE
and CE statistics accumulate as device tensors and become Python floats once per
logical update, after the GPU work they summarize; ``router_token_count`` and
``router_slot_count`` are static counts derived from tensor shapes.  The host
transfers that remain are the ones that must exist: CUDA event waits at stage
boundaries, and the single per-layer per-expert count transfer inside the MoE
dispatch.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn

from .optim import clip_gradient_groups
from .training import _as_float

# Execution-policy identifiers recorded in run metadata and checkpoints.
SERIAL_SCHEDULE = "serial_microbatch_v1"
OVERLAPPED_SCHEDULE = "overlapped_2gpu_v1"
MONOLITHIC_SCHEDULE = "monolithic"
# Value written by checkpoints created before the executor transition.
LEGACY_SCHEDULE = "gpipe"
MICROBATCH_SCHEDULES = (SERIAL_SCHEDULE, OVERLAPPED_SCHEDULE, LEGACY_SCHEDULE)


class _StatsAccumulator:
    """Accumulate per-microbatch, per-layer sufficient statistics in place.

    Everything stays a device tensor.  ``router_token_count`` and
    ``router_slot_count`` are static counts derived from tensor shapes and are
    supplied by the caller; only legacy callers that pass no static count fall
    back to reading them off the routing tensors.
    """

    def __init__(self, num_experts: int, device: torch.device | None):
        self.num_experts = num_experts
        self.device = device
        self.exp_counts: dict[int, torch.Tensor] = {}
        self.prob_sum: dict[int, torch.Tensor] = {}
        self.token_count: dict[int, int] = {}
        self.slot_count: dict[int, int] = {}
        self.expert_load_sum: dict[int, torch.Tensor] = {}

    def _place(self, value: torch.Tensor) -> torch.Tensor:
        """Move a statistic to the accumulator device (a pure device copy)."""
        if self.device is not None and value.device != self.device:
            return value.to(self.device)
        return value

    def add_layer_stats(
        self,
        stats: dict[str, Any],
        *,
        token_count: int | None = None,
        slot_count: int | None = None,
    ) -> None:
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
                    value = self._place(value)
                    base = self.exp_counts.get(layer)
                    self.exp_counts[layer] = (base + value) if base is not None else value
                    # Deterministic expert load (counts / slots) for metrics.
                    load = self.expert_load_sum.get(layer)
                    self.expert_load_sum[layer] = (load + value) if load is not None else value
                elif key == "router_prob_sum":
                    value = self._place(value)
                    prev = self.prob_sum.get(layer)
                    self.prob_sum[layer] = (prev + value) if prev is not None else value
                elif key == "router_token_count":
                    self.token_count[layer] = self.token_count.get(layer, 0) + (
                        int(token_count) if token_count is not None else int(value.item())
                    )
                elif key == "router_slot_count":
                    self.slot_count[layer] = self.slot_count.get(layer, 0) + (
                        int(slot_count) if slot_count is not None else int(value.item())
                    )



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


def _engine_settings(model: nn.Module, aux_loss_coef: float | None) -> tuple[int, int, float]:
    """Return ``(num_experts, top_k, aux_loss_coef)`` from the model config."""
    config = getattr(model, "config", None)
    arch = getattr(config, "architecture_version", 2)
    moe_cfg = getattr(config, "moe", None) if config is not None else None
    num_experts = getattr(moe_cfg, "num_experts", 0)
    top_k = getattr(moe_cfg, "top_k", 1)
    if aux_loss_coef is None:
        aux_loss_coef = moe_cfg.aux_loss_coef if (arch >= 3 and moe_cfg is not None) else 0.01
    return num_experts, top_k, aux_loss_coef


class _LogicalBatchObjective:
    """The single logical-batch objective built from per-microbatch results.

    Every accumulator is a device tensor, so a step never reads a device scalar
    until :meth:`metrics` runs after the optimizer update.  Statistics are
    accumulated in block order then microbatch order — the same order the
    monolithic path produces them in — so the logical aux rebuilds identically.
    """

    def __init__(self, num_experts: int, microbatch_count: int, device: torch.device | None):
        self.num_experts = num_experts
        self.microbatch_count = microbatch_count
        self.device = device
        self.ce = torch.zeros((), device=device, requires_grad=True)
        self.entropy_sum = torch.zeros((), device=device)
        self.entropy_queries = 0
        self.ple_norm_sum = torch.zeros((), device=device)
        self.ple_scale = None
        self.acc = _StatsAccumulator(num_experts, device)

    def _place(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        if self.device is not None and value.device != self.device:
            return value.to(self.device)
        return value

    def add_stats(self, stats: dict[str, Any], *, token_count: int, slot_count: int) -> None:
        """Accumulate one microbatch's (or one stage's) routing/PLE statistics."""
        self.acc.add_layer_stats(stats, token_count=token_count, slot_count=slot_count)
        entropy = stats.get("router_entropy")
        if entropy:
            for value in entropy:
                self.entropy_sum = self.entropy_sum + self._place(value)
            # Matches the historical normalization: one query per MoE layer per
            # routed token.
            self.entropy_queries += len(entropy) * token_count
        if "ple_scale" in stats:
            self.ple_scale = stats["ple_scale"]
        if "ple_norm_ratio" in stats:
            self.ple_norm_sum = self.ple_norm_sum + self._place(stats["ple_norm_ratio"])

    def add_microbatch(self, ce: torch.Tensor, stats: dict[str, Any], *,
                       token_count: int, slot_count: int) -> None:
        """Accumulate one microbatch's CE contribution and statistics."""
        self.ce = self.ce + ce / self.microbatch_count
        self.add_stats(stats, token_count=token_count, slot_count=slot_count)

    def build(self, aux_loss_coef: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, logical_aux)`` after the finite checks."""
        from .training import _finite_tensor

        aux = _logical_aux(
            self.num_experts, self.acc.exp_counts, self.acc.prob_sum,
            self.acc.token_count, self.acc.slot_count,
        )
        if not _finite_tensor(aux):
            raise FloatingPointError("router auxiliary loss is non-finite")
        total_loss = self.ce + aux_loss_coef * aux
        if not _finite_tensor(total_loss):
            raise FloatingPointError("total loss is non-finite before backward")
        return total_loss, aux

    def metrics(self, aux: torch.Tensor, clipping: dict[str, Any], grad_norm: float,
                total_loss: torch.Tensor) -> dict[str, Any]:
        """Materialize the Python-facing metrics once per logical update."""
        metrics: dict[str, Any] = {
            "loss": _as_float(self.ce, "loss"),
            "total_loss": _as_float(total_loss, "total loss"),
            "grad_norm": grad_norm,
            "router_aux_loss": _as_float(aux, "router auxiliary loss"),
            **clipping,
        }
        if self.entropy_queries > 0:
            metrics["router_entropy"] = _as_float(
                self.entropy_sum / self.entropy_queries, "router entropy"
            )
        if self.acc.expert_load_sum:
            load_tensors = [
                v / self.acc.slot_count[layer] for layer, v in self.acc.expert_load_sum.items()
            ]
            stacked = torch.stack(load_tensors).mean(0)
            load_mean = float(stacked.mean().item())
            metrics["expert_load_max"] = float(stacked.max().item())
            metrics["expert_load_mean"] = load_mean
            metrics["expert_load_ratio"] = float(stacked.max().item() / (load_mean + 1e-9))
            metrics["expert_load_dist"] = [float(x) for x in stacked.detach().cpu().tolist()]
        if self.ple_scale is not None:
            metrics["ple_scale"] = _as_float(self.ple_scale, "ple_scale")
        if self.microbatch_count > 0:
            metrics["ple_norm_ratio"] = _as_float(
                self.ple_norm_sum / self.microbatch_count, "ple_norm_ratio"
            )
        return metrics


def _finish_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    objective: _LogicalBatchObjective,
    aux_loss_coef: float,
    grad_clip: float,
    *,
    before_step=None,
) -> dict[str, Any]:
    """One backward, one clip, one optimizer step for the logical batch.

    ``before_step`` lets an engine order gradient consumption after the streams
    that produced the gradients, without a host synchronization.
    """
    import math

    total_loss, aux = objective.build(aux_loss_coef)
    total_loss.backward()
    if before_step is not None:
        before_step()

    config = getattr(model, "config", None)
    arch = getattr(config, "architecture_version", 2)
    clipping: dict[str, Any] = {}
    if arch >= 3:
        clipping = clip_gradient_groups(model, grad_clip)
        grad_norm = math.sqrt(sum(
            clipping[f"grad_norm_{group}_preclip"] ** 2
            for group in ("shared", "ple_dense", "ple_sparse")
        ))
    else:
        from .optim import clip_gradients
        grad_norm = clip_gradients(model, grad_clip)
    grad_norm_value = _as_float(grad_norm, "gradient norm")
    optimizer.step()
    return objective.metrics(aux, clipping, grad_norm_value, total_loss)


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

    This is the ``serial_microbatch_v1`` engine: every microbatch runs the whole
    model in turn, so there is no stage overlap.  It is the reference engine and
    the only one that works without two distinct CUDA devices.
    """
    m = len(microbatches)
    if not microbatches:
        raise ValueError("pipeline_train_step requires at least one microbatch")
    num_experts, top_k, aux_loss_coef = _engine_settings(model, aux_loss_coef)

    input_device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    objective = _LogicalBatchObjective(num_experts, m, input_device)
    for input_ids, labels in microbatches:
        input_ids = input_ids.to(input_device)
        labels = labels.to(input_device)
        tokens = int(input_ids.numel())
        if use_amp and input_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
        else:
            out = model(input_ids, labels=labels)
        objective.add_microbatch(
            out["loss"], out.get("stats") or {}, token_count=tokens, slot_count=tokens * top_k
        )
    return _finish_update(model, optimizer, objective, aux_loss_coef, grad_clip)


class _PipelineRuntime:
    """CUDA streams and events for one overlapped pipeline configuration."""

    def __init__(self, stage_a_device: torch.device, stage_b_device: torch.device,
                 microbatch_count: int):
        self.stage_a_device = stage_a_device
        self.stage_b_device = stage_b_device
        self.stream_a = torch.cuda.Stream(device=stage_a_device)
        self.stream_b = torch.cuda.Stream(device=stage_b_device)
        self.stage_a_done = [torch.cuda.Event() for _ in range(microbatch_count)]
        self.stage_b_done = [torch.cuda.Event() for _ in range(microbatch_count)]
        self.tail_a = torch.cuda.Event()
        self.tail_b = torch.cuda.Event()
        # Recorded after each step's optimizer update so the next step's stage
        # work cannot overtake it.
        self.step_a = torch.cuda.Event()
        self.step_b = torch.cuda.Event()

    def covers(self, microbatch_count: int) -> bool:
        return len(self.stage_a_done) >= microbatch_count


def _pipeline_runtime(model: nn.Module, stage_a_device: torch.device,
                      stage_b_device: torch.device, microbatch_count: int) -> _PipelineRuntime:
    """Return the runtime cached on the model, creating it when placement changes."""
    runtime = getattr(model, "_pipeline_runtime", None)
    if (
        runtime is None
        or runtime.stage_a_device != stage_a_device
        or runtime.stage_b_device != stage_b_device
        or not runtime.covers(microbatch_count)
    ):
        runtime = _PipelineRuntime(stage_a_device, stage_b_device, microbatch_count)
        model._pipeline_runtime = runtime
    return runtime


def _timing_event(enabled: bool):
    """Record a start event on the current stream, or nothing when disabled."""
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _record_stage_ms(timings, key: str, start, enabled: bool) -> None:
    """Close one stage's timing pair (no synchronization; read after the step)."""
    if not enabled or start is None:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    timings.setdefault(key, []).append((start, end))


def _sum_stage_ms(timings) -> dict[str, float]:
    """Materialize stage timings; requires the enclosing streams to be idle."""
    result: dict[str, float] = {}
    for key, pairs in timings.items():
        result[key] = float(sum(start.elapsed_time(end) for start, end in pairs))
    return result


def overlapped_pipeline_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    aux_loss_coef: float | None = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    timing: bool = False,
) -> dict:
    """Run one optimizer update with a real staged, overlapping two-GPU pipeline.

    Device A owns the embedding, PLE and blocks ``[0, stage_split)`` and also the
    output stage (final residual combination, normalization, tied head, CE);
    device B owns blocks ``[stage_split, num_layers)``.  Stage-A forwards are all
    issued first, so device A starts stage A for microbatch *i+1* while device B
    is still running stage B for microbatch *i*.

    Dependencies are explicit CUDA events: one per microbatch at the stage
    boundary.  The activation and the shared KVC bank are transferred exactly
    once per microbatch (``non_blocking=True``), the pinned embedding/head stay
    where they are, and no global synchronization happens inside the schedule.

    This is the ``overlapped_2gpu_v1`` engine.  It falls back to the serial
    engine when the placement does not provide two distinct CUDA devices (CPU
    tests, single-GPU), which keeps the identical math and the same metric
    contract available everywhere.
    """
    if not microbatches:
        raise ValueError("overlapped_pipeline_train_step requires at least one microbatch")
    num_experts, top_k, aux_loss_coef = _engine_settings(model, aux_loss_coef)

    stage_a_device = model.embed.weight.device
    block_devices = list(getattr(model, "block_devices", []) or [])
    stage_split = getattr(model, "stage_split", None)
    usable = (
        stage_a_device.type == "cuda"
        and stage_split is not None
        and 0 < stage_split < len(block_devices)
        and block_devices[stage_split].type == "cuda"
        and block_devices[stage_split] != stage_a_device
    )
    if not usable:
        # Nothing can overlap without two distinct CUDA devices; keep the
        # identical objective on the reference engine rather than failing.
        return pipeline_train_step(
            model, optimizer, microbatches,
            aux_loss_coef=aux_loss_coef, grad_clip=grad_clip, use_amp=use_amp,
        )
    stage_b_device = block_devices[stage_split]

    m = len(microbatches)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    runtime = _pipeline_runtime(model, stage_a_device, stage_b_device, m)
    stream_a, stream_b = runtime.stream_a, runtime.stream_b
    objective = _LogicalBatchObjective(num_experts, m, stage_a_device)
    timings: dict[str, list] = {}

    def _autocast(enable: bool):
        if enable:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    slots: list[dict] = []
    for input_ids, labels in microbatches:
        tokens = int(input_ids.numel())
        slots.append({
            "input_ids": input_ids.to(stage_a_device),
            "labels": labels.to(stage_a_device),
            "tokens": tokens,
            "slot_count": tokens * top_k,
            "stage_a": None,
            "stage_b": None,
        })

    # Phase 1 — every stage-A forward. Nothing here waits on device B, so device
    # A runs stage A for microbatch i+1 while device B is still finishing
    # microbatch i.
    with torch.cuda.stream(stream_a):
        for index, slot in enumerate(slots):
            start = _timing_event(timing)
            with _autocast(use_amp):
                slot["stage_a"] = model.pipeline_stage0(
                    slot["input_ids"], stage_end=stage_split
                )
            runtime.stage_a_done[index].record(stream_a)
            _record_stage_ms(timings, "stage0_ms", start, timing)

    # Phase 2 — stage B on device B. The activation and the shared KVC bank cross
    # the boundary exactly once per microbatch, on device B's stream, ordered by
    # that microbatch's stage-A event.
    with torch.cuda.stream(stream_b):
        for index, slot in enumerate(slots):
            stream_b.wait_event(runtime.stage_a_done[index])
            source = slot["stage_a"]
            hidden = source["hidden"].to(stage_b_device, non_blocking=True)
            bank = source["kvc_bank"]
            if bank is not None:
                bank = (bank[0].to(stage_b_device, non_blocking=True),
                        bank[1].to(stage_b_device, non_blocking=True))
            start = _timing_event(timing)
            with _autocast(use_amp):
                slot["stage_b"] = model.pipeline_stage1(
                    hidden, stage_start=stage_split, kvc_bank=bank
                )
            runtime.stage_b_done[index].record(stream_b)
            _record_stage_ms(timings, "stage1_ms", start, timing)
        runtime.tail_b.record(stream_b)

    # Phase 3 — output stage on device A, gated on each microbatch's stage-B
    # event. The hidden activation is brought back once and the CE is added to the
    # single logical objective; the accumulated statistics keep block order then
    # microbatch order, exactly as the monolithic path produces them.
    with torch.cuda.stream(stream_a):
        for index, slot in enumerate(slots):
            stream_a.wait_event(runtime.stage_b_done[index])
            hidden = slot["stage_b"]["hidden"].to(stage_a_device, non_blocking=True)
            start = _timing_event(timing)
            with _autocast(use_amp):
                logits = model.pipeline_output(hidden)["logits"]
                ce = model.pipeline_loss(logits, slot["labels"])
            _record_stage_ms(timings, "output_ms", start, timing)
            objective.add_microbatch(
                ce, slot["stage_a"]["stats"],
                token_count=slot["tokens"], slot_count=slot["slot_count"],
            )
            objective.add_stats(
                slot["stage_b"]["stats"],
                token_count=slot["tokens"], slot_count=slot["slot_count"],
            )
            # Drop the stage payloads as soon as their only consumer has been
            # queued, so peak activation memory stays at stage granularity.
            slot["stage_a"] = None
            slot["stage_b"] = None
        runtime.tail_a.record(stream_a)

    default_a = torch.cuda.current_stream(stage_a_device)
    default_b = torch.cuda.current_stream(stage_b_device)
    # Both default streams must see every pipeline result before the backward
    # consumes them. Event waits only: no host synchronization.
    default_a.wait_event(runtime.tail_a)
    default_b.wait_event(runtime.tail_b)

    def _order_gradients_after_streams() -> None:
        # The clip and the optimizer read gradients that the backward produced
        # on the pipeline streams; order them without draining the device.
        default_a.wait_stream(stream_a)
        default_b.wait_stream(stream_b)

    metrics = _finish_update(
        model, optimizer, objective, aux_loss_coef, grad_clip,
        before_step=_order_gradients_after_streams,
    )

    # The next step's stage work must not overtake this step's optimizer update.
    runtime.step_a.record(default_a)
    runtime.step_b.record(default_b)
    stream_a.wait_event(runtime.step_a)
    stream_b.wait_event(runtime.step_b)

    if timing and timings:
        # Reading CUDA event timings needs completed events. This is a real
        # synchronization, so the training loop only enables timing on steps it
        # is already going to log.
        torch.cuda.synchronize(stage_a_device)
        torch.cuda.synchronize(stage_b_device)
        metrics["stage_ms"] = _sum_stage_ms(timings)
    return metrics





