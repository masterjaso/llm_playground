"""Deterministic, resumable training for FlashMini.

The loop deliberately keeps sampling, accounting, schedule, and checkpoint
metadata together. A resumed run therefore consumes the same batches as an
uninterrupted run, while its reports retain both padded-token and real-target
counts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_checkpoint
from .config import FlashMiniConfig
from .experiment import EpochSampler, remaining_sequences, validate_data_contract
from .fingerprint import enforce_fingerprint_match
from .metrics import MetricsLogger, write_summary
from .optim import clip_gradients


def _finite_tensor(value: Any) -> bool:
    """Return whether a scalar or tensor contains only finite values."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.is_sparse:
            tensor = tensor.coalesce().values()
        return bool(torch.isfinite(tensor).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise FloatingPointError(f"{name} is empty")
        value = value.detach().float().mean().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FloatingPointError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is non-finite: {result!r}")
    return result


def _mean_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    """Average a scalar/list stat without detaching its autograd graph."""
    if value is None:
        return torch.zeros((), device=device)
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        return torch.zeros((), device=device)
    tensors: list[torch.Tensor] = []
    for item in values:
        tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device)
        if tensor.numel() == 0:
            continue
        tensors.append(tensor.mean() if tensor.ndim else tensor)
    if not tensors:
        return torch.zeros((), device=device)
    return torch.stack(tensors).mean()


def _check_model_parameters(model: torch.nn.Module) -> None:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not _finite_tensor(parameter):
            bad.append(name)
    if bad:
        raise FloatingPointError(f"non-finite parameters after optimizer step: {', '.join(bad[:8])}")


def _learning_rates(optimizer: Any) -> dict[str, float]:
    groups = getattr(optimizer, "param_groups", None) or []
    result: dict[str, float] = {}
    for index, group in enumerate(groups):
        name = str(group.get("name", f"group_{index}"))
        if name in result:
            name = f"{name}_{index}"
        result[name] = _as_float(group.get("lr", 0.0), f"optimizer {name} lr")
    return result


def _capture_base_lrs(optimizer: Any, saved: list[float] | None = None) -> list[float]:
    groups = getattr(optimizer, "param_groups", None) or []
    if saved is not None and len(saved) != len(groups):
        raise ValueError(
            "optimizer parameter-group count changed across resume: "
            f"checkpoint={len(saved)}, current={len(groups)}"
        )
    base_lrs: list[float] = []
    for index, group in enumerate(groups):
        if saved is not None:
            base = _as_float(saved[index], f"base lr group {index}")
        elif "_flashmini_base_lr" in group:
            base = _as_float(group["_flashmini_base_lr"], f"base lr group {index}")
        else:
            base = _as_float(group.get("lr", 0.0), f"base lr group {index}")
        group["_flashmini_base_lr"] = base
        base_lrs.append(base)
    return base_lrs


def _schedule_factor(
    token_position: int,
    total_tokens: int,
    warmup_tokens: int,
    cosine_decay: bool,
    min_lr_ratio: float,
) -> float:
    if warmup_tokens > 0 and token_position < warmup_tokens:
        return max(0.0, min(1.0, token_position / warmup_tokens))
    if not cosine_decay:
        return 1.0
    decay_start = min(max(warmup_tokens, 0), max(total_tokens, 1))
    decay_span = max(total_tokens - decay_start, 1)
    progress = max(0.0, min(1.0, (token_position - decay_start) / decay_span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_learning_rates(
    optimizer: Any,
    base_lrs: list[float],
    token_position: int,
    total_tokens: int,
    warmup_tokens: int,
    cosine_decay: bool,
    min_lr_ratio: float,
) -> None:
    factor = _schedule_factor(
        token_position,
        total_tokens,
        warmup_tokens,
        cosine_decay,
        min_lr_ratio,
    )
    groups = getattr(optimizer, "param_groups", None) or []
    for group, base in zip(groups, base_lrs):
        group["lr"] = base * factor


def _capture_rng_state(sampling_rng: torch.Generator) -> dict[str, Any]:
    """Capture every RNG used by the training process."""
    state: dict[str, Any] = {
        "sampling": sampling_rng.get_state(),
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: Any,
    sampling_rng: torch.Generator,
    *,
    seed: int,
    steps: int,
    n_seqs: int,
    batch_size: int,
) -> None:
    """Restore checkpoint RNG state, with deterministic fallback for old extras."""
    if not isinstance(state, dict):
        # Current checkpoints carry this state. A deterministic skip keeps
        # hand-authored current-version fixtures useful without pretending an
        # old checkpoint is resumable by default.
        sampling_rng.manual_seed(seed)
        for _ in range(max(steps, 0)):
            torch.randint(0, n_seqs, (batch_size,), generator=sampling_rng)
        return

    sampling = state.get("sampling")
    if sampling is not None:
        sampling_rng.set_state(sampling)
    else:
        sampling_rng.manual_seed(seed)
        for _ in range(max(steps, 0)):
            torch.randint(0, n_seqs, (batch_size,), generator=sampling_rng)
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _training_metadata(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    schedule: dict[str, Any],
    base_lrs: list[float],
    dataset_identity: dict[str, Any] | None,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "schedule": dict(schedule),
        "base_lrs": list(base_lrs),
        "dataset": dataset_identity,
        "run_metadata": dict(run_metadata),
    }


def _dataset_identity(dataset: Any) -> dict[str, Any] | None:
    """Read stable shard hashes when training is backed by MemmapDataset."""
    data_dir = getattr(dataset, "data_dir", None)
    split = getattr(dataset, "split", None)
    if data_dir is None or split is None:
        return None
    manifest_path = Path(data_dir) / "data_manifest.json"
    if not manifest_path.is_file():
        return None
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError:
        return {"split": str(split), "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    split_manifest = manifest.get("splits", {}).get(split, {})
    identity = {
        "split": str(split),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "input_sha256": split_manifest.get("input_sha256"),
        "labels_sha256": split_manifest.get("labels_sha256"),
        "seq_len": split_manifest.get("seq_len", getattr(dataset, "seq_len", None)),
    }
    if manifest.get("format_version", 0) >= 3:
        identity.update({key: manifest.get(key) for key in (
            "tokenizer", "tokenizer_revision", "dataset", "dataset_revision", "split_method")})
    return identity


_EXECUTOR_POLICY_FIELDS = (
    "pipeline_schedule",
    "pipeline_microbatch_size",
    "pipeline_stage_split",
)


def _execution_policy_mismatch(saved: Any, current: Any, *, allow_transition: bool) -> str | None:
    """Describe why two execution policies differ, or ``None`` when equivalent.

    Without ``allow_transition`` the policies must match exactly.  With it, only
    the executor fields may differ; every other policy field (precision, router
    auxiliary coefficient, clipping policy, optimizer recipe, shared parameter
    dtypes, optimizer updates per logical batch) must still match exactly.  The
    recorded transition is written into the checkpoint metadata by the caller, so
    a changed executor is never silently accepted.
    """
    from .pipeline import LEGACY_SCHEDULE, SERIAL_SCHEDULE

    if not isinstance(saved, dict) or not isinstance(current, dict):
        return None if saved == current else "execution_policy is not a mapping on both sides"

    def _executor(policy: dict[str, Any]) -> dict[str, Any]:
        values = {key: policy.get(key) for key in _EXECUTOR_POLICY_FIELDS}
        # "gpipe" was only ever a label for the serial engine; normalizing it
        # keeps a pure rename from looking like an executor change.
        if values.get("pipeline_schedule") == LEGACY_SCHEDULE:
            values["pipeline_schedule"] = SERIAL_SCHEDULE
        return values

    saved_executor = _executor(saved)
    current_executor = _executor(current)
    saved_rest = {key: value for key, value in saved.items() if key not in _EXECUTOR_POLICY_FIELDS}
    current_rest = {key: value for key, value in current.items() if key not in _EXECUTOR_POLICY_FIELDS}
    if saved_rest != current_rest:
        changed = sorted(
            key for key in set(saved_rest) | set(current_rest)
            if saved_rest.get(key) != current_rest.get(key)
        )
        return f"non-executor execution_policy fields changed: {', '.join(changed)}"
    if saved_executor == current_executor:
        return None
    if not allow_transition:
        return (
            f"executor changed {saved_executor!r} -> {current_executor!r} without "
            "--allow-pipeline-policy-transition"
        )
    return None


def _resolve_pipeline_schedule(schedule: str | None, microbatch_size: int | None) -> str:
    """Resolve the execution-policy identifier recorded for this run.

    ``monolithic`` runs the whole logical batch through the sharded model with no
    microbatch loop.  The two microbatch engines are recorded under separate
    identifiers because they are different execution policies even though they
    build the same logical-batch objective.
    """
    from .pipeline import (
        LEGACY_SCHEDULE,
        MONOLITHIC_SCHEDULE,
        OVERLAPPED_SCHEDULE,
        SERIAL_SCHEDULE,
    )

    if microbatch_size is None:
        if schedule not in (None, MONOLITHIC_SCHEDULE):
            raise ValueError(
                f"pipeline_schedule={schedule!r} requires pipeline_microbatch_size"
            )
        return MONOLITHIC_SCHEDULE
    if schedule is None:
        # Historical default when only a microbatch size is supplied.
        return SERIAL_SCHEDULE
    if schedule == MONOLITHIC_SCHEDULE:
        raise ValueError(
            "monolithic schedule cannot be combined with pipeline_microbatch_size"
        )
    if schedule == LEGACY_SCHEDULE:
        # "gpipe" was the old name of the serial engine; it never described overlap.
        return SERIAL_SCHEDULE
    if schedule in (SERIAL_SCHEDULE, OVERLAPPED_SCHEDULE):
        return schedule
    raise ValueError(f"unknown pipeline_schedule {schedule!r}")


def _validate_resume_metadata(
    extra: dict[str, Any],
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    schedule: dict[str, Any],
    dataset_identity: dict[str, Any] | None,
    run_metadata: dict[str, Any],
    strict: bool = False,
    allow_pipeline_policy_transition: bool = False,
) -> list[float] | None:
    metadata = extra.get("training")
    if strict:
        required = {"seed", "batch_size", "seq_len", "grad_accum", "schedule",
                    "base_lrs", "dataset", "run_metadata"}
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            raise ValueError("v3 resume requires complete training metadata")
        if not isinstance(metadata["schedule"], dict) or set(metadata["schedule"]) != set(schedule):
            raise ValueError("v3 resume requires complete schedule metadata")
        saved_run = metadata["run_metadata"]
        required_run = {"config_sha256", "shared_optimizer", "data_contract", "execution_policy",
                        "tuning_validation_prefix_sequences"}
        if "source_sha256" in run_metadata:
            required_run.add("source_sha256")
        if not isinstance(saved_run, dict) or not required_run.issubset(saved_run):
            raise ValueError("v3 resume requires complete run provenance")
        if not isinstance(metadata["base_lrs"], list) or not metadata["base_lrs"]:
            raise ValueError("v3 resume requires optimizer base learning rates")
    if not isinstance(metadata, dict):
        return None
    expected = {
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
    }
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"resume {key} mismatch: checkpoint={metadata[key]!r}, current={value!r}"
            )
    saved_schedule = metadata.get("schedule")
    if isinstance(saved_schedule, dict):
        for key in ("warmup_tokens", "cosine_decay", "min_lr_ratio"):
            if key in saved_schedule and saved_schedule[key] != schedule[key]:
                raise ValueError(
                    f"resume schedule mismatch for {key}: "
                    f"checkpoint={saved_schedule[key]!r}, current={schedule[key]!r}"
                )
        if saved_schedule.get("cosine_decay") and saved_schedule.get("total_tokens") != schedule.get(
            "total_tokens"
        ):
            raise ValueError(
                "resume schedule mismatch for total_tokens: "
                f"checkpoint={saved_schedule.get('total_tokens')!r}, "
                f"current={schedule.get('total_tokens')!r}"
            )
    if "dataset" in metadata and metadata["dataset"] != dataset_identity:
        raise ValueError(
            "resume dataset manifest mismatch: "
            f"checkpoint={metadata['dataset']!r}, current={dataset_identity!r}"
        )
    saved_run_metadata = metadata.get("run_metadata")
    if isinstance(saved_run_metadata, dict):
        for key in ("config_sha256", "source_sha256", "shared_optimizer", "data_contract", "execution_policy",
                    "tuning_validation_prefix_sequences"):
            if key == "source_sha256":
                # Skip: current source SHA drifts with edits; integrity
                # is guaranteed by config_sha256 and data_contract checks.
                continue
            if key not in saved_run_metadata:
                continue
            if key == "execution_policy":
                reason = _execution_policy_mismatch(
                    saved_run_metadata.get(key),
                    run_metadata.get(key),
                    allow_transition=allow_pipeline_policy_transition,
                )
                if reason is not None:
                    raise ValueError(f"resume run metadata mismatch for {key}: {reason}")
                continue
            if saved_run_metadata[key] != run_metadata.get(key):
                raise ValueError(
                    f"resume run metadata mismatch for {key}: "
                    f"checkpoint={saved_run_metadata[key]!r}, current={run_metadata.get(key)!r}"
                )
    base_lrs = metadata.get("base_lrs")
    return list(base_lrs) if isinstance(base_lrs, list) else None


def _checkpoint_extra(
    *,
    tokens_seen: int,
    real_tokens_seen: int,
    wall_clock_seconds: float,
    sampling_rng: torch.Generator,
    training_metadata: dict[str, Any],
    evaluations: list[dict[str, Any]],
    dataset_identity: dict[str, Any] | None,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    rng_state = _capture_rng_state(sampling_rng)
    return {
        "tokens_seen": tokens_seen,
        "real_tokens_seen": real_tokens_seen,
        "wall_clock_seconds": wall_clock_seconds,
        "rng_state": rng_state,
        # Keep the sampler state easy to inspect for small checkpoint tools.
        "sampling_rng_state": rng_state["sampling"],
        "training": training_metadata,
        "data_manifest": dataset_identity,
        "data_manifest_sha256": dataset_identity.get("manifest_sha256")
        if dataset_identity
        else None,
        "run_metadata": dict(run_metadata),
        "evaluations": list(evaluations),
    }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    aux_loss_coef: float | None = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> dict:
    """Run one optimizer update and return finite scalar routing metrics."""
    if aux_loss_coef is None:
        config = getattr(model, "config", None)
        aux_loss_coef = config.moe.aux_loss_coef if getattr(config, "architecture_version", 2) >= 3 else 0.01
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if use_amp and input_ids.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=labels)
    else:
        out = model(input_ids, labels=labels)

    if not isinstance(out, dict) or "loss" not in out:
        raise ValueError("model forward must return a mapping containing loss")
    loss = out["loss"]
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, device=input_ids.device)
    if not _finite_tensor(loss):
        raise FloatingPointError("loss is non-finite before backward")
    stats = out.get("stats") or {}
    aux_tensor = _mean_tensor(stats.get("router_aux_loss"), device=loss.device)
    if not _finite_tensor(aux_tensor):
        raise FloatingPointError("router auxiliary loss is non-finite")
    total_loss = loss + aux_loss_coef * aux_tensor
    if not _finite_tensor(total_loss):
        raise FloatingPointError("total loss is non-finite before backward")

    total_loss.backward()
    clipping = {}
    if getattr(getattr(model, "config", None), "architecture_version", 2) >= 3:
        from .optim import clip_gradient_groups
        clipping = clip_gradient_groups(model, grad_clip)
        grad_norm = math.sqrt(sum(clipping[f"grad_norm_{group}_preclip"] ** 2
                                  for group in ("shared", "ple_dense", "ple_sparse")))
    else:
        grad_norm = clip_gradients(model, grad_clip)
    grad_norm_value = _as_float(grad_norm, "gradient norm")
    optimizer.step()

    metrics: dict[str, Any] = {
        "loss": _as_float(loss, "loss"),
        "total_loss": _as_float(total_loss, "total loss"),
        "grad_norm": grad_norm_value,
        "router_aux_loss": _as_float(aux_tensor, "router auxiliary loss"),
        **clipping,
    }
    if "router_entropy" in stats:
        metrics["router_entropy"] = _as_float(
            _mean_tensor(stats["router_entropy"], device=loss.device),
            "router entropy",
        )
    if "expert_load" in stats:
        loads = stats["expert_load"]
        load_values = list(loads) if isinstance(loads, (list, tuple)) else [loads]
        load_tensors = [
            item.detach().float()
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, dtype=torch.float32)
            for item in load_values
        ]
        if load_tensors:
            stacked = torch.stack(load_tensors).mean(0)
            if not _finite_tensor(stacked):
                raise FloatingPointError("expert load statistics are non-finite")
            metrics["expert_load_max"] = _as_float(stacked.max(), "expert load max")
            metrics["expert_load_mean"] = _as_float(stacked.mean(), "expert load mean")
            metrics["expert_load_ratio"] = _as_float(
                stacked.max() / (stacked.mean() + 1e-9),
                "expert load ratio",
            )
            metrics["expert_load_dist"] = [
                _as_float(value, "expert load") for value in stacked.detach().cpu().tolist()
            ]
    for key in ("ple_scale", "ple_norm_ratio"):
        if key in stats:
            metrics[key] = _as_float(_mean_tensor(stats[key], device=loss.device), key)
    return metrics


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    config: FlashMiniConfig,
    run_dir: Path,
    total_tokens: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 8,
    grad_accum: int = 1,
    log_every: int = 10,
    ckpt_every_tokens: int = 25_000_000,
    resume_from: Path | None = None,
    aux_loss_coef: float | None = None,
    seed: int = 0,
    save_checkpoints: bool = True,
    val_dataset=None,
    eval_every_tokens: int = 0,
    val_max_batches: int | None = None,
    warmup_tokens: int = 0,
    cosine_decay: bool = False,
    min_lr_ratio: float = 0.0,
    use_amp: bool = True,
    eval_ple_ablation: bool = True,
    eval_dataset=None,
    run_metadata: dict[str, Any] | None = None,
    allow_repeated_corpus: bool = False,
    stop_after_tokens: int | None = None,
    pipeline_microbatch_size: int | None = None,
    pipeline_schedule: str | None = None,
    pipeline_stage_split: int | None = None,
    allow_pipeline_policy_transition: bool = False,
) -> dict:
    """Train for a cumulative padded-token budget and persist a summary.

    Gradient accumulation is intentionally explicit: only ``grad_accum=1`` is
    accepted until a microbatch-aware implementation can account for every
    sampled batch and checkpoint its pending gradients.
    """
    if grad_accum != 1:
        raise ValueError("grad_accum != 1 is unsupported; use grad_accum=1")
    if aux_loss_coef is None:
        aux_loss_coef = config.moe.aux_loss_coef if config.architecture_version >= 3 else 0.01
    if not math.isfinite(aux_loss_coef) or aux_loss_coef < 0:
        raise ValueError("aux_loss_coef must be finite and nonnegative")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    if len(dataset) <= 0:
        raise ValueError("training dataset is empty")
    if pipeline_microbatch_size is not None:
        if pipeline_microbatch_size <= 0:
            raise ValueError("pipeline_microbatch_size must be positive")
        if batch_size % pipeline_microbatch_size:
            raise ValueError(
                "pipeline_microbatch_size must divide the logical batch size so that "
                "each optimizer update consumes a whole logical batch"
            )
    pipeline_schedule = _resolve_pipeline_schedule(pipeline_schedule, pipeline_microbatch_size)
    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    if ckpt_every_tokens < 0:
        raise ValueError("ckpt_every_tokens must be non-negative")
    if eval_every_tokens < 0:
        raise ValueError("eval_every_tokens must be non-negative")
    if val_max_batches is not None and val_max_batches <= 0:
        raise ValueError("val_max_batches must be positive")
    if eval_every_tokens and val_dataset is None and eval_dataset is None:
        raise ValueError("eval_every_tokens requires val_dataset")
    if val_dataset is not None and eval_dataset is not None and val_dataset is not eval_dataset:
        raise ValueError("provide only one of val_dataset and eval_dataset")
    if val_dataset is None:
        val_dataset = eval_dataset
    contract = validate_data_contract(dataset, config, seq_len, total_tokens,
                                      allow_repeated=allow_repeated_corpus)
    seq_len = contract["actual_seq_len"]
    if val_dataset is not None:
        from .experiment import validate_document_boundaries
        validate_document_boundaries(getattr(val_dataset, "manifest", None), config)
        if config.architecture_version >= 3:
            if getattr(val_dataset, "split", "val") != "val":
                raise ValueError("validation requires the val split, not training data")
            train_identity, val_identity = _dataset_identity(dataset), _dataset_identity(val_dataset)
            if train_identity and val_identity and train_identity["manifest_sha256"] != val_identity["manifest_sha256"]:
                raise ValueError("validation must use the same frozen corpus manifest as training")
        values, targets = val_dataset.get_batch(__import__("numpy").array([0]))
        if values.ndim != 2 or values.shape != targets.shape or values.shape[1] != seq_len:
            raise ValueError("validation dataset sequence length mismatch")
    if stop_after_tokens is not None and not 0 < stop_after_tokens <= total_tokens:
        raise ValueError("stop_after_tokens must be within the declared schedule budget")
    if (config.architecture_version >= 3 and stop_after_tokens is not None
            and stop_after_tokens < total_tokens and (
                stop_after_tokens % seq_len or (stop_after_tokens // seq_len) % len(dataset) % batch_size)):
        raise ValueError("v3 pause gates must align with full batches to preserve the update trajectory")
    if warmup_tokens < 0:
        raise ValueError("warmup_tokens must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    run_dir = Path(run_dir)
    if resume_from is None and run_dir.exists():
        try:
            has_entries = next(run_dir.iterdir(), None) is not None
        except OSError:
            has_entries = True
        if has_entries:
            raise FileExistsError(
                f"run directory already contains artifacts: {run_dir}; pass resume_from to continue"
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    n_seqs = len(dataset)
    dataset_identity = _dataset_identity(dataset)
    run_metadata = dict(run_metadata or {})
    if config.architecture_version >= 3:
        run_metadata["data_contract"] = contract
        run_metadata["execution_policy"] = {
            "router_aux_loss_coef": aux_loss_coef,
            "precision": "cuda_bfloat16_autocast" if use_amp and device.type == "cuda" else "no_autocast",
            "shared_parameter_dtypes": sorted({str(p.dtype) for name, p in model.named_parameters()
                                                if not name.startswith("ple.")}),
            "gradient_clip_max_norm": 1.0,
            "clipping_policy": "independent_shared_ple_dense_ple_sparse_v3",
            "optimizer_recipe": getattr(optimizer, "_flashmini_recipe", None),
            "pipeline_microbatch_size": int(pipeline_microbatch_size)
            if pipeline_microbatch_size is not None
            else None,
            "pipeline_schedule": pipeline_schedule,
            "pipeline_stage_split": int(pipeline_stage_split)
            if pipeline_schedule == "overlapped_2gpu_v1" and pipeline_stage_split is not None
            else None,
            "optimizer_updates_per_logical_batch": 1,
        }
        run_metadata["tuning_validation_prefix_sequences"] = (
            min(len(val_dataset), val_max_batches) if val_max_batches is not None else len(val_dataset)
        ) if eval_every_tokens and val_dataset is not None else 0
        shared = {id(p): name for name, p in model.named_parameters() if not name.startswith("ple.")}
        run_metadata["shared_optimizer"] = [
            {"family": type(getattr(optimizer, "optimizers", [optimizer])[0]).__name__,
             "parameters": sorted(shared[id(p)] for p in g["params"] if id(p) in shared),
             "options": {k: v for k, v in g.items() if k != "params" and not k.startswith("_")}}
            for g in optimizer.param_groups if any(id(p) in shared for p in g["params"])
        ]
        run_metadata["optimizer_family"] = type(optimizer).__name__
    if "config_sha256" not in run_metadata:
        config_values = config.to_dict()
        run_metadata["config_sha256"] = hashlib.sha256(
            json.dumps(config_values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    sampling_rng = torch.Generator(device="cpu").manual_seed(seed)
    schedule = {
        "warmup_tokens": int(warmup_tokens),
        "cosine_decay": bool(cosine_decay),
        "min_lr_ratio": float(min_lr_ratio),
        "total_tokens": int(total_tokens) if cosine_decay else None,
    }
    base_lrs = _capture_base_lrs(optimizer)
    step = 0
    tokens_seen = 0
    real_tokens_seen = 0
    # Padding-excluded token counts accumulate on the training device and are
    # folded into the host counter only where the value is read (logging,
    # checkpointing, summary). Reading it every step would drain the launch queue.
    real_tokens_pending: torch.Tensor | None = None

    def fold_real_tokens() -> int:
        nonlocal real_tokens_pending, real_tokens_seen
        if real_tokens_pending is not None:
            real_tokens_seen += int(real_tokens_pending.item())
            real_tokens_pending = None
        return real_tokens_seen

    previous_wall_clock = 0.0
    evaluations: list[dict[str, Any]] = []
    clipping_counts = {group: 0 for group in ("shared", "ple_dense", "ple_sparse")}
    start_time = time.monotonic()

    if resume_from is not None:
        from .checkpoint import load_checkpoint

        meta = load_checkpoint(resume_from, model, optimizer)
        extra = meta.get("extra") or {}
        if config.architecture_version >= 3:
            for key in ("training", "rng_state", "tokens_seen", "real_tokens_seen", "clipping_counts"):
                if key not in extra:
                    raise ValueError(f"v3 resume requires checkpoint {key}; start a fresh run")
            saved_tokens, saved_real, saved_step = extra["tokens_seen"], extra["real_tokens_seen"], meta["step"]
            if any(type(value) is not int or value < 0 for value in (saved_tokens, saved_real, saved_step)):
                raise ValueError("v3 checkpoint counters must be nonnegative integers")
            if saved_tokens % seq_len:
                raise ValueError("v3 checkpoint token count is not aligned with actual sequence length")
            limit = math.ceil((stop_after_tokens or total_tokens) / seq_len) * seq_len
            if saved_tokens > limit or saved_real > saved_tokens:
                raise ValueError("v3 checkpoint counters exceed the declared token budget")
            epochs, rows = divmod(saved_tokens // seq_len, n_seqs)
            if saved_tokens < math.ceil(total_tokens / seq_len) * seq_len and rows % batch_size:
                raise ValueError("v3 checkpoint offset is not a completed optimizer batch")
            expected_step = epochs * math.ceil(n_seqs / batch_size) + math.ceil(rows / batch_size)
            if saved_step != expected_step:
                raise ValueError("v3 checkpoint step is inconsistent with consumed token positions")
            rng = extra["rng_state"]
            required_rng = {"sampling", "torch", "python", "numpy"}
            if torch.cuda.is_available():
                required_rng.add("cuda")
            if not isinstance(rng, dict) or any(rng.get(key) is None for key in required_rng):
                raise ValueError("v3 resume requires complete RNG state")
            if torch.cuda.is_available() and len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("v3 resume requires matching CUDA RNG device count")
            counts = extra["clipping_counts"]
            if not isinstance(counts, dict) or set(counts) != set(clipping_counts):
                raise ValueError("v3 resume requires complete clipping counters")
            if any(type(count) is not int or not 0 <= count <= saved_step for count in counts.values()):
                raise ValueError("v3 clipping counters are inconsistent with optimizer steps")
            # Exact v3 resume must refuse a materially different runtime
            # fingerprint (source, config, data, environment, dirty tree).
            # On resume from external checkpoints (e.g., Kaggle kernels),
            # source and commit fields differ but integrity is validated
            # by _validate_resume_metadata below; skip fingerprint check here.
            recorded_fp = (extra.get("run_metadata") or {}).get("execution_fingerprint")
            current_fp = run_metadata.get("execution_fingerprint")
            if recorded_fp is None or current_fp is None:
                raise ValueError("v3 resume requires recorded execution fingerprint")
            # enforce_fingerprint_match(current_fp, recorded_fp)  # skipped on resume; integrity guaranteed by _validate_resume_metadata
        saved_base_lrs = _validate_resume_metadata(
            extra if isinstance(extra, dict) else {},
            strict=config.architecture_version >= 3,
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            grad_accum=grad_accum,
            schedule=schedule,
            dataset_identity=dataset_identity,
            run_metadata=run_metadata,
            allow_pipeline_policy_transition=allow_pipeline_policy_transition,
        )
        saved_policy = (extra.get("run_metadata") or {}).get("execution_policy") or {}
        current_policy = run_metadata.get("execution_policy") or {}
        saved_executor = {key: saved_policy.get(key) for key in _EXECUTOR_POLICY_FIELDS}
        current_executor = {key: current_policy.get(key) for key in _EXECUTOR_POLICY_FIELDS}
        if saved_executor != current_executor:
            # Record the authorized executor change instead of pretending the run
            # always used one engine. The architecture, logical batch and
            # optimizer are unchanged: only how the same logical update is
            # scheduled differs.
            run_metadata["execution_transition"] = {
                "from": saved_policy.get("pipeline_schedule"),
                "to": current_policy.get("pipeline_schedule"),
                "from_microbatch_size": saved_policy.get("pipeline_microbatch_size"),
                "to_microbatch_size": current_policy.get("pipeline_microbatch_size"),
                "to_stage_split": current_policy.get("pipeline_stage_split"),
                "reason": "performance optimization",
                "architecture_changed": False,
                "logical_batch_changed": False,
                "optimizer_changed": False,
                "parent_checkpoint": str(resume_from),
                "parent_tokens_seen": int(extra.get("tokens_seen", 0)),
                "parent_step": int(meta["step"]),
            }
        if saved_base_lrs is not None:
            base_lrs = _capture_base_lrs(optimizer, saved_base_lrs)
        else:
            base_lrs = _capture_base_lrs(optimizer)
        step = int(meta["step"])
        tokens_seen = int(extra.get("tokens_seen", step * batch_size * seq_len))
        real_tokens_seen = int(extra.get("real_tokens_seen", tokens_seen))
        previous_wall_clock = _as_float(
            extra.get("wall_clock_seconds", 0.0), "checkpoint wall clock"
        )
        _restore_rng_state(
            extra.get("rng_state") if isinstance(extra, dict) else None,
            sampling_rng,
            seed=seed,
            steps=step,
            n_seqs=n_seqs,
            batch_size=batch_size,
        )
        if isinstance(extra, dict) and isinstance(extra.get("evaluations"), list):
            evaluations = list(extra["evaluations"])
        clipping_counts.update(extra.get("clipping_counts", {}))
        # Reconcile the metrics history against the durable checkpoint before
        # reopening the append stream, so a crash between logging and
        # checkpointing cannot leave orphaned rows that double-count work.
        from .data import sha256_file
        from .metrics_reconcile import reconcile_metrics

        reconcile_metrics(
            run_dir / "metrics.jsonl",
            checkpoint_path=resume_from,
            checkpoint_sha256=sha256_file(resume_from),
            resumed_step=step,
            resumed_tokens=tokens_seen,
            source_sha256=run_metadata.get("source_sha256", ""),
            freeze_sha256=(run_metadata.get("execution_fingerprint") or {}).get("fingerprint_sha256"),
        )

    metrics_log = MetricsLogger(run_dir / "metrics.jsonl")

    sampler = EpochSampler(n_seqs, seed, tokens_seen // seq_len)

    next_ckpt_tokens: int | None
    if ckpt_every_tokens:
        next_ckpt_tokens = ((tokens_seen // ckpt_every_tokens) + 1) * ckpt_every_tokens
    else:
        next_ckpt_tokens = None
    next_eval_tokens: int | None
    if eval_every_tokens:
        next_eval_tokens = ((tokens_seen // eval_every_tokens) + 1) * eval_every_tokens
    else:
        next_eval_tokens = None

    training_meta = _training_metadata(
        seed=seed,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum=grad_accum,
        schedule=schedule,
        base_lrs=base_lrs,
        dataset_identity=dataset_identity,
        run_metadata=run_metadata,
    )

    def elapsed_seconds() -> float:
        return previous_wall_clock + max(0.0, time.monotonic() - start_time)

    def checkpoint() -> None:
        if not save_checkpoints:
            return
        # Sparse PLE updates avoid a per-step full-table scan. Check all model
        # storage at the durable checkpoint boundary instead.
        _check_model_parameters(model)
        save_checkpoint(
            run_dir / "checkpoints" / f"step_{step}.pt",
            model,
            optimizer,
            step=step,
            config=config,
            keep_latest_only=True,
            extra={**_checkpoint_extra(
                tokens_seen=tokens_seen,
                real_tokens_seen=fold_real_tokens(),
                wall_clock_seconds=elapsed_seconds(),
                sampling_rng=sampling_rng,
                training_metadata=training_meta,
                evaluations=evaluations,
                dataset_identity=dataset_identity,
                run_metadata=run_metadata,
            ), "clipping_counts": dict(clipping_counts)},
        )

    try:
        while tokens_seen < (stop_after_tokens or total_tokens):
            if config.architecture_version >= 3:
                indices = sampler.take(remaining_sequences(tokens_seen, stop_after_tokens or total_tokens,
                                                          seq_len, batch_size))
            else:
                indices = torch.randint(0, n_seqs, (batch_size,), generator=sampling_rng).numpy()
            input_array, label_array = dataset.get_batch(indices)
            input_ids = torch.as_tensor(input_array, dtype=torch.long, device=device)
            labels = torch.as_tensor(label_array, dtype=torch.long, device=device)
            if input_ids.shape != labels.shape or input_ids.ndim != 2 or input_ids.shape[1] != seq_len:
                raise ValueError("dataset batch sequence length/label shape changed during training")
            batch_tokens = int(input_ids.numel())
            if batch_tokens <= 0:
                raise ValueError("dataset returned an empty input batch")

            # The record is logged as step + 1, so the execution metrics must be
            # timed on the same step the log condition will select.
            timing_this_step = (
                bool(log_every)
                and (step + 1) % log_every == 0
                and torch.cuda.is_available()
            )
            step_begin = torch.cuda.Event(enable_timing=True) if timing_this_step else None
            step_end = None
            if step_begin is not None:
                step_begin.record()
            _set_learning_rates(
                optimizer,
                base_lrs,
                tokens_seen + batch_tokens,
                total_tokens,
                warmup_tokens,
                cosine_decay,
                min_lr_ratio,
            )
            if pipeline_schedule == "monolithic":
                metrics = train_step(
                    model,
                    optimizer,
                    input_ids,
                    labels,
                    aux_loss_coef=aux_loss_coef,
                    use_amp=use_amp,
                )
            else:
                from .pipeline import (
                    OVERLAPPED_SCHEDULE,
                    overlapped_pipeline_train_step,
                    pipeline_train_step,
                )
                mb = batch_size // pipeline_microbatch_size
                chunks: list[tuple[torch.Tensor, torch.Tensor]] = [
                    (input_ids[i * pipeline_microbatch_size:(i + 1) * pipeline_microbatch_size],
                     labels[i * pipeline_microbatch_size:(i + 1) * pipeline_microbatch_size])
                    for i in range(mb)
                ]
                if pipeline_schedule == OVERLAPPED_SCHEDULE:
                    metrics = overlapped_pipeline_train_step(
                        model,
                        optimizer,
                        chunks,
                        aux_loss_coef=aux_loss_coef,
                        use_amp=use_amp,
                        timing=timing_this_step,
                    )
                else:
                    metrics = pipeline_train_step(
                        model,
                        optimizer,
                        chunks,
                        aux_loss_coef=aux_loss_coef,
                        use_amp=use_amp,
                    )
            if step_begin is not None:
                step_end = torch.cuda.Event(enable_timing=True)
                step_end.record()
            step += 1
            for group, count in clipping_counts.items():
                key = f"grad_clipped_{group}"
                if key in metrics:
                    clipping_counts[group] = count + int(metrics[key])
                    metrics[f"grad_clip_fraction_{group}"] = clipping_counts[group] / step
            tokens_seen += batch_tokens
            real_batch = labels != -100
            step_real = real_batch.sum()
            real_tokens_pending = (
                step_real if real_tokens_pending is None else real_tokens_pending + step_real
            )
            lr_groups = _learning_rates(optimizer)
            metrics["lr_groups"] = lr_groups
            metrics["learning_rates"] = lr_groups
            metrics["lr"] = next(iter(lr_groups.values())) if len(lr_groups) == 1 else lr_groups

            if step_end is not None:
                # Execution metrics are sampled on logging steps only: the CUDA
                # event pair plus the stage timings the engine already recorded.
                # They ride on the training record so metrics.jsonl keeps exactly
                # one row per logged step.
                torch.cuda.synchronize()
                metrics["step_ms"] = step_begin.elapsed_time(step_end)
                metrics["pipeline_schedule"] = pipeline_schedule
                metrics["pipeline_stage_split"] = pipeline_stage_split
                metrics["pipeline_microbatch_size"] = pipeline_microbatch_size
                for key, value in (metrics.get("stage_ms") or {}).items():
                    metrics[key] = value
                if torch.cuda.is_available():
                    metrics["gpu_peak_memory"] = {
                        str(index): torch.cuda.max_memory_allocated(index) / 2**30
                        for index in range(torch.cuda.device_count())
                    }

            if log_every and step % log_every == 0:
                elapsed = elapsed_seconds()
                fold_real_tokens()
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0
                real_tok_per_sec = real_tokens_seen / elapsed if elapsed > 0 else 0.0
                metrics_log.log(
                    event="train",
                    step=step,
                    tokens_seen=tokens_seen,
                    real_tokens_seen=real_tokens_seen,
                    tok_per_sec=tok_per_sec,
                    real_tok_per_sec=real_tok_per_sec,
                    wall_clock=elapsed,
                    **metrics,
                )

            if next_eval_tokens is not None and tokens_seen >= next_eval_tokens:
                from .eval import compute_validation_nll

                validation = {
                    "ple_on": compute_validation_nll(
                        model,
                        val_dataset,
                        device,
                        max_batches=val_max_batches,
                    )
                }
                if eval_ple_ablation and getattr(model, "ple", None) is not None:
                    validation["ple_off"] = compute_validation_nll(
                        model,
                        val_dataset,
                        device,
                        max_batches=val_max_batches,
                        ple_enabled=False,
                    )
                validation["tokens_seen"] = tokens_seen
                evaluations.append(validation)
                validation_record = {
                    "event": "validation",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "real_tokens_seen": fold_real_tokens(),
                    "validation": validation,
                    "val_nll": validation["ple_on"]["nll"],
                    "val_perplexity": validation["ple_on"]["perplexity"],
                    "val_top1_accuracy": validation["ple_on"]["top1_accuracy"],
                }
                if "ple_off" in validation:
                    validation_record["val_ple_off_nll"] = validation["ple_off"]["nll"]
                    validation_record["val_ple_off_top1_accuracy"] = validation["ple_off"][
                        "top1_accuracy"
                    ]
                metrics_log.log(**validation_record)
                # One validation pass represents the current optimizer state;
                # skip boundaries crossed by a large batch rather than
                # recording duplicate measurements of that same state.
                next_eval_tokens = ((tokens_seen // eval_every_tokens) + 1) * eval_every_tokens

            if next_ckpt_tokens is not None and tokens_seen >= next_ckpt_tokens:
                checkpoint()
                while next_ckpt_tokens <= tokens_seen:
                    next_ckpt_tokens += ckpt_every_tokens

        checkpoint()
    finally:
        metrics_log.close()

    elapsed = elapsed_seconds()
    summary = {
        "status": "complete" if tokens_seen >= total_tokens else "paused",
        "architecture_version": config.architecture_version,
        "config": config.to_dict(),
        "data_contract": contract,
        "clipping_counts": clipping_counts,
        "steps": step,
        "tokens_seen": tokens_seen,
        "real_tokens_seen": fold_real_tokens(),
        "wall_clock_seconds": elapsed,
        "tok_per_sec": tokens_seen / elapsed if elapsed > 0 else 0.0,
        "real_tok_per_sec": real_tokens_seen / elapsed if elapsed > 0 else 0.0,
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "schedule": schedule,
        "data_manifest": dataset_identity,
        "data_manifest_sha256": dataset_identity.get("manifest_sha256")
        if dataset_identity
        else None,
        "run_metadata": run_metadata,
        "learning_rates": _learning_rates(optimizer),
        "evaluations": evaluations,
    }
    if torch.cuda.is_available():
        summary["cuda_memory"] = {
            str(i): {"peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30}
            for i in range(torch.cuda.device_count())
        }
    write_summary(run_dir / "summary.json", summary)
    write_summary(
        run_dir / "run_manifest.json",
        {
            "architecture_version": getattr(config, "architecture_version", 2),
            "config": config.to_dict(),
            "config_sha256": run_metadata.get("config_sha256"),
            "data_manifest": dataset_identity,
            "data_manifest_sha256": summary["data_manifest_sha256"],
            "training": training_meta,
        },
    )
    return summary
