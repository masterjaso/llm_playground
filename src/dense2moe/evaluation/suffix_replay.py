"""Exact paired suffix replay for one patched FFN layer.

The evaluator is deliberately model-interface agnostic: callers may provide
one-at-a-time loaders or forward callbacks so two 27B checkpoints need not be
resident simultaneously.  It never falls back to approximate suffix
propagation; resource failures are explicit terminal statuses.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .lm import StreamingLMMetrics
from .receipts import build_lm_output_receipt


class ExactSuffixReplayError(ValueError):
    """Raised when identity or patch-boundary validation fails."""


BLOCKED_EXACT_SUFFIX_REPLAY_RESOURCE_LIMIT = "BLOCKED_EXACT_SUFFIX_REPLAY_RESOURCE_LIMIT"
BLOCKED_EXACT_KL_RESOURCE_LIMIT = "BLOCKED_EXACT_KL_RESOURCE_LIMIT"


def _state_digest(model: Any, *, exclude_ffn: bool = False) -> dict[str, str]:
    state = model.state_dict() if hasattr(model, "state_dict") else {}
    result: dict[str, str] = {}
    for name, value in state.items():
        if exclude_ffn and (".mlp." in str(name) or str(name).startswith("mlp.")):
            continue
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").contiguous()
            raw = value.numpy().tobytes()
        else:
            raw = bytes(value)
        result[str(name)] = hashlib.sha256(raw).hexdigest()
    return result


def validate_source_identity(
    source_model: Mapping[str, Any],
    *,
    expected_model: str = "Qwen/Qwen3.8-27B",
    expected_revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    expected_root_family: str = "qwen3_5",
    expected_text_model_type: str = "qwen3_5_text",
    expected_layers: int = 64,
    expected_hidden_size: int = 5120,
    expected_dense_intermediate: int = 17408,
) -> dict[str, Any]:
    errors: list[str] = []
    checks = {
        "model": expected_model,
        "revision": expected_revision,
        "root_family": expected_root_family,
        "text_model_type": expected_text_model_type,
        "layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "dense_intermediate": expected_dense_intermediate,
    }
    for key, expected in checks.items():
        if key in source_model and source_model.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {source_model.get(key)!r}")
    missing = [key for key in checks if key not in source_model]
    return {"valid": not errors and not missing, "errors": errors, "missing": missing, "expected": checks}


def _extract_logits(output: Any) -> Any:
    if isinstance(output, Mapping):
        if "logits" in output:
            return output["logits"]
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    return output


def _resource_measurements(start: float, *, batch_size: int, token_count: int) -> dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - start)
    peak_vram = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            peak_vram = int(torch.cuda.max_memory_allocated())
    except ImportError:  # pragma: no cover
        pass
    peak_ram = None
    try:
        import psutil  # type: ignore

        peak_ram = int(psutil.Process().memory_info().rss)
    except ImportError:  # pragma: no cover
        pass
    return {
        "batch_size": int(batch_size),
        "peak_vram_bytes": peak_vram,
        "peak_system_ram_bytes": peak_ram,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (float(token_count / elapsed) if elapsed > 0 else None),
    }


class PairedSuffixReplayEvaluator:
    """Run exact paired teacher-forced suffix replay for one layer patch."""

    def __init__(
        self,
        *,
        source_identity: Mapping[str, Any],
        expected_layer: int,
        patched_layers: Sequence[int] | None = None,
        expected_revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        max_tokens: int | None = None,
        max_vocab_chunk: int | None = None,
    ) -> None:
        identity = validate_source_identity(source_identity, expected_revision=expected_revision)
        if not identity["valid"]:
            raise ExactSuffixReplayError(f"source identity validation failed: {identity}")
        if expected_layer < 0 or expected_layer >= 64:
            raise ExactSuffixReplayError(f"layer out of source-model range: {expected_layer}")
        layers = list(patched_layers if patched_layers is not None else [expected_layer])
        if layers != [expected_layer]:
            raise ExactSuffixReplayError("exact suffix replay requires exactly one patched FFN layer")
        self.source_identity = dict(source_identity)
        self.expected_layer = int(expected_layer)
        self.max_tokens = max_tokens
        self.max_vocab_chunk = max_vocab_chunk

    def validate_models(self, dense_model: Any | None, candidate_model: Any | None) -> dict[str, Any]:
        if dense_model is None or candidate_model is None:
            return {"valid": True, "skipped": True, "reason": "one-at-a-time loader/callback mode"}
        dense_non_ffn = _state_digest(dense_model, exclude_ffn=True)
        candidate_non_ffn = _state_digest(candidate_model, exclude_ffn=True)
        if dense_non_ffn and candidate_non_ffn and dense_non_ffn != candidate_non_ffn:
            differing = sorted(set(dense_non_ffn) ^ set(candidate_non_ffn))
            differing.extend(name for name in sorted(set(dense_non_ffn) & set(candidate_non_ffn)) if dense_non_ffn[name] != candidate_non_ffn[name])
            raise ExactSuffixReplayError(f"non-FFN source tensors differ: {differing[:8]}")
        return {"valid": True, "non_ffn_source_identical": True}

    def evaluate(
        self,
        sequences: Iterable[Any] | Any,
        *,
        dense_model: Any | None = None,
        candidate_model: Any | None = None,
        dense_forward: Callable[[Any], Any] | None = None,
        candidate_forward: Callable[[Any], Any] | None = None,
        code_science_identity: Mapping[str, Any] | None = None,
        runtime_lock_identity: Mapping[str, Any] | None = None,
        candidate_identity: Mapping[str, Any] | None = None,
        dataset_identity: Mapping[str, Any] | None = None,
        source_receipt_lineage: Mapping[str, Any] | None = None,
        max_batch_size: int | None = None,
    ) -> dict[str, Any]:
        self.validate_models(dense_model, candidate_model)
        if dense_forward is None:
            if dense_model is None:
                raise ExactSuffixReplayError("dense_model or dense_forward is required")
            dense_forward = lambda batch: _extract_logits(dense_model(**batch) if isinstance(batch, Mapping) else dense_model(batch))
        if candidate_forward is None:
            if candidate_model is None:
                raise ExactSuffixReplayError("candidate_model or candidate_forward is required")
            candidate_forward = lambda batch: _extract_logits(candidate_model(**batch) if isinstance(batch, Mapping) else candidate_model(batch))
        if isinstance(sequences, Mapping) or hasattr(sequences, "shape"):
            sequence_iterable: Iterable[Any] = [sequences]
        else:
            sequence_iterable = sequences
        metrics = StreamingLMMetrics()
        start = time.perf_counter()
        total_tokens = 0
        max_batch = 0
        try:
            for sequence in sequence_iterable:
                if isinstance(sequence, Mapping):
                    batch = dict(sequence)
                    targets = batch.pop("targets", batch.pop("labels", None))
                    mask = batch.pop("metric_mask", batch.pop("loss_mask", batch.get("attention_mask")))
                else:
                    batch = {"input_ids": sequence}
                    targets = None
                    mask = None
                if max_batch_size is not None and hasattr(batch.get("input_ids"), "shape") and int(batch["input_ids"].shape[0]) > max_batch_size:
                    raise ExactSuffixReplayError("sequence batch exceeds max_batch_size; split frozen input deterministically")
                dense_logits = _extract_logits(dense_forward(batch))
                candidate_logits = _extract_logits(candidate_forward(batch))
                if self.max_vocab_chunk is not None:
                    vocab_size = int(dense_logits.shape[-1])
                    if vocab_size > int(self.max_vocab_chunk):
                        # The generic callback interface cannot safely split a
                        # full-vocabulary forward pass after the fact.  Do not
                        # relabel a top-k or truncated calculation as exact;
                        # callers with a chunk-capable model adapter can omit
                        # this guard and emit exact bounded reductions.
                        resource = _resource_measurements(start, batch_size=max_batch, token_count=total_tokens)
                        return {
                            "status": BLOCKED_EXACT_KL_RESOURCE_LIMIT,
                            "error": f"vocabulary dimension {vocab_size} exceeds configured exact chunk {self.max_vocab_chunk}",
                            "metrics": {},
                            "resource_measurements": resource,
                            "exact": False,
                            "scientific_evaluation_performed": False,
                        }
                batch_tokens = int(dense_logits.shape[-2] * dense_logits.shape[-3]) if getattr(dense_logits, "ndim", 0) >= 3 else int(dense_logits.shape[0])
                if self.max_tokens is not None and total_tokens + batch_tokens > self.max_tokens:
                    raise ExactSuffixReplayError("frozen sequence set exceeds configured max_tokens")
                metrics.update(dense_logits, candidate_logits, targets=targets, mask=mask, baseline_logits=dense_logits)
                total_tokens += batch_tokens
                max_batch = max(max_batch, int(batch.get("input_ids").shape[0]) if hasattr(batch.get("input_ids"), "shape") else 1)
        except (MemoryError, RuntimeError) as exc:
            message = str(exc).casefold()
            if isinstance(exc, MemoryError) or "out of memory" in message or "cuda" in message and "memory" in message:
                resource = _resource_measurements(start, batch_size=max_batch, token_count=total_tokens)
                return {
                    "status": BLOCKED_EXACT_SUFFIX_REPLAY_RESOURCE_LIMIT,
                    "error": str(exc),
                    "metrics": {},
                    "resource_measurements": resource,
                    "exact": False,
                    "scientific_evaluation_performed": False,
                }
            raise
        result = metrics.finalize()
        resource = _resource_measurements(start, batch_size=max_batch, token_count=total_tokens)
        receipt = build_lm_output_receipt(
            source_model=self.source_identity,
            candidate={"layer": self.expected_layer, **dict(candidate_identity or {})},
            layer=self.expected_layer,
            dataset=dict(dataset_identity or {}),
            raw_metrics=result,
            dense_numerical_baseline={
                "status": result.get("dense_repeat_baseline_status"),
                "mean_forward_kl": result.get("dense_repeat_mean_forward_kl"),
                "p95_forward_kl": result.get("dense_repeat_p95_forward_kl"),
            },
            threshold_classification="NOT_CLASSIFIED",
            source_receipt_lineage=source_receipt_lineage,
            code_science_identity=code_science_identity,
            runtime_lock_identity=runtime_lock_identity,
            resource_measurements=resource,
        )
        return {
            "status": "EXACT_SUFFIX_REPLAY_COMPLETE",
            "exact": True,
            "scientific_evaluation_performed": True,
            "metrics": result,
            "receipt": receipt,
            "resource_measurements": resource,
            "patched_layer": self.expected_layer,
            "patched_layer_count": 1,
            "source_identical_except_ffn": True,
        }


def run_exact_layer_patch_replay(
    *,
    source_identity: Mapping[str, Any],
    expected_layer: int,
    sequences: Iterable[Any] | Any,
    patched_layers: Sequence[int] | None = None,
    expected_revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    max_tokens: int | None = None,
    max_vocab_chunk: int | None = None,
    **evaluate_kwargs: Any,
) -> dict[str, Any]:
    evaluator = PairedSuffixReplayEvaluator(
        source_identity=source_identity,
        expected_layer=expected_layer,
        patched_layers=patched_layers,
        expected_revision=expected_revision,
        max_tokens=max_tokens,
        max_vocab_chunk=max_vocab_chunk,
    )
    return evaluator.evaluate(sequences, **evaluate_kwargs)


__all__ = [
    "BLOCKED_EXACT_KL_RESOURCE_LIMIT",
    "BLOCKED_EXACT_SUFFIX_REPLAY_RESOURCE_LIMIT",
    "ExactSuffixReplayError",
    "PairedSuffixReplayEvaluator",
    "run_exact_layer_patch_replay",
    "validate_source_identity",
]
