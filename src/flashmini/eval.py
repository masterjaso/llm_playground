"""Validation and downstream evaluation helpers for FlashMini."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F


def _snapshot_training_modes(model: torch.nn.Module) -> dict[torch.nn.Module, bool]:
    """Remember mixed parent/child modes so evaluation can restore them exactly."""
    return {module: module.training for module in model.modules()}


def _restore_training_modes(modes: dict[torch.nn.Module, bool]) -> None:
    for module, training in modes.items():
        module.training = training


def _snapshot_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def compute_validation_nll(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    max_batches: int | None = None,
    ple_enabled: bool | None = None,
) -> dict[str, float | int]:
    """Compute masked-token NLL, perplexity, and top-1 accuracy.

    ``-100`` labels are ignored for both NLL and accuracy. The model's complete
    training-mode tree and all process RNGs are restored even when evaluation
    raises, so periodic validation cannot perturb a resumed trajectory.
    """
    if len(dataset) <= 0:
        raise ValueError("validation dataset is empty")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    modes = _snapshot_training_modes(model)
    rng_state = _snapshot_rng()
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0
    n_batches = 0
    try:
        model.eval()
        with torch.no_grad():
            for index in range(len(dataset)):
                if max_batches is not None and n_batches >= max_batches:
                    break
                input_array, label_array = dataset.get_batch(torch.tensor([index]).numpy())
                input_ids = torch.as_tensor(input_array, device=device)
                labels = torch.as_tensor(label_array, device=device)
                kwargs: dict[str, Any] = {"labels": labels}
                if ple_enabled is not None:
                    kwargs["ple_enabled"] = ple_enabled
                out = model(input_ids, **kwargs)
                if not isinstance(out, dict) or "logits" not in out:
                    raise ValueError("model forward must return logits for validation")
                logits = out["logits"]
                if logits.shape[:-1] != labels.shape:
                    raise ValueError(
                        "validation logits/labels shape mismatch: "
                        f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
                    )
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError("validation logits are non-finite")

                flat_labels = labels.reshape(-1)
                valid = flat_labels != -100
                valid_count = int(valid.sum().item())
                if valid_count:
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    token_nll = F.cross_entropy(
                        flat_logits,
                        flat_labels,
                        reduction="none",
                        ignore_index=-100,
                    )
                    total_nll += float(token_nll[valid].sum().item())
                    predictions = flat_logits.argmax(dim=-1)
                    correct_tokens += int((predictions[valid] == flat_labels[valid]).sum().item())
                    total_tokens += valid_count
                n_batches += 1
    finally:
        _restore_training_modes(modes)
        _restore_rng(rng_state)

    if total_tokens <= 0:
        raise ValueError("validation dataset has no non-ignored target tokens")
    nll = total_nll / total_tokens
    if not math.isfinite(nll):
        raise FloatingPointError("validation NLL is non-finite")
    perplexity = math.exp(min(nll, math.log(torch.finfo(torch.float64).max)))
    accuracy = correct_tokens / total_tokens
    return {
        "nll": nll,
        "perplexity": perplexity,
        "top1_accuracy": accuracy,
        "top1_token_accuracy": accuracy,
        "token_accuracy": accuracy,
        "correct_tokens": correct_tokens,
        "tokens": total_tokens,
        "batches": n_batches,
    }


def run_lm_eval(
    model: torch.nn.Module,
    tokenizer,
    tasks: list[str],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Report downstream status without invoking an unbound HF stub adapter.

    ``lm-eval`` needs a real model adapter/tokenizer integration. A placeholder
    model would produce a report that looks like a benchmark for this model
    while evaluating something else, so downstream scores remain
    explicitly unsupported until that adapter exists.
    """
    del model, tokenizer, device, batch_size
    return {
        "status": "unsupported",
        "adapter_status": "unsupported",
        "supported": False,
        "reason": "downstream lm-eval adapter is not implemented for FlashMini",
        "error": "downstream lm-eval adapter is not implemented for FlashMini",
        "tasks": {},
        "requested_tasks": list(tasks),
    }


def aggregate_composite(scores: dict[str, dict]) -> dict[str, float]:
    """Normalized aggregate/composite of downstream task scores."""
    accs = []
    for score in scores.values():
        for key in ("acc_norm", "acc", "perplexity"):
            if key in score:
                accs.append(float(score[key]))
                break
    if not accs:
        return {"composite": 0.0, "n_tasks": 0}
    return {"composite": sum(accs) / len(accs), "n_tasks": len(accs)}
