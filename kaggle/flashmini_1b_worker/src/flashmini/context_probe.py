"""Run a structural causal probe at a configurable sequence length.

The probe answers only whether a FlashMini model can execute at the requested
length and whether suffix changes leave prefix logits unchanged.  It records
that limited scope explicitly; a passing probe is not a long-context quality
or capability result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from .checkpoint import load_checkpoint
from .config import FlashMiniConfig
from .models import FlashMiniModel


def _load_config(config: FlashMiniConfig | Path | str) -> tuple[FlashMiniConfig, str | None, str | None]:
    """Load a config and retain its source path/hash when one was supplied."""
    if isinstance(config, FlashMiniConfig):
        return copy.deepcopy(config), None, None
    path = Path(config)
    raw = path.read_bytes()
    values = yaml.safe_load(raw)
    if not isinstance(values, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    return FlashMiniConfig.from_dict(values), str(path), hashlib.sha256(raw).hexdigest()


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expand_causal_masks(model: torch.nn.Module, seq_len: int) -> dict[str, Any]:
    """Expand known fixed causal masks after a checkpoint has been validated."""
    expanded: list[str] = []
    capacities: list[int] = []
    for module_name, module in model.named_modules():
        mask = getattr(module, "causal_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
            continue
        capacity = int(mask.shape[-1])
        capacities.append(capacity)
        if capacity >= seq_len:
            continue
        shape = (seq_len, seq_len)
        replacement = torch.tril(torch.ones(shape, dtype=mask.dtype, device=mask.device))
        replacement = replacement.reshape((1,) * (mask.ndim - 2) + shape)
        module.register_buffer("causal_mask", replacement, persistent=False)
        expanded.append(module_name or module.__class__.__name__)
    return {
        "fixed_mask_modules_expanded": expanded,
        "fixed_mask_capacities_before": capacities,
    }


def _ensure_capacity(model: torch.nn.Module, seq_len: int, *, checkpoint: bool) -> dict[str, Any]:
    """Ensure attention masks cover ``seq_len`` and report what was changed."""
    details = _expand_causal_masks(model, seq_len) if checkpoint else {
        "fixed_mask_modules_expanded": [],
        "fixed_mask_capacities_before": [],
    }
    insufficient: list[str] = []
    for module_name, module in model.named_modules():
        mask = getattr(module, "causal_mask", None)
        if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and mask.shape[-1] < seq_len:
            insufficient.append(module_name or module.__class__.__name__)
    if insufficient:
        raise ValueError(
            "model causal masks do not cover requested sequence length: "
            + ", ".join(insufficient)
        )
    details["fixed_mask_capacities_after"] = [
        int(mask.shape[-1])
        for module in model.modules()
        if isinstance(mask := getattr(module, "causal_mask", None), torch.Tensor)
        and mask.ndim >= 2
    ]
    return details


def _tokens(config: FlashMiniConfig, seq_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(config.vocab_size, (1, seq_len), generator=generator, dtype=torch.long)
    eos_id = config.ple.eos_id
    if eos_id is not None and config.vocab_size > 1:
        # Keep EOS out of the random structural input so PLE document resets
        # do not turn this into a boundary-specific test.
        ids.masked_fill_(ids == eos_id, (eos_id + 1) % config.vocab_size)
    return ids


def _prefix_check(
    model: torch.nn.Module,
    ids: torch.Tensor,
    prefix_len: int,
) -> dict[str, Any]:
    changed = ids.clone()
    changed[:, prefix_len:] = (changed[:, prefix_len:] + 1) % model.config.vocab_size
    with torch.no_grad():
        original = model(ids)
        perturbed = model(changed)
    if not isinstance(original, dict) or not isinstance(perturbed, dict):
        raise TypeError("model forward must return mappings for the structural probe")
    logits = original.get("logits")
    changed_logits = perturbed.get("logits")
    if not isinstance(logits, torch.Tensor) or not isinstance(changed_logits, torch.Tensor):
        raise TypeError("model forward must return logits for the structural probe")
    if logits.shape != changed_logits.shape or logits.ndim < 2:
        raise ValueError("structural probe logits shape changed between inputs")
    prefix = logits[..., :prefix_len, :]
    changed_prefix = changed_logits[..., :prefix_len, :]
    if not torch.isfinite(prefix).all().item() or not torch.isfinite(changed_prefix).all().item():
        raise FloatingPointError("structural probe produced non-finite prefix logits")
    difference = (prefix.float() - changed_prefix.float()).abs()
    max_difference = float(difference.max().item()) if difference.numel() else 0.0
    passed = bool(torch.allclose(prefix, changed_prefix, rtol=1e-5, atol=1e-5))
    return {
        "prefix_length": prefix_len,
        "suffix_length": int(ids.shape[1] - prefix_len),
        "max_prefix_logit_abs_diff": max_difference,
        "passed": passed,
    }


def run_context_probe(
    config: FlashMiniConfig | Path | str,
    *,
    seq_len: int,
    checkpoint: Path | str | None = None,
    device: str | torch.device = "auto",
    seed: int = 0,
    prefix_len: int | None = None,
) -> dict[str, Any]:
    """Execute the structural probe and return JSON-serializable evidence."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    config_obj, config_path, config_sha256 = _load_config(config)
    resolved_device = _resolve_device(device)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    requested_prefix = prefix_len if prefix_len is not None else max(1, seq_len // 2)
    if requested_prefix <= 0 or requested_prefix >= seq_len:
        raise ValueError("prefix_len must be between 1 and seq_len - 1")

    checkpoint_config_max_seq_len = config_obj.max_seq_len
    if checkpoint_path is None:
        runtime_config = copy.deepcopy(config_obj)
        runtime_config.max_seq_len = seq_len
        runtime_config.__post_init__()
    else:
        # Checkpoint config validation must happen against the original config
        # before fixed masks are enlarged for this read-only structural probe.
        runtime_config = config_obj

    _seed(seed)
    model = FlashMiniModel(runtime_config)
    checkpoint_metadata: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint_metadata = load_checkpoint(checkpoint_path, model)
    capacity_details = _ensure_capacity(model, seq_len, checkpoint=checkpoint_path is not None)
    model.to(resolved_device)
    model.eval()
    ids = _tokens(runtime_config, seq_len, seed).to(resolved_device)
    check = _prefix_check(model, ids, requested_prefix)
    result: dict[str, Any] = {
        "purpose": "structural_causal_execution_probe",
        "status": "PASS" if check["passed"] else "FAIL",
        "structural_only": True,
        "long_context_capability_claim": False,
        "interpretation": "Causality and execution only; no long-context quality or capability claim.",
        "config_path": config_path,
        "config_sha256": config_sha256,
        "architecture_version": int(runtime_config.architecture_version),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_step": checkpoint_metadata.get("step") if checkpoint_metadata else None,
        "checkpoint_config_max_seq_len": int(checkpoint_config_max_seq_len),
        "requested_seq_len": int(seq_len),
        "device": str(resolved_device),
        "seed": int(seed),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "capacity": capacity_details,
        "causality": check,
    }
    return result


def _write_result(result: dict[str, Any], out: Path | None) -> None:
    if out is None:
        return
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = run_context_probe(
        args.config,
        seq_len=args.seq_len,
        checkpoint=args.checkpoint,
        device=args.device,
        seed=args.seed,
        prefix_len=args.prefix_len,
    )
    _write_result(result, Path(args.out) if args.out else None)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
