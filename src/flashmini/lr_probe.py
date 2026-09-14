"""Bounded LR probe for a FlashMini variant.

Runs log-spaced candidate learning rates from identical initialization and reports
the best stable candidate. A failed or out-of-memory candidate is recorded without
preventing the remaining candidates from running.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import FlashMiniConfig
from .data import MemmapDataset
from .models import FlashMiniModel
from .training import train
from .optim import build_optimizer


_MAX_STABLE_GRAD_NORM = 10_000.0


def _seed_all(seed: int) -> None:
    """Reset every RNG used by model construction and the data sampler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as metrics_file:
        for line in metrics_file:
            # Skip blank or truncated lines left by interrupted candidates.
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _candidate_result(
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    *,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    losses = [row.get("loss") for row in rows if _finite(row.get("loss"))]
    grad_norms = [row.get("grad_norm") for row in rows if _finite(row.get("grad_norm"))]
    final_loss = losses[-1] if losses else None
    max_grad_norm = max(grad_norms) if grad_norms else None
    finite = bool(losses) and all(_finite(loss) for loss in losses) and all(_finite(grad) for grad in grad_norms)
    stable = finite and (max_grad_norm is None or float(max_grad_norm) <= _MAX_STABLE_GRAD_NORM)
    if status != "ok":
        stable = False
    return {
        "status": status,
        "stable": stable,
        "final_loss": final_loss,
        "initial_loss": losses[0] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "max_grad_norm": max_grad_norm,
        "tok_per_sec": summary.get("tok_per_sec") if summary else None,
        "steps": summary.get("steps") if summary else None,
        "tokens_seen": summary.get("tokens_seen") if summary else None,
        "error": error,
    }


def _release_candidate(model: Any, optimizer: Any, device: torch.device) -> None:
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def probe(
    config_path: str,
    data_dir: str,
    run_dir: str,
    tokens: int,
    lrs: list[float],
    batch_size: int = 4,
    seed: int = 0,
) -> dict:
    import yaml

    if not lrs:
        raise ValueError("at least one learning rate is required")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    config = FlashMiniConfig.from_dict(yaml.safe_load(Path(config_path).read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MemmapDataset(Path(data_dir), split="train")
    results: dict[str, dict[str, Any]] = {}

    for lr in lrs:
        model = None
        optimizer = None
        summary = None
        sub = Path(run_dir) / f"lr_{lr:.0e}"
        sub.mkdir(parents=True, exist_ok=True)
        _seed_all(seed)
        try:
            model = FlashMiniModel(config).to(device)
            optimizer = build_optimizer(model, lr=lr)
            summary = train(
                model,
                optimizer,
                dataset,
                config,
                sub,
                total_tokens=tokens,
                seq_len=config.max_seq_len,
                device=device,
                batch_size=batch_size,
                log_every=1,
                seed=seed,
            )
            rows = _read_metrics(sub / "metrics.jsonl")
            results[str(lr)] = _candidate_result(rows, summary)
        except torch.OutOfMemoryError as exc:
            results[str(lr)] = _candidate_result(
                _read_metrics(sub / "metrics.jsonl"),
                summary,
                status="oom",
                error=str(exc),
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            results[str(lr)] = _candidate_result(
                _read_metrics(sub / "metrics.jsonl"),
                summary,
                status="oom",
                error=str(exc),
            )
        finally:
            _release_candidate(model, optimizer, device)

    stable = {key: value for key, value in results.items() if value["stable"]}
    best_lr = min(stable, key=lambda key: stable[key]["final_loss"]) if stable else None
    return {
        "results": results,
        "best_lr": best_lr,
        "selection": "lowest finite, stable final loss" if best_lr is not None else "no stable candidate",
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lrs = [float(x) for x in args.lrs.split(",")]
    result = probe(
        args.config,
        args.data_dir,
        args.run_dir,
        args.tokens,
        lrs,
        args.batch_size,
        seed=args.seed,
    )
    out = Path(args.run_dir) / "lr_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
