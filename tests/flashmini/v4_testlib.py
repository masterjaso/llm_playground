"""Shared fixtures for v4 runner tests: synthetic packed data and surrogate train configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from flashmini.v4_data import write_packed
from flashmini.v4_train import Runner, TrainConfig

FINGERPRINT = "0" * 64
VOCAB = 64


def write_synthetic_data(root: Path, *, documents: int = 400, seed: int = 0) -> Path:
    """Learnable documents: arithmetic progressions mod 62 over ids 2..63, EOS-separated."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(documents):
        start, step, length = int(rng.integers(0, 62)), int(rng.choice([1, 3, 5])), int(rng.integers(8, 40))
        docs.append([2 + (start + step * i) % 62 for i in range(length)])
    write_packed(docs, root / "data", tokenizer_fingerprint=FINGERPRINT, vocab_size=VOCAB, eos_id=0)
    return root / "data" / "packed_manifest.json"


def train_config(root: Path, manifest: Path, *, steps_a: int, steps_b: int, strategy: str = "single", nodes: int = 1,
                 per_node: int = 1, micro: tuple[int, int] = (2, 4), accumulation: tuple[int, int] = (2, 2),
                 seq: tuple[int, int] = (16, 8), mode: str = "exact_prepass", lr_scale: float = 1.0) -> dict[str, Any]:
    world = nodes * per_node
    tokens_a = world * accumulation[0] * micro[0] * seq[0]
    tokens_b = world * accumulation[1] * micro[1] * seq[1]
    planned = steps_a * tokens_a + steps_b * tokens_b
    return {
        "run": {"name": "test", "purpose": "engineering_test", "output_dir": str(root), "seed": 7, "deterministic": True},
        "model": {"config": "unused", "surrogate": True, "gdn_kernel": "chunked", "activation_checkpointing": False, "compute_dtype": "float32"},
        "tokenizer": {"fingerprint": FINGERPRINT},
        "distributed": {"strategy": strategy, "nodes": nodes, "gpus_per_node": per_node, "ple_backing": "process", "force_cpu": True},
        "schedule": {"planned_total_tokens": planned, "lr": {"kind": "cosine", "warmup_tokens": tokens_a, "min_lr_ratio": 0.1}},
        "optimizer": {"muon": {"lr": 0.05 * lr_scale, "weight_decay": 0.0, "ns_dtype": "float32"},
                      "adamw": {"lr": 1e-2 * lr_scale, "betas": [0.9, 0.95], "eps": 1e-8,
                                "weight_decay": {"embedding": 0.0, "control_matrix": 0.01, "no_decay": 0.0}},
                      "ple": {"lr": 1e-2 * lr_scale, "betas": [0.9, 0.95], "eps": 1e-8}, "grad_clip": 1.0},
        "loss": {"router_aux_coefficient": 0.01, "router_balance_mode": mode},
        "data": {"teacher_mixture": {"real": 1.0}, "phases": [
            {"name": "short", "manifest": str(manifest), "sequence_length": seq[0], "micro_batch_sequences": micro[0],
             "gradient_accumulation": accumulation[0], "until_tokens": steps_a * tokens_a},
            {"name": "shorter", "manifest": str(manifest), "sequence_length": seq[1], "micro_batch_sequences": micro[1],
             "gradient_accumulation": accumulation[1], "until_tokens": planned}]},
        "checkpoint": {"dir": str(root / "checkpoints"), "interval_optimizer_steps": 1000, "keep_last": 2},
        "observability": {"metrics_path": str(root / "metrics.jsonl"), "log_expert_distribution": True},
        "watchdog": {"step_timeout_seconds": 900},
    }


def with_overrides(config: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for section, values in sections.items():
        result.setdefault(section, {}).update(values)
    return result


def write_config(path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


def run(path: Path, *, keep_runner: bool = False):
    runner = Runner(TrainConfig.load(path))
    try:
        runner.setup()
        runner.run()
    finally:
        runner.close()
    return runner if keep_runner else None


def metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def full_state(runner) -> dict[str, torch.Tensor]:
    """Every trainable and optimizer tensor, gathered to full tensors."""
    from torch.distributed.tensor import DTensor

    def full(value):
        return value.full_tensor() if isinstance(value, DTensor) else value

    state = {f"param/{name}": full(p).detach().clone() for name, p in runner.model.named_parameters()}
    state.update({f"ple/{name}": table.clone() for name, table in runner.model.ple.store.named_tables()})
    names = {id(p): name for name, p in runner.model.named_parameters()}
    for optimizer, label in ((runner.stack.muon, "muon"), (runner.stack.adamw, "adamw")):
        for param, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    state[f"{label}/{names[id(param)]}/{key}"] = full(value).detach().clone()
    for head, item in runner.stack.ple.state.items():
        state[f"ple_adam/{head}/exp_avg"] = item.exp_avg.clone()
        state[f"ple_adam/{head}/exp_avg_sq"] = item.exp_avg_sq.clone()
        state[f"ple_adam/{head}/step"] = item.step.clone()
    return state
