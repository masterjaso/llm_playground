"""Evaluate an existing fresh p16 selector checkpoint on A and B only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.config import load_config
from dense2moe.training.torch_distill import ActivationShardDataset
from scripts.train_fresh_p16_selector import (
    DEFAULT_FRESH,
    DEFAULT_RUN,
    DEFAULT_SOURCE,
    _load_ab,
    _load_checkpoint_model,
    _load_dense_mlp,
    _hash_indices,
    _route_metrics,
    current_git_commit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--microbatch", type=int, default=512)
    args = parser.parse_args()

    fresh_dir = Path(args.fresh_dir)
    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or run_dir / "layer-checkpoints/fresh-selector/p16-top4-hard-regret-bce"
    activation = fresh_dir / "capture/layer-0000-train.json"
    ab_path = fresh_dir / "capture/fresh-selector-validation-ab.json"
    dataset = ActivationShardDataset(activation, split="train", microbatch=args.microbatch)
    a_indices, b_indices = _load_ab(ab_path, dataset.count)
    profile = load_config("configs/qwen38_p16s1_top4.yaml")
    partition = run_dir / "partitions/high-sparsity-p16-top4.json"
    weights = _load_dense_mlp(Path(args.source_dir))
    model = _load_checkpoint_model(weights, profile, partition, checkpoint, args.device)
    route_a = _route_metrics(model, dataset, a_indices, weights, microbatch=args.microbatch, device=args.device)
    route_b = _route_metrics(model, dataset, b_indices, weights, microbatch=args.microbatch, device=args.device)
    report_path = fresh_dir / "reports/p16-top4-fresh-selector-hard-regret-bce.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["evaluation_code_commit"] = current_git_commit()
    payload["metrics"]["route_validation_a"] = route_a
    payload["metrics"]["route_validation_b"] = route_b
    payload["fresh_selector_metrics"] = {
        "validation_a": {**payload["fresh_selector_metrics"]["validation_a"], **{key: route_a[key] for key in ("hard_quartile_recall", "hard_quartile_cosine", "hard_quartile_mean_jaccard", "hard_quartile_count", "hard_quartile_threshold")}},
        "validation_b": {**payload["fresh_selector_metrics"]["validation_b"], **{key: route_b[key] for key in ("hard_quartile_recall", "hard_quartile_cosine", "hard_quartile_mean_jaccard", "hard_quartile_count", "hard_quartile_threshold")}},
    }
    payload["telemetry"] = {
        "cosine": "metrics.validation_a/validation_b",
        "nmse": "metrics.validation_a/validation_b",
        "load_cv": "metrics.validation_a/validation_b.load_cv",
        "dead_experts": "metrics.validation_a/validation_b.dead_experts",
        "exact_set_match": "metrics.route_validation_a/validation_b.oracle_exact_set_match",
        "topk_recall": "metrics.route_validation_a/validation_b.oracle_route_recall",
        "mean_jaccard": "metrics.route_validation_a/validation_b.oracle_mean_jaccard",
        "router_entropy": "metrics.validation_a/validation_b.router_entropy",
        "topk_margin": "metrics.validation_a/validation_b.topk_logit_margin",
        "hard_quartile_recall": "metrics.route_validation_a/validation_b.hard_quartile_recall",
        "hard_quartile_cosine": "metrics.route_validation_a/validation_b.hard_quartile_cosine",
    }
    payload["validation_protocol"]["selection_indices_hash"] = _hash_indices(a_indices)
    payload["validation_protocol"]["validation_b_indices_hash"] = _hash_indices(b_indices)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FRESH_P16_SELECTOR_EVALUATION_COMPLETE", "route_validation_a": route_a, "route_validation_b": route_b, "code_commit": payload["evaluation_code_commit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
