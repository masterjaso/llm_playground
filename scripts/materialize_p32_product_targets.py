"""Promote a FIT/dev-selected p32/top4 or p32/top5 partition.

Unlike the historical ``materialize_p32_top5_candidate`` helper, this command
does not silently reuse a top6 artifact.  It requires a row evaluated for the
requested ``top_k`` in ``high-sparsity-partition-search.json`` and records the
selection hash and oracle evidence alongside the capacity-exact plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Keep the FIT/dev materializer usable before package installation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.provenance import current_git_commit


def materialize(*, run: Path, top_k: int, strategy: str | None = None, output_name: str | None = None) -> Path:
    if top_k not in {4, 5}:
        raise ValueError("p32 product targets are top4 and top5")
    report_path = run / "reports" / "high-sparsity-partition-search.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in report.get("results", [])
        if row.get("profile") == "p32" and int(row.get("top_k", -1)) == top_k
    ]
    if strategy is not None:
        rows = [row for row in rows if row.get("partition_strategy") == strategy]
    if not rows:
        raise ValueError(f"no FIT/dev p32/top{top_k} row found in {report_path}")
    # Lower reconstruction error is the primary oracle criterion; load CV is a
    # deterministic tie-breaker, never a hidden post-hoc validation signal.
    chosen: dict[str, Any] = min(rows, key=lambda row: (float(row["normalized_mse"]), float(row.get("expert_usage_cv", float("inf")))))
    partition = dict(chosen["partition"])
    profile = {
        "name": f"p32_top{top_k}",
        "hidden_size": 5120,
        "dense_intermediate_size": 17408,
        "routed_experts": 32,
        "expert_intermediate_size": 512,
        "shared_intermediate_size": 1024,
        "top_k": top_k,
    }
    partition.update(
        {
            "schema_version": 4,
            "status": "PARTITION_READY_P32_PRODUCT_TARGET_FIT_ONLY",
            "profile": profile,
            "profile_name": profile["name"],
            "top_k": top_k,
            "routing_mode": "independent_positive",
            "strategy": chosen.get("partition_strategy"),
            "initial_expert_scales": [max(float(value), 1e-3) for value in chosen.get("learned_global_scales", [])],
            "architecture_dev_selection_hash": report.get("dev_subset", {}).get("selected_row_key_hash"),
            "architecture_dev_tokens": int(report.get("dev_tokens", report.get("dev_subset", {}).get("selected_count", 0))),
            "selection_contract": "FIT/dev only; topology-specific p32 target; holdout closed",
            "oracle_result": chosen,
            "materialized_from_report": str(report_path),
            "code_commit": current_git_commit(),
        }
    )
    output = run / "partitions" / (output_name or f"high-sparsity-p32-top{top_k}-product.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(partition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/20260815-184644-windows-real-d2m-v4-streaming")
    parser.add_argument("--top-k", type=int, choices=(4, 5), required=True)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    output = materialize(run=Path(args.run_dir), top_k=args.top_k, strategy=args.strategy, output_name=args.output_name)
    print(json.dumps({"status": "MATERIALIZED", "partition": str(output), "top_k": args.top_k}, indent=2), flush=True)


if __name__ == "__main__":
    main()
