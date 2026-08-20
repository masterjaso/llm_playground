"""Materialize one already-evaluated TRAIN/dev partition-search row.

The high-sparsity search keeps every evaluated plan in its report, while the
default finalist artifact contains only the NMSE-selected plan.  This helper
promotes a named evaluated row (for example ``activation_magnitude``) to a
standalone partition artifact without rerunning the expensive search or
opening the holdout split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dense2moe.provenance import current_git_commit


DEFAULT_RUN = Path("runs/20260815-184644-windows-real-d2m-v4-streaming")


def _write_partition(*, run: Path, profile: str, top_k: int, strategy: str, output_name: str) -> Path:
    report_path = run / "reports/high-sparsity-partition-search.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in report["results"]
        if row.get("profile") == profile
        and int(row.get("top_k", -1)) == int(top_k)
        and row.get("partition_strategy") == strategy
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one evaluated row for profile={profile!r}, top_k={top_k}, "
            f"strategy={strategy!r}; found {len(rows)}"
        )
    row: dict[str, Any] = rows[0]
    partition = dict(row["partition"])
    partition.update(
        {
            "schema_version": 3,
            "status": "PARTITION_READY_HIGH_SPARSITY_ALTERNATIVE_TRAINING",
            "layer": int(report.get("layer", 0)),
            "profile": {
                "name": profile,
                "routed_experts": int(partition["routed_experts"]),
                "expert_intermediate_size": int(partition["expert_intermediate_size"]),
                "shared_intermediate_size": int(partition["shared_intermediate_size"]),
                "dense_intermediate_size": int(partition["dense_intermediate_size"]),
                "hidden_size": 5120,
            },
            "profile_name": f"{profile}_top{top_k}",
            "top_k": int(top_k),
            "routing_mode": "independent_positive",
            "strategy": strategy,
            "initial_expert_scales": [max(float(value), 1e-3) for value in row["learned_global_scales"]],
            "oracle_learned_global_scales": row["learned_global_scales"],
            "architecture_dev_selection_hash": report["dev_subset"]["selected_row_key_hash"],
            "architecture_dev_tokens": int(report["dev_tokens"]),
            "selection_contract": "TRAIN/dev only; full holdout not opened by this materializer",
            "source_manifest": report.get("run_dir"),
            "oracle_result": row,
            "materialized_from_report": str(report_path),
            "code_commit": current_git_commit(),
        }
    )
    output_path = run / "partitions" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(partition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--profile", default="p16")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--strategy", default="activation_magnitude")
    parser.add_argument("--output-name", default="high-sparsity-p16-top4-activation-magnitude.json")
    args = parser.parse_args()
    path = _write_partition(
        run=Path(args.run_dir),
        profile=args.profile,
        top_k=args.top_k,
        strategy=args.strategy,
        output_name=args.output_name,
    )
    print(json.dumps({"status": "MATERIALIZED", "partition": str(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
