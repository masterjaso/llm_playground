"""Materialize the bounded p32/top5 candidate after protocol machinery is fixed.

The existing residual-correlation/refined p32 partition is capacity-exact for
all p32 top-k values.  Reusing that immutable partition lets top5 be measured
without reopening the holdout or silently changing neuron assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dense2moe.provenance import current_git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/20260815-184644-windows-real-d2m-v4-streaming")
    parser.add_argument("--source-name", default="high-sparsity-p32-top6.json")
    parser.add_argument("--output-name", default="high-sparsity-p32-top5.json")
    args = parser.parse_args()
    run = Path(args.run_dir)
    source_path = run / "partitions" / args.source_name
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if int(payload.get("routed_experts", 0)) != 32 or int(payload.get("expert_intermediate_size", 0)) != 512:
        raise ValueError("source partition is not the p32/512 candidate")
    payload = dict(payload)
    payload.update(
        {
            "status": "PARTITION_READY_P32_TOP5_PROTOCOL_FIXED",
            "profile_name": "p32_top5",
            "top_k": 5,
            "strategy": "residual_swap_refined_partition_reused_for_top5",
            "selection_contract": "TRAIN/dev bounded residual-correlation partition; true validation training; full holdout closed",
            "materialized_from": str(source_path),
            "code_commit": current_git_commit(),
        }
    )
    output = run / "partitions" / args.output_name
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "partition": str(output), "top_k": payload["top_k"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
