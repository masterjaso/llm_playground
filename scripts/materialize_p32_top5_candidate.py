"""Compatibility entry point for topology-specific p32/top5 materialization.

The old helper silently copied the top6 partition.  That path is retained only
as a command name for existing handoffs; it now requires a FIT/dev search
report and promotes a row evaluated specifically at top5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.materialize_p32_product_targets import materialize
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from materialize_p32_product_targets import materialize  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/20260815-184644-windows-real-d2m-v4-streaming")
    parser.add_argument("--source-name", default=None, help="deprecated; top6 reuse is no longer permitted")
    parser.add_argument("--output-name", default="high-sparsity-p32-top5-product.json")
    args = parser.parse_args()
    run = Path(args.run_dir)
    if args.source_name is not None:
        raise ValueError("top6 partition reuse is prohibited; rerun FIT/dev search and materialize its top5 row")
    output = materialize(run=run, top_k=5, output_name=args.output_name)
    print(json.dumps({"status": "MATERIALIZED", "partition": str(output), "top_k": 5}, indent=2), flush=True)


if __name__ == "__main__":
    main()
