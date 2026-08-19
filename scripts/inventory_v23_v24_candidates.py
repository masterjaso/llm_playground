"""Freeze a bounded inventory of all discoverable V2.3/V2.4 candidate receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dense2moe.evaluation.inventory import discover_candidate_inventory, inventory_payload, write_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[".nsp/artifacts/runs", "runs"])
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()
    records = discover_candidate_inventory(args.root)
    payload = inventory_payload(records)
    if args.output:
        payload = write_inventory(records, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
