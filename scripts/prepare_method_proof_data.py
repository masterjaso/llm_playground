#!/usr/bin/env python3
"""Prepare the clean, receipt-backed METHOD_PROOF_ONLY subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.method_proof import (
    METHOD_PROOF_DEFAULT_MIN_TOKENS,
    MethodProofBlocked,
    prepare_method_proof_data,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=METHOD_PROOF_DEFAULT_MIN_TOKENS)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        result = prepare_method_proof_data(args.corpus_manifest, args.output, min_tokens=args.min_tokens)
    except MethodProofBlocked as exc:
        print(json.dumps(exc.payload, ensure_ascii=False, sort_keys=True))
        return 2
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "METHOD_PROOF_DATA_BLOCKED", "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"{result['status']}: {result['manifest']['path']} ({result['selection']['selected_tokens']} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
