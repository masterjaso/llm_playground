#!/usr/bin/env python3
"""Run the real Qwen layer-0 p16/top4 method proof.

Unlike the legacy synthetic smoke, this entrypoint requires a frozen
METHOD_PROOF_ONLY receipt, a real Qwen capture receipt, and a token budget.
Every provenance/runtime gate runs before the source basis or optimizer is
constructed.  ``--max-tokens`` means captured tokens/samples; there is no
``--rows`` alias because random tensor rows are not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import QWEN_SOURCE_REVISION
from dense2moe.provenance import current_git_commit
from dense2moe.training.real_method_proof import (
    _load_source_ffn_model,
    run_real_method_proof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-proof-receipt", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--topology", choices=("p16/top4",), required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=False)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--source-revision", default=QWEN_SOURCE_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--assignment-refresh-steps", type=int, default=1)
    parser.add_argument("--m-step-repeats", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--result-receipt", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.source_revision != QWEN_SOURCE_REVISION:
        result = {
            "status": "BLOCKED",
            "phase_state": "PHASE_01_BLOCKED_INVALID_CAPTURE",
            "blocker": "SOURCE_REVISION_MISMATCH",
            "requested_revision": args.source_revision,
            "required_revision": QWEN_SOURCE_REVISION,
            "optimizer_steps": 0,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 2
    factory = None
    if args.source_snapshot is not None:
        factory = lambda: _load_source_ffn_model(args.source_snapshot, device=args.device)
    result = run_real_method_proof(
        args.method_proof_receipt,
        args.capture_receipt,
        topology=args.topology,
        max_tokens=args.max_tokens,
        runtime_lock_path=args.runtime_lock,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        assignment_refresh_steps=args.assignment_refresh_steps,
        m_step_repeats=args.m_step_repeats,
        batch_rows=args.batch_rows,
        candidate_pool_size=args.candidate_pool_size,
        max_combinations=args.max_combinations,
        model_factory=factory,
        result_receipt=args.result_receipt,
        checkpoint_dir=args.checkpoint_dir,
        require_native_windows=True,
    )
    result.setdefault("code_commit", current_git_commit())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0 if result.get("status") == "REAL_QWEN_P16_METHOD_PROOF_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
