#!/usr/bin/env python3
"""Run the tiny synthetic oracle-routed basis smoke test.

This command deliberately exercises only a random, tiny SwiGLU fixture.  Its
receipt is implementation evidence and is never eligible for Phase 01 or
production promotion, even when ``--samples 32768`` is used.  Real method
proofs use ``run_real_oracle_routed_basis_refinement.py`` instead.
"""

from __future__ import annotations

import argparse
import json

from run_oracle_routed_basis_refinement import run_synthetic_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), default="p16/top4")
    parser.add_argument("--samples", "--rows", dest="samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--assignment-refresh-steps", type=int, default=1)
    parser.add_argument("--m-step-repeats", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_synthetic_smoke(
        topology=args.topology,
        seed=args.seed,
        rows=args.samples,
        epochs=args.epochs,
        assignment_refresh_steps=args.assignment_refresh_steps,
        m_step_repeats=args.m_step_repeats,
        candidate_pool_size=args.candidate_pool_size,
        max_combinations=args.max_combinations,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
