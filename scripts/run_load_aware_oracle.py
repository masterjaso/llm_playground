"""Run bounded load-aware oracle diagnostics from contribution arrays.

The expensive teacher/model stage is intentionally kept separate.  A caller
materializes FIT/validation-only ``.npz`` files containing ``shared``,
``routed`` and ``target`` arrays, then this script evaluates p16/top4 and/or
p32/top4/top5 with the same deterministic Lagrangian implementation.  This
separation makes it impossible for the diagnostic to silently open the full
holdout while still producing a concise quality/load Pareto report.

Example::

    python scripts/run_load_aware_oracle.py \
      --input p16_top4.npz --topology p16/top4 \
      --input p32_top5.npz --topology p32/top5 \
      --output reports/load-aware-oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow checkout-local execution before ``pip install -e .`` (the same path
# bootstrap used by the guarded-command entry point).
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.partition import frozen_slice_load_aware_oracle
from dense2moe.provenance import current_git_commit


def _load_arrays(path: Path) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("numpy is required for load-aware oracle diagnostics") from exc
    with np.load(path) as values:
        required = {"shared", "routed", "target"}
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"{path} is missing contribution arrays: {missing}")
        return values["shared"], values["routed"], values["target"]


def run(
    inputs: list[Path],
    topologies: list[str],
    *,
    target_load_cv: float,
    candidate_pool_size: int | None,
    max_combinations: int,
    iterations: int,
    batch_size: int,
    max_in_memory_bytes: int,
    storage_dir: Path | None,
) -> dict[str, Any]:
    if len(inputs) != len(topologies):
        raise ValueError("each --input requires a matching --topology")
    rows: list[dict[str, Any]] = []
    for row_index, (path, topology) in enumerate(zip(inputs, topologies)):
        if "/" not in topology:
            raise ValueError(f"topology must look like p16/top4: {topology!r}")
        profile, raw_top_k = topology.split("/", 1)
        top_k = int(raw_top_k.removeprefix("top"))
        shared, routed, target = _load_arrays(path)
        expected_experts = {"p16": 16, "p32": 32}.get(profile)
        if expected_experts is None:
            raise ValueError(f"unsupported topology profile: {profile!r}")
        if top_k not in ({4} if profile == "p16" else {4, 5}):
            raise ValueError(f"unsupported product topology: {topology!r}")
        if int(routed.shape[1]) != expected_experts:
            raise ValueError(f"{topology} expects {expected_experts} routed experts, received {routed.shape[1]}")
        input_storage = (
            storage_dir / f"{row_index:02d}-{profile}-top{top_k}"
            if storage_dir is not None
            else None
        )
        result = frozen_slice_load_aware_oracle(
            shared,
            routed,
            target,
            top_k=top_k,
            target_load_cv=target_load_cv,
            candidate_pool_size=candidate_pool_size,
            max_combinations=max_combinations,
            iterations=iterations,
            batch_size=batch_size,
            max_in_memory_bytes=max_in_memory_bytes,
            storage_dir=input_storage,
            materialize_outputs=False,
        )
        rows.append(
            {
                "topology": topology,
                "profile": profile,
                "top_k": top_k,
                "active_intermediate_width": (1024 + top_k * (1024 if profile == "p16" else 512)),
                "ffn_reduction": 1.0 - (1024 + top_k * (1024 if profile == "p16" else 512)) / 17408.0,
                "input": str(path),
                "tokens": int(result["indices"].shape[0]),
                "exact_or_bounded": result["assurance"],
                "candidate_pool_size": result["candidate_pool_size"],
                "effective_candidate_pool_size": result["effective_candidate_pool_size"],
                "combinations_considered_per_token": result["combinations_considered_per_token"],
                "max_combinations": result["max_combinations"],
                "assurance": result["assurance"],
                "candidate_error_storage": result["candidate_error_storage"],
                "candidate_id_storage": result["candidate_id_storage"],
                "candidate_storage_ephemeral": result["candidate_storage_ephemeral"],
                "coefficient_solver": result["coefficient_solver"],
                "candidate_fit_exact": result["candidate_fit_exact"],
                "candidate_chunk_size": result["candidate_chunk_size"],
                "exact_oracle_global_nmse": result["unconstrained"]["global_nmse"],
                "exact_oracle_cosine": result["unconstrained"]["cosine"],
                "oracle_hard_quartile_cosine": result["unconstrained"]["hard_quartile_cosine"],
                "oracle_load_cv": result["unconstrained"]["load_cv"],
                "global_nmse": result["global_nmse"],
                "cosine": result["cosine"],
                "hard_quartile_cosine": result["hard_quartile_cosine"],
                "load_constrained_oracle_cosine": result["cosine"],
                "load_constrained_oracle_cv": result["load_cv"],
                "load_cv": result["load_cv"],
                "expert_usage_counts": result["expert_usage_counts"],
                "dead_experts": result["dead_experts"],
                "feasible_load_target": result["feasible_load_target"],
                "green_gate": result["green_gate"],
                "hard_feasible": result["hard_feasible"],
                "selected_penalty": result["selected_penalty"],
                "pareto": result["pareto"],
            }
        )
    return {
        "schema_version": 1,
        "status": "LOAD_AWARE_ORACLE_COMPLETE",
        "classification": "FIT_OR_VALIDATION_ARRAYS_ONLY_HOLDOUT_NOT_OPENED",
        "target_load_cv": float(target_load_cv),
        "candidate_pool_size": candidate_pool_size,
        "max_combinations": int(max_combinations),
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "max_in_memory_bytes": int(max_in_memory_bytes),
        "storage_dir": str(storage_dir) if storage_dir is not None else None,
        "results": rows,
        "code_commit": current_git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--topology", action="append", required=True, help="p16/top4, p32/top5, or p32/top4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-load-cv", type=float, default=0.50)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-in-memory-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--storage-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = run(
        args.input,
        args.topology,
        target_load_cv=args.target_load_cv,
        candidate_pool_size=args.candidate_pool_size,
        max_combinations=args.max_combinations,
        iterations=args.iterations,
        batch_size=args.batch_size,
        max_in_memory_bytes=args.max_in_memory_bytes,
        storage_dir=args.storage_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": payload["results"], "output": str(args.output)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
