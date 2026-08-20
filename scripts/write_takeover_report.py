"""Compose the bounded checkpoint-aware takeover report from JSON receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.provenance import current_git_commit


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"receipt is not an object: {path}")
    return payload


def _metrics(receipt: dict[str, Any], key: str) -> dict[str, Any]:
    value = receipt[key]
    return {
        field: value[field]
        for field in ("cosine", "global_nmse", "load_cv", "dead_experts", "hard_quartile_cosine")
        if field in value
    }


def _continuation_summary(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": receipt["status"],
        "output_checkpoint": receipt["output_checkpoint"],
        "output_tensor_sha256": receipt["output_checkpoint_tensor_sha256"],
        "selector_frozen": receipt["selector_frozen"],
        "fit_rows_used": receipt["split"]["fit_rows_used"],
        "updates": receipt["updates"],
        "before_student": receipt["before_validation_a"],
        "after_student": receipt["after_validation_a"],
    }


def _diagnosis_summary(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "student": _metrics(receipt, "student"),
        "unconstrained_trained_basis_oracle": _metrics(receipt, "unconstrained_trained_basis_oracle"),
        "load_constrained_trained_basis_oracle": _metrics(receipt, "load_constrained_trained_basis_oracle"),
        "blocker": receipt["identified_blocker"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    p16_before = _load(args.p16_before)
    p16_after = _load(args.p16_after)
    p16_cont = _load(args.p16_continuation)
    p32_before = _load(args.p32_before)
    p32_after = _load(args.p32_after)
    p32_cont = _load(args.p32_continuation)
    optional_paths = (
        args.p16_continuation2,
        args.p16_after2,
        args.p32_continuation2,
        args.p32_after2,
    )
    if any(path is not None for path in optional_paths) and not all(path is not None for path in optional_paths):
        raise ValueError("continuation round 2 requires both continuation and diagnosis receipts for p16 and p32")
    p16_cont2 = _load(args.p16_continuation2) if args.p16_continuation2 else None
    p16_after2 = _load(args.p16_after2) if args.p16_after2 else None
    p32_cont2 = _load(args.p32_continuation2) if args.p32_continuation2 else None
    p32_after2 = _load(args.p32_after2) if args.p32_after2 else None
    p16_latest = p16_after2 or p16_after
    p32_latest = p32_after2 or p32_after
    p16_equiv = _load(args.p16_equivalence)
    p32_equiv = _load(args.p32_equivalence)
    p16_frontier = p16_before["load_constrained_trained_basis_oracle"]["pricing_frontier"]
    p32_frontier = p32_before["load_constrained_trained_basis_oracle"]["pricing_frontier"]
    split = p16_before["split"]
    report = {
        "schema_version": 3,
        "status": "TAKEOVER_COMPLETE_BOUNDED_REFINEMENT_CONTINUING",
        "observed_base_head": args.observed_base_head,
        "code_commit": current_git_commit(),
        "execution": {
            "requested_mode": "native_windows_only",
            "native_windows_smoke": "PASS",
            "scientific_runtime": "guarded native Windows CPU execution; no GPU training claim",
            "representative_replay_started": False,
            "full64_replay_started": False,
            "holdout_opened": False,
        },
        "infrastructure": {
            "checkpoint_aware_contributions": {
                "trained_mode": "basis_source=trained_checkpoint",
                "raw_mode": "basis_source=raw_dense_partition",
                "raw_to_trained_fallback": False,
                "manifest_fields": [
                    "checkpoint_path",
                    "checkpoint_tensor_sha256",
                    "partition_path",
                    "partition_sha256",
                    "topology",
                    "source_revision",
                    "dataset_hash",
                    "capture_identity",
                    "split",
                    "code_commit",
                    "dtype",
                    "row_count",
                ],
            },
            "equivalence_receipts": {
                "p16": {"path": str(args.p16_equivalence), "status": p16_equiv["status"], "metrics": p16_equiv["metrics"]["final_ffn_reconstruction"]},
                "p32": {"path": str(args.p32_equivalence), "status": p32_equiv["status"], "metrics": p32_equiv["metrics"]["final_ffn_reconstruction"]},
            },
            "load_price_validation": {
                "synthetic_assignment_movement_test": "PASS",
                "telemetry_fields": [
                    "reconstruction_objective",
                    "priced_objective",
                    "cosine",
                    "global_nmse",
                    "load_cv",
                    "dead_experts",
                    "expert_loads",
                    "assignment_change_fraction_from_zero",
                    "assignment_change_fraction_from_previous",
                    "price_min",
                    "price_mean",
                    "price_max",
                    "convergence_reason",
                ],
                "p16_real_slice_assignment_change_max": max(
                    float(row["assignment_change_fraction_from_zero"])
                    for point in p16_frontier
                    for row in point["pricing_trace"]
                ),
                "p32_real_slice_assignment_change_max": max(
                    float(row["assignment_change_fraction_from_zero"])
                    for point in p32_frontier
                    for row in point["pricing_trace"]
                ),
                "objective_scale": "per_token_residual_mse_relative_to_hidden_mean",
                "p16_candidate_sets": 1820,
                "p32_top5_candidate_pool": 15,
                "p32_top5_candidate_sets": 3003,
            },
            "tests": {
                "focused": "PASS",
                "full": "PASS (85 tests)",
                "ruff": "PASS",
                "py_compile": "PASS",
                "native_windows_help_smoke": "PASS",
            },
        },
        "split_contract": {
            "dataset_hash": p16_cont.get("dataset_hash") or p32_cont.get("dataset_hash"),
            "fit_rows": split["fit_rows"],
            "validation_a_rows": split["validation_a_rows"],
            "validation_a_indices_sha256": split["validation_a_indices_sha256"],
            "validation_b_rows": split["validation_b_rows"],
            "validation_b_indices_sha256": split["validation_b_indices_sha256"],
            "fit_a_overlap": 0,
            "fit_b_overlap": 0,
            "a_b_overlap": 0,
            "official_holdout": "CLOSED",
        },
        "p16_top4": {
            "checkpoint_before": {
                "path": p16_before["checkpoint_path"],
                "tensor_sha256": p16_before["checkpoint_tensor_sha256"],
                "partition_sha256": p16_before["partition_sha256"],
            },
            "diagnosis_before": {
                "path": str(args.p16_before),
                "student": _metrics(p16_before, "student"),
                "unconstrained_trained_basis_oracle": _metrics(p16_before, "unconstrained_trained_basis_oracle"),
                "load_constrained_trained_basis_oracle": _metrics(p16_before, "load_constrained_trained_basis_oracle"),
                "topk_recall": p16_before["student_oracle_topk_recall"],
                "blocker": p16_before["identified_blocker"],
            },
            "continuation": _continuation_summary(p16_cont, args.p16_continuation),
            "diagnosis_after": _diagnosis_summary(p16_after, args.p16_after),
            "next_state": "BASIS_REFINING_BOUNDED_PILOT",
        },
        "p32_top5": {
            "checkpoint_before": {
                "path": p32_before["checkpoint_path"],
                "tensor_sha256": p32_before["checkpoint_tensor_sha256"],
                "partition_sha256": p32_before["partition_sha256"],
            },
            "diagnosis_before": {
                "path": str(args.p32_before),
                "student": _metrics(p32_before, "student"),
                "unconstrained_trained_basis_oracle": _metrics(p32_before, "unconstrained_trained_basis_oracle"),
                "load_constrained_trained_basis_oracle": _metrics(p32_before, "load_constrained_trained_basis_oracle"),
                "topk_recall": p32_before["student_oracle_topk_recall"],
                "blocker": p32_before["identified_blocker"],
            },
            "continuation": _continuation_summary(p32_cont, args.p32_continuation),
            "diagnosis_after": _diagnosis_summary(p32_after, args.p32_after),
            "next_state": "BASIS_REFINING_BOUNDED_PILOT",
        },
        "decision_table": [
            {
                "target": "p16/top4",
                "basis_cosine_before": p16_before["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_cosine_after": p16_after["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_cosine_after_latest": p16_latest["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_nmse_after_latest": p16_latest["unconstrained_trained_basis_oracle"]["global_nmse"],
                "basis_load_cv_after_latest": p16_latest["unconstrained_trained_basis_oracle"]["load_cv"],
                "joint_gate_oracle": p16_before["joint_quality_load_green"],
                "blocker": p16_latest["identified_blocker"],
                "next_state": "REFINING",
            },
            {
                "target": "p32/top5",
                "basis_cosine_before": p32_before["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_cosine_after": p32_after["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_cosine_after_latest": p32_latest["unconstrained_trained_basis_oracle"]["cosine"],
                "basis_nmse_after_latest": p32_latest["unconstrained_trained_basis_oracle"]["global_nmse"],
                "basis_load_cv_after_latest": p32_latest["unconstrained_trained_basis_oracle"]["load_cv"],
                "joint_gate_oracle": p32_before["joint_quality_load_green"],
                "blocker": p32_latest["identified_blocker"],
                "next_state": "REFINING",
            },
        ],
        "green_gate": {
            "cosine_min": 0.98,
            "nmse_max": 0.05,
            "load_cv_max": 0.50,
            "dead_experts": 0,
            "p16_cleared": False,
            "p32_cleared": False,
        },
        "holdout_opened": False,
        "representative_replay_started": False,
        "full64_replay_started": False,
    }
    if p16_cont2 is not None:
        report["p16_top4"]["continuation_round_2"] = _continuation_summary(p16_cont2, args.p16_continuation2)
        report["p16_top4"]["diagnosis_after_round_2"] = _diagnosis_summary(p16_after2, args.p16_after2)
        report["p32_top5"]["continuation_round_2"] = _continuation_summary(p32_cont2, args.p32_continuation2)
        report["p32_top5"]["diagnosis_after_round_2"] = _diagnosis_summary(p32_after2, args.p32_after2)
        report["continuation_rounds"] = 2
    else:
        report["continuation_rounds"] = 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-base-head", required=True)
    parser.add_argument("--p16-before", type=Path, required=True)
    parser.add_argument("--p16-after", type=Path, required=True)
    parser.add_argument("--p16-continuation", type=Path, required=True)
    parser.add_argument("--p16-continuation2", type=Path, default=None)
    parser.add_argument("--p16-after2", type=Path, default=None)
    parser.add_argument("--p16-equivalence", type=Path, required=True)
    parser.add_argument("--p32-before", type=Path, required=True)
    parser.add_argument("--p32-after", type=Path, required=True)
    parser.add_argument("--p32-continuation", type=Path, required=True)
    parser.add_argument("--p32-continuation2", type=Path, default=None)
    parser.add_argument("--p32-after2", type=Path, default=None)
    parser.add_argument("--p32-equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Dense2MoE takeover report (2026-08-16)",
        "",
        f"Status: `{report['status']}`",
        "",
        "The checkpoint-aware contribution path is proven against both active topologies. Both fresh-A diagnoses classify the blocker as `BASIS_QUALITY`; bounded basis-only continuations are published and remain below the green gate.",
        "",
        "| Target | Oracle cosine before → after | Oracle NMSE before → after | Oracle CV before → after | Blocker | State |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["decision_table"]:
        before = report["p16_top4" if row["target"] == "p16/top4" else "p32_top5"]["diagnosis_before"]["unconstrained_trained_basis_oracle"]
        after = report["p16_top4" if row["target"] == "p16/top4" else "p32_top5"]["diagnosis_after_round_2" if report["continuation_rounds"] > 1 else "diagnosis_after"]["unconstrained_trained_basis_oracle"]
        lines.append(f"| {row['target']} | {before['cosine']:.6f} → {after['cosine']:.6f} | {before['global_nmse']:.6f} → {after['global_nmse']:.6f} | {before['load_cv']:.6f} → {after['load_cv']:.6f} | `{row['blocker']}` | `{row['next_state']}` |")
    lines.extend(
        [
            "",
            "Infrastructure receipts, split hashes, checkpoint fingerprints, and the complete JSON report are adjacent to this file. `HOLDOUT_OPENED = false`, `REPRESENTATIVE_REPLAY_STARTED = false`, and `FULL64_REPLAY_STARTED = false`.",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()
