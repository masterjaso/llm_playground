"""Build a conservative cohort analysis from the frozen candidate inventory.

Legacy V2.3/V2.4 metrics are retained as diagnostic observations only.  They
are never relabeled as V2 receipts; a row receives a V2 generalization class
only when a validated V2 receipt is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from dense2moe.evaluation.analysis import analyze_cohort
from dense2moe.evaluation.receipts import classify_receipt_compatibility


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = Path.cwd() / path
    return candidate if candidate.exists() else path


def _metric_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, Any] = {}
    aliases = {
        "cosine": "cosine_similarity",
        "normalized_mse": "normalized_mse",
        "load_cv": "learned_load_cv",
        "dead_experts": "dead_expert_count",
        "router_entropy": "routing_entropy",
        "token_count": "scored_token_count",
    }
    for key, value in raw.items():
        target = aliases.get(str(key), str(key))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[target] = float(value)
        elif target in {"scored_token_count", "dead_expert_count"} and isinstance(value, int):
            result[target] = int(value)
    result.setdefault("independent_group_count", 1)
    result.setdefault("dropped_token_count", 0)
    result.setdefault("invalid_token_count", 0)
    result.setdefault("non_finite_token_count", 0)
    return result


def _find_payload(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    fallback: tuple[dict[str, Any], str] | None = None
    for location in record.get("receipt_locations", []):
        path = _resolve(str(location))
        if path is None:
            continue
        payload = _read_json(path)
        if payload is not None:
            if fallback is None:
                fallback = (payload, str(path))
            metrics = payload.get("metrics")
            if isinstance(metrics, Mapping) and ("final_fit" in metrics or "final_dev" in metrics):
                return payload, str(path)
    return fallback if fallback is not None else (None, None)


def _row(record: Mapping[str, Any]) -> dict[str, Any]:
    payload, payload_path = _find_payload(record)
    metrics = payload.get("metrics", {}) if isinstance(payload, Mapping) else {}
    fit_raw = metrics.get("final_fit") if isinstance(metrics, Mapping) else None
    dev_raw = metrics.get("final_dev") if isinstance(metrics, Mapping) else None
    fit = _metric_payload(fit_raw)
    dev = _metric_payload(dev_raw)
    v2_receipt = False
    if isinstance(payload, Mapping):
        compatibility = classify_receipt_compatibility(payload)
        v2_receipt = compatibility.get("classification") == "V2_COMPLETE"
    generalization: dict[str, Any] = {
        "classification": "V2_COMPLETE" if v2_receipt else "LEGACY_METRICS_ONLY",
        "absolute_dev_governs": bool(v2_receipt),
        "fit_dev_averaged": False,
    }
    if fit and dev and "cosine_similarity" in fit and "cosine_similarity" in dev:
        generalization["cosine_gap"] = fit["cosine_similarity"] - dev["cosine_similarity"]
    if fit and dev and "normalized_mse" in fit and "normalized_mse" in dev:
        generalization["absolute_nmse_increase"] = dev["normalized_mse"] - fit["normalized_mse"]
        generalization["nmse_ratio"] = dev["normalized_mse"] / max(fit["normalized_mse"], 1e-8)
    return {
        "candidate_id": record.get("candidate_id"),
        "design_id": record.get("design_id"),
        "topology": record.get("topology"),
        "routing_mode": record.get("routing_mode"),
        "seed": record.get("seed"),
        "stage": record.get("stage"),
        "checkpoint": record.get("checkpoint"),
        "checkpoint_hash": record.get("checkpoint_hash"),
        "fit_train": fit,
        "fit_dev": dev,
        "generalization": generalization,
        "lm_metrics": {},
        "lm_evaluation_eligibility": "ELIGIBLE_ONLY_AFTER_VALID_V2_STRUCTURAL_RECEIPT" if not v2_receipt else "PENDING",
        "metric_policy_version": record.get("existing_metric_policy_version"),
        "policy_hash": None,
        "rerun_status": record.get("rerun_status"),
        "rerun_eligibility": record.get("rerun_eligibility"),
        "legacy_metric_source": payload_path,
        "v2_receipt_valid": v2_receipt,
        "source_receipt_lineage": record.get("source_receipt_lineage", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = _read_json(args.inventory)
    if inventory is None:
        raise SystemExit(f"invalid inventory: {args.inventory}")
    rows = [_row(record) for record in inventory.get("records", [])]
    analysis = analyze_cohort(rows)
    fit_dev_rows = [row for row in rows if row["fit_train"] and row["fit_dev"]]
    legacy_gaps = [row["generalization"] for row in fit_dev_rows if "cosine_gap" in row["generalization"]]
    payload = {
        "schema_version": 2,
        "receipt_type": "dense2moe-v2-cohort-analysis-v2",
        "source_inventory": str(args.inventory),
        "candidate_count": len(rows),
        "rows_with_fit_dev_metrics": len(fit_dev_rows),
        "valid_v2_receipt_count": sum(int(row["v2_receipt_valid"]) for row in rows),
        "legacy_metrics_are_diagnostic_only": True,
        "protected_tiers_opened": False,
        "cohort_complete": bool(inventory.get("cohort_complete", False)),
        "rerun_statuses": inventory.get("terminal_statuses", {}),
        "legacy_gap_summary": {
            "max_cosine_gap": max((item.get("cosine_gap", float("-inf")) for item in legacy_gaps), default=None),
            "max_absolute_nmse_increase": max((item.get("absolute_nmse_increase", float("-inf")) for item in legacy_gaps), default=None),
            "rows": legacy_gaps,
        },
        "analysis": analysis,
        "rows": rows,
        "causation_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = _read_json(args.output)
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite existing analysis: {args.output}")
    else:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("candidate_count", "rows_with_fit_dev_metrics", "valid_v2_receipt_count", "cohort_complete", "rerun_statuses")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
