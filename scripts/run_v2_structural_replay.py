"""Execute retained V2.3/V2.4 structural FIT/DEV replay on native Windows.

The status-only inventory remains unchanged.  This command consumes its
``PARTIAL_METRICS_ONLY`` records, writes resumable run-scoped result files, and
publishes one immutable V2 structural receipt per successful record.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dense2moe.evaluation.replay import (
    ReplayInputError,
    ReplayValidationError,
    replay_candidate_record,
    resolve_artifact_path,
)
from dense2moe.evaluation.receipts import write_immutable_receipt
from dense2moe.hardware import check_runtime_lock, collect_environment, load_runtime_lock
from dense2moe.provenance import current_git_commit
from dense2moe.evaluation.registry import POLICY_HASH, STRUCTURAL_POLICY_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="fresh inventory_v23_v24_candidates.py JSON")
    parser.add_argument("--output", required=True, help="run-scoped replay summary JSON")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--record-index", type=int, action="append", default=[])
    return parser.parse_args()


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "record"))
    return text[:80] or "record"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _runtime_preflight(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_runtime_lock(repo_root / "runs" / "windows-runtime-lock.json")
    if lock.get("status") != "LOCKED" or not lock.get("ok"):
        raise RuntimeError(f"WINDOWS_RUNTIME_LOCK_REQUIRED: {lock}")
    payload = lock.get("payload", {})
    current_commit = current_git_commit()
    locked_commit = str(payload.get("code_commit", ""))
    if locked_commit != current_commit:
        try:
            changed = {
                line.strip().replace("/", "\\")
                for line in subprocess.check_output(
                    ["git", "diff", "--name-only", f"{locked_commit}..{current_commit}"],
                    cwd=repo_root,
                    text=True,
                ).splitlines()
                if line.strip()
            }
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"WINDOWS_RUNTIME_LOCK_CODE_COMMIT_UNRESOLVED: lock={locked_commit} current={current_commit}") from exc
        allowed_runtime_receipt_changes = {"runs\\windows-runtime-lock.json", "runs\\windows-environment-receipt.json"}
        if not changed or not changed.issubset(allowed_runtime_receipt_changes):
            raise RuntimeError(
                "WINDOWS_RUNTIME_LOCK_CODE_COMMIT_MISMATCH: "
                f"lock={locked_commit} current={current_commit} changed={sorted(changed)}; refresh the lock after source changes"
            )
    environment = collect_environment(repo_root=repo_root)
    checked = check_runtime_lock(environment, repo_root / "runs" / "windows-runtime-lock.json")
    if checked.get("status") != "LOCKED" or not checked.get("ok"):
        raise RuntimeError(f"WINDOWS_RUNTIME_DRIFT: {checked.get('drift', [])}")
    runtime_identity = {
        "status": "LOCKED",
        "lock_path": str(repo_root / "runs" / "windows-runtime-lock.json"),
        "lock_sha256": payload.get("lock_sha256"),
        "code_commit": locked_commit,
        "platform": payload.get("platform"),
        "python_version": payload.get("python_version"),
        "torch_version": payload.get("torch_version"),
        "compiled_cuda": payload.get("compiled_cuda"),
        "selected_training_gpus": payload.get("selected_training_gpus"),
        "runtime_fingerprint": payload.get("runtime_fingerprint"),
    }
    return runtime_identity, {"environment_status": "LOCKED", "drift": checked.get("drift", [])}


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    inventory_path = resolve_artifact_path(args.inventory, repo_root=repo_root)
    output_path = resolve_artifact_path(args.output, repo_root=repo_root)
    output_root = output_path.parent
    records_root = output_root / "records"
    receipts_root = output_root / "receipts"
    try:
        runtime_identity, runtime_check = _runtime_preflight(repo_root)
    except Exception as exc:
        payload = {"status": "BLOCKED_RUNTIME", "error": str(exc), "inventory": str(inventory_path), "code_commit": current_git_commit()}
        _write_json(output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 2
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        records = [item for item in inventory.get("records", []) if item.get("rerun_status") == "PARTIAL_METRICS_ONLY"]
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        payload = {"status": "BLOCKED_INVENTORY", "error": str(exc), "inventory": str(inventory_path)}
        _write_json(output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 2
    selected_pairs = list(enumerate(records))
    if args.record_index:
        wanted = set(int(index) for index in args.record_index)
        selected_pairs = [(index, item) for index, item in selected_pairs if index in wanted]
    if args.max_records is not None:
        selected_pairs = selected_pairs[: max(0, int(args.max_records))]
    code_commit = str(runtime_identity.get("code_commit") or current_git_commit())
    code_identity = {
        "code_commit": code_commit,
        "metric_policy_version": STRUCTURAL_POLICY_VERSION,
        "policy_hash": POLICY_HASH,
        "runner": "scripts/run_v2_structural_replay.py",
    }
    summary_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    checkpoint_hash_cache: dict[str, str] = {}
    for index, record in selected_pairs:
        checkpoint_hash = str(record.get("checkpoint_hash", ""))
        stem = f"{index:03d}-{_safe_name(record.get('candidate_id'))}-{checkpoint_hash[:12]}"
        result_path = records_root / f"{stem}.json"
        if result_path.exists():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("code_commit") == code_commit:
                    summary_rows.append(previous)
                    counts[str(previous.get("status", "UNKNOWN"))] += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            result_path.unlink(missing_ok=True)
        try:
            result = replay_candidate_record(
                record,
                repo_root=repo_root,
                device=args.device,
                batch_tokens=args.batch_tokens,
                runtime_lock_identity=runtime_identity,
                code_science_identity=code_identity,
                checkpoint_hash_cache=checkpoint_hash_cache,
            )
            receipt_path = receipts_root / f"{stem}.json"
            receipt = write_immutable_receipt(result.pop("receipt"), receipt_path)
            result["receipt_path"] = str(receipt_path)
            result["receipt_id"] = receipt.get("receipt_id")
            result["receipt_sha256"] = receipt.get("receipt_sha256")
        except (ReplayInputError, ReplayValidationError) as exc:
            result = {
                "status": "BLOCKED_INPUT" if isinstance(exc, ReplayInputError) else "FAILED_VALIDATION",
                "candidate_id": record.get("candidate_id"),
                "design_id": record.get("design_id"),
                "seed": record.get("seed"),
                "checkpoint": record.get("checkpoint"),
                "checkpoint_sha256": checkpoint_hash,
                "error": str(exc),
                "receipt_emitted": False,
            }
        except Exception as exc:  # keep the 70-record sweep resumable and explicit
            result = {
                "status": "FAILED_EXECUTION",
                "candidate_id": record.get("candidate_id"),
                "design_id": record.get("design_id"),
                "seed": record.get("seed"),
                "checkpoint": record.get("checkpoint"),
                "checkpoint_sha256": checkpoint_hash,
                "error": f"{type(exc).__name__}: {exc}",
                "receipt_emitted": False,
            }
        result["code_commit"] = code_commit
        _write_json(result_path, result)
        summary_rows.append(result)
        counts[str(result.get("status", "UNKNOWN"))] += 1
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print(json.dumps({"index": index, "status": result.get("status"), "candidate_id": result.get("candidate_id"), "design_id": result.get("design_id"), "lm_evaluation_eligibility": result.get("lm_evaluation_eligibility")}, sort_keys=True), flush=True)
    valid_receipts = [row for row in summary_rows if row.get("receipt_id")]
    eligible = [row for row in summary_rows if row.get("lm_evaluation_eligibility") == "ELIGIBLE"]
    final_status = "REPLAY_COMPLETE" if len(summary_rows) == len(selected_pairs) and all(row.get("status") == "REPLAY_COMPLETE" for row in summary_rows) else "REPLAY_PARTIAL"
    summary = {
        "status": final_status,
        "inventory": str(inventory_path),
        "selected_record_count": len(selected_pairs),
        "eligible_inventory_count": len(records),
        "result_counts": dict(sorted(counts.items())),
        "valid_receipt_count": len(valid_receipts),
        "lm_eligible_count": len(eligible),
        "lm_eligible_records": [{"candidate_id": row.get("candidate_id"), "design_id": row.get("design_id"), "seed": row.get("seed"), "receipt_id": row.get("receipt_id")} for row in eligible],
        "runtime": runtime_identity,
        "runtime_check": runtime_check,
        "code_science_identity": code_identity,
        "results_root": str(records_root),
        "receipts_root": str(receipts_root),
    }
    _write_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if final_status == "REPLAY_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
