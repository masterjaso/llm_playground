#!/usr/bin/env python3
"""Capture real Qwen layer-0 FFN inputs/targets for development or promotion.

The command reuses :func:`dense2moe.capture.stream_teacher_split`, which
loads the pinned Qwen embedding and one decoder layer at a time.  It refuses
to run outside native Windows or without an approved runtime lock and never
falls back to a synthetic tensor fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import (
    QWEN_SOURCE_REVISION,
    RealCaptureBlocked,
    build_real_capture_receipt,
    capture_frozen_evaluation_tier,
    stream_teacher_split,
    validate_method_proof_receipt,
    write_real_capture_receipt,
)
from dense2moe.hardware import load_runtime_lock
from dense2moe.provenance import current_git_commit
from dense2moe.state import atomic_write_json


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "content", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        answer = row.get("answer")
        return f"{question}\n{answer}" if isinstance(answer, str) and answer else question
    messages = row.get("messages", row.get("conversations"))
    if isinstance(messages, list):
        chunks: list[str] = []
        for message in messages:
            if isinstance(message, Mapping):
                role = message.get("role", message.get("from", ""))
                content = message.get("content", message.get("value", ""))
                if content:
                    chunks.append(f"{role}: {content}" if role else str(content))
            elif message:
                chunks.append(str(message))
        if chunks:
            return "\n".join(chunks)
    return ""


def _write_method_proof_plan(receipt: dict[str, Any], output: Path, *, source_revision: str, sequence_length: int) -> Path:
    rows = list(receipt["_rows"])
    source_manifest = Path(str(receipt["method_proof_policy"]["source_manifest"]))
    if not source_manifest.is_absolute():
        for candidate in (Path(receipt["_path"]).parent / source_manifest, Path.cwd() / source_manifest):
            if candidate.is_file():
                source_manifest = candidate.resolve()
                break
    if not source_manifest.is_file():
        raise RealCaptureBlocked("METHOD_PROOF_SOURCE_MANIFEST_MISSING", details={"path": str(source_manifest)})
    plan_rows: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["split"] = "FIT-TRAIN"
        # ``resolve_corpus_records`` reopens and hashes the exact text before
        # tokenization.  METHOD_PROOF_ONLY receipts retain the text in their
        # manifest, so make that immutable identity explicit in the capture
        # plan instead of relying on an unverified row count.
        text = _row_text(value)
        if not text.strip():
            raise RealCaptureBlocked("METHOD_PROOF_ROW_TEXT_MISSING", details={"id": value.get("id")})
        value["text"] = text
        value["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        value["text_sha256"] = value["content_sha256"]
        plan_rows.append(value)
    plan = {
        "status": "CALIBRATION_READY",
        "schema_version": 1,
        "source": {"path": str(source_manifest)},
        "resolvability": {"base_dir": str(source_manifest.parent)},
        "source_revision": source_revision,
        "tokenizer_revision": source_revision,
        "sequence_length": sequence_length,
        "dataset_hash": _canonical_hash(receipt["_selected_record_ids"]),
        "train_tokens": int(receipt["_selected_tokens"]),
        "FIT-TRAIN": plan_rows,
        "method_proof_only": True,
        "method_proof_receipt": str(receipt["_path"]),
        "method_proof_receipt_sha256": hashlib.sha256(Path(receipt["_path"]).read_bytes()).hexdigest(),
        "evaluation_splits_excluded": list(receipt["method_proof_policy"].get("excluded_splits", [])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, plan)
    return output


def capture_real_qwen_layer0(
    method_proof_receipt: str | Path,
    source_snapshot: str | Path,
    run_dir: str | Path,
    *,
    runtime_lock: str | Path,
    source_revision: str = QWEN_SOURCE_REVISION,
    device: str = "cuda:1",
    compute_dtype: str = "bfloat16",
    shard_tokens: int = 2048,
    sequence_length: int = 2048,
    attention_implementation: str = "sdpa",
    resume: bool = False,
) -> dict[str, Any]:
    """Run the bounded native layer-major capture and publish its receipt."""

    if os.name != "nt" or platform.system() != "Windows":
        return {
            "status": "BLOCKED",
            "blocker_code": "NATIVE_WINDOWS_REQUIRED",
            "message": "real Qwen scientific capture is authorized only on native Windows",
            "native_windows": False,
            "code_commit": current_git_commit(),
        }
    lock = load_runtime_lock(runtime_lock)
    if lock.get("status") != "LOCKED" or not lock.get("ok"):
        return {
            "status": "BLOCKED",
            "blocker_code": "RUNTIME_LOCK_INVALID",
            "message": "an approved current runtime lock is required before capture",
            "runtime_lock": lock,
            "code_commit": current_git_commit(),
        }
    try:
        method = validate_method_proof_receipt(method_proof_receipt)
        if source_revision != QWEN_SOURCE_REVISION:
            raise RealCaptureBlocked("SOURCE_REVISION_MISMATCH")
        run = Path(run_dir)
        plan_path = _write_method_proof_plan(method, run / "capture" / "method-proof-data-plan.json", source_revision=source_revision, sequence_length=sequence_length)
        stream_result = stream_teacher_split(
            source_snapshot,
            plan_path,
            run,
            split="FIT-TRAIN",
            layers=(0,),
            device=device,
            compute_dtype=compute_dtype,
            shard_tokens=shard_tokens,
            attention_implementation=attention_implementation,
        )
        stream_manifest = run / "capture" / "layer-0000-FIT-TRAIN.json"
        receipt = build_real_capture_receipt(
            stream_manifest,
            method_proof_receipt=method_proof_receipt,
            runtime_lock=runtime_lock,
            source_snapshot=source_snapshot,
            source_revision=source_revision,
        )
        receipt_path = run / "capture" / "real-qwen-layer0-receipt.json"
        write_real_capture_receipt(receipt, receipt_path)
        return {
            "status": receipt["status"],
            "receipt": str(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "stream": stream_result,
            "method_proof_receipt": str(method_proof_receipt),
            "code_commit": current_git_commit(),
        }
    except (OSError, TypeError, ValueError, RuntimeError, KeyError, RealCaptureBlocked) as exc:
        result = {
            "status": "BLOCKED",
            "blocker_code": getattr(exc, "reason", "REAL_CAPTURE_FAILED"),
            "message": str(exc),
            "native_windows": True,
            "code_commit": current_git_commit(),
        }
        atomic_write_json(Path(run_dir) / "capture" / "real-qwen-layer0-blocked.json", result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-proof-receipt", type=Path)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--promotion-mode", action="store_true", help="capture one explicitly named frozen evaluation tier")
    parser.add_argument("--corpus-root", type=Path, help="frozen Corpus V2.2 development-internal-final root for promotion capture")
    parser.add_argument("--tier", help="explicit frozen evaluation tier for promotion capture")
    parser.add_argument("--promotion-output-root", type=Path, help="promotion output directory containing only activation shards and layer-0000.json")
    parser.add_argument("--staging-root", type=Path, help="resumable capture staging directory outside the published output")
    parser.add_argument("--contamination-ledger", type=Path, help="sealed promotion contamination ledger to bind without modifying")
    parser.add_argument("--source-revision", default=QWEN_SOURCE_REVISION)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--shard-tokens", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.promotion_mode:
        if args.method_proof_receipt is not None:
            parser.error("--method-proof-receipt is not used with --promotion-mode")
        if args.corpus_root is None or args.tier is None or args.promotion_output_root is None:
            parser.error("--promotion-mode requires --corpus-root, --tier, and --promotion-output-root")
        try:
            result = capture_frozen_evaluation_tier(
                args.corpus_root,
                args.source_snapshot,
                args.promotion_output_root,
                args.runtime_lock,
                tier=args.tier,
                source_revision=args.source_revision,
                device=args.device,
                compute_dtype=args.compute_dtype,
                shard_tokens=args.shard_tokens,
                sequence_length=args.sequence_length,
                attention_implementation=args.attention_implementation,
                staging_root=args.staging_root or args.run_dir / "capture-work" / args.tier,
                contamination_ledger=args.contamination_ledger,
            )
        except (OSError, TypeError, ValueError, RuntimeError, KeyError, RealCaptureBlocked) as exc:
            result = {
                "status": "BLOCKED",
                "blocker_code": getattr(exc, "reason", "PROMOTION_CAPTURE_FAILED"),
                "message": str(exc),
                "code_commit": current_git_commit(),
            }
    else:
        if args.method_proof_receipt is None:
            parser.error("--method-proof-receipt is required unless --promotion-mode is set")
        result = capture_real_qwen_layer0(
            args.method_proof_receipt,
            args.source_snapshot,
            args.run_dir,
            runtime_lock=args.runtime_lock,
            source_revision=args.source_revision,
            device=args.device,
            compute_dtype=args.compute_dtype,
            shard_tokens=args.shard_tokens,
            sequence_length=args.sequence_length,
            attention_implementation=args.attention_implementation,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
