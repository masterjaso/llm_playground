#!/usr/bin/env python3
"""Capture identical native Qwen layer-0 inputs/targets for V2.3 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import QWEN_SOURCE_REVISION, stream_teacher_split
from dense2moe.data import sha256_file, write_immutable_json
from dense2moe.hardware import load_runtime_lock
from dense2moe.provenance import current_git_commit
from dense2moe.state import atomic_write_json

METHOD_VERSION = "moe-v23-m01"
CAPTURE_SCHEMA_VERSION = 1
CAPTURE_RECEIPT_TYPE = "dense2moe-v23-real-qwen-layer0-capture"
CAPTURE_STATUS = "V23_TEACHER_CAPTURE_READY"
SPLITS = ("FIT-TRAIN", "FIT-DEV")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _validate_plan(plan_path: Path, source_revision: str) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if plan.get("status") != "V23_CAPTURE_PLAN_READY":
        raise ValueError(f"V2.3 capture plan is not ready: {plan.get('status')!r}")
    if str(plan.get("method_version")) != METHOD_VERSION:
        raise ValueError("capture plan method version does not match moe-v23-m01")
    if str(plan.get("source_revision")) != source_revision or str(plan.get("tokenizer_revision")) != source_revision:
        raise ValueError("capture plan source/tokenizer revision mismatch")
    if not all(isinstance(plan.get(split), list) and plan[split] for split in SPLITS):
        raise ValueError("capture plan must contain non-empty FIT-TRAIN and FIT-DEV rows")
    source = plan.get("source")
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise ValueError("capture plan source is not content-addressed")
    source_path = Path(str(source["path"]))
    if not source_path.is_file() or sha256_file(source_path) != str(source["sha256"]):
        raise ValueError("capture plan source hash does not resolve")
    return plan


def _manifest_summary(path: Path, *, expected_dataset_hash: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("status") != "CAPTURE_COMPLETE":
        raise ValueError(f"capture manifest is not complete: {path}")
    if payload.get("split") not in SPLITS or int(payload.get("layer", -1)) != 0:
        raise ValueError(f"capture manifest split/layer mismatch: {path}")
    if str(payload.get("dataset_hash")) != expected_dataset_hash:
        raise ValueError(f"capture dataset hash mismatch: {path}")
    if payload.get("input_tensor") != "ffn_input" or payload.get("target_tensor") != "dense_ffn_target":
        raise ValueError(f"capture manifest is not a paired FFN input/target capture: {path}")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"capture manifest has no shards: {path}")
    checked: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict) or not shard.get("path") or not shard.get("sha256"):
            raise ValueError(f"capture shard metadata is incomplete: {path}")
        shard_path = path.parent / str(shard["path"])
        if not shard_path.is_file() or sha256_file(shard_path) != str(shard["sha256"]):
            # stream manifests store paths relative to capture/, while the
            # manifest itself lives in run/capture; retry the run/capture root.
            shard_path = path.parent / "streaming" / str(shard["path"])
        if not shard_path.is_file() or sha256_file(shard_path) != str(shard["sha256"]):
            raise ValueError(f"capture shard hash mismatch: {shard.get('path')}")
        checked.append(
            {
                "shard_id": int(shard.get("shard_id", -1)),
                "path": str(shard["path"]),
                "sha256": str(shard["sha256"]),
                "count": int(shard.get("count", 0)),
                "shape": list(shard.get("shape", [])),
                "dtype": str(shard.get("dtype", "")),
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "split": str(payload["split"]),
        "layer": 0,
        "count": int(payload["count"]),
        "dataset_hash": str(payload["dataset_hash"]),
        "tokenizer_hash": str(payload.get("tokenizer_hash", "")),
        "shards": checked,
        "capture_kind": str(payload.get("capture_kind", "")),
        "dtype": str(payload.get("dtype", "")),
    }


def capture_v23_teacher_layer0(
    *,
    data_plan: str | Path,
    source_snapshot: str | Path,
    run_dir: str | Path,
    runtime_lock: str | Path,
    source_revision: str = QWEN_SOURCE_REVISION,
    device: str = "cuda:1",
    compute_dtype: str = "bfloat16",
    shard_tokens: int = 2_048,
    attention_implementation: str = "sdpa",
    resume: bool = False,
) -> dict[str, Any]:
    """Replay both fixed V2.3 splits through the native streaming teacher."""

    run = Path(run_dir)
    plan_path = Path(data_plan).resolve()
    source_path = Path(source_snapshot).resolve()
    lock_path = Path(runtime_lock).resolve()
    try:
        if os.name != "nt" or platform.system() != "Windows":
            raise RuntimeError("NATIVE_WINDOWS_REQUIRED")
        if source_revision != QWEN_SOURCE_REVISION:
            raise RuntimeError("SOURCE_REVISION_MISMATCH")
        lock = load_runtime_lock(lock_path)
        if lock.get("status") != "LOCKED" or not lock.get("ok"):
            raise RuntimeError("RUNTIME_LOCK_INVALID")
        if not source_path.is_dir() or not (source_path / "model.safetensors.index.json").is_file():
            raise RuntimeError("SOURCE_SNAPSHOT_MISSING")
        plan = _validate_plan(plan_path, source_revision)
        stream_reports: dict[str, Any] = {}
        for split in SPLITS:
            stream_reports[split] = stream_teacher_split(
                source_path,
                plan_path,
                run,
                split=split,
                layers=(0,),
                device=device,
                compute_dtype=compute_dtype,
                shard_tokens=shard_tokens,
                attention_implementation=attention_implementation,
            )
        manifests = {
            split: _manifest_summary(
                run / "capture" / f"layer-0000-{split}.json",
                expected_dataset_hash=str(plan["dataset_hash"]),
            )
            for split in SPLITS
        }
        command = (
            f"{sys.executable} scripts/capture_v23_teacher_layer0.py "
            f"--data-plan {plan_path} --source-snapshot {source_path} --run-dir {run} "
            f"--runtime-lock {lock_path} --source-revision {source_revision} "
            f"--device {device} --compute-dtype {compute_dtype} --shard-tokens {shard_tokens} "
            f"--attention-implementation {attention_implementation} --resume"
        )
        payload: dict[str, Any] = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "receipt_type": CAPTURE_RECEIPT_TYPE,
            "status": CAPTURE_STATUS,
            "evidence_class": "v23-real-qwen-layer0-capture",
            "method_version": METHOD_VERSION,
            "source": {
                "snapshot": str(source_path),
                "revision": source_revision,
                "index_sha256": sha256_file(source_path / "model.safetensors.index.json"),
                "config_sha256": sha256_file(source_path / "config.json"),
            },
            "runtime": {
                "lock_path": str(lock_path),
                "lock_sha256": sha256_file(lock_path),
                "status": lock["status"],
                "device": device,
                "compute_dtype": compute_dtype,
                "attention_implementation": attention_implementation,
            },
            "data_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
                "dataset_hash": str(plan["dataset_hash"]),
                "selected_tokens": plan.get("selected_tokens", {}),
                "sequence_length": int(plan.get("sequence_length", 0)),
            },
            "layers": [0],
            "shard_tokens": int(shard_tokens),
            "splits": manifests,
            "streams": stream_reports,
            "evaluation_tiers_opened": [],
            "promotion_data_used": False,
            "retired_v22_data_used": False,
            "exact_command": command,
            "code_commit": current_git_commit(),
        }
        payload["receipt_sha256"] = _canonical_hash(payload)
        receipt_path = run / "capture" / "v23-teacher-layer0-receipt.json"
        write_immutable_json(receipt_path, payload)
        return {"status": CAPTURE_STATUS, "receipt": str(receipt_path), "receipt_sha256": payload["receipt_sha256"], "splits": manifests}
    except Exception as exc:  # noqa: BLE001 - publish a fail-closed blocked receipt for native runtime errors
        blocked = {
            "status": "BLOCKED",
            "blocker_code": str(exc) if str(exc) else "V23_CAPTURE_FAILED",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "data_plan": str(plan_path),
            "source_snapshot": str(source_path),
            "runtime_lock": str(lock_path),
            "code_commit": current_git_commit(),
        }
        atomic_write_json(run / "capture" / "v23-teacher-layer0-blocked.json", blocked)
        return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--source-revision", default=QWEN_SOURCE_REVISION)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--shard-tokens", type=int, default=2_048)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = capture_v23_teacher_layer0(
        data_plan=args.data_plan,
        source_snapshot=args.source_snapshot,
        run_dir=args.run_dir,
        runtime_lock=args.runtime_lock,
        source_revision=args.source_revision,
        device=args.device,
        compute_dtype=args.compute_dtype,
        shard_tokens=args.shard_tokens,
        attention_implementation=args.attention_implementation,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
