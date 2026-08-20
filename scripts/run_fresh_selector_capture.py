#!/usr/bin/env python3
"""Run and receipt a layer-0-only fresh selector activation capture."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.capture import stream_teacher_split
from dense2moe.provenance import current_git_commit


def _heartbeat(stop: threading.Event, progress: Path) -> None:
    started = time.monotonic()
    while not stop.wait(30.0):
        age = None
        if progress.exists():
            age = round(time.time() - progress.stat().st_mtime, 1)
        print(
            json.dumps(
                {
                    "status": "HEARTBEAT",
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "progress_path": str(progress),
                    "progress_age_seconds": age,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--shard-tokens", type=int, default=2048)
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()
    payload = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "CALIBRATION_READY":
        raise ValueError("fresh selector capture requires a CALIBRATION_READY manifest")
    progress = args.run_dir / "capture" / "streaming" / "train" / "progress.json"
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat, args=(stop, progress), daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = stream_teacher_split(
            args.source_dir,
            args.dataset_manifest,
            args.run_dir,
            split="train",
            layers=(0,),
            device=args.device,
            compute_dtype=args.compute_dtype,
            shard_tokens=args.shard_tokens,
            attention_implementation=args.attention_implementation,
        )
    finally:
        stop.set()
        thread.join(timeout=2.0)
    if result.get("status") != "STREAMING_CAPTURE_COMPLETE":
        raise RuntimeError(f"fresh layer-0 capture did not complete: {result.get('status')}")
    if result.get("layers") != [0] or int(result.get("replayed_layers", -1)) != 1:
        raise RuntimeError("fresh selector capture replayed more than layer 0")
    expected = int(payload.get("train_tokens", 0))
    if int(result.get("tokens", -1)) != expected:
        raise RuntimeError(f"fresh capture token count mismatch: {result.get('tokens')} != {expected}")
    receipt = {
        "schema_version": 1,
        "status": "FRESH_LAYER0_CAPTURE_COMPLETE",
        "classification": "NEW_DIVERSE_TEXT_LAYER0_ONLY_SELECTOR_DATA",
        "run_dir": str(args.run_dir),
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_hash": payload.get("dataset_hash"),
        "source_revision": args.source_revision,
        "source_dir": str(args.source_dir),
        "layers": [0],
        "replayed_layers": 1,
        "tokens": int(result["tokens"]),
        "examples": int(result["examples"]),
        "streaming_result": result,
        "existing_holdout_opened": False,
        "historical_replay_touched": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "code_commit": current_git_commit(),
    }
    report_path = args.run_dir / "reports" / "fresh-selector-layer0-capture.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "report": str(report_path), "tokens": receipt["tokens"], "code_commit": receipt["code_commit"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
