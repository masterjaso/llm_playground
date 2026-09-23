"""Crash-safe metrics reconciliation for resumable v3 training.

The append-only ``metrics.jsonl`` can retain rows that describe work performed
after the newest durable checkpoint if the process crashes between logging and
checkpointing. On resume this module reconciles the metrics file so that the
retained history is exactly consistent with the durable checkpoint, preserving
orphaned rows for forensic use and recording an explicit resume-lineage event.

The reconciliation is idempotent: running it again on an already-reconciled
file produces the same retained history and does not duplicate the lineage
event.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a JSONL file, separating complete records from a partial final line.

    Returns ``(records, partial)`` where ``partial`` is a list containing at
    most one record that could not be parsed as complete JSON (a truncated
    final line). A trailing newline means the final line is complete.
    """
    records: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    if not path.is_file():
        return records, partial
    text = path.read_text()
    lines = text.split("\n")
    # A trailing newline produces a final empty string; drop it.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            partial.append({"_partial": True, "_raw": line})
    return records, partial


def _is_consistent(record: dict[str, Any], ckpt_step: int, ckpt_tokens: int) -> bool:
    """A record is consistent with the durable checkpoint if its counters do
    not exceed the checkpoint's counters."""
    step = record.get("step")
    tokens = record.get("tokens_seen")
    if step is None or tokens is None:
        # Validation/summary records without a step are retained only if they
        # carry a tokens_seen that is within the checkpoint.
        if tokens is None:
            return False
        return tokens <= ckpt_tokens
    return step <= ckpt_step and tokens <= ckpt_tokens


def reconcile_metrics(
    metrics_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    resumed_step: int,
    resumed_tokens: int,
    source_sha256: str,
    freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Reconcile ``metrics.jsonl`` against the durable checkpoint.

    Retains only records consistent with the checkpoint, moves orphaned and
    partial records to a timestamped forensic file, atomically rewrites the
    metrics file, and appends exactly one resume-lineage event. Idempotent.
    """
    metrics_path = Path(metrics_path)
    records, partial = _parse_jsonl(metrics_path)

    retained: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []
    for record in records:
        if _is_consistent(record, resumed_step, resumed_tokens):
            retained.append(record)
        else:
            orphaned.append(record)
    orphaned.extend(partial)

    # Idempotency: drop any pre-existing resume-lineage events so a repeated
    # reconciliation does not accumulate duplicate lineage records.
    retained = [r for r in retained if r.get("event") != "resume_lineage"]

    lineage = {
        "event": "resume_lineage",
        "ts": time.time(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "resumed_step": int(resumed_step),
        "resumed_tokens": int(resumed_tokens),
        "source_sha256": source_sha256,
        "freeze_sha256": freeze_sha256,
    }
    retained.append(lineage)

    # Preserve orphaned records for forensic use in a timestamped side file.
    if orphaned:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        forensic_path = metrics_path.with_name(f"metrics_orphaned_{stamp}.jsonl")
        with open(forensic_path, "w") as f:
            for record in orphaned:
                f.write(json.dumps(record) + "\n")

    # Atomically replace the metrics file.
    tmp = metrics_path.with_name(metrics_path.name + ".tmp")
    with open(tmp, "w") as f:
        for record in retained:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, metrics_path)
    return {
        "retained": len(retained),
        "orphaned": len(orphaned),
        "lineage_appended": True,
        "checkpoint_sha256": checkpoint_sha256,
    }
