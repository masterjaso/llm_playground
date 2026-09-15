"""Regression coverage for crash-safe metrics/resume reconciliation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from flashmini.metrics_reconcile import reconcile_metrics


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)


class MetricsReconcileTests(unittest.TestCase):
    def _ckpt(self, tmp: Path) -> Path:
        ckpt = tmp / "step_10.pt"
        ckpt.write_bytes(b"fake-checkpoint-bytes")
        return ckpt

    def test_retains_only_rows_consistent_with_checkpoint(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metrics = tmp / "metrics.jsonl"
            ckpt = self._ckpt(tmp)
            # Rows up to step 10 are consistent; rows 11-12 are orphaned
            # (logged after the durable checkpoint, before a crash).
            _write_jsonl(metrics, [
                {"event": "train", "step": 8, "tokens_seen": 8 * 4096},
                {"event": "train", "step": 10, "tokens_seen": 10 * 4096},
                {"event": "train", "step": 11, "tokens_seen": 11 * 4096},
                {"event": "train", "step": 12, "tokens_seen": 12 * 4096},
            ])
            result = reconcile_metrics(
                metrics, checkpoint_path=ckpt, checkpoint_sha256="abc",
                resumed_step=10, resumed_tokens=10 * 4096, source_sha256="src",
            )
            rows = [json.loads(l) for l in metrics.read_text().splitlines()]
            train_rows = [r for r in rows if r.get("event") == "train"]
            self.assertEqual([r["step"] for r in train_rows], [8, 10])
            self.assertEqual(result["orphaned"], 2)
            lineage = [r for r in rows if r.get("event") == "resume_lineage"]
            self.assertEqual(len(lineage), 1)
            self.assertEqual(lineage[0]["resumed_step"], 10)
            self.assertEqual(lineage[0]["checkpoint_sha256"], "abc")

    def test_removes_partially_written_final_line(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metrics = tmp / "metrics.jsonl"
            ckpt = self._ckpt(tmp)
            # A complete row followed by a truncated final JSON line.
            with open(metrics, "w") as f:
                f.write(json.dumps({"event": "train", "step": 5, "tokens_seen": 5 * 4096}) + "\n")
                f.write('{"event": "train", "step": 6, "tokens_seen": 6 * 4096, "loss": ')
            result = reconcile_metrics(
                metrics, checkpoint_path=ckpt, checkpoint_sha256="abc",
                resumed_step=5, resumed_tokens=5 * 4096, source_sha256="src",
            )
            rows = [json.loads(l) for l in metrics.read_text().splitlines()]
            train_rows = [r for r in rows if r.get("event") == "train"]
            self.assertEqual([r["step"] for r in train_rows], [5])
            self.assertEqual(result["orphaned"], 1)

    def test_repeated_resume_is_idempotent(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metrics = tmp / "metrics.jsonl"
            ckpt = self._ckpt(tmp)
            _write_jsonl(metrics, [
                {"event": "train", "step": 10, "tokens_seen": 10 * 4096},
            ])
            reconcile_metrics(metrics, checkpoint_path=ckpt, checkpoint_sha256="abc",
                             resumed_step=10, resumed_tokens=10 * 4096, source_sha256="src")
            # Simulate continued training that logs a new row, then a second
            # resume from the same checkpoint.
            with open(metrics, "a") as f:
                f.write(json.dumps({"event": "train", "step": 11, "tokens_seen": 11 * 4096}) + "\n")
            reconcile_metrics(metrics, checkpoint_path=ckpt, checkpoint_sha256="abc",
                             resumed_step=10, resumed_tokens=10 * 4096, source_sha256="src")
            rows = [json.loads(l) for l in metrics.read_text().splitlines()]
            lineage = [r for r in rows if r.get("event") == "resume_lineage"]
            self.assertEqual(len(lineage), 1)
            train_rows = [r for r in rows if r.get("event") == "train"]
            self.assertEqual([r["step"] for r in train_rows], [10])

    def test_no_duplicate_effective_trajectory_after_resume(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metrics = tmp / "metrics.jsonl"
            ckpt = self._ckpt(tmp)
            _write_jsonl(metrics, [
                {"event": "train", "step": 1, "tokens_seen": 4096},
                {"event": "train", "step": 2, "tokens_seen": 8192},
                {"event": "train", "step": 3, "tokens_seen": 12288},
            ])
            reconcile_metrics(metrics, checkpoint_path=ckpt, checkpoint_sha256="abc",
                             resumed_step=2, resumed_tokens=8192, source_sha256="src")
            rows = [json.loads(l) for l in metrics.read_text().splitlines()]
            train_rows = [r for r in rows if r.get("event") == "train"]
            # The retained trajectory is strictly increasing and contains no
            # step beyond the checkpoint, so a naive downstream analysis cannot
            # count abandoned work twice.
            steps = [r["step"] for r in train_rows]
            self.assertEqual(steps, [1, 2])
            self.assertEqual(steps, sorted(steps))


if __name__ == "__main__":
    unittest.main()
