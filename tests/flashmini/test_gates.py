"""Tests for the gate engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flashmini.gates import GateDecision, GateLogger, classify_relative


class GateTests(unittest.TestCase):
    def test_gate_logger_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gates.jsonl"
            with GateLogger(path) as gl:
                gl.record(
                    GateDecision(
                        gate_id="G1",
                        metric_values={"nll": 3.0},
                        thresholds={"nll": 0.02},
                        verdict="PASS",
                        checkpoint="step_100",
                        action="continue",
                    )
                )
            content = path.read_text()
            self.assertIn("G1", content)
            self.assertIn("PASS", content)

    def test_classify_relative(self):
        # metric improves (lower) by >= threshold -> PASS
        self.assertEqual(classify_relative(3.0, 3.1, 0.02), "PASS")
        # not enough improvement -> FAIL
        self.assertEqual(classify_relative(3.09, 3.1, 0.02), "FAIL")
        # baseline zero -> AMBIGUOUS
        self.assertEqual(classify_relative(1.0, 0.0, 0.02), "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()