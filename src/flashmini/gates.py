"""Gate engine — records gate decisions with metric values, thresholds, and verdicts.

Gate decisions must record: metric values, threshold, PASS/FAIL/AMBIGUOUS,
timestamp, checkpoint, and resulting action.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GateDecision:
    gate_id: str
    metric_values: dict[str, float]
    thresholds: dict[str, float]
    verdict: str  # PASS | FAIL | AMBIGUOUS
    checkpoint: Optional[str]
    action: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "metric_values": self.metric_values,
            "thresholds": self.thresholds,
            "verdict": self.verdict,
            "checkpoint": self.checkpoint,
            "action": self.action,
            "timestamp": self.timestamp,
        }


class GateLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def record(self, decision: GateDecision) -> None:
        self._f.write(json.dumps(decision.to_dict()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "GateLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def classify_relative(metric: float, baseline: float, threshold: float) -> str:
    """Classify a relative improvement vs baseline against a threshold.

    Returns PASS if metric improves by >= threshold (relative), else FAIL.
    """
    if baseline == 0:
        return "AMBIGUOUS"
    rel = (baseline - metric) / abs(baseline)  # positive = improvement
    if rel >= threshold:
        return "PASS"
    return "FAIL"