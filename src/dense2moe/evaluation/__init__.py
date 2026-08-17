"""Quality metrics and terminal-state classification."""

from .metrics import (
    PROMOTION_THRESHOLDS,
    amplitude_metrics,
    classify_metric,
    evaluate_promotion_metrics,
    quality_gate,
)
from .pilot import run_oracle_ablation, run_real_layer_pilot

__all__ = [
    "PROMOTION_THRESHOLDS",
    "amplitude_metrics",
    "classify_metric",
    "evaluate_promotion_metrics",
    "quality_gate",
    "run_oracle_ablation",
    "run_real_layer_pilot",
]
