"""Pre-registered, machine-readable v3 gate policy.

The policy file (``configs/flashmini/v3_gate_policy.json``) is hashed into the
execution freeze manifest and must not change after freeze. This module loads
and validates it, and maps gate observations to the pre-registered verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import sha256_file

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "flashmini" / "v3_gate_policy.json"

_REQUIRED_GATE_NAMES = ("preflight_2p1m", "gate_100m", "final_250m")
_REQUIRED_VERDICTS = (
    "implementation_failure",
    "training_instability",
    "catastrophic_architecture_failure",
    "quality_win",
    "quality_parity",
    "ple_pass",
    "ple_unproven",
    "ple_fail",
    "needs_seed_confirmation",
)


def load_gate_policy(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the pre-registered gate policy.

    Fails closed if the file is missing, malformed, or is missing any required
    gate, verdict, or threshold.
    """
    path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    policy = json.loads(path.read_text(encoding="utf-8"))
    _validate_policy(policy)
    return policy


def _validate_policy(policy: dict[str, Any]) -> None:
    gates = policy.get("gates")
    if not isinstance(gates, list):
        raise TypeError("gate policy must contain a 'gates' list")
    names = {g.get("name") for g in gates if isinstance(g, dict)}
    missing = set(_REQUIRED_GATE_NAMES) - names
    if missing:
        raise ValueError(f"gate policy missing required gates: {sorted(missing)}")
    verdicts = policy.get("verdicts")
    if not isinstance(verdicts, dict):
        raise TypeError("gate policy must contain a 'verdicts' mapping")
    missing_verdicts = set(_REQUIRED_VERDICTS) - set(verdicts)
    if missing_verdicts:
        raise ValueError(f"gate policy missing required verdicts: {sorted(missing_verdicts)}")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict):
        raise TypeError("gate policy must contain a 'thresholds' mapping")
    for key in ("max_nan", "max_nll_implementation_failure",
                "loss_spike_ratio", "loss_spike_window_updates"):
        if key not in thresholds:
            raise ValueError(f"gate policy thresholds missing '{key}'")


def gate_policy_sha256(path: Path | None = None) -> str:
    """Return the SHA-256 of the gate policy file."""
    path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    return sha256_file(path)


def verdict_for(check: str, policy: dict[str, Any]) -> str:
    """Map a gate check name to its pre-registered verdict.

    Fails closed if the check is not in the policy.
    """
    verdicts = policy.get("verdicts", {})
    if check not in verdicts:
        raise KeyError(f"gate check '{check}' has no pre-registered verdict")
    return verdicts[check]
