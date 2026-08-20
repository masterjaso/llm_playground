"""Authoritative layer-worker helpers."""

from .distill import train_real_layer
from .oracle_refinement import (
    OracleAssignments,
    oracle_assignments,
    oracle_routed_forward,
    train_oracle_routed_basis,
)
from .real_method_proof import (
    PHASE_01_BLOCKED_INVALID_CAPTURE,
    PHASE_01_BLOCKED_NO_REAL_CAPTURE,
    PHASE_01_REAL_METHOD_PROOF_FAILED,
    PHASE_01_REAL_METHOD_PROOF_GREEN,
    PHASE_01_REAL_METHOD_PROOF_RUNNING,
    preflight_real_method_proof,
    run_real_method_proof,
    validate_result_receipt,
)
from .torch_distill import (
    ActivationShardDataset,
    deterministic_shadow_validation_indices,
    load_fixed_activation_splits,
    train_torch_layer,
    validate_split_contract,
)
from .worker import OOMBackoff, train_tiny_layer

__all__ = [
    "PHASE_01_BLOCKED_INVALID_CAPTURE",
    "PHASE_01_BLOCKED_NO_REAL_CAPTURE",
    "PHASE_01_REAL_METHOD_PROOF_FAILED",
    "PHASE_01_REAL_METHOD_PROOF_GREEN",
    "PHASE_01_REAL_METHOD_PROOF_RUNNING",
    "ActivationShardDataset",
    "OOMBackoff",
    "OracleAssignments",
    "deterministic_shadow_validation_indices",
    "load_fixed_activation_splits",
    "oracle_assignments",
    "oracle_routed_forward",
    "preflight_real_method_proof",
    "run_real_method_proof",
    "train_oracle_routed_basis",
    "train_real_layer",
    "train_tiny_layer",
    "train_torch_layer",
    "validate_result_receipt",
    "validate_split_contract",
]
