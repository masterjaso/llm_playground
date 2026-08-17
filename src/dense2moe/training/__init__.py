"""Authoritative layer-worker helpers."""

from .distill import train_real_layer
from .torch_distill import (
    ActivationShardDataset,
    deterministic_shadow_validation_indices,
    load_fixed_activation_splits,
    train_torch_layer,
    validate_split_contract,
)
from .oracle_refinement import (
    OracleAssignments,
    oracle_assignments,
    oracle_routed_forward,
    train_oracle_routed_basis,
)
from .worker import OOMBackoff, train_tiny_layer

__all__ = [
    "ActivationShardDataset",
    "OOMBackoff",
    "deterministic_shadow_validation_indices",
    "load_fixed_activation_splits",
    "train_real_layer",
    "train_tiny_layer",
    "train_torch_layer",
    "validate_split_contract",
    "OracleAssignments",
    "oracle_assignments",
    "oracle_routed_forward",
    "train_oracle_routed_basis",
]
