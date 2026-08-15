"""Exact FFN neuron partitioning and deterministic activation assignment."""

from .ffn import (
    ActivationPartitioner,
    PartitionPlan,
    pack_experts,
    partition_ffn_weights,
    partition_indices,
    reconstruct_ffn_weights,
    unpack_experts,
)

__all__ = [
    "ActivationPartitioner",
    "PartitionPlan",
    "pack_experts",
    "partition_ffn_weights",
    "partition_indices",
    "reconstruct_ffn_weights",
    "unpack_experts",
]
