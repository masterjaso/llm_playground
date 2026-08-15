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
from .oracle import oracle_topk, sparse_baseline, swiglu_contributions

__all__ = [
    "ActivationPartitioner",
    "PartitionPlan",
    "oracle_topk",
    "pack_experts",
    "partition_ffn_weights",
    "partition_indices",
    "reconstruct_ffn_weights",
    "sparse_baseline",
    "swiglu_contributions",
    "unpack_experts",
]
