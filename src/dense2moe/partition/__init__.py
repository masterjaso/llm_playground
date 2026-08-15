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
from .oracle import (
    frozen_slice_positive_oracle,
    frozen_slice_scaled_router_oracle,
    frozen_slice_simplex_oracle,
    oracle_topk,
    sparse_baseline,
    swiglu_contributions,
    trainable_student_proxy,
)

__all__ = [
    "ActivationPartitioner",
    "PartitionPlan",
    "frozen_slice_positive_oracle",
    "frozen_slice_scaled_router_oracle",
    "frozen_slice_simplex_oracle",
    "oracle_topk",
    "pack_experts",
    "partition_ffn_weights",
    "partition_indices",
    "reconstruct_ffn_weights",
    "sparse_baseline",
    "swiglu_contributions",
    "trainable_student_proxy",
    "unpack_experts",
]
