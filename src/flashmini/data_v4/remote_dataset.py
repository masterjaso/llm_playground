"""Alias kept for the proposed module name; implementation lives in sampler."""

from .sampler import RemoteShardDataset, SamplerState, sequence_permutation, shard_permutation

__all__ = ["RemoteShardDataset", "SamplerState", "sequence_permutation", "shard_permutation"]

