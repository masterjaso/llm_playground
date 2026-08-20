"""Compatibility shim for :mod:`dense2moe.training.worker_impl`."""

from dense2moe.training.worker_impl import OOMBackoff, train_tiny_layer

__all__ = ["OOMBackoff", "train_tiny_layer"]

