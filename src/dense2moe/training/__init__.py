"""Compatibility namespace for layer-worker helpers."""

from training.worker import OOMBackoff, train_tiny_layer

__all__ = ["OOMBackoff", "train_tiny_layer"]

