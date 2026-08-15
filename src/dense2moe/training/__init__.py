"""Authoritative layer-worker helpers."""

from .distill import train_real_layer
from .worker import OOMBackoff, train_tiny_layer

__all__ = ["OOMBackoff", "train_real_layer", "train_tiny_layer"]
