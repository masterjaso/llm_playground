"""Durable dense-to-MoE experiment primitives."""

from .config import MoEProfile, load_config
from .provenance import current_git_commit

__all__ = ["MoEProfile", "current_git_commit", "load_config"]
__version__ = "0.1.0"
