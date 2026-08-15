"""Reference routing and tiny models used by structural tests."""

from .router import normalize_topk_weights, shared_gate_initialization, topk_router
from .tiny import TinyDenseFFN, TinyMoE

__all__ = ["TinyDenseFFN", "TinyMoE", "normalize_topk_weights", "shared_gate_initialization", "topk_router"]

