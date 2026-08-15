"""Reference routing and tiny models used by structural tests."""

from .moe import ReferenceMoE
from .qwen_moe import DenseSwiGLU, Qwen35SwiGLUMoE
from .router import normalize_topk_weights, shared_gate_initialization, topk_router
from .tiny import TinyDenseFFN, TinyMoE

__all__ = [
    "DenseSwiGLU",
    "Qwen35SwiGLUMoE",
    "ReferenceMoE",
    "TinyDenseFFN",
    "TinyMoE",
    "normalize_topk_weights",
    "shared_gate_initialization",
    "topk_router",
]
