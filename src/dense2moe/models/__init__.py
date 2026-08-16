"""Reference routing and tiny models used by structural tests."""

from .full_text import TinyQwen35TextConfig, TinyQwen35TextMoE, run_full_model_spike
from .moe import ReferenceMoE
from .qwen_moe import DenseSwiGLU, Qwen35SwiGLUMoE
from .router import normalize_topk_weights, shared_gate_initialization, topk_router
from .tiny import TinyDenseFFN, TinyMoE

try:
    from .torch_moe import SharedOutputFeatureRouter, TorchQwen35SwiGLUMoE
except RuntimeError:  # Optional PyTorch dependency is absent in minimal installs.
    SharedOutputFeatureRouter = None  # type: ignore[assignment,misc]
    TorchQwen35SwiGLUMoE = None  # type: ignore[assignment,misc]

__all__ = [
    "DenseSwiGLU",
    "Qwen35SwiGLUMoE",
    "ReferenceMoE",
    "SharedOutputFeatureRouter",
    "TinyDenseFFN",
    "TinyMoE",
    "TinyQwen35TextConfig",
    "TinyQwen35TextMoE",
    "TorchQwen35SwiGLUMoE",
    "normalize_topk_weights",
    "run_full_model_spike",
    "shared_gate_initialization",
    "topk_router",
]
