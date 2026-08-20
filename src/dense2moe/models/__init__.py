"""Reference routing and tiny models used by structural tests."""

from .full_text import TinyQwen35TextConfig, TinyQwen35TextMoE, run_full_model_spike
from .moe import ReferenceMoE
from .qwen35_full import Qwen35DenseToMoE, apply_layer_checkpoints, replace_qwen35_ffns
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
    "Qwen35DenseToMoE",
    "Qwen35SwiGLUMoE",
    "ReferenceMoE",
    "SharedOutputFeatureRouter",
    "TinyDenseFFN",
    "TinyMoE",
    "TinyQwen35TextConfig",
    "TinyQwen35TextMoE",
    "TorchQwen35SwiGLUMoE",
    "apply_layer_checkpoints",
    "normalize_topk_weights",
    "replace_qwen35_ffns",
    "run_full_model_spike",
    "shared_gate_initialization",
    "topk_router",
]
