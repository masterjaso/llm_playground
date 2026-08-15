"""Reference sparse MoE forward pass for local equivalence checks."""

from __future__ import annotations

from typing import Any

from .router import topk_router


class ReferenceMoE:
    """A framework-neutral routed/shared FFN reference.

    Weights use the conventional `[experts, input, intermediate]` and
    `[experts, intermediate, output]` layouts.  The class is intentionally
    small: it is a correctness oracle for converted layers, not a production
    dispatcher.
    """

    def __init__(self, w1: Any, w2: Any, router: Any, *, top_k: int = 2, shared_w1: Any | None = None, shared_w2: Any | None = None):
        self.w1, self.w2, self.router = w1, w2, router
        self.top_k = top_k
        self.shared_w1, self.shared_w2 = shared_w1, shared_w2

    def _expert_outputs(self, inputs: Any) -> Any:
        module = getattr(inputs.__class__, "__module__", "")
        if module.startswith("torch"):
            import torch  # type: ignore

            hidden = torch.einsum("ti,eih->teh", inputs, self.w1).clamp_min(0)
            return torch.einsum("teh,eho->teo", hidden, self.w2)
        import numpy as np  # type: ignore

        hidden = np.maximum(0, np.einsum("ti,eih->teh", inputs, self.w1))
        return np.einsum("teh,eho->teo", hidden, self.w2)

    def __call__(self, inputs: Any, *, all_experts: bool = False) -> Any:
        module = getattr(inputs.__class__, "__module__", "")
        outputs = self._expert_outputs(inputs)
        if module.startswith("torch"):
            import torch  # type: ignore

            if all_experts:
                # Partitioned dense neurons have one home expert, so the
                # all-expert oracle sums disjoint expert contributions.
                routed = outputs.sum(dim=1)
            else:
                indices, weights = topk_router(inputs @ self.router, self.top_k)
                routed = torch.zeros((inputs.shape[0], outputs.shape[-1]), dtype=outputs.dtype, device=outputs.device)
                for token in range(inputs.shape[0]):
                    for slot in range(self.top_k):
                        routed[token] += weights[token, slot] * outputs[token, indices[token, slot]]
            if self.shared_w1 is not None and self.shared_w2 is not None:
                shared = (inputs @ self.shared_w1).clamp_min(0) @ self.shared_w2
                routed = routed + shared
            return routed
        import numpy as np  # type: ignore

        if all_experts:
            routed = outputs.sum(axis=1)
        else:
            indices, weights = topk_router(inputs @ self.router, self.top_k)
            routed = np.zeros((inputs.shape[0], outputs.shape[-1]), dtype=outputs.dtype)
            for token in range(inputs.shape[0]):
                for slot in range(self.top_k):
                    routed[token] += weights[token, slot] * outputs[token, indices[token, slot]]
        if self.shared_w1 is not None and self.shared_w2 is not None:
            routed = routed + np.maximum(0, inputs @ self.shared_w1) @ self.shared_w2
        return routed
