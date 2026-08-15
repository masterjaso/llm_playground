"""Exact SwiGLU dense-to-MoE building block used by the target spike.

The NumPy implementation is deliberately dependency-light and is also useful
for oracle/partition experiments.  It follows Hugging Face linear weight
layouts (`[out_features, in_features]`) and exposes a strict safetensors
save/reload path.  A full Qwen 3.5 hybrid transformer wrapper can compose this
module without changing its semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..partition import PartitionPlan, partition_indices


def _np():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError as exc:
        raise RuntimeError("numpy is required for the local architecture spike") from exc


def _silu(value: Any) -> Any:
    np = _np()
    values = np.asarray(value)
    return values / (1.0 + np.exp(-values))


def _topk(logits: Any, top_k: int) -> tuple[Any, Any]:
    np = _np()
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if top_k <= 0 or top_k > values.shape[-1]:
        raise ValueError("top_k must be in [1, num_experts]")
    order = np.argsort(-values, axis=-1, kind="stable")[:, :top_k]
    selected = np.take_along_axis(values, order, axis=-1)
    selected -= selected.max(axis=-1, keepdims=True)
    weights = np.exp(selected)
    weights /= weights.sum(axis=-1, keepdims=True)
    return order, weights.astype(np.float32)


class DenseSwiGLU:
    """Reference dense SwiGLU FFN with HF-compatible weight orientation."""

    def __init__(self, gate_proj: Any, up_proj: Any, down_proj: Any):
        np = _np()
        self.gate_proj = np.asarray(gate_proj)
        self.up_proj = np.asarray(up_proj)
        self.down_proj = np.asarray(down_proj)
        if self.gate_proj.ndim != 2 or self.up_proj.shape != self.gate_proj.shape:
            raise ValueError("gate/up projection shape mismatch")
        if self.down_proj.shape != (self.gate_proj.shape[1], self.gate_proj.shape[0]):
            raise ValueError("down projection shape mismatch")

    @property
    def hidden_size(self) -> int:
        return int(self.gate_proj.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_proj.shape[0])

    def __call__(self, inputs: Any) -> Any:
        x = _np().asarray(inputs)
        hidden = _silu(x @ self.gate_proj.T) * (x @ self.up_proj.T)
        return hidden @ self.down_proj.T


class Qwen35SwiGLUMoE:
    """Capacity-preserving shared+routed SwiGLU MoE.

    The partition is exhaustive.  `all_experts=True` is therefore an exact
    dense reconstruction (within the input dtype); sparse mode is normalized
    top-k routing with no token dropping.
    """

    architecture = "qwen3_5_text_swiglu_moe_v1"

    def __init__(
        self,
        gate_proj: Any,
        up_proj: Any,
        down_proj: Any,
        *,
        routed_experts: int,
        shared_intermediate_size: int,
        top_k: int = 2,
        partition: PartitionPlan | None = None,
        router: Any | None = None,
    ):
        np = _np()
        dense = DenseSwiGLU(gate_proj, up_proj, down_proj)
        if partition is None:
            if (dense.intermediate_size - shared_intermediate_size) % routed_experts:
                raise ValueError("dense width is not divisible by routed expert capacity")
            partition = partition_indices(
                dense.intermediate_size,
                routed_experts,
                (dense.intermediate_size - shared_intermediate_size) // routed_experts,
                shared_intermediate_size,
            )
        partition.validate()
        if top_k > routed_experts:
            raise ValueError("top_k cannot exceed routed_experts")
        self.partition = partition
        self.top_k = int(top_k)
        self.routed_experts = int(routed_experts)
        self.hidden_size = dense.hidden_size
        self.intermediate_size = dense.intermediate_size
        self.shared_gate_proj = dense.gate_proj[list(partition.shared_indices)].copy()
        self.shared_up_proj = dense.up_proj[list(partition.shared_indices)].copy()
        self.shared_down_proj = dense.down_proj[:, list(partition.shared_indices)].copy()
        self.expert_gate_proj = np.stack([dense.gate_proj[list(group)] for group in partition.expert_indices])
        self.expert_up_proj = np.stack([dense.up_proj[list(group)] for group in partition.expert_indices])
        self.expert_down_proj = np.stack([dense.down_proj[:, list(group)] for group in partition.expert_indices])
        self.router = np.zeros((self.hidden_size, self.routed_experts), dtype=np.float32) if router is None else np.asarray(router).copy()
        if self.router.shape != (self.hidden_size, self.routed_experts):
            raise ValueError("router shape mismatch")

    @classmethod
    def from_dense(cls, dense: DenseSwiGLU, **kwargs: Any) -> Qwen35SwiGLUMoE:
        return cls(dense.gate_proj, dense.up_proj, dense.down_proj, **kwargs)

    def _shared(self, x: Any) -> Any:
        return (_silu(x @ self.shared_gate_proj.T) * (x @ self.shared_up_proj.T)) @ self.shared_down_proj.T

    def expert_outputs(self, inputs: Any) -> Any:
        np = _np()
        x = np.asarray(inputs)
        if x.ndim < 2:
            raise ValueError("inputs must have a hidden dimension")
        x = x.reshape(-1, x.shape[-1])
        gated = _silu(np.einsum("th,eih->tei", x, self.expert_gate_proj))
        up = np.einsum("th,eih->tei", x, self.expert_up_proj)
        hidden = gated * up
        return np.einsum("tei,eoi->teo", hidden, self.expert_down_proj)

    def __call__(self, inputs: Any, *, all_experts: bool = False, return_router: bool = False) -> Any:
        np = _np()
        x = np.asarray(inputs)
        if x.ndim < 2:
            raise ValueError("inputs must have a hidden dimension")
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        outputs = self.expert_outputs(x)
        if all_experts:
            routed = outputs.sum(axis=1)
            router_info: dict[str, Any] = {"indices": np.tile(np.arange(self.routed_experts), (x.shape[0], 1)), "weights": np.ones((x.shape[0], self.routed_experts), dtype=np.float32) / self.routed_experts}
        else:
            indices, weights = _topk(x @ self.router, self.top_k)
            routed = np.zeros((x.shape[0], self.hidden_size), dtype=outputs.dtype)
            for token in range(x.shape[0]):
                for slot in range(self.top_k):
                    routed[token] += weights[token, slot] * outputs[token, indices[token, slot]]
            router_info = {"indices": indices, "weights": weights}
        result = (routed + self._shared(x)).reshape(*original_shape[:-1], self.hidden_size)
        if return_router:
            router_info["indices"] = router_info["indices"].reshape(*original_shape[:-1], -1)
            router_info["weights"] = router_info["weights"].reshape(*original_shape[:-1], -1)
            return result, router_info
        return result

    def state_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "mlp.router.weight": self.router,
            "mlp.shared_expert.gate_proj.weight": self.shared_gate_proj,
            "mlp.shared_expert.up_proj.weight": self.shared_up_proj,
            "mlp.shared_expert.down_proj.weight": self.shared_down_proj,
        }
        for expert in range(self.routed_experts):
            prefix = f"mlp.experts.{expert}"
            values.update({
                f"{prefix}.gate_proj.weight": self.expert_gate_proj[expert],
                f"{prefix}.up_proj.weight": self.expert_up_proj[expert],
                f"{prefix}.down_proj.weight": self.expert_down_proj[expert],
            })
        return values

    def state_dict_inventory(self) -> dict[str, Any]:
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in self.state_dict().items()}

    def save_pretrained(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.numpy import save_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required to save the target spike") from exc
        np = _np()
        save_file({name: np.ascontiguousarray(value) for name, value in self.state_dict().items()}, str(destination / "model.safetensors"))
        config = {
            "architectures": [self.architecture],
            "model_type": "qwen3_5_text",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "routed_experts": self.routed_experts,
            "shared_intermediate_size": self.partition.shared_intermediate_size,
            "expert_intermediate_size": self.partition.expert_intermediate_size,
            "top_k": self.top_k,
            "partition": self.partition.as_dict(),
            "state_dict_inventory": self.state_dict_inventory(),
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_pretrained(cls, source: str | Path, *, strict: bool = True) -> Qwen35SwiGLUMoE:
        source = Path(source)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        partition_payload = config.get("partition", {})
        partition = PartitionPlan(
            int(partition_payload["dense_intermediate_size"]),
            int(partition_payload["routed_experts"]),
            int(partition_payload["expert_intermediate_size"]),
            int(partition_payload["shared_intermediate_size"]),
            tuple(int(v) for v in partition_payload["shared_indices"]),
            tuple(tuple(int(v) for v in group) for group in partition_payload["expert_indices"]),
        )
        try:
            from safetensors.numpy import load_file  # type: ignore
        except ImportError as exc:
            raise RuntimeError("safetensors is required to reload the target spike") from exc
        values = load_file(str(source / "model.safetensors"))
        expected = set(config.get("state_dict_inventory", {}))
        if strict and set(values) != expected:
            raise ValueError(f"strict state-dict mismatch: missing={sorted(expected - set(values))}, unexpected={sorted(set(values) - expected)}")
        router = values["mlp.router.weight"]
        # Reconstruct dense-shaped arrays from the partitioned tensors so the
        # constructor can apply the same validation and slicing logic.
        np = _np()
        # The constructor path is more robust when given partitioned tensors;
        # instantiate a shell and then restore exact arrays below.
        hidden = int(router.shape[0])
        dense_i = partition.dense_intermediate_size
        gate_dense = np.zeros((dense_i, hidden), dtype=values["mlp.shared_expert.gate_proj.weight"].dtype)
        up_dense = np.zeros_like(gate_dense)
        down_dense = np.zeros((hidden, dense_i), dtype=values["mlp.shared_expert.down_proj.weight"].dtype)
        gate_dense[list(partition.shared_indices)] = values["mlp.shared_expert.gate_proj.weight"]
        up_dense[list(partition.shared_indices)] = values["mlp.shared_expert.up_proj.weight"]
        down_dense[:, list(partition.shared_indices)] = values["mlp.shared_expert.down_proj.weight"]
        for expert, group in enumerate(partition.expert_indices):
            gate_dense[list(group)] = values[f"mlp.experts.{expert}.gate_proj.weight"]
            up_dense[list(group)] = values[f"mlp.experts.{expert}.up_proj.weight"]
            down_dense[:, list(group)] = values[f"mlp.experts.{expert}.down_proj.weight"]
        return cls(
            gate_dense,
            up_dense,
            down_dense,
            routed_experts=partition.routed_experts,
            shared_intermediate_size=partition.shared_intermediate_size,
            top_k=int(config["top_k"]),
            partition=partition,
            router=router,
        )


__all__ = ["DenseSwiGLU", "Qwen35SwiGLUMoE"]
