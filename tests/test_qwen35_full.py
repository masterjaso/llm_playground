from __future__ import annotations

import torch
from torch import nn

from dense2moe.config import TopologyContract
from dense2moe.models.qwen35_full import replace_qwen35_ffns


class _DenseMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(8, 8)
        self.mlp = _DenseMLP(8, 16)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.embed = nn.Embedding(16, 8)


def test_full_replacement_preserves_non_ffn_and_replaces_every_layer() -> None:
    model = _Model().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items() if ".mlp." not in name}
    topology = TopologyContract(
        topology_id="p4/top2",
        profile_name="fixture",
        role="test",
        routed_experts=4,
        expert_intermediate_size=2,
        shared_intermediate_size=8,
        top_k=2,
        dense_intermediate_size=16,
    )
    receipt = replace_qwen35_ffns(model, topology=topology, strict_layer_count=2)
    assert receipt["replaced_layer_count"] == 2
    assert receipt["non_ffn_preserved"] is True
    assert all("TorchQwen35SwiGLUMoE" in type(layer.mlp).__name__ for layer in model.model.layers)
    after = {name: value.detach() for name, value in model.state_dict().items() if ".mlp." not in name}
    assert before.keys() == after.keys()
    for name, value in before.items():
        assert torch.equal(value, after[name])
