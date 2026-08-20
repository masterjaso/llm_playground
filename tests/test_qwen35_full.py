from __future__ import annotations

import json

import torch
from torch import nn

from dense2moe.checkpoint.layer import (
    LayerCheckpoint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from dense2moe.config import TopologyContract
from dense2moe.models.qwen35_full import (
    Qwen35DenseToMoE,
    apply_layer_checkpoints,
    replace_qwen35_ffns,
)
from dense2moe.provenance import current_git_commit


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


def test_full_assembly_applies_only_complete_validated_layer_manifest(tmp_path) -> None:
    model = _Model().eval()
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
    wrapper = Qwen35DenseToMoE(model, receipt=receipt)
    layer_entries = []
    for layer_index, layer in enumerate(model.model.layers):
        layer_dir = tmp_path / f"layer-{layer_index}"
        layer_dir.mkdir()
        tensor_map = {
            f"model.layers.{layer_index}.{name}": value.detach().cpu().numpy()
            for name, value in layer.mlp.state_dict().items()
        }
        tensor_path, inventory, tensor_hash = publish_tensor_artifact(tensor_map, layer_dir / "layer.safetensors")
        metadata_path = layer_dir / "layer.json"
        checkpoint = LayerCheckpoint(
            layer=layer_index,
            profile="fixture",
            status="TRAINED_VALIDATED",
            profile_hash="fixture-profile",
            source_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            source_config_hash="config-hash",
            source_index_hash="index-hash",
            dataset_hash="dataset-hash",
            partition_hash="partition-hash",
            tensor_file=tensor_path.name,
            tensor_sha256=tensor_hash,
            tensor_inventory=inventory,
            quality_gate={"overall": "green"},
            code_commit=current_git_commit(),
        )
        save_layer_checkpoint(checkpoint, metadata_path)
        layer_entries.append({"layer": layer_index, "profile": "fixture", "metadata": str(metadata_path)})
    manifest_path = tmp_path / "checkpoints-manifest.json"
    manifest_path.write_text(json.dumps({"complete": True, "layers": layer_entries}), encoding="utf-8")
    applied = apply_layer_checkpoints(wrapper, manifest_path, expected_profile="fixture", strict_layer_count=2)
    assert applied["status"] == "FULL64_LAYER_CHECKPOINTS_APPLIED"
    assert applied["applied_layer_count"] == 2
    assert wrapper.receipt["layer_checkpoint_application"]["source_backbone_preserved"] is True
