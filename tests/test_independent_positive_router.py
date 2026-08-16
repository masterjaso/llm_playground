from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.checkpoint.layer import LayerCheckpoint, load_layer_checkpoint, save_layer_checkpoint


def _model(*, top_k: int = 2) -> TorchQwen35SwiGLUMoE:
    gate = torch.randn(6, 3)
    up = torch.randn(6, 3)
    down = torch.randn(3, 6)
    return TorchQwen35SwiGLUMoE.from_dense(
        gate,
        up,
        down,
        routed_experts=2,
        shared_intermediate_size=2,
        top_k=top_k,
        routing_mode="independent_positive",
        learnable_scales=False,
    )


def test_known_selected_experts_and_positive_amplitudes_reconstruct_exactly() -> None:
    model = _model()
    inputs = torch.eye(2, 3)
    expected_contributions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]
    )
    with torch.no_grad():
        for module in (model.shared_gate_proj, model.shared_up_proj, model.shared_down_proj):
            module.weight.zero_()
        model.router.weight.zero_()  # stable top-k tie selects both experts
        model.amplitude_router.weight.zero_()
        model.amplitude_router.bias.copy_(torch.tensor([math.log(math.expm1(2.0)), math.log(math.expm1(3.0))]))
        model._expert_output = lambda _x, expert: expected_contributions[:, expert]  # type: ignore[method-assign]
    output, info = model(inputs, return_router=True)
    assert info["routing_mode"] == "independent_positive"
    assert torch.equal(info["indices"], torch.tensor([[0, 1], [0, 1]]))
    assert torch.allclose(info["weights"], torch.tensor([[2.0, 3.0], [2.0, 3.0]]), atol=1e-5)
    assert torch.allclose(output, torch.tensor([[2.0, 3.0, 0.0], [4.0, 9.0, 0.0]]), atol=1e-5)


def test_independent_positive_gradients_and_amplitudes_are_finite() -> None:
    model = _model()
    inputs = torch.randn(4, 3, requires_grad=True)
    output, info = model(inputs, return_router=True)
    assert torch.isfinite(info["weights"]).all()
    loss = output.square().mean()
    loss.backward()
    assert model.amplitude_router.weight.grad is not None
    assert torch.isfinite(model.amplitude_router.weight.grad).all()
    assert torch.isfinite(model.amplitude_router.bias.grad).all()


def test_independent_positive_mode_strict_reload(tmp_path) -> None:
    model = _model()
    inputs = torch.randn(3, 3)
    expected = model(inputs)
    destination = model.save_pretrained(tmp_path / "positive")
    reloaded = TorchQwen35SwiGLUMoE.from_pretrained(destination, strict=True)
    assert reloaded.routing_mode == "independent_positive"
    config = (destination / "config.json").read_text(encoding="utf-8")
    assert '"routing_mode": "independent_positive"' in config
    assert torch.allclose(expected, reloaded(inputs), atol=1e-6)


def test_layer_checkpoint_preserves_routing_mode(tmp_path) -> None:
    path = tmp_path / "layer.json"
    save_layer_checkpoint(
        LayerCheckpoint(layer=0, profile="fixture", status="RESEARCH_CANDIDATE", routing_mode="independent_positive"),
        path,
    )
    assert load_layer_checkpoint(path).routing_mode == "independent_positive"
