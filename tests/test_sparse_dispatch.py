from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE


def _model(*, routing_mode: str = "independent_positive") -> TorchQwen35SwiGLUMoE:
    torch.manual_seed(7)
    return TorchQwen35SwiGLUMoE(
        hidden_size=8,
        intermediate_size=12,
        routed_experts=4,
        expert_intermediate_size=2,
        shared_intermediate_size=4,
        top_k=2,
        routing_mode=routing_mode,
        learnable_scales=True,
    )


def _reference(model: TorchQwen35SwiGLUMoE, inputs: torch.Tensor) -> torch.Tensor:
    x = inputs.reshape(-1, model.hidden_size)
    shared = model.shared_down_proj(torch.nn.functional.silu(model.shared_gate_proj(x)) * model.shared_up_proj(x))
    logits = model.router(x)
    values, indices = torch.topk(logits, model.top_k, dim=-1)
    if model.routing_mode == "normalized_softmax":
        weights = torch.softmax(values, dim=-1)
    else:
        weights = torch.nn.functional.softplus(model.amplitude_router(x).gather(-1, indices))
    routed, _ = model._routed_dense(x, indices, weights, return_contributions=False)
    return (shared + routed).reshape_as(inputs)


@pytest.mark.parametrize("routing_mode", ["normalized_softmax", "independent_positive"])
def test_sparse_dispatch_matches_dense_reference(routing_mode: str) -> None:
    model = _model(routing_mode=routing_mode).eval()
    inputs = torch.randn(5, 3, 8)
    sparse = model(inputs)
    dense = _reference(model, inputs)
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-6)


def test_sparse_dispatch_evaluates_only_selected_rows_and_reports_reduction() -> None:
    model = _model().eval()
    inputs = torch.randn(9, 8)
    calls: list[tuple[int, int]] = []
    original = model._expert_output

    def spy(x: torch.Tensor, expert: int) -> torch.Tensor:
        calls.append((int(x.shape[0]), expert))
        return original(x, expert)

    with patch.object(model, "_expert_output", side_effect=spy):
        output, info = model(inputs, return_router=True)

    assert output.shape == inputs.shape
    assert info["dispatch_mode"] == "sparse_token_dispatch"
    assert info["dense_fallback_used"] is False
    assert info["selected_dispatches"] == inputs.shape[0] * model.top_k
    assert sum(count for count, _ in calls) == inputs.shape[0] * model.top_k
    assert info["active_intermediate_width"] == 8
    assert info["dense_intermediate_width"] == 12
    assert info["estimated_ffn_reduction"] == pytest.approx(1.0 - 8.0 / 12.0)
    assert all(count > 0 for count, _ in calls)


def test_contribution_diagnostics_are_explicit_dense_fallback() -> None:
    model = _model().eval()
    _, info = model(torch.randn(4, 8), return_router=True, return_contributions=True)
    assert info["dispatch_mode"] == "dense_contributions"
    assert info["dense_fallback_used"] is True
    assert info["selected_dispatches"] == 8
    assert len(info["contributions"]) == model.routed_experts


def test_sparse_dispatch_preserves_gradients() -> None:
    model = _model()
    inputs = torch.randn(6, 8, requires_grad=True)
    loss = model(inputs).square().mean()
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.expert_gate_proj[0].weight.grad is not None
    assert torch.isfinite(model.expert_gate_proj[0].weight.grad).all()
