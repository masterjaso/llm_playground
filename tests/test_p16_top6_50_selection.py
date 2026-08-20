from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")

from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.training.torch_distill import _optimizer_parameter_groups, _quantile_balance_weights
from scripts.run_p16_top6_50_selection import _inject_policy_learning_rates


def _model() -> TorchQwen35SwiGLUMoE:
    torch.manual_seed(13)
    gate = torch.randn(32, 8)
    up = torch.randn(32, 8)
    down = torch.randn(8, 32)
    return TorchQwen35SwiGLUMoE.from_dense(
        gate,
        up,
        down,
        routed_experts=4,
        shared_intermediate_size=16,
        top_k=2,
        routing_mode="independent_positive",
        learnable_scales=True,
        residual_intermediate_size=4,
        residual_scope="static",
        fallback_mode="none",
        fallback_rate_budget=0.0,
    )


def test_static_residual_is_executed_and_accounted() -> None:
    model = _model().eval()
    output, info = model(torch.randn(7, 8), return_router=True)
    assert output.shape == (7, 8)
    assert info["residual_executed"] is True
    assert info["residual_parameter_count"] > 0
    assert info["active_intermediate_width"] == 28
    assert info["active_intermediate_width_mean"] == 28
    assert info["active_intermediate_width_p50"] == 28
    assert info["active_intermediate_width_p95"] == 28
    assert info["active_intermediate_width_max"] == 28
    assert info["average_ffn_reduction"] == pytest.approx(1.0 - 28.0 / 32.0)


def test_residual_parameters_are_in_optimizer_group_and_receive_state() -> None:
    model = _model()
    groups = _optimizer_parameter_groups(model, learning_rate=1e-3, learning_rates=None)
    residual_ids = {id(parameter) for parameter in model.residual_corrector.parameters()}
    residual_group = next(group for group in groups if group["group"] == "residual")
    assert residual_ids == {id(parameter) for parameter in residual_group["params"]}
    optimizer = torch.optim.AdamW(groups)
    loss = model(torch.randn(6, 8)).square().mean()
    loss.backward()
    optimizer.step()
    assert all(parameter in optimizer.state for parameter in model.residual_corrector.parameters())


def test_residual_checkpoint_reload_preserves_output_and_routing(tmp_path) -> None:
    model = _model().eval()
    inputs = torch.randn(5, 8)
    expected, expected_info = model(inputs, return_router=True)
    destination = model.save_pretrained(tmp_path / "candidate")
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["residual_intermediate_size"] == 4
    restored = TorchQwen35SwiGLUMoE.from_pretrained(destination, strict=True)
    observed, observed_info = restored(inputs, return_router=True)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed_info["indices"], expected_info["indices"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed_info["weights"], expected_info["weights"], rtol=0.0, atol=0.0)


def test_quantile_balance_weights_are_finite_unit_mean_and_teacher_training_only() -> None:
    target = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    weights = _quantile_balance_weights(target, (1.75, 2.5, 3.25))
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="quantile"):
        _quantile_balance_weights(target, None)


def test_runner_applies_preregistered_router_learning_rate() -> None:
    prereg = {"training_policy": {"base_learning_rate": 1e-4, "router_learning_rate": 2e-4}}
    stages = _inject_policy_learning_rates([{"name": "capacity", "learning_rates": {"experts": 1e-4}}], prereg)
    assert stages[0]["learning_rates"]["selection_router"] == pytest.approx(2e-4)
    assert stages[0]["learning_rates"]["amplitude_router"] == pytest.approx(2e-4)
    assert stages[0]["learning_rates"]["experts"] == pytest.approx(1e-4)
