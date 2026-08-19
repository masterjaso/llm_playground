from __future__ import annotations

import json

import pytest

from dense2moe.science.hard_tail import (
    HardTailConfig,
    make_static_config,
    summarize_active_widths,
    validate_inference_features,
)

torch = pytest.importorskip("torch")

from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import partition_indices


def _dense(*, hidden: int = 8, intermediate: int = 32):
    torch.manual_seed(7)
    return (
        torch.randn(intermediate, hidden),
        torch.randn(intermediate, hidden),
        torch.randn(hidden, intermediate),
    )


def _model(**kwargs):
    gate, up, down = _dense()
    return TorchQwen35SwiGLUMoE.from_dense(
        gate,
        up,
        down,
        routed_experts=16,
        shared_intermediate_size=16,
        top_k=6,
        **kwargs,
    )


def test_static_geometry_and_compute_budget_are_exact() -> None:
    config = make_static_config(2048, residual_width=0)
    assert config.expert_width == 960
    assert config.static_active_width == 7808
    assert config.static_reduction == pytest.approx(1.0 - 7808 / 17408)
    assert config.configuration_id == make_static_config(2048).configuration_id
    assert config.as_dict()["configuration_id"].startswith("ht-")

    over_budget = make_static_config(3456, residual_width=512)
    with pytest.raises(ValueError, match="below"):
        over_budget.require_compute_budget()


def test_percentiles_use_nearest_rank_and_empty_is_rejected() -> None:
    summary = summarize_active_widths([10, 20, 30, 40])
    assert summary.mean == pytest.approx(25.0)
    assert summary.p50 == 20
    assert summary.p95 == 40
    assert summary.maximum == 40
    with pytest.raises(ValueError, match="at least one"):
        summarize_active_widths([])


def test_disabled_fallback_is_dispatch_equivalent() -> None:
    baseline = _model()
    extended = _model()
    extended.load_state_dict(baseline.state_dict())
    inputs = torch.randn(5, 8)
    expected, expected_info = baseline(inputs, return_router=True)
    observed, observed_info = extended(
        inputs,
        return_router=True,
        fallback_mode="top8",
        fallback_mask=torch.zeros(5, dtype=torch.bool),
    )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed_info["weights"], expected_info["weights"], rtol=0.0, atol=0.0)
    assert observed_info["fallback_token_count"] == 0
    assert observed_info["active_intermediate_width_mean"] == expected_info["active_intermediate_width_mean"]


def test_residual_branch_is_real_trainable_and_reloadable(tmp_path) -> None:
    model = _model(residual_intermediate_size=2)
    assert any(name.startswith("residual_corrector.") for name in model.state_dict())
    with torch.no_grad():
        model.residual_corrector.output_proj.weight.fill_(0.05)
    inputs = torch.randn(4, 8, requires_grad=True)
    with torch.no_grad():
        model.residual_corrector.output_proj.weight.zero_()
        without_residual = model(inputs.detach())
        model.residual_corrector.output_proj.weight.fill_(0.05)
        with_residual = model(inputs.detach())
    assert not torch.equal(with_residual, without_residual)
    prediction = model(inputs)
    prediction.square().mean().backward()
    assert model.residual_corrector.input_proj.weight.grad is not None
    assert float(model.residual_corrector.input_proj.weight.grad.abs().sum()) > 0.0
    assert float(model.residual_corrector.output_proj.weight.grad.abs().sum()) > 0.0

    destination = model.save_pretrained(tmp_path / "residual")
    config = json.loads((destination / "config.json").read_text())
    assert config["residual_intermediate_size"] == 2
    restored = TorchQwen35SwiGLUMoE.from_pretrained(destination)
    torch.testing.assert_close(restored(inputs.detach()), model(inputs.detach()), rtol=0.0, atol=0.0)


def test_topk_fallback_is_selected_and_accounted() -> None:
    model = _model(fallback_mode="top8")
    inputs = torch.randn(8, 8)
    mask = torch.tensor([True, False, False, False, True, False, False, False])
    output, info = model(inputs, return_router=True, fallback_mask=mask)
    assert output.shape == inputs.shape
    assert info["indices"].shape[-1] == 8
    assert info["fallback_token_count"] == 2
    assert info["fallback_rate"] == pytest.approx(0.25)
    assert info["static_active_intermediate_width"] == 22
    assert info["fallback_active_intermediate_width"] == 24
    assert info["active_intermediate_width_mean"] == pytest.approx(22.5)
    assert info["active_intermediate_width_p50"] == 22
    assert info["active_intermediate_width_p95"] == 24
    assert info["active_intermediate_width_max"] == 24
    assert info["dropped_token_count"] == info["invalid_token_count"] == info["non_finite_token_count"] == 0


def test_residual_fallback_only_executes_selected_rows() -> None:
    model = _model(residual_intermediate_size=2, residual_scope="selected", fallback_mode="residual")
    with torch.no_grad():
        model.residual_corrector.output_proj.weight.fill_(0.05)
    inputs = torch.randn(4, 8)
    mask = torch.tensor([True, False, False, False])
    selected = model(inputs, fallback_mask=mask)
    unselected = model(inputs, fallback_mask=torch.zeros_like(mask))
    assert not torch.equal(selected[0], unselected[0])
    assert torch.equal(selected[1:], unselected[1:])
    _, info = model(inputs, return_router=True, fallback_mask=mask)
    assert info["fallback_token_count"] == 1
    assert info["residual_scope"] == "selected"
    assert info["fallback_active_intermediate_width"] == 24


def test_fallback_rate_and_nonfinite_inputs_fail_closed() -> None:
    model = _model(fallback_mode="top8")
    with pytest.raises(ValueError, match="fallback rate"):
        model(torch.randn(4, 8), fallback_mask=torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[float("nan")] + [0.0] * 7]))
    with torch.no_grad():
        model.shared_gate_proj.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="outputs must be finite"):
        model(torch.zeros(2, 8))


def test_teacher_dependent_predictor_features_are_rejected() -> None:
    assert validate_inference_features(["hidden_state", "router_entropy"]) == ("hidden_state", "router_entropy")
    for feature in ("teacher_ffn_output", "target_norm", "realized_reconstruction_error", "posthoc_dense_moe_output_difference"):
        with pytest.raises(ValueError, match="teacher-dependent"):
            validate_inference_features(["hidden_state", feature])


def test_hard_tail_config_rejects_invalid_fallback_budget() -> None:
    config = HardTailConfig(shared_width=16, expert_width=1, dense_width=32, fallback_mode="top10")
    with pytest.raises(ValueError, match="below"):
        config.require_compute_budget(0.30)
    with pytest.raises(ValueError, match="fallback rate"):
        config.mean_active_width(0.31)


def test_partition_is_still_exact_for_residual_configs() -> None:
    plan = partition_indices(32, 16, 1, 16)
    plan.validate()
    assert plan.shared_intermediate_size == 16

