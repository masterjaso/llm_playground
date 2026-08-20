from __future__ import annotations

import json

import pytest

from dense2moe.science.v23_accounting import (
    V23AccountingSpec,
    account_v23,
    estimate_active_flops,
)


@pytest.mark.parametrize(
    ("topology", "active_width", "top_k", "expert_width", "routed_experts"),
    (
        ("p16/top4", 5120, 4, 1024, 16),
        ("p32/top5", 3584, 5, 512, 32),
    ),
)
def test_active_topology_arithmetic_is_capacity_preserving(
    topology: str,
    active_width: int,
    top_k: int,
    expert_width: int,
    routed_experts: int,
) -> None:
    accounting = account_v23(topology)

    assert accounting.dense_intermediate_size == 17_408
    assert accounting.active_intermediate_size == active_width
    assert accounting.top_k == top_k
    assert accounting.expert_intermediate_size == expert_width
    assert accounting.routed_experts == routed_experts
    assert accounting.active_width_reduction == pytest.approx(1 - active_width / 17_408)

    hidden = accounting.hidden_size
    assert accounting.shared_parameters == 3 * hidden * 1_024
    assert accounting.routed_parameters == 3 * hidden * expert_width * routed_experts
    assert accounting.active_routed_parameters == 3 * hidden * expert_width * top_k
    assert accounting.dense_ffn_parameters == (
        accounting.shared_parameters + accounting.routed_parameters
    )
    assert accounting.parameters.active_ffn_parameters == (
        accounting.shared_parameters + accounting.active_routed_parameters
    )


def test_forbidden_topology_fails_closed() -> None:
    with pytest.raises(ValueError, match="explicitly forbidden"):
        account_v23("p32/top4")


def test_residual_router_and_scale_calibration_are_separate() -> None:
    accounting = account_v23(
        V23AccountingSpec(
            topology="p16/top4",
            hidden_size=64,
            routing_mode="independent_positive",
            residual_intermediate_size=8,
            router_architecture="linear",
            learnable_scales=True,
            scale_calibration_parameters=7,
            scale_calibration_flops_per_token=11,
            normalization_parameters=128,
            normalization_flops_per_token=64,
        )
    )

    assert accounting.residual_parameters == 3 * 64 * 8
    # Independent-positive routing has a selector and a hidden-to-expert
    # amplitude projection, including one bias per routed expert.
    assert accounting.router_parameters == (64 * 16) + (64 * 16 + 16)
    assert accounting.scale_parameters == 16
    assert accounting.calibration_parameters == 7
    assert accounting.parameters.active_ffn_parameters == (
        accounting.shared_parameters
        + accounting.active_routed_parameters
        + accounting.residual_parameters
        + 128
        + 7
    )
    assert accounting.flops.calibration_flops_per_token == 11
    assert accounting.flops.router_flops_per_token > 0
    assert accounting.flops.scale_flops_per_token == 4 * 64


def test_active_flops_are_monotonic_with_work_and_budget() -> None:
    p16 = account_v23("p16/top4", hidden_size=64, num_hidden_layers=1)
    p32 = account_v23("p32/top5", hidden_size=64, num_hidden_layers=1)
    extra = account_v23(
        "p16/top4",
        hidden_size=64,
        num_hidden_layers=1,
        residual_intermediate_size=8,
        router_hidden_size=16,
        router_architecture="low_rank_silu",
        learnable_scales=True,
        scale_calibration_parameters=4,
    )

    assert p32.active_flops < p16.active_flops
    assert extra.active_flops > p16.active_flops
    assert estimate_active_flops(p16, tokens=3, num_hidden_layers=2) == (
        p16.flops.active_flops_per_token * 3 * 2
    )


def test_accounting_serializes_to_deterministic_json() -> None:
    accounting = account_v23("p32/top5", tokens=3)
    payload = accounting.as_dict()
    encoded = accounting.to_json()

    assert json.loads(encoded) == payload
    assert encoded == accounting.to_json()
    assert payload["version"] == "dense2moe-v2.3"
    assert payload["topology"]["id"] == "p32/top5"
    assert payload["parameters"]["router_parameters"] == accounting.router_parameters
