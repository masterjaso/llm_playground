from __future__ import annotations

import numpy as np
import torch

from dense2moe.partition.contributions import raw_dense_partition_contributions
from scripts.run_v24_residual_partition_oracle import (
    _build_residual_aware_plan,
    _dense_partition_contributions,
)


def test_residual_aware_plan_is_capacity_preserving_and_deterministic() -> None:
    torch.manual_seed(17)
    hidden = torch.randn(9, 12)
    down = torch.randn(4, 12)
    targets = torch.randn(9, 4)
    first = _build_residual_aware_plan(
        hidden,
        targets,
        down,
        routed_experts=2,
        expert_intermediate_size=3,
        shared_intermediate_size=6,
    )
    second = _build_residual_aware_plan(
        hidden,
        targets,
        down,
        routed_experts=2,
        expert_intermediate_size=3,
        shared_intermediate_size=6,
    )
    first.validate()
    assert first == second
    assert len(first.shared_indices) == 6
    assert all(len(group) == 3 for group in first.expert_indices)
    assert sorted(first.all_indices) == list(range(12))


def test_residual_aware_plan_handles_numpy_targets() -> None:
    hidden = torch.ones(5, 8)
    down = torch.ones(3, 8)
    targets = np.zeros((5, 3), dtype=np.float32)
    plan = _build_residual_aware_plan(
        hidden,
        targets,
        down,
        routed_experts=2,
        expert_intermediate_size=2,
        shared_intermediate_size=4,
    )
    plan.validate()
    assert plan.total_capacity == 8


def test_gpu_formula_matches_existing_raw_dense_contribution_contract() -> None:
    torch.manual_seed(23)
    inputs = torch.randn(7, 3).numpy()
    gate = torch.randn(8, 3).numpy()
    up = torch.randn(8, 3).numpy()
    down = torch.randn(3, 8).numpy()
    plan = _build_residual_aware_plan(
        torch.nn.functional.silu(torch.from_numpy(inputs) @ torch.from_numpy(gate).T)
        * (torch.from_numpy(inputs) @ torch.from_numpy(up).T),
        np.zeros((7, 3), dtype=np.float32),
        torch.from_numpy(down),
        routed_experts=2,
        expert_intermediate_size=2,
        shared_intermediate_size=4,
    )
    expected_shared, expected_routed = raw_dense_partition_contributions(
        inputs,
        {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
        plan,
    )
    observed_shared, observed_routed = _dense_partition_contributions(
        torch.nn.functional.silu(torch.from_numpy(inputs) @ torch.from_numpy(gate).T)
        * (torch.from_numpy(inputs) @ torch.from_numpy(up).T),
        torch.from_numpy(down),
        plan,
    )
    np.testing.assert_allclose(observed_shared, expected_shared, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(observed_routed, expected_routed, rtol=1e-5, atol=1e-5)
