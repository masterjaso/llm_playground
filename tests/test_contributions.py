from __future__ import annotations

import json

import numpy as np
import pytest

from dense2moe.partition import (
    PartitionPlan,
    basis_outputs_from_state,
    contribution_manifest,
    raw_dense_partition_contributions,
    reconstruct_selected,
)


def _plan() -> PartitionPlan:
    plan = PartitionPlan(
        dense_intermediate_size=6,
        routed_experts=2,
        expert_intermediate_size=2,
        shared_intermediate_size=2,
        shared_indices=(0, 1),
        expert_indices=((2, 3), (4, 5)),
    )
    plan.validate()
    return plan


def _state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    hidden = 4
    values: dict[str, np.ndarray] = {
        "shared_gate_proj.weight": rng.normal(size=(2, hidden)).astype(np.float32),
        "shared_up_proj.weight": rng.normal(size=(2, hidden)).astype(np.float32),
        "shared_down_proj.weight": rng.normal(size=(hidden, 2)).astype(np.float32),
        "expert_scales": np.asarray([1.0, 1.5], dtype=np.float32),
    }
    for expert in range(2):
        values[f"expert_gate_proj.{expert}.weight"] = rng.normal(size=(2, hidden)).astype(np.float32)
        values[f"expert_up_proj.{expert}.weight"] = rng.normal(size=(2, hidden)).astype(np.float32)
        values[f"expert_down_proj.{expert}.weight"] = rng.normal(size=(hidden, 2)).astype(np.float32)
    return values


def test_checkpoint_basis_outputs_and_arbitrary_reconstruction_are_deterministic() -> None:
    plan = _plan()
    state = _state()
    inputs = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
    shared, routed = basis_outputs_from_state(inputs, state, plan, batch_size=2)
    shared_again, routed_again = basis_outputs_from_state(inputs, state, plan, batch_size=5)
    np.testing.assert_allclose(shared, shared_again, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(routed, routed_again, rtol=1e-6, atol=1e-6)
    ids = np.asarray([[0], [1], [0], [1], [0]], dtype=np.int64)
    weights = np.asarray([[0.5], [1.0], [1.5], [0.25], [2.0]], dtype=np.float32)
    expected = shared + np.take_along_axis(routed, ids[:, :, None], axis=1)[:, 0] * weights
    np.testing.assert_allclose(reconstruct_selected(shared, routed, ids, weights), expected, rtol=1e-6, atol=1e-6)


def test_safetensors_checkpoint_loader_matches_independent_state_path(tmp_path) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    from dense2moe.partition import trained_checkpoint_contributions

    plan = _plan()
    state = _state()
    checkpoint = tmp_path / "layer-0000.safetensors"
    save_file({key: np.ascontiguousarray(value) for key, value in state.items()}, str(checkpoint))
    inputs = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
    expected_shared, expected_routed = basis_outputs_from_state(inputs, state, plan)
    shared, routed, metadata = trained_checkpoint_contributions(inputs, checkpoint, plan, batch_size=2)
    np.testing.assert_allclose(shared, expected_shared, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(routed, expected_routed, rtol=1e-6, atol=1e-6)
    assert metadata["basis_source"] == "trained_checkpoint"
    assert metadata["checkpoint_tensor_sha256"]


def test_raw_partition_and_trained_checkpoint_manifest_modes_are_distinct(tmp_path) -> None:
    plan = _plan()
    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    checkpoint_path = tmp_path / "trained.safetensors"
    checkpoint_path.write_bytes(b"checkpoint-fingerprint")
    raw = contribution_manifest(
        basis_source="raw_dense_partition",
        plan=plan,
        row_count=3,
        split="train",
        dataset_hash="dataset",
        partition_path=partition_path,
        top_k=4,
    )
    trained = contribution_manifest(
        basis_source="trained_checkpoint",
        plan=plan,
        row_count=3,
        split="validation_a",
        dataset_hash="dataset",
        partition_path=partition_path,
        top_k=4,
        checkpoint=checkpoint_path,
        source_revision="rev",
        capture_identity={"manifest": "capture.json"},
        dtype="float32",
        code_commit="commit",
    )
    assert raw["basis_source"] == "raw_dense_partition"
    assert raw["checkpoint_tensor_sha256"] is None
    assert trained["basis_source"] == "trained_checkpoint"
    assert trained["checkpoint_path"].endswith("trained.safetensors")
    assert trained["checkpoint_tensor_sha256"]
    assert trained["topology"] == {"expert_count": 2, "expert_width": 2, "shared_width": 2, "top_k": 4}
    assert trained["split"] == "validation_a"


def test_raw_dense_mode_remains_explicitly_separate_from_checkpoint_basis() -> None:
    plan = _plan()
    rng = np.random.default_rng(5)
    dense = {
        "gate_proj.weight": rng.normal(size=(6, 4)).astype(np.float32),
        "up_proj.weight": rng.normal(size=(6, 4)).astype(np.float32),
        "down_proj.weight": rng.normal(size=(4, 6)).astype(np.float32),
    }
    inputs = rng.normal(size=(3, 4)).astype(np.float32)
    shared, routed = raw_dense_partition_contributions(inputs, dense, plan)
    assert shared.shape == (3, 4)
    assert routed.shape == (3, 2, 4)
