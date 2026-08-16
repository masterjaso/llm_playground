"""Analytically solvable tests for the separated frozen-slice diagnostics."""

from __future__ import annotations

import numpy as np

from dense2moe.partition import (
    frozen_slice_load_aware_oracle,
    frozen_slice_positive_oracle,
    frozen_slice_scaled_router_oracle,
    frozen_slice_simplex_oracle,
    sparse_baseline,
    trainable_student_proxy,
)
from dense2moe.partition.oracle import _pareto_points


def test_load_aware_oracle_reports_balanced_pareto_assignment() -> None:
    # Every expert is an exact reconstruction, so the load-aware tie-breaker
    # can balance dispatches without sacrificing quality.
    shared = np.zeros((8, 1))
    routed = np.ones((8, 4, 1))
    target = np.ones((8, 1))

    result = frozen_slice_load_aware_oracle(shared, routed, target, top_k=1, iterations=8)

    assert result["method"] == "frozen_slice_load_aware_oracle"
    assert result["global_nmse"] < 1e-12
    assert result["load_cv"] == 0.0
    assert result["dead_experts"] == 0
    assert result["feasible_load_target"] is True
    assert result["pareto"]


def test_load_aware_oracle_uses_bounded_float32_blocks_for_float64_inputs(tmp_path) -> None:
    rng = np.random.default_rng(41)
    shared = rng.normal(size=(8, 3)).astype(np.float64)
    routed = rng.normal(size=(8, 16, 3)).astype(np.float64)
    target = rng.normal(size=(8, 3)).astype(np.float64)

    result = frozen_slice_load_aware_oracle(
        shared,
        routed,
        target,
        top_k=4,
        iterations=1,
        batch_size=2,
        max_in_memory_bytes=4096,
        storage_dir=tmp_path,
        materialize_outputs=False,
    )

    assert result["assurance"] == "exact_candidate_sets"
    assert result["candidate_fit_exact"] is False
    assert result["coefficient_solver"].endswith("float32")
    assert result["weights"].dtype == np.float32
    assert result["candidate_error_storage"] == "memmap"
    assert result["candidate_batch_size"] < 1024
    assert result["input_batch_bytes"] <= 4096


def test_load_aware_oracle_marks_p32_as_bounded_and_reports_candidate_count() -> None:
    shared = np.zeros((2, 2), dtype=np.float32)
    routed = np.ones((2, 32, 2), dtype=np.float32)
    target = np.ones((2, 2), dtype=np.float32)

    result = frozen_slice_load_aware_oracle(shared, routed, target, top_k=5, iterations=1, batch_size=1)

    assert result["assurance"] == "bounded_correlation_candidate_pool"
    assert result["combinations_considered_per_token"] == 2002
    assert result["candidate_error_storage"] in {"ram", "memmap"}
    assert result["candidate_id_storage"] in {"ram", "memmap"}


def test_pareto_frontier_keeps_cosine_dimension() -> None:
    points = [
        {"cosine": 0.978, "global_nmse": 0.015, "load_cv": 0.40},
        {"cosine": 0.982, "global_nmse": 0.025, "load_cv": 0.45},
        {"cosine": 0.977, "global_nmse": 0.015, "load_cv": 0.50},
    ]

    frontier = _pareto_points(points)

    assert points[0] in frontier
    assert points[1] in frontier
    assert points[2] not in frontier


def test_simplex_oracle_uses_exact_pairwise_clipped_solution() -> None:
    shared = np.zeros((1, 2))
    routed = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    target = np.asarray([[0.25, 0.75]])

    result = frozen_slice_simplex_oracle(shared, routed, target, top_k=2)

    assert result["indices"].tolist() == [[0, 1]]
    np.testing.assert_allclose(result["weights"], [[0.25, 0.75]], atol=1e-12)
    assert result["mse"] < 1e-24
    assert result["method"] == "frozen_slice_simplex_oracle"


def test_simplex_oracle_clips_to_boundary() -> None:
    shared = np.zeros((1, 2))
    routed = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    target = np.asarray([[-2.0, 0.5]])

    result = frozen_slice_simplex_oracle(shared, routed, target, top_k=2)

    np.testing.assert_allclose(result["weights"], [[0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(result["reconstruction"], [[0.0, 1.0]], atol=1e-12)


def test_positive_oracle_solves_interior_two_variable_nnls() -> None:
    shared = np.zeros((1, 2))
    routed = np.asarray([[[1.0, 0.0], [0.0, 2.0]]])
    target = np.asarray([[2.0, 3.0]])

    result = frozen_slice_positive_oracle(shared, routed, target, top_k=2)

    np.testing.assert_allclose(result["weights"], [[2.0, 1.5]], atol=1e-12)
    assert result["mse"] < 1e-24
    assert not np.isclose(result["weights"].sum(), 1.0)


def test_positive_oracle_selects_nonnegative_boundary_solution() -> None:
    shared = np.zeros((1, 2))
    routed = np.asarray([[[1.0, 0.0], [-1.0, 0.0]]])
    target = np.asarray([[2.0, 0.0]])

    result = frozen_slice_positive_oracle(shared, routed, target, top_k=2)

    np.testing.assert_allclose(result["weights"], [[2.0, 0.0]], atol=1e-12)
    assert result["mse"] < 1e-24


def test_scaled_router_fits_scales_on_train_only_and_scores_holdout() -> None:
    train_shared = np.zeros((2, 1))
    train_routed = np.asarray([[[1.0], [0.0]], [[0.0], [1.0]]])
    train_target = np.asarray([[2.0], [3.0]])
    holdout_shared = np.zeros((2, 1))
    holdout_routed = np.asarray([[[1.0], [0.0]], [[0.0], [1.0]]])
    holdout_target = np.asarray([[4.0], [5.0]])

    result = frozen_slice_scaled_router_oracle(
        train_shared,
        train_routed,
        train_target,
        holdout_shared,
        holdout_routed,
        holdout_target,
        top_k=1,
    )

    np.testing.assert_allclose(result["scales"], [2.0, 3.0], atol=1e-12)
    assert result["fit_scope"] == "train_only"
    assert result["train"]["mse"] < 1e-24
    np.testing.assert_allclose(result["holdout"]["reconstruction"], [[2.0], [3.0]], atol=1e-12)
    assert result["holdout"]["mse"] > 0.0


def test_capacity_scaled_uses_e_over_k_per_selected_contribution() -> None:
    shared = np.zeros((1, 1))
    routed = np.asarray([[[1.0], [2.0], [3.0], [4.0]]])

    result = sparse_baseline(shared, routed, top_k=2, mode="capacity_scaled")

    np.testing.assert_allclose(result["weights"], [[2.0, 2.0]], atol=1e-12)
    # Norm-ranked experts are 3 and 2; each gets E/k = 2.
    np.testing.assert_allclose(result["reconstruction"], [[14.0]], atol=1e-12)


def test_trainable_student_proxy_is_not_a_frozen_hard_ceiling() -> None:
    result = trainable_student_proxy()

    assert result["method"] == "trainable_student_proxy"
    assert result["frozen_slice_is_hard_ceiling"] is False
    assert result["genuine_constraints"]
