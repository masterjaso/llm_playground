"""Analytically solvable tests for the separated frozen-slice diagnostics."""

from __future__ import annotations

import numpy as np

from dense2moe.partition import (
    frozen_slice_positive_oracle,
    frozen_slice_scaled_router_oracle,
    frozen_slice_simplex_oracle,
    sparse_baseline,
    trainable_student_proxy,
)


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
