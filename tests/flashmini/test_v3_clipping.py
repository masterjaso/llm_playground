"""Regression coverage for v3 independent gradient clipping."""

import pytest
import torch
from torch import nn

from flashmini.optim import clip_gradient_groups


class _PLE(nn.Module):
    def __init__(self, rows: int = 128):
        super().__init__()
        self.dense = nn.Linear(2, 2, bias=False)
        self.table = nn.Embedding(rows, 2, sparse=True)


class _WithPLE(nn.Module):
    def __init__(self, rows: int = 128):
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        self.ple = _PLE(rows)


class _WithoutPLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)


def _set_shared_gradient(model: nn.Module) -> None:
    model.shared.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])


def _set_sparse_gradient(parameter: nn.Parameter, values: torch.Tensor) -> None:
    indices = torch.tensor([[1, 5, 1]], dtype=torch.long)
    parameter.grad = torch.sparse_coo_tensor(
        indices, values, size=parameter.shape, check_invariants=True
    )


def test_huge_ple_gradient_cannot_change_shared_clipping_coefficient():
    baseline = _WithoutPLE()
    _set_shared_gradient(baseline)
    baseline_metrics = clip_gradient_groups(baseline, max_norm=1.0)

    candidate = _WithPLE()
    _set_shared_gradient(candidate)
    candidate.ple.dense.weight.grad = torch.full_like(candidate.ple.dense.weight, 10_000.0)
    _set_sparse_gradient(candidate.ple.table.weight, torch.tensor([[10_000.0, 0.0],
                                                                     [0.0, 10_000.0],
                                                                     [10_000.0, 0.0]]))
    candidate_metrics = clip_gradient_groups(candidate, max_norm=1.0)

    assert candidate_metrics["grad_clip_coefficient_shared"] == pytest.approx(
        baseline_metrics["grad_clip_coefficient_shared"]
    )
    assert candidate_metrics["grad_clipped_shared"] is True
    assert candidate_metrics["grad_clipped_ple_dense"] is True
    assert candidate_metrics["grad_clipped_ple_sparse"] is True


def test_each_gradient_group_is_clipped_independently():
    model = _WithPLE()
    model.shared.weight.grad = torch.full_like(model.shared.weight, 0.25)
    model.ple.dense.weight.grad = torch.full_like(model.ple.dense.weight, 1.0)
    _set_sparse_gradient(model.ple.table.weight, torch.tensor([[3.0, 0.0],
                                                                [0.0, 4.0],
                                                                [3.0, 0.0]]))

    metrics = clip_gradient_groups(model, max_norm=1.0)

    assert metrics["grad_norm_shared_preclip"] == pytest.approx(0.5)
    assert metrics["grad_norm_ple_dense_preclip"] == pytest.approx(2.0)
    # Duplicate sparse row 1 is coalesced before its norm is calculated.
    assert metrics["grad_norm_ple_sparse_preclip"] == pytest.approx(7.21110255)
    assert metrics["grad_clipped_shared"] is False
    assert metrics["grad_clipped_ple_dense"] is True
    assert metrics["grad_clipped_ple_sparse"] is True
    assert model.ple.table.weight.grad.is_sparse
    assert model.ple.table.weight.grad.is_coalesced()
    assert model.ple.table.weight.grad._nnz() == 2


def test_sparse_adam_receives_sparse_rows_after_grouped_clipping():
    model = _WithPLE(rows=100_000)
    _set_sparse_gradient(model.ple.table.weight, torch.tensor([[3.0, 0.0],
                                                                [0.0, 4.0],
                                                                [3.0, 0.0]]))

    metrics = clip_gradient_groups(model, max_norm=1.0)
    gradient = model.ple.table.weight.grad
    assert metrics["grad_clipped_ple_sparse"] is True
    assert gradient.is_sparse
    assert gradient.is_coalesced()
    assert gradient._nnz() == 2
    assert gradient.shape == model.ple.table.weight.shape

    optimizer = torch.optim.SparseAdam([model.ple.table.weight], lr=0.01)
    optimizer.step()
    assert model.ple.table.weight.grad.is_sparse
    assert model.ple.table.weight.grad._nnz() == 2


def test_grouped_clipping_reports_empty_groups_and_rejects_nonfinite_values():
    model = _WithoutPLE()
    _set_shared_gradient(model)
    metrics = clip_gradient_groups(model, max_norm=1.0)

    assert metrics["grad_norm_ple_dense_preclip"] == 0.0
    assert metrics["grad_clip_coefficient_ple_dense"] == 1.0
    assert metrics["grad_clipped_ple_dense"] is False
    assert metrics["grad_norm_ple_sparse_preclip"] == 0.0
    assert metrics["grad_clip_coefficient_ple_sparse"] == 1.0
    assert metrics["grad_clipped_ple_sparse"] is False

    model.shared.weight.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite gradient"):
        clip_gradient_groups(model, max_norm=1.0)
