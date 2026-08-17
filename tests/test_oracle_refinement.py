from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
from dense2moe.partition import partition_indices
from dense2moe.training.oracle_refinement import (
    oracle_assignments,
    oracle_routed_forward,
    train_oracle_routed_basis,
)


def _model(*, experts: int = 3, top_k: int = 2) -> TorchQwen35SwiGLUMoE:
    rng = np.random.default_rng(17)
    hidden = 5
    shared = 1
    width = 2
    intermediate = shared + experts * width
    return TorchQwen35SwiGLUMoE.from_dense(
        rng.normal(size=(intermediate, hidden)).astype("float32"),
        rng.normal(size=(intermediate, hidden)).astype("float32"),
        rng.normal(size=(hidden, intermediate)).astype("float32"),
        routed_experts=experts,
        shared_intermediate_size=shared,
        top_k=top_k,
        routing_mode="independent_positive",
        partition=partition_indices(intermediate, experts, width, shared),
        learnable_scales=True,
    )


def test_oracle_assignments_bypass_selector_and_fit_positive_coefficients() -> None:
    model = _model()
    inputs = torch.randn(3, 5)
    with torch.no_grad():
        values = inputs.reshape(-1, 5)
        import torch.nn.functional as F

        shared_value = model.shared_down_proj(
            F.silu(model.shared_gate_proj(values)) * model.shared_up_proj(values)
        )
        routed_value = torch.stack(
            [
                model._expert_output(values, expert) * model.expert_scales[expert]
                for expert in range(model.routed_experts)
            ],
            dim=1,
        )
        target = shared_value + 2.0 * routed_value[:, 0] + 3.0 * routed_value[:, 1]

    original_router = model.router

    class _FailingRouter(torch.nn.Module):
        def forward(self, _inputs: object) -> object:
            raise AssertionError("oracle E/M path called learned selector")

    model.router = _FailingRouter()  # type: ignore[assignment]
    assignment = oracle_assignments(model, inputs, target, max_combinations=100)
    prediction = oracle_routed_forward(model, inputs, assignment)
    assert assignment.method == "exact_all_combinations"
    assert assignment.coefficient_constraint == "nonnegative"
    assert float(torch.mean((prediction - target).square())) < 1e-8
    model.router = original_router


def test_p32_style_assignment_is_explicitly_bounded() -> None:
    model = _model(experts=8, top_k=3)
    inputs = torch.randn(2, 5)
    target = torch.randn(2, 5)
    assignment = oracle_assignments(
        model,
        inputs,
        target,
        candidate_pool_size=5,
        max_combinations=4,
        max_candidate_bytes=1 << 20,
    )
    assert assignment.method == "bounded_correlation_candidate_pool"
    assert assignment.candidate_count <= 4
    assert assignment.effective_candidate_pool_size == 5
    assert assignment.indices.shape == (2, 3)
    assert assignment.coefficients.shape == (2, 3)
    assert torch.isfinite(assignment.coefficients).all()
    assert (assignment.coefficients >= 0).all()


def test_training_updates_basis_only_and_reopens_callable_per_epoch() -> None:
    model = _model()
    inputs = torch.randn(4, 5)
    target = torch.randn(4, 5)
    router_before = {
        name: value.detach().clone() for name, value in model.router.state_dict().items()
    }
    amplitude_before = {
        name: value.detach().clone() for name, value in model.amplitude_router.state_dict().items()
    }
    calls = 0

    def batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
        nonlocal calls
        calls += 1
        return [(inputs, target)]

    result = train_oracle_routed_basis(
        model,
        batches,
        epochs=2,
        learning_rate=1e-3,
        assignment_refresh_steps=1,
        m_step_repeats=2,
        device="cpu",
    )
    assert result["status"] == "ORACLE_ROUTED_BASIS_REFINEMENT_COMPLETE"
    assert result["updates"] == 4
    assert result["assignment_refreshes"] == 2
    assert calls == 2
    assert result["selector_frozen"] is True
    for name, value in model.router.state_dict().items():
        assert torch.equal(router_before[name], value)
    for name, value in model.amplitude_router.state_dict().items():
        assert torch.equal(amplitude_before[name], value)


def test_corpus_gate_requires_v2_frozen_success_checks(tmp_path: Path) -> None:
    from scripts.run_oracle_routed_basis_refinement import require_frozen_corpus_v2

    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({"receipt_type": "dense2moe-corpus-receipt", "schema_version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corpus_version"):
        require_frozen_corpus_v2(old)

    receipt = tmp_path / "v2.json"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"records": {"FIT-TRAIN": ["a"]}}), encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "corpus_version": "corpus-v2",
                "status": "CORPUS_V2_FROZEN",
                "manifest": {
                    "path": str(manifest),
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "records": 1,
                },
                "split_identity": {
                    "path": str(splits),
                    "sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
                    "record_ids_sha256": hashlib.sha256(b"a").hexdigest(),
                },
                "checks": {
                    "provenance": "PASS",
                    "repo_document_overlap": "PASS",
                    "benchmark_denylist": "PASS",
                },
            }
        ),
        encoding="utf-8",
    )
    assert require_frozen_corpus_v2(receipt)["corpus_version"] == "corpus-v2"

    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            {
                "corpus_version": "corpus-v2",
                "status": "CORPUS_V2_FROZEN",
                "checks": {
                    "provenance": "PASS",
                    "repo_document_overlap": "PASS",
                    "benchmark_denylist": "PASS",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hashed corpus manifest"):
        require_frozen_corpus_v2(forged)
