from __future__ import annotations

import json

import pytest

from dense2moe.science.v23_designs import (
    DENSE_INTERMEDIATE_SIZE,
    DESIGN_A,
    DESIGN_B,
    DESIGN_C,
    DESIGN_D,
    DESIGN_E,
    MIN_ACTIVE_FFN_REDUCTION,
    derive_expert_width,
    design_from_mapping,
    get_design,
    is_capacity_valid,
    iter_designs,
    registry_as_dict,
    registry_to_json,
    validate_design_registry,
)


def test_all_five_designs_have_declared_strategy_and_valid_capacity() -> None:
    designs = validate_design_registry()

    assert [design.design_id for design in designs] == ["A", "B", "C", "D", "E"]
    assert [design.topology_id for design in designs] == [
        "p16/top6",
        "p16/top5",
        "p16/top4",
        "p32/top8",
        "p16/top5",
    ]
    assert all(design.total_capacity == DENSE_INTERMEDIATE_SIZE for design in designs)
    assert all(design.active_ffn_reduction >= MIN_ACTIVE_FFN_REDUCTION for design in designs)
    assert DESIGN_A.loss.norm_aware
    assert DESIGN_B.loss.covariance_partition and DESIGN_B.router.load_penalty == "hard_load"
    assert DESIGN_C.residual.kind == "low_rank_silu"
    assert DESIGN_D.loss.load_aware
    assert DESIGN_E.router.nonlinear and DESIGN_E.loss.mixed_amplitude_supervision


def test_width_derivation_and_bounded_geometry_retry_are_capacity_valid() -> None:
    assert derive_expert_width(17_408, 16, 2_048) == 960
    assert derive_expert_width(17_408, 32, 2_048) == 480
    assert is_capacity_valid(17_408, 16, 2_048, 960)
    assert not is_capacity_valid(17_408, 16, 2_048, 961)

    retry = DESIGN_A.with_geometry(shared_intermediate_size=1_024, top_k=6)
    assert retry.expert_intermediate_size == 1_024
    assert retry.total_capacity == DENSE_INTERMEDIATE_SIZE
    assert retry.active_ffn_reduction >= 0.50


def test_registry_receipt_is_deterministic_and_round_trips() -> None:
    payload = registry_as_dict()
    encoded = registry_to_json()

    assert json.loads(encoded) == payload
    assert encoded == registry_to_json()
    restored = design_from_mapping(payload["designs"][2])
    assert restored.as_dict() == get_design("C").as_dict()
    assert tuple(iter_designs()) == validate_design_registry()


def test_design_validation_rejects_bad_capacity_and_reduction() -> None:
    with pytest.raises(ValueError, match="capacity"):
        DESIGN_A.with_geometry(expert_intermediate_size=961)

    with pytest.raises(ValueError, match="below"):
        DESIGN_A.with_geometry(shared_intermediate_size=8_192, top_k=6)
