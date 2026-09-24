"""Frozen v4 parameter classification for donor optimizer construction.

The init checkpoint intentionally carries no optimizer state.  This module
classifies logical operators by stable tensor name so a donor can build Muon,
AdamW, and PLE Adam groups without materializing the full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base_init_accounting import classify_tensor


MUON_OPERATOR_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "in_proj_qkvz.weight", "in_proj_ba.weight", "out_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    "key_proj.weight", "value_proj.weight", "fc_hidden.weight", "fc_embedding.weight",
)


@dataclass(frozen=True)
class ParameterClass:
    name: str
    family: str
    logical_operator: str
    shape: tuple[int, ...]
    numel: int
    weight_decay: float | None


def _operator(name: str) -> str:
    for suffix in MUON_OPERATOR_SUFFIXES:
        if name.endswith(suffix):
            return suffix.removesuffix(".weight")
    return "non_matrix_control_or_state"


def classify_parameter(name: str, shape: Iterable[int]) -> ParameterClass:
    """Classify one v4 tensor without loading its storage."""
    shape_tuple = tuple(int(value) for value in shape)
    category = classify_tensor(name)
    if category == "ple_table":
        return ParameterClass(name, "Adam", "ple_hash_table", shape_tuple, int(torch_numel(shape_tuple)), 0.0)
    if name.endswith(".shared_gate.weight"):
        return ParameterClass(name, "AdamW", "shared_expert_gate", shape_tuple, int(torch_numel(shape_tuple)), 0.1)
    operator = _operator(name)
    if operator != "non_matrix_control_or_state" and len(shape_tuple) == 2:
        return ParameterClass(name, "Muon", operator, shape_tuple, int(torch_numel(shape_tuple)), None)
    if category in {"input_embeddings", "lm_head", "routers", "shared_experts_and_gates", "hyperconnections", "ple_dense_machinery"}:
        return ParameterClass(name, "AdamW", operator, shape_tuple, int(torch_numel(shape_tuple)), 0.0 if len(shape_tuple) < 2 else 0.1)
    if category in {"attention", "gdn", "routed_experts"}:
        return ParameterClass(name, "AdamW", operator, shape_tuple, int(torch_numel(shape_tuple)), 0.0 if len(shape_tuple) < 2 else 0.1)
    return ParameterClass(name, "AdamW", operator, shape_tuple, int(torch_numel(shape_tuple)), 0.0)


def torch_numel(shape: Iterable[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def classify_parameters(manifest: Iterable[dict[str, Any]]) -> list[ParameterClass]:
    return [classify_parameter(item["name"], item["shape"]) for item in manifest]


def optimizer_contract() -> dict[str, Any]:
    return {
        "muon": {
            "momentum": 0.95,
            "nesterov": True,
            "iterations": 8,
            "epsilon": 1e-14,
            "update_scale": "0.2 * sqrt(max(A, B))",
            "logical_operator_rule": "never orthogonalize a physical fusion across gate/up/down or attention Q/K/V operators",
        },
        "adamw": {
            "parameters": "embeddings, lm_head, routers, shared gates, HC controls, attention/GDN controls, norms, scalars, biases",
        },
        "ple": {"optimizer": "Adam", "weight_decay": 0, "storage": "host/distributed off-accelerator"},
        "batch_size_warmup": False,
    }


__all__ = ["ParameterClass", "classify_parameter", "classify_parameters", "optimizer_contract"]
