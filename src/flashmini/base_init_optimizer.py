"""Frozen v4 optimizer taxonomy at the logical-operator level.

Every learned tensor is partitioned into one or more *logical slices* (row sets of
dimension 0).  A physical tensor that fuses several operators (attention
``q_proj`` = query + output gate; GDN ``in_proj_qkvz`` = q/k/v/z; GDN
``in_proj_ba`` = beta/decay-control) yields one slice per operator, so Muon never
orthogonalizes across operators.  Classification fails closed: an unknown name,
a slice set that does not partition the rows exactly, or a Muon slice that is
not a genuine 2-D transformation raises.

Families: ``Muon`` (genuine learned 2-D transformations), ``AdamW`` (embeddings,
LM head, routers, attention/GDN output gates, shared gates, HC controls,
GDN beta/decay controls, norms, scalars, short convolutions), ``PLE_Adam``
(hash-head tables, weight decay 0).
AdamW weight-decay *classes* are classified here; their values are donor
training parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .base_init_accounting import classify_tensor
from .base_init_config import FlashMini50BConfig

MUON, ADAMW, PLE_ADAM = "Muon", "AdamW", "PLE_Adam"
DECAY_CLASSES = ("muon_matrix", "embedding", "control_matrix", "no_decay", "ple_table")


@dataclass(frozen=True)
class LogicalSlice:
    """Rows ``[b * block + offset, b * block + offset + width)`` for ``b in range(blocks)``."""

    tensor: str
    logical_operator: str
    family: str
    decay_class: str
    block: int
    offset: int
    width: int
    blocks: int
    columns: int

    @property
    def rows(self) -> int:
        return self.width * self.blocks

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    def row_index(self, device: torch.device | str | None = None) -> torch.Tensor:
        base = torch.arange(self.blocks, device=device) * self.block + self.offset
        return (base[:, None] + torch.arange(self.width, device=device)[None, :]).reshape(-1)

    def as_dict(self) -> dict[str, Any]:
        return {"logical_operator": self.logical_operator, "family": self.family, "decay_class": self.decay_class,
                "block": self.block, "offset": self.offset, "width": self.width, "blocks": self.blocks,
                "rows": self.rows, "columns": self.columns}


@dataclass(frozen=True)
class ParameterClass:
    name: str
    family: str
    logical_operator: str
    shape: tuple[int, ...]
    numel: int
    decay_class: str
    slices: tuple[LogicalSlice, ...]


def _whole(name: str, shape: tuple[int, ...], operator: str, family: str, decay: str) -> tuple[LogicalSlice, ...]:
    rows = shape[0]
    columns = int(torch.Size(shape[1:]).numel()) if len(shape) > 1 else 1
    return (LogicalSlice(name, operator, family, decay, rows, 0, rows, 1, columns),)


def _interleaved(name: str, shape: tuple[int, ...], blocks: int, parts: list[tuple[str, int]], family: str, decay: str) -> tuple[LogicalSlice, ...]:
    return _interleaved_mixed(name, shape, blocks, [(operator, width, family, decay) for operator, width in parts])


def _interleaved_mixed(
    name: str,
    shape: tuple[int, ...],
    blocks: int,
    parts: list[tuple[str, int, str, str]],
) -> tuple[LogicalSlice, ...]:
    block = sum(width for _, width, _, _ in parts)
    if block * blocks != shape[0]:
        raise ValueError(f"{name}: fused layout {[(op, width) for op, width, _, _ in parts]} x {blocks} does not cover {shape[0]} rows")
    slices, offset = [], 0
    for operator, width, family, decay in parts:
        slices.append(LogicalSlice(name, operator, family, decay, block, offset, width, blocks, shape[1]))
        offset += width
    return tuple(slices)


class OptimizerTaxonomy:
    """Name -> logical slices for one v4 geometry (production or surrogate)."""

    def __init__(self, config: FlashMini50BConfig):
        attention, gdn = config.section("attention"), config.section("gdn")
        self.q_heads, self.head_dim = attention["query_heads"], attention["head_dim"]
        self.k_heads, self.head_k = gdn["key_query_heads"], gdn["key_head_dim"]
        self.v_ratio, self.head_v = gdn["value_heads"] // gdn["key_query_heads"], gdn["value_head_dim"]

    def classify(self, name: str, shape: Iterable[int]) -> ParameterClass:
        shape = tuple(int(value) for value in shape)
        slices = self._slices(name, shape)
        covered = torch.zeros(shape[0], dtype=torch.int64)
        for item in slices:
            covered.index_add_(0, item.row_index(), torch.ones(item.rows, dtype=torch.int64))
            if item.family == MUON and (len(shape) != 2 or min(item.matrix_shape) < 1):
                raise ValueError(f"{name}: Muon slice {item.logical_operator} is not a 2-D transformation")
        if not bool((covered == 1).all()):
            raise ValueError(f"{name}: logical slices do not partition the tensor rows exactly")
        families = {item.family for item in slices}
        family = next(iter(families)) if len(families) == 1 else "mixed"
        decay_classes = {item.decay_class for item in slices}
        decay_class = next(iter(decay_classes)) if len(decay_classes) == 1 else "mixed"
        operator = slices[0].logical_operator if len(slices) == 1 else "+".join(item.logical_operator for item in slices)
        numel = int(torch.Size(shape).numel())
        return ParameterClass(name, family, operator, shape, numel, decay_class, slices)

    def _slices(self, name: str, shape: tuple[int, ...]) -> tuple[LogicalSlice, ...]:
        local = name[4:] if name.startswith("mtp.") else name
        prefix = "mtp_" if name.startswith("mtp.") else ""
        if re.fullmatch(r"ple\.tables\.\d+\.weight", local):
            return _whole(name, shape, "ple_hash_table", PLE_ADAM, "ple_table")
        rules: list[tuple[str, Any]] = [
            (r"(block|blocks\.\d+)\.mixer\.q_proj\.weight", lambda: _interleaved_mixed(name, shape, self.q_heads, [
                (prefix + "attention_query", self.head_dim, MUON, "muon_matrix"),
                (prefix + "attention_output_gate", self.head_dim, ADAMW, "control_matrix"),
            ])),
            (r"(block|blocks\.\d+)\.mixer\.k_proj\.weight", lambda: _whole(name, shape, prefix + "attention_key", MUON, "muon_matrix")),
            (r"(block|blocks\.\d+)\.mixer\.v_proj\.weight", lambda: _whole(name, shape, prefix + "attention_value", MUON, "muon_matrix")),
            (r"(block|blocks\.\d+)\.mixer\.o_proj\.weight", lambda: _whole(name, shape, prefix + "attention_output", MUON, "muon_matrix")),
            (r"blocks\.\d+\.mixer\.in_proj_qkvz\.weight", lambda: _interleaved_mixed(name, shape, self.k_heads, [
                ("gdn_query", self.head_k, MUON, "muon_matrix"),
                ("gdn_key", self.head_k, MUON, "muon_matrix"),
                ("gdn_value", self.v_ratio * self.head_v, MUON, "muon_matrix"),
                ("gdn_output_gate_z", self.v_ratio * self.head_v, ADAMW, "control_matrix"),
            ])),
            (r"blocks\.\d+\.mixer\.in_proj_ba\.weight", lambda: _interleaved(name, shape, self.k_heads, [("gdn_beta", self.v_ratio), ("gdn_decay_a", self.v_ratio)], ADAMW, "control_matrix")),
            (r"blocks\.\d+\.mixer\.out_proj\.weight", lambda: _whole(name, shape, "gdn_output", MUON, "muon_matrix")),
            (r"blocks\.\d+\.mixer\.(A_log|dt_bias)", lambda: _whole(name, shape, "gdn_decay_control", ADAMW, "no_decay")),
            (r"blocks\.\d+\.mixer\.norm_offset", lambda: _whole(name, shape, "gdn_gated_norm", ADAMW, "no_decay")),
            (r"blocks\.\d+\.mixer\.conv1d\.weight", lambda: _whole(name, shape, "gdn_short_conv", ADAMW, "no_decay")),
            (r"(block|blocks\.\d+)\.mixer\.(q_norm|k_norm)\.offset", lambda: _whole(name, shape, prefix + "attention_qk_norm", ADAMW, "no_decay")),
            (r"(block|blocks\.\d+)\.moe\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight", lambda: _whole(name, shape, prefix + "routed_expert_" + name.split(".")[-2], MUON, "muon_matrix")),
            (r"(block|blocks\.\d+)\.moe\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight", lambda: _whole(name, shape, prefix + "shared_expert_" + name.split(".")[-2], MUON, "muon_matrix")),
            (r"(block|blocks\.\d+)\.moe\.router\.weight", lambda: _whole(name, shape, prefix + "router", ADAMW, "control_matrix")),
            (r"(block|blocks\.\d+)\.moe\.shared_gate\.weight", lambda: _whole(name, shape, prefix + "shared_expert_gate", ADAMW, "control_matrix")),
            (r"((block|blocks\.\d+)\.)?(mixer_hc|moe_hc|final_hc)\.(read_down|read_up|write_gate)\.weight", lambda: _whole(name, shape, prefix + "hc_" + name.split(".")[-2], ADAMW, "control_matrix")),
            (r"((block|blocks\.\d+)\.)?(mixer_hc|moe_hc|final_hc)\.norm_offset", lambda: _whole(name, shape, prefix + "hc_norm", ADAMW, "no_decay")),
            (r"ple\.(key_proj|value_proj)\.weight", lambda: _whole(name, shape, "ple_" + name.split(".")[-2], MUON, "muon_matrix")),
            (r"ple\.(key_norm|query_norm|conv_norm)\.offset", lambda: _whole(name, shape, "ple_norm", ADAMW, "no_decay")),
            (r"ple\.conv1d\.weight", lambda: _whole(name, shape, "ple_short_conv", ADAMW, "no_decay")),
            (r"fusion\.(fc_hidden|fc_embedding)\.weight", lambda: _whole(name, shape, "mtp_fusion_" + name.split(".")[-2], MUON, "muon_matrix")),
            (r"fusion\.(hidden_norm|embedding_norm)\.offset", lambda: _whole(name, shape, "mtp_fusion_norm", ADAMW, "no_decay")),
            (r"embed_tokens\.weight", lambda: _whole(name, shape, "input_embedding", ADAMW, "embedding")),
            (r"lm_head\.weight", lambda: _whole(name, shape, "lm_head", ADAMW, "embedding")),
        ]
        matches = [build for pattern, build in rules if re.fullmatch(pattern, local)]
        if len(matches) != 1:
            raise KeyError(f"unclassified or ambiguous v4 optimizer tensor {name!r} ({len(matches)} rules)")
        classify_tensor(name)
        return matches[0]()


_DEFAULT_TAXONOMY: OptimizerTaxonomy | None = None


def default_taxonomy() -> OptimizerTaxonomy:
    global _DEFAULT_TAXONOMY
    if _DEFAULT_TAXONOMY is None:
        from .base_init_config import load_config

        _DEFAULT_TAXONOMY = OptimizerTaxonomy(load_config())
    return _DEFAULT_TAXONOMY


def classify_parameter(name: str, shape: Iterable[int], taxonomy: OptimizerTaxonomy | None = None) -> ParameterClass:
    return (taxonomy or default_taxonomy()).classify(name, shape)


def classify_parameters(manifest: Iterable[dict[str, Any]], taxonomy: OptimizerTaxonomy | None = None) -> list[ParameterClass]:
    return [classify_parameter(item["name"], item["shape"], taxonomy) for item in manifest]


def optimizer_contract() -> dict[str, Any]:
    return {
        "muon": {
            "parameters": "genuine 2-D learned transformations, one Newton-Schulz orthogonalization per logical slice",
            "momentum": 0.95, "nesterov": True, "iterations": 8, "epsilon": 1e-14,
            "update_scale": "0.2 * sqrt(max(rows, columns)) of the logical slice",
            "logical_operator_rule": "physical_fusion_allowed_but_no_cross_operator_orthogonalization",
        },
        "adamw": {
            "parameters": "embeddings, lm_head, routers, attention/GDN output gates, shared gates, HC controls, GDN beta/decay controls, norms, scalars, short convolutions",
            "weight_decay_classes": ["embedding", "control_matrix", "no_decay"],
        },
        "ple": {"optimizer": "Adam", "weight_decay": 0, "update": "row-sparse, per-row step count",
                "storage": "host BF16 master tables; fp32 moments on owner ranks"},
        "batch_size_warmup": False,
        "hyperparameters": "donor training configuration (explicit; startup fails if unset)",
    }


__all__ = [
    "ADAMW", "DECAY_CLASSES", "LogicalSlice", "MUON", "OptimizerTaxonomy", "PLE_ADAM", "ParameterClass",
    "classify_parameter", "classify_parameters", "default_taxonomy", "optimizer_contract",
]
