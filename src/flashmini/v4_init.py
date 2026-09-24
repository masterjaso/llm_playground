"""Authoritative FlashMini v4 initializer shared by direct construction and donor materialization.

Every learned tensor name must match exactly one rule.  Values are generated in
fp32 per name/chunk by the frozen ``flashmini-sha256-name-seed-v1`` algorithm on
a CPU ``torch.Generator`` and rounded once to BF16, the checkpoint dtype.  A
model constructed directly therefore holds exactly the BF16 values a donor
materializes from the manifest, whatever its parameter dtype.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import torch

INIT_DTYPE = torch.bfloat16
CHUNK_ELEMENTS = 1_048_576
GLOBAL_SEED = 500_277
SEED_ALGORITHM = "flashmini-sha256-name-seed-v1"

_NORMAL = {"kind": "normal", "mean": 0.0, "std": 0.02}
_ZEROS = {"kind": "zeros"}

# (rule id, full-name regex, law).  Names may carry the ``mtp.`` namespace; MTP
# mirrors the corresponding backbone rule (initialization.mtp contract).
INIT_RULES: tuple[tuple[str, re.Pattern[str], dict[str, Any]], ...] = tuple(
    (rule, re.compile(pattern), law) for rule, pattern, law in (
        ("gdn_A_log_log_uniform", r"^(mtp\.)?blocks\.\d+\.mixer\.A_log$", {"kind": "log_uniform", "low": 0.01, "high": 16.0}),
        ("gdn_dt_bias_ones", r"^(mtp\.)?blocks\.\d+\.mixer\.dt_bias$", {"kind": "ones"}),
        ("gdn_gated_norm_offset_zero", r"^(mtp\.)?blocks\.\d+\.mixer\.norm_offset$", _ZEROS),
        ("gdn_conv_normal", r"^(mtp\.)?blocks\.\d+\.mixer\.conv1d\.weight$", _NORMAL),
        ("gdn_projection_normal", r"^(mtp\.)?blocks\.\d+\.mixer\.(in_proj_qkvz|in_proj_ba|out_proj)\.weight$", _NORMAL),
        ("attention_projection_normal", r"^(mtp\.block|blocks\.\d+)\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.weight$", _NORMAL),
        ("attention_qk_norm_offset_zero", r"^(mtp\.block|blocks\.\d+)\.mixer\.(q_norm|k_norm)\.offset$", _ZEROS),
        ("hc_norm_offset_zero", r"^(mtp\.block\.|blocks\.\d+\.|mtp\.)?(mixer_hc|moe_hc|final_hc)\.norm_offset$", _ZEROS),
        ("hc_projection_normal", r"^(mtp\.block\.|blocks\.\d+\.|mtp\.)?(mixer_hc|moe_hc|final_hc)\.(read_down|read_up|write_gate)\.weight$", _NORMAL),
        ("router_normal", r"^(mtp\.block|blocks\.\d+)\.moe\.router\.weight$", _NORMAL),
        ("shared_gate_normal", r"^(mtp\.block|blocks\.\d+)\.moe\.shared_gate\.weight$", _NORMAL),
        ("routed_expert_normal", r"^(mtp\.block|blocks\.\d+)\.moe\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", _NORMAL),
        ("shared_expert_normal", r"^(mtp\.block|blocks\.\d+)\.moe\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$", _NORMAL),
        ("ple_table_normal", r"^ple\.tables\.\d+\.weight$", _NORMAL),
        ("ple_dense_projection_normal", r"^ple\.(key_proj|value_proj)\.weight$", _NORMAL),
        ("ple_grouped_norm_offset_zero", r"^ple\.(key_norm|query_norm|conv_norm)\.offset$", _ZEROS),
        ("ple_conv_exact_zero", r"^ple\.conv1d\.weight$", _ZEROS),
        ("mtp_fusion_norm_offset_zero", r"^mtp\.fusion\.(hidden_norm|embedding_norm)\.offset$", _ZEROS),
        ("mtp_fusion_projection_normal", r"^mtp\.fusion\.(fc_hidden|fc_embedding)\.weight$", _NORMAL),
        ("input_embedding_normal", r"^embed_tokens\.weight$", _NORMAL),
        ("lm_head_normal", r"^lm_head\.weight$", _NORMAL),
    )
)


def init_law(name: str) -> dict[str, Any]:
    """Return the unique init law for ``name``; fail closed on zero or several matches."""
    matches = [(rule, law) for rule, pattern, law in INIT_RULES if pattern.fullmatch(name)]
    if len(matches) != 1:
        raise KeyError(f"tensor {name!r} matches {len(matches)} init rules: {[rule for rule, _ in matches]}")
    rule, law = matches[0]
    return {"rule": rule, **law, "dtype": "bfloat16", "generator_dtype": "float32"}


def name_seed(name: str, chunk_index: int, global_seed: int = GLOBAL_SEED) -> int:
    payload = f"flashmini-init-v1\0{global_seed}\0{name}\0{chunk_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little") % (2**63 - 1)


def generate_chunk(name: str, law: dict[str, Any], chunk_index: int, count: int, *, global_seed: int = GLOBAL_SEED) -> torch.Tensor:
    """fp32 values for one chunk; RNG is always a CPU generator."""
    kind = law["kind"]
    if kind == "zeros":
        return torch.zeros(count, dtype=torch.float32)
    if kind == "ones":
        return torch.ones(count, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(name_seed(name, chunk_index, global_seed))
    if kind == "log_uniform":
        values = torch.rand(count, generator=generator, dtype=torch.float32)
        return (law["low"] + values * (law["high"] - law["low"])).log()
    if kind == "normal":
        values = torch.randn(count, generator=generator, dtype=torch.float32)
        return values * law["std"] + law["mean"]
    raise ValueError(f"unknown init law kind {kind!r}")


def fill_(target: torch.Tensor, name: str, law: dict[str, Any] | None = None, *,
          global_seed: int = GLOBAL_SEED, chunk_elements: int = CHUNK_ELEMENTS) -> torch.Tensor:
    """Fill ``target`` in place with the BF16-rounded init values for ``name``."""
    law = law or init_law(name)
    flat = target.reshape(-1)
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, flat.numel(), chunk_elements)):
            stop = min(flat.numel(), start + chunk_elements)
            values = generate_chunk(name, law, chunk_index, stop - start, global_seed=global_seed)
            flat[start:stop].copy_(values.to(INIT_DTYPE).to(device=target.device, dtype=target.dtype))
    return target


def materialize(name: str, shape: tuple[int, ...] | list[int], law: dict[str, Any] | None = None, *,
                global_seed: int = GLOBAL_SEED, chunk_elements: int = CHUNK_ELEMENTS) -> torch.Tensor:
    output = torch.empty(tuple(shape), dtype=INIT_DTYPE)
    return fill_(output, name, law, global_seed=global_seed, chunk_elements=chunk_elements)


__all__ = ["CHUNK_ELEMENTS", "GLOBAL_SEED", "INIT_DTYPE", "INIT_RULES", "SEED_ALGORITHM", "fill_", "generate_chunk", "init_law", "materialize", "name_seed"]
