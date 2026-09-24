"""Authoritative FlashMini-50B v4 parameter and active-work accounting."""

from __future__ import annotations

from typing import Any

import torch

from .base_init_config import FlashMini50BConfig
from .base_init_model import FlashMini50BBaseInit


_BASE_CATEGORIES = (
    "input_embeddings", "lm_head", "attention", "gdn", "hyperconnections",
    "routers", "routed_experts", "shared_experts_and_gates", "ple_table",
    "ple_dense_machinery", "other_base",
)
_ALL_CATEGORIES = _BASE_CATEGORIES + tuple(f"mtp_{name}" for name in _BASE_CATEGORIES)


def _is_attention_tensor(name: str) -> bool:
    return any(token in name for token in (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".q_norm.", ".k_norm."))


def _is_gdn_tensor(name: str) -> bool:
    return any(token in name for token in (
        ".in_proj_qkvz.", ".in_proj_ba.", ".dt_bias", ".A_log", ".norm_offset", ".conv1d.", ".out_proj."
    )) and not _is_attention_tensor(name)


def classify_tensor(name: str) -> str:
    """Classify a v4 named parameter without relying on construction order."""
    namespace = "mtp" if name.startswith("mtp.") else "base"
    local = name[4:] if namespace == "mtp" else name
    if local.startswith("lm_head."):
        category = "lm_head"
    elif local.startswith("embed_tokens."):
        category = "input_embeddings"
    elif local.startswith("ple.tables."):
        category = "ple_table"
    elif local.startswith("ple."):
        category = "ple_dense_machinery"
    elif _is_attention_tensor(local):
        category = "attention"
    elif _is_gdn_tensor(local):
        category = "gdn"
    elif local.startswith("fusion.") or "hyper" in local or ".hc." in local:
        category = "hyperconnections"
    elif ".router." in local:
        category = "routers"
    elif ".experts." in local:
        category = "routed_experts"
    elif ".shared_expert" in local or ".shared_gate." in local:
        category = "shared_experts_and_gates"
    elif any(token in local for token in ("hyper", ".hc.", "hc.")):
        category = "hyperconnections"
    else:
        category = "other_base"
    return category if namespace == "base" else f"mtp_{category}"


def build_meta_model(config: FlashMini50BConfig) -> FlashMini50BBaseInit:
    with torch.device("meta"):
        return FlashMini50BBaseInit(config)


def _empty_counts(names: tuple[str, ...]) -> dict[str, int]:
    return {name: 0 for name in names}


def parameter_report(config: FlashMini50BConfig, model: FlashMini50BBaseInit | None = None) -> dict[str, Any]:
    """Return exact base, MTP, total, category, and active-per-token counts."""
    model = model or build_meta_model(config)
    categories = _empty_counts(_ALL_CATEGORIES)
    tensors: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = classify_tensor(name)
        count = int(parameter.numel())
        categories[category] += count
        tensors.append({"name": name, "shape": list(parameter.shape), "numel": count, "category": category})
    base_total = sum(categories[name] for name in _BASE_CATEGORIES)
    mtp_total = sum(categories[name] for name in _ALL_CATEGORIES if name.startswith("mtp_"))
    if sum(categories.values()) != base_total + mtp_total:
        raise ValueError("v4 category counts do not reconcile")
    if categories["other_base"] or any(value for name, value in categories.items() if name.startswith("mtp_other_base")):
        raise ValueError(f"unclassified v4 tensors remain: {categories}")

    d = config.d_model
    ple = config.section("ple")
    gdn = config.section("gdn")
    attention = config.section("attention")
    hc = config.section("hyperconnections")
    moe = config.section("moe")
    base_tensor_count = sum(1 for tensor in tensors if not tensor["name"].startswith("mtp."))
    mtp_tensor_count = sum(1 for tensor in tensors if tensor["name"].startswith("mtp."))
    # Input embeddings contribute one row per token; the untied LM head is a
    # dense matrix used for every token. PLE contributes one row per head. The
    # routed-expert fraction is exact integer arithmetic, not a float estimate.
    active = {
        "input_embedding_row": d,
        "lm_head": categories["lm_head"],
        "attention": categories["attention"],
        "gdn": categories["gdn"],
        "hyperconnections": categories["hyperconnections"],
        "routers": categories["routers"],
        "routed_experts": categories["routed_experts"] * moe["top_k"] // moe["routed_experts"],
        "shared_experts_and_gates": categories["shared_experts_and_gates"],
        "ple_table": ple["total_heads"] * ple["head_dim"],
        "ple_dense_machinery": categories["ple_dense_machinery"],
    }
    active_total = sum(active.values())
    attention_roles = {
        "source_layers_one_based": [source + 1 for source, _ in config.kvc_pairs],
        "reuse_layers_one_based": [reuse + 1 for _, reuse in config.kvc_pairs],
        "reuse_has_no_kv": True,
    }
    return {
        "model_id": "FlashMini-50B-Base",
        "checkpoint_id": "FlashMini-50B-Base-Init-v1",
        "architecture_version": 4,
        "base_total_learned_parameters": base_total,
        "mtp_total_learned_parameters": mtp_total,
        "checkpoint_total_learned_parameters": base_total + mtp_total,
        "base_tensor_count": base_tensor_count,
        "mtp_tensor_count": mtp_tensor_count,
        "base_active_parameters_per_token": active_total,
        "active_parameters_per_token_breakdown": active,
        "active_definition": "one input embedding row; full untied LM head; all attention/GDN/HC/router/shared/dense PLE tensors; top-6 routed experts; one PLE row per hash head",
        "categories": {name: value for name, value in categories.items() if value},
        "categories_sum": sum(categories.values()),
        "architecture": {
            "d_model": d, "decoder_layers": config.num_layers,
            "attention_query_heads": attention["query_heads"],
            "attention_kv_heads": attention["kv_heads"],
            "attention_head_dim": attention["head_dim"],
            "gdn_key_query_heads": gdn["key_query_heads"],
            "gdn_value_heads": gdn["value_heads"],
            "gdn_key_head_dim": gdn["key_head_dim"],
            "gdn_value_head_dim": gdn["value_head_dim"],
            "hc_streams": hc["streams"], "hc_lowrank": hc["lowrank"],
            "moe_routed_experts": moe["routed_experts"], "moe_shared_experts": moe["shared_experts"],
            "moe_top_k": moe["top_k"], "ple_heads": ple["total_heads"],
            "ple_table_rows": ple["hash_head_rows"],
        },
        "kvc": attention_roles,
        "tensors": tensors,
    }


def category_summary(report: dict[str, Any]) -> dict[str, int]:
    return dict(report["categories"])


def compare_target(report: dict[str, Any], target: int = 50_277_000_000, tolerance: float = 0.005) -> dict[str, Any]:
    actual = int(report["base_total_learned_parameters"])
    difference = actual - target
    fraction = difference / target
    return {
        "target_base_parameters": target,
        "actual_base_parameters": actual,
        "difference": difference,
        "relative_difference": fraction,
        "tolerance": tolerance,
        "within_tolerance": abs(fraction) <= tolerance,
    }


__all__ = ["build_meta_model", "category_summary", "classify_tensor", "compare_target", "parameter_report"]
