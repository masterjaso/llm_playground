"""Frozen FlashMini-50B-Base architecture version 4 contract.

The v4 schema is intentionally separate from the historical v2/v3 configuration.
Every field consumed by the model, accountant, manifest, or materializer is
required in the YAML artifact; production construction fails closed on missing
or unknown fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

MODEL_ID = "FlashMini-50B-Base"
CHECKPOINT_ID = "FlashMini-50B-Base-Init-v1"
ARCHITECTURE_VERSION = 4
DEFAULT_CONFIG_PATH = Path("configs/flashmini/flashmini_50b_base_init_v1.yaml")

_ATTENTION_LAYERS_0B = [3, 7, 11, 15, 19, 23, 28, 33, 38, 43]
_KVC_PAIRS_0B = [[3, 7], [11, 15], [19, 23], [28, 33], [38, 43]]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


# Fields excluded from the architecture fingerprint: training/inference policy,
# storage, initialization, optimizer, and freeze metadata.  Everything else in
# these sections defines forward semantics or tensor geometry.
_ARCHITECTURE_SECTION_EXCLUSIONS = {
    "architecture": {"parameter_target"},
    "mixer": set(),
    "attention": set(),
    "kvc": {"inference_representation"},
    "gdn": set(),
    "moe": set(),
    "hyperconnections": set(),
    "ple": {"training_storage", "inference_storage"},
    "mtp": {"teacher_forcing", "main_loss_weight", "auxiliary_loss", "coefficient_schedule", "inference_precision"},
}


def architecture_fingerprint_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Architecture-defining fields only (geometry, topology, sharing, forward semantics)."""
    payload: dict[str, Any] = {
        "model_id": raw["model_id"],
        "architecture_version": raw["architecture_version"],
        "tokenizer_geometry": {
            "family": raw["tokenizer"]["family"],
            "vocab_size": raw["tokenizer"]["vocab_size"],
            "ple_reset_eos_id": raw["tokenizer"]["special_token_ids"].get("eos"),
        },
    }
    for section, excluded in _ARCHITECTURE_SECTION_EXCLUSIONS.items():
        payload[section] = {key: value for key, value in raw[section].items() if key not in excluded}
    return payload


def _require_exact_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ValueError(f"{path} fields mismatch; missing={missing}, unknown={unknown}")


def _validate_positive_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")
    return value


class FlashMini50BConfig:
    """Validated mapping wrapper for the immutable v4 architecture."""

    def __init__(self, raw: Mapping[str, Any]):
        self._raw = copy.deepcopy(dict(raw))
        self._validate()

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    @property
    def architecture_version(self) -> int:
        return int(self._raw["architecture_version"])

    @property
    def d_model(self) -> int:
        return int(self._raw["architecture"]["d_model"])

    @property
    def vocab_size(self) -> int:
        return int(self._raw["tokenizer"]["vocab_size"])

    @property
    def num_layers(self) -> int:
        return int(self._raw["architecture"]["decoder_layers"])

    @property
    def ple_injection_layer(self) -> int:
        return int(self._raw["ple"]["injection_layer_zero_based"])

    @property
    def attention_layers(self) -> list[int]:
        return list(self._raw["mixer"]["attention_layers_zero_based"])

    @property
    def kvc_pairs(self) -> list[list[int]]:
        return copy.deepcopy(self._raw["kvc"]["pairs_zero_based"])

    @property
    def ple_table_sizes(self) -> list[int]:
        return list(self._raw["ple"]["hash_head_rows"])

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self._raw)

    @property
    def architecture_sha256(self) -> str:
        return canonical_sha256(architecture_fingerprint_payload(self._raw))

    def section(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._raw[name])

    def _validate(self) -> None:
        root_required = {
            "schema_version", "model_id", "checkpoint_id", "architecture_version",
            "tokenizer", "architecture", "mixer", "attention", "kvc", "gdn", "moe",
            "hyperconnections", "ple", "mtp", "initialization", "optimizer", "storage",
            "architecture_contract",
        }
        _require_exact_keys(self._raw, root_required, "config")
        if self._raw["schema_version"] != 1:
            raise ValueError("v4 config schema_version must be 1")
        if self._raw["model_id"] != MODEL_ID or self._raw["checkpoint_id"] != CHECKPOINT_ID:
            raise ValueError("v4 model/checkpoint identity mismatch")
        if self._raw["architecture_version"] != ARCHITECTURE_VERSION:
            raise ValueError("v4 architecture_version must be 4")

        tokenizer = self._raw["tokenizer"]
        _require_exact_keys(tokenizer, {
            "family", "vocab_size", "special_token_ids", "special_token_slots", "identity_status",
            "freeze_before_optimizer_step", "manifest", "fingerprint_algorithm", "fingerprint",
        }, "tokenizer")
        if tokenizer["family"] != "custom_byte_level_bpe" or tokenizer["vocab_size"] != 131_072:
            raise ValueError("v4 requires custom byte-level BPE with 131072 vocabulary slots")
        if tokenizer["identity_status"] != "frozen":
            raise ValueError("v4 tokenizer must be frozen before training")
        if tokenizer["freeze_before_optimizer_step"] is not True:
            raise ValueError("tokenizer must freeze before the first optimizer step")
        if tokenizer["fingerprint_algorithm"] != "sha256_canonical_tokenizer_json_v1":
            raise ValueError("v4 tokenizer fingerprint algorithm mismatch")
        fingerprint = tokenizer["fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("tokenizer.fingerprint must be lowercase SHA256 hex")
        special = tokenizer["special_token_ids"]
        slots = tokenizer["special_token_slots"]
        if not isinstance(special, dict) or not {"eos", "pad"} <= set(special):
            raise ValueError("tokenizer.special_token_ids must define at least eos and pad")
        ids = list(special.values())
        if len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < slots for i in ids):
            raise ValueError("tokenizer special IDs must be distinct integers inside the special slot range")

        architecture = self._raw["architecture"]
        _require_exact_keys(architecture, {
            "d_model", "decoder_layers", "tie_word_embeddings", "input_embedding_shape",
            "lm_head_shape", "native_context", "final_mixer", "parameter_target"
        }, "architecture")
        d = _validate_positive_int(architecture, "d_model", "architecture")
        layers = _validate_positive_int(architecture, "decoder_layers", "architecture")
        if (d, layers) != (2048, 48):
            raise ValueError("frozen base geometry requires d_model=2048 and 48 decoder layers")
        if architecture["tie_word_embeddings"] is not False:
            raise ValueError("v4 input embedding and LM head must be untied")
        if architecture["input_embedding_shape"] != [131_072, 2048]:
            raise ValueError("v4 input embedding shape mismatch")
        if architecture["lm_head_shape"] != [2048, 131_072]:
            raise ValueError("v4 LM head shape mismatch")
        if architecture["native_context"] != 262_144 or architecture["final_mixer"] != "read_only_hc_8192_to_2048":
            raise ValueError("v4 context/final mixer contract mismatch")
        target = architecture["parameter_target"]
        _require_exact_keys(target, {"base_approximate", "tolerance_fraction", "resize_prohibited"}, "architecture.parameter_target")
        if target["base_approximate"] != 50_277_000_000 or target["tolerance_fraction"] != 0.005 or target["resize_prohibited"] is not True:
            raise ValueError("v4 parameter reconciliation gate mismatch")

        mixer = self._raw["mixer"]
        _require_exact_keys(mixer, {
            "gated_delta_net_layers", "full_attention_layers", "attention_layers_one_based",
            "attention_layers_zero_based", "tail_rule"
        }, "mixer")
        if mixer["gated_delta_net_layers"] != 38 or mixer["full_attention_layers"] != 10:
            raise ValueError("v4 requires 38 GDN and 10 full-attention layers")
        if mixer["attention_layers_zero_based"] != _ATTENTION_LAYERS_0B:
            raise ValueError("v4 attention layer positions mismatch")
        if mixer["attention_layers_one_based"] != [x + 1 for x in _ATTENTION_LAYERS_0B]:
            raise ValueError("v4 one-based attention layer positions mismatch")
        if mixer["tail_rule"] != "layers_45_through_48_are_gated_delta_net":
            raise ValueError("v4 final four layers must remain GDN")

        attention = self._raw["attention"]
        _require_exact_keys(attention, {
            "query_heads", "kv_heads", "head_dim", "query_width", "kv_projection_width",
            "partial_rotary_fraction", "rotary_dimensions", "output_gated", "rope_theta",
            "bias", "norm_offsets"
        }, "attention")
        expected_attention = {
            "query_heads": 16, "kv_heads": 2, "head_dim": 256, "query_width": 4096,
            "kv_projection_width": 512, "partial_rotary_fraction": 0.25,
            "rotary_dimensions": 64, "output_gated": True, "rope_theta": 10_000_000,
            "bias": False, "norm_offsets": "zero_identity_semantics",
        }
        if attention != expected_attention:
            raise ValueError("v4 full-attention geometry mismatch")

        kvc = self._raw["kvc"]
        _require_exact_keys(kvc, {
            "enabled", "share_group_size", "pairs_one_based", "pairs_zero_based",
            "kv_bits", "kv_format", "scale_format", "scale_group_size", "qat",
            "training_representation", "inference_representation"
        }, "kvc")
        if kvc["enabled"] is not True or kvc["share_group_size"] != 2 or kvc["pairs_zero_based"] != _KVC_PAIRS_0B:
            raise ValueError("v4 KVC must use all five frozen source/reuse pairs")
        if kvc["pairs_one_based"] != [[x + 1 for x in pair] for pair in _KVC_PAIRS_0B]:
            raise ValueError("v4 KVC one-based pairs mismatch")
        if (kvc["kv_bits"], kvc["kv_format"], kvc["scale_format"], kvc["scale_group_size"], kvc["qat"]) != (4, "e2m1", "e4m3", 16, True):
            raise ValueError("v4 KVC quantization contract mismatch")
        if kvc["training_representation"] != "fake_quantize_dequantize_straight_through":
            raise ValueError("v4 KVC training representation mismatch")
        if kvc["inference_representation"] != "packed_fp4_e4m3_scales_fused_or_on_demand_dequantize":
            raise ValueError("v4 KVC inference representation mismatch")

        gdn = self._raw["gdn"]
        _require_exact_keys(gdn, {
            "key_query_heads", "key_head_dim", "value_heads", "value_head_dim",
            "short_conv_kernel", "residual_owner", "recurrent_arithmetic", "projection_layout",
            "special_state_shapes"
        }, "gdn")
        if (gdn["key_query_heads"], gdn["key_head_dim"], gdn["value_heads"], gdn["value_head_dim"]) != (16, 128, 32, 128):
            raise ValueError("v4 GDN head geometry mismatch")
        if gdn["short_conv_kernel"] != 4 or gdn["residual_owner"] != "hyperconnection":
            raise ValueError("v4 GDN convolution/residual contract mismatch")
        if gdn["recurrent_arithmetic"] != "float32_stable_state_and_triangular_solve":
            raise ValueError("v4 GDN arithmetic precision mismatch")
        if gdn["projection_layout"] != "qkvz_8192_plus_ba_64_plus_depthwise_conv_plus_gated_norm_plus_out":
            raise ValueError("v4 GDN projection layout mismatch")
        if gdn["special_state_shapes"] != {"dt_bias": [32], "A_log": [32], "gated_norm_offset": [128]}:
            raise ValueError("v4 GDN special state shape mismatch")

        moe = self._raw["moe"]
        _require_exact_keys(moe, {
            "routed_experts", "shared_experts", "top_k", "expert_intermediate", "activation",
            "router", "routing", "shared_expert_gate", "load_balance_scope", "physical_fusion"
        }, "moe")
        if (moe["routed_experts"], moe["shared_experts"], moe["top_k"], moe["expert_intermediate"]) != (80, 1, 6, 1280):
            raise ValueError("v4 MoE topology mismatch")
        if moe["activation"] != "swiglu" or moe["router"] != "learned_linear_2048_to_80_bias_false":
            raise ValueError("v4 MoE activation/router mismatch")
        if moe["routing"] != "softmax_all_experts_then_top6_then_renormalize_selected":
            raise ValueError("v4 routing contract mismatch")
        if moe["shared_expert_gate"] != "learned_sigmoid_2048_to_1_bias_false":
            raise ValueError("v4 shared expert gate contract mismatch")
        if moe["load_balance_scope"] != "global_or_logical_batch_not_microbatch_local":
            raise ValueError("v4 load balance scope mismatch")
        if moe["physical_fusion"] != "runtime_only_logical_gate_up_down_tensors_preserved":
            raise ValueError("v4 expert fusion contract mismatch")

        hc = self._raw["hyperconnections"]
        _require_exact_keys(hc, {
            "streams", "persistent_residual_width", "lowrank", "per_decoder_boundary",
            "final", "norm_offsets"
        }, "hyperconnections")
        if hc != {
            "streams": 4, "persistent_residual_width": 8192, "lowrank": 256,
            "per_decoder_boundary": ["mixer", "moe"],
            "final": "read_only_8192_to_2048", "norm_offsets": "zero",
        }:
            raise ValueError("v4 HyperConnection contract mismatch")

        ple = self._raw["ple"]
        _require_exact_keys(ple, {
            "injections", "injection_layer_zero_based", "ngram_orders", "heads_per_order",
            "total_heads", "aggregate_width", "head_dim", "ngram_vocab_size_base",
            "table_capacity_policy", "hash_head_rows", "table_tensor_layout",
            "dense_machinery", "convolution", "reset_isolation", "hash", "training_storage",
            "inference_storage"
        }, "ple")
        if ple["hash"] != {
            "algorithm": "xor_of_odd_splitmix64_multiplier_products_mod_head_prime_v1",
            "preimage": "x_t_x_t_minus_1_x_t_minus_2_segment_local",
            "reset_sentinel": "tokenizer_eos_id",
            "seed": 500_277,
        }:
            raise ValueError("v4 PLE hash contract mismatch")
        if (ple["injections"], ple["injection_layer_zero_based"], ple["ngram_orders"], ple["heads_per_order"], ple["total_heads"]) != (1, 1, [2, 3], 8, 16):
            raise ValueError("v4 PLE topology/injection mismatch")
        if (ple["aggregate_width"], ple["head_dim"], ple["ngram_vocab_size_base"]) != (2048, 128, 8_388_608):
            raise ValueError("v4 PLE embedding/table base mismatch")
        if ple["table_capacity_policy"] != "sixteen_distinct_primes_at_or_above_base_in_ascending_head_order":
            raise ValueError("v4 PLE prime table policy mismatch")
        expected_rows = [8_388_617, 8_388_619, 8_388_623, 8_388_637, 8_388_673, 8_388_683, 8_388_691, 8_388_697, 8_388_733, 8_388_739, 8_388_761, 8_388_763, 8_388_791, 8_388_811, 8_388_833, 8_388_841]
        if ple["hash_head_rows"] != expected_rows:
            raise ValueError("v4 PLE final hash-head capacities mismatch")
        if ple["table_tensor_layout"] != "one_independently_addressable_bf16_tensor_per_hash_head":
            raise ValueError("v4 PLE table must be segmented per hash head")
        if ple["dense_machinery"] != "key_2048_to_8192_value_2048_to_2048_three_grouped_norms_depthwise_conv_8192x4":
            raise ValueError("v4 PLE dense machinery mismatch")
        if ple["convolution"] != {"kind": "causal_depthwise", "kernel": 4, "dilation": 1, "init": "exact_zero"}:
            raise ValueError("v4 PLE convolution mismatch")
        if ple["reset_isolation"] != "eos_segment_local_history_and_convolution":
            raise ValueError("v4 PLE EOS reset isolation mismatch")
        if ple["training_storage"] != "bf16_master_host_or_distributed_off_accelerator":
            raise ValueError("v4 PLE training storage mismatch")
        if ple["inference_storage"] != "fp8_e4m3_host_ram_mmap_async_row_prefetch_optional_hot_rows_4bit_optional":
            raise ValueError("v4 PLE inference storage mismatch")

        mtp = self._raw["mtp"]
        _require_exact_keys(mtp, {
            "hidden_layers", "layer_type", "d_model", "embedding", "lm_head", "moe",
            "hyperconnections", "kv", "prediction_window", "teacher_forcing", "fusion",
            "main_loss_weight", "auxiliary_loss", "coefficient_schedule", "inference_precision"
        }, "mtp")
        if mtp["hidden_layers"] != 1 or mtp["layer_type"] != "full_attention" or mtp["d_model"] != 2048:
            raise ValueError("v4 MTP layer contract mismatch")
        if mtp["embedding"] != "shared_main_model" or mtp["lm_head"] != "shared_main_model":
            raise ValueError("v4 MTP must share the main embedding and LM head")
        if mtp["moe"] != copy.deepcopy(moe) or mtp["hyperconnections"] != {"streams": 4, "lowrank": 256}:
            raise ValueError("v4 MTP MoE/HC contract mismatch")
        if mtp["kv"] != "own_full_attention_kv_no_pairwise_kvc":
            raise ValueError("v4 MTP KVC contract mismatch")
        if mtp["prediction_window"] != {"maximum": 4, "runtime_range": [0, 1, 2, 3, 4], "recursive_steps": 3}:
            raise ValueError("v4 MTP prediction window mismatch")
        if mtp["teacher_forcing"] != "ground_truth_shifted_embeddings_with_recursive_hidden_propagation_no_sampling":
            raise ValueError("v4 MTP teacher forcing mismatch")
        if mtp["fusion"] != "grouped_hc_rms_8192_embedding_rms_2048_fc_hidden_2048_to_2048_shared_per_stream_fc_embedding_2048_to_2048_add_each_stream_final_read_only_hc":
            raise ValueError("v4 MTP fusion contract mismatch")
        if mtp["main_loss_weight"] != 1.0 or mtp["auxiliary_loss"] != "equal_mean_t_plus_2_t_plus_3_t_plus_4":
            raise ValueError("v4 MTP loss contract mismatch")
        if mtp["coefficient_schedule"] != {"first_70_percent": 0.30, "final_30_percent": 0.10}:
            raise ValueError("v4 MTP coefficient schedule mismatch")
        if mtp["inference_precision"] != {
            "routed_experts": "fp8", "attention_qkvo": "bf16", "router": "bf16",
            "shared_expert_and_gate": "bf16", "hc": "bf16", "fusion": "bf16", "norms": "bf16"
        }:
            raise ValueError("v4 MTP inference precision mismatch")

        initialization = self._raw["initialization"]
        _require_exact_keys(initialization, {
            "global_seed", "ordinary_matrices", "biases", "norms", "identity_algorithm",
            "dtype", "gdn", "hc", "moe", "ple", "mtp", "chunk_elements"
        }, "initialization")
        if initialization["global_seed"] != 500_277:
            raise ValueError("v4 global initialization seed mismatch")
        if initialization["ordinary_matrices"] != "normal_mean_0_std_0.02_fp32_then_cast":
            raise ValueError("v4 ordinary matrix initializer mismatch")
        if initialization["biases"] != "zero" or initialization["norms"] != "identity_semantics_offset_zero":
            raise ValueError("v4 bias/norm initializer mismatch")
        if initialization["identity_algorithm"] != {
            "version": "flashmini-sha256-name-seed-v1",
            "hash": "sha256",
            "input": "flashmini-init-v1\0<global_seed_decimal>\0<tensor_name>\0<chunk_index_decimal>",
            "seed_bytes": "first_16_bytes_little_endian_mod_2^63_minus_1",
            "generator": "torch.Generator(device=cpu).manual_seed(seed)",
            "chunk_elements": 1_048_576,
        }:
            raise ValueError("v4 deterministic name-seed algorithm mismatch")
        if initialization["dtype"] != "bfloat16" or initialization["chunk_elements"] != 1_048_576:
            raise ValueError("v4 init dtype/chunk contract mismatch")
        if initialization["gdn"] != {"dt_bias": "ones_32", "A_log": "log_uniform_fp32_0.01_to_16_shape_32"}:
            raise ValueError("v4 GDN init override mismatch")
        if initialization["hc"] != {"norm_offset": "zeros", "projections": "normal_0.02"}:
            raise ValueError("v4 HC init override mismatch")
        if initialization["moe"] != {"experts": "normal_0.02", "router": "normal_0.02", "shared_gate": "normal_0.02"}:
            raise ValueError("v4 MoE init override mismatch")
        if initialization["ple"] != {"table": "normal_0.02", "dense_projections": "normal_0.02", "convolution": "exact_zero"}:
            raise ValueError("v4 PLE init override mismatch")
        if initialization["mtp"] != "mirror_corresponding_backbone_rules_plus_fusion_rules":
            raise ValueError("v4 MTP init override mismatch")

        optimizer = self._raw["optimizer"]
        _require_exact_keys(optimizer, {"state_in_checkpoint", "muon", "adamw", "ple_table", "logical_operator_rule", "batch_size_warmup"}, "optimizer")
        if optimizer["state_in_checkpoint"] is not False or optimizer["batch_size_warmup"] is not False:
            raise ValueError("v4 init checkpoint must omit optimizer state and batch-size warmup")
        if optimizer["muon"] != {
            "parameters": "genuine_2d_learned_linear_transformations_by_logical_operator",
            "momentum": 0.95, "nesterov": True, "newton_schulz_or_polar_express_iterations": 8,
            "epsilon": 1e-14, "update_scale": "0.2*sqrt(max(A,B))"
        }:
            raise ValueError("v4 Muon contract mismatch")
        if optimizer["adamw"] != {
            "parameters": "embeddings_lm_head_routers_gates_hc_controls_norms_biases_scalars_equivalent_mtp_controls"
        }:
            raise ValueError("v4 AdamW classification mismatch")
        if optimizer["ple_table"] != {"optimizer": "Adam", "weight_decay": 0}:
            raise ValueError("v4 PLE table optimizer mismatch")
        if optimizer["logical_operator_rule"] != "physical_fusion_allowed_but_no_cross_operator_orthogonalization":
            raise ValueError("v4 logical operator optimizer rule mismatch")

        storage = self._raw["storage"]
        _require_exact_keys(storage, {
            "format", "index", "target_shard_bytes", "ple_layout", "expert_layout",
            "materialization", "precision_policy"
        }, "storage")
        if storage["format"] != "safetensors" or storage["index"] != "model.safetensors.index.json":
            raise ValueError("v4 checkpoint format/index mismatch")
        if storage["target_shard_bytes"] != 4 * 1024**3:
            raise ValueError("v4 target shard size must be 4 GiB")
        if storage["ple_layout"] != "one_hash_head_per_tensor_and_never_monolithic":
            raise ValueError("v4 PLE checkpoint layout mismatch")
        if storage["expert_layout"] != "logical_expert_tensors_groupable_by_layer_or_expert_group":
            raise ValueError("v4 expert checkpoint layout mismatch")
        if storage["materialization"] != "meta_manifest_then_name_seeded_chunk_fill_immediate_shard_write_release":
            raise ValueError("v4 materialization contract mismatch")
        if storage["precision_policy"] != "never_change_precision_or_geometry_for_local_capacity":
            raise ValueError("v4 local-capacity policy mismatch")

        contract = self._raw["architecture_contract"]
        _require_exact_keys(contract, {"status", "source", "changes_require", "training_started"}, "architecture_contract")
        if contract != {
            "status": "frozen",
            "source": "FlashMini-50B Base Init hardened execution contract",
            "changes_require": "new_model_and_init_version",
            "training_started": False,
        }:
            raise ValueError("v4 architecture freeze declaration mismatch")

    def is_attention_layer(self, layer_index: int) -> bool:
        return layer_index in self.attention_layers

    def kvc_role(self, layer_index: int) -> str | None:
        for source, reuse in self.kvc_pairs:
            if layer_index == source:
                return "source"
            if layer_index == reuse:
                return "reuse"
        return None

    def kvc_source_for(self, layer_index: int) -> int | None:
        for source, reuse in self.kvc_pairs:
            if layer_index == reuse:
                return source
        return None


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FlashMini50BConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("FlashMini-50B config must be a mapping")
    return FlashMini50BConfig(raw)


__all__ = [
    "ARCHITECTURE_VERSION", "CHECKPOINT_ID", "DEFAULT_CONFIG_PATH", "MODEL_ID",
    "FlashMini50BConfig", "architecture_fingerprint_payload", "canonical_sha256", "load_config",
]
