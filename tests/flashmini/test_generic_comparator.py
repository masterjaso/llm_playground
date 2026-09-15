"""Tests for the generic fail-closed A/B/C comparator."""

from __future__ import annotations

import unittest

from flashmini.comparison import validate_generic_pair


def _config(gdn: int, use_ple: bool) -> dict:
    return {
        "architecture_version": 3,
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 10,
        "num_heads": 6,
        "head_dim": 128,
        "gdn_per_attention": gdn,
        "max_seq_len": 256,
        "use_ple": use_ple,
        "moe": {"num_experts": 16, "top_k": 2, "shared_experts": 1,
                "expert_intermediate": 1536, "aux_loss_coef": 0.01},
        "gdn": {"d_state": 128, "chunk_size": 64, "use_short_conv": True,
                "short_conv_kernel": 4, "residual_in_mixer": False},
        "ple": {"ngram": 3, "ngram_vocab_size_base": 50257, "heads_per_ngram": 8,
                "embed_dim": 2048, "hash_seed": 1234, "conv_kernel_size": 4,
                "conv_dilation": 3, "injection_layer": 1, "eos_id": 50256,
                "sparse": True, "offload": "cpu"},
    }


def _envelope(config: dict, *, tokens: int, seed: int, env_sha: str,
              optimizer_params: list[str] | None = None) -> dict:
    params = optimizer_params if optimizer_params is not None else ["a"]
    return {
        "config": config,
        "architecture_version": 3,
        "extra": {
            "tokens_seen": tokens,
            "real_tokens_seen": tokens,
            "data_manifest_sha256": "manifest",
            "training": {
                "seed": seed,
                "batch_size": 16,
                "seq_len": 256,
                "grad_accum": 1,
                "schedule": {"warmup_tokens": 524288, "cosine_decay": True,
                            "min_lr_ratio": 0.1, "total_tokens": 250000000},
                "dataset": {"tokenizer": "t", "tokenizer_revision": "tr",
                            "dataset_revision": "dr", "manifest_sha256": "manifest"},
                "run_metadata": {
                    "source_sha256": "src",
                    "shared_optimizer": [{"family": "AdamW", "parameters": params, "options": {}}],
                    "data_contract": {"actual_seq_len": 256},
                    "execution_policy": {
                        "router_aux_loss_coef": 0.01, "precision": "cuda_bfloat16_autocast",
                        "shared_parameter_dtypes": ["torch.bfloat16"],
                        "gradient_clip_max_norm": 1.0,
                        "clipping_policy": "independent_shared_ple_dense_ple_sparse_v3",
                        "optimizer_recipe": {"dense_family": "AdamW", "table_family": "AdamW",
                                             "base_lr": 3e-4, "ple_lr_multiplier": 5,
                                             "dense_weight_decay": 0.1, "table_weight_decay": 0.0,
                                             "betas": [0.9, 0.999], "eps": 1e-8},
                    },
                    "execution_fingerprint": {"environment_fingerprint_sha256": env_sha},
                },
            },
        },
    }


class GenericComparatorTests(unittest.TestCase):
    def test_a_vs_b_passes(self):
        # A and B have different mixer parameter names, so the optimizer
        # parameter inventories legitimately differ.
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["attn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        result = validate_generic_pair(a, b, "manifest")
        self.assertEqual(result["comparison_type"], "mixer_treatment")

    def test_b_vs_c_passes(self):
        # B and C have the same mixer, so the optimizer parameter inventory
        # must match exactly.
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        c = _envelope(_config(3, True), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        result = validate_generic_pair(b, c, "manifest")
        self.assertEqual(result["comparison_type"], "ple_treatment")

    def test_a_vs_c_passes(self):
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["attn_0", "attn_1"])
        c = _envelope(_config(3, True), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        result = validate_generic_pair(a, c, "manifest")
        self.assertEqual(result["comparison_type"], "mixer_and_ple")

    def test_rejects_unrelated_config_difference(self):
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["attn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        b["config"]["d_model"] = 512  # unrelated backbone change
        with self.assertRaises(ValueError):
            validate_generic_pair(a, b, "manifest")

    def test_rejects_seed_mismatch(self):
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["attn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=18, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        with self.assertRaises(ValueError):
            validate_generic_pair(a, b, "manifest")

    def test_rejects_environment_fingerprint_mismatch(self):
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env1",
                      optimizer_params=["attn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env2",
                      optimizer_params=["gdn_0", "attn_1"])
        with self.assertRaises(ValueError):
            validate_generic_pair(a, b, "manifest")

    def test_rejects_same_treatment(self):
        a = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        with self.assertRaises(ValueError):
            validate_generic_pair(a, b, "manifest")

    def test_rejects_c_ple_off_as_b(self):
        # C with PLE disabled is not B: it still has gdn_per_attention=3 and
        # use_ple=False, which is exactly B's signature, so a C-off vs B pair
        # has no valid treatment difference and must be rejected.
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        c_off = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                          optimizer_params=["gdn_0", "attn_1"])
        with self.assertRaises(ValueError):
            validate_generic_pair(b, c_off, "manifest")

    def test_rejects_optimizer_semantics_mismatch(self):
        # A and B have different mixer names (allowed), but if the optimizer
        # options (e.g. base LR) differ, the comparison must fail.
        a = _envelope(_config(0, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["attn_0", "attn_1"])
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        b["extra"]["training"]["run_metadata"]["shared_optimizer"][0]["options"]["lr"] = 1e-3
        with self.assertRaises(ValueError):
            validate_generic_pair(a, b, "manifest")

    def test_rejects_bc_optimizer_inventory_mismatch(self):
        # B and C have the same mixer, so the optimizer parameter inventory
        # must match exactly. A different parameter name must fail.
        b = _envelope(_config(3, False), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_1"])
        c = _envelope(_config(3, True), tokens=2097152, seed=17, env_sha="env",
                      optimizer_params=["gdn_0", "attn_2"])
        with self.assertRaises(ValueError):
            validate_generic_pair(b, c, "manifest")


if __name__ == "__main__":
    unittest.main()
