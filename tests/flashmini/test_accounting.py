"""Tests for FlashMini parameter/FLOP accounting."""

from __future__ import annotations

import unittest

from flashmini.accounting import count_parameters, estimate_flops_per_token, ParameterBudget
from flashmini.config import FlashMiniConfig
from flashmini.models import FlashMiniModel


class AccountingTests(unittest.TestCase):
    def test_parameter_counts(self):
        config = FlashMiniConfig(
            vocab_size=512,
            d_model=64,
            num_layers=2,
            num_heads=2,
            head_dim=16,
            max_seq_len=32,
            use_ple=True,
        )
        model = FlashMiniModel(config)
        counts = count_parameters(model, config)
        self.assertGreater(counts.total, 0)
        self.assertGreater(counts.core, 0)
        self.assertGreater(counts.embedding_head, 0)
        self.assertGreater(counts.ple, 0)
        # total = core + embedding_head + ple
        self.assertEqual(counts.total, counts.core + counts.embedding_head + counts.ple)
        # active <= total
        self.assertLessEqual(counts.active_per_token, counts.total)

    def test_ple_off_counts_zero(self):
        config = FlashMiniConfig(
            vocab_size=512, d_model=64, num_layers=2, num_heads=2, head_dim=16, max_seq_len=32, use_ple=False
        )
        model = FlashMiniModel(config)
        counts = count_parameters(model, config)
        self.assertEqual(counts.ple, 0)

    def test_flops_positive(self):
        config = FlashMiniConfig(vocab_size=512, d_model=64, num_layers=2, num_heads=2, head_dim=16, max_seq_len=32)
        flops = estimate_flops_per_token(config)
        self.assertGreater(flops, 0)

    def test_ple_active_rows_and_budget_match_actual_shapes(self):
        from flashmini.config import MoEConfig, PLEConfig
        cfg = FlashMiniConfig(vocab_size=31, d_model=8, num_layers=1, num_heads=1,
                              head_dim=8, max_seq_len=8, gdn_per_attention=0,
                              use_ple=True,
                              moe=MoEConfig(num_experts=2, top_k=1, shared_experts=1,
                                            expert_intermediate=4),
                              ple=PLEConfig(num_heads=2, head_dim=4, table_size=101))
        model = FlashMiniModel(cfg)
        counts = count_parameters(model, cfg)
        inactive_experts = 2 * 8 * 4
        inactive_table = model.ple.value_embed.weight.numel() - 2 * 4
        self.assertEqual(counts.active_per_token, counts.total - inactive_experts - inactive_table)
        budget = ParameterBudget.from_config(cfg)
        self.assertEqual(budget.core_params(), counts.core)
        self.assertEqual(budget.ple_params(), counts.ple)
        self.assertEqual(budget.total_params(), counts.total)
        with_ple = estimate_flops_per_token(cfg)
        cfg.use_ple = False
        self.assertEqual(with_ple - estimate_flops_per_token(cfg), 4 * 8 * 2 * 4)


if __name__ == "__main__":
    unittest.main()
