"""Tests for the mandatory micro-overfit test."""

from __future__ import annotations

import unittest

import torch

from flashmini.config import FlashMiniConfig
from flashmini.overfit import run_overfit_test


class OverfitTests(unittest.TestCase):
    def test_micro_overfit_passes(self):
        config = FlashMiniConfig(
            vocab_size=256,
            d_model=32,
            num_layers=2,
            num_heads=2,
            head_dim=16,
            max_seq_len=16,
            use_ple=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = run_overfit_test(config, device=device, steps=150)
        self.assertTrue(result["pass"], f"overfit failed: {result}")
        self.assertGreaterEqual(result["loss_drop"], 0.40)
        self.assertTrue(result["ple_active"])
        self.assertTrue(result["moe_active"])
        self.assertTrue(result["resume_ok"])


if __name__ == "__main__":
    unittest.main()