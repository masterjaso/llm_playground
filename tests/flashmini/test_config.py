"""Tests for A/B/C config invariants and routing."""

from __future__ import annotations

import unittest

import torch

from flashmini.config import FlashMiniConfig
from flashmini.models import FlashMiniModel, topk_router


class ConfigInvariantTests(unittest.TestCase):
    def test_attention_layer_pattern(self):
        # 3 GDN : 1 attention for 8 layers -> layers 3, 7 are attention
        config = FlashMiniConfig(num_layers=8, gdn_per_attention=3)
        self.assertEqual(config.attention_layers, [3, 7])
        self.assertTrue(config.is_attention_layer(3))
        self.assertFalse(config.is_attention_layer(0))

    def test_abc_configs_share_shape(self):
        # A (control), B (flash), C (flash+ple) share core shape
        base = dict(vocab_size=512, d_model=64, num_layers=4, num_heads=2, head_dim=16, max_seq_len=32)
        a = FlashMiniConfig(**base, use_hyperconnection=False, use_ple=False)
        b = FlashMiniConfig(**base, use_hyperconnection=True, use_ple=False)
        c = FlashMiniConfig(**base, use_hyperconnection=True, use_ple=True)
        self.assertEqual(a.num_layers, b.num_layers)
        self.assertEqual(b.num_layers, c.num_layers)
        self.assertEqual(a.attention_layers, b.attention_layers)
        self.assertEqual(b.attention_layers, c.attention_layers)

    def test_router_topk(self):
        router = torch.nn.Linear(16, 8, bias=False)
        x = torch.randn(10, 16)
        indices, weights, logits, aux = topk_router(x, router, 8, 2)
        self.assertEqual(indices.shape, (10, 2))
        self.assertEqual(weights.shape, (10, 2))
        self.assertTrue(torch.isfinite(aux))

    def test_forward_shapes(self):
        config = FlashMiniConfig(vocab_size=256, d_model=32, num_layers=2, num_heads=2, head_dim=16, max_seq_len=16)
        model = FlashMiniModel(config)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        self.assertEqual(out["logits"].shape, (2, 16, 256))


if __name__ == "__main__":
    unittest.main()