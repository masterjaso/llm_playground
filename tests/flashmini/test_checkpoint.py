"""Tests for checkpoint save/resume."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from flashmini.checkpoint import save_checkpoint, load_checkpoint, verify_resume_continuity
from flashmini.config import FlashMiniConfig
from flashmini.models import FlashMiniModel


class CheckpointTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        config = FlashMiniConfig(vocab_size=256, d_model=32, num_layers=1, num_heads=1, head_dim=16, max_seq_len=16)
        model = FlashMiniModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, step=10, config=config)
            model2 = FlashMiniModel(config)
            meta = load_checkpoint(path, model2)
            self.assertEqual(meta["step"], 10)
            # weights match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                self.assertTrue(torch.equal(p1, p2), f"mismatch {n1}")

    def test_resume_continuity(self):
        config = FlashMiniConfig(vocab_size=256, d_model=32, num_layers=1, num_heads=1, head_dim=16, max_seq_len=16)
        model = FlashMiniModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            c1 = Path(tmp) / "c1.pt"
            c2 = Path(tmp) / "c2.pt"
            save_checkpoint(c1, model, optimizer, step=5, config=config)
            # train a bit
            x = torch.randint(0, 256, (2, 16))
            y = torch.randint(0, 256, (2, 16))
            optimizer.zero_grad()
            out = model(x, labels=y)
            out["loss"].backward()
            optimizer.step()
            save_checkpoint(c2, model, optimizer, step=6, config=config)
            self.assertTrue(verify_resume_continuity(c1, c2, model))


if __name__ == "__main__":
    unittest.main()