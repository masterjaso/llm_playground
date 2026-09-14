"""Changing an unseen suffix must never change a language model's prefix logits."""

import pytest
import torch

from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel


@pytest.mark.parametrize("hybrid,ple", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("cut", [1, 7, 8, 9, 15])
def test_full_model_suffix_invariance(hybrid, ple, cut):
    torch.manual_seed(11)
    cfg = FlashMiniConfig(vocab_size=31, d_model=16, num_layers=4, num_heads=2,
                          head_dim=8, max_seq_len=19, gdn_per_attention=3 if hybrid else 0,
                          use_ple=ple,
                          moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
                          gdn=GatedDeltaNetConfig(d_state=8, chunk_size=8),
                          ple=PLEConfig(num_heads=4, head_dim=4, table_size=101))
    model = FlashMiniModel(cfg).eval()
    ids = torch.randint(0, 31, (2, 19))
    changed = ids.clone()
    changed[:, cut:] = (changed[:, cut:] + 7) % 31
    with torch.no_grad():
        for enabled in ([True, False] if ple else [False]):
            original = model(ids, ple_enabled=enabled)["logits"][:, :cut]
            perturbed = model(changed, ple_enabled=enabled)["logits"][:, :cut]
            torch.testing.assert_close(original, perturbed, rtol=1e-5, atol=1e-5)
