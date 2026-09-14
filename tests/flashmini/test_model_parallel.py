import copy

import pytest
import torch

from flashmini.config import FlashMiniConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel
from flashmini.optim import build_optimizer
from flashmini.training import train_step


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two CUDA GPUs")
def test_sharded_ple_matches_single_gpu_and_trains(tmp_path):
    cfg = FlashMiniConfig(vocab_size=31, d_model=16, num_layers=2, num_heads=2,
                          head_dim=8, max_seq_len=8, use_ple=True,
                          gdn_per_attention=0,
                          moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
                          ple=PLEConfig(offload="cpu", num_heads=4, head_dim=4,
                                        table_size=101, injection_layer=1))
    reference = FlashMiniModel(cfg).cuda(0)
    sharded = copy.deepcopy(reference).parallelize([torch.device("cuda:0"), torch.device("cuda:1")])
    ids = torch.tensor([[1, 2, 7, 4, 5, 7, 8, 9]], device="cuda:0")
    assert sharded.head.weight is sharded.embed.weight
    assert sharded.blocks[1].norm1.weight.device == torch.device("cuda:1")
    assert sharded.ple.value_embed.weight.device.type == "cpu"
    torch.testing.assert_close(reference(ids)["logits"], sharded(ids)["logits"], atol=2e-6, rtol=2e-5)
    opt_a, opt_b = build_optimizer(reference, 0.001), build_optimizer(sharded, 0.001)
    train_step(reference, opt_a, ids, ids, use_amp=False)
    train_step(sharded, opt_b, ids, ids, use_amp=False)
    for a, b in zip(reference.parameters(), sharded.parameters()):
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=2e-5, rtol=2e-4)
    train_step(sharded, opt_b, ids, ids, use_amp=True)
    from flashmini.checkpoint import load_checkpoint, save_checkpoint

    checkpoint = tmp_path / "sharded.pt"
    save_checkpoint(checkpoint, sharded, opt_b, 2, cfg)
    restored = FlashMiniModel(cfg).parallelize([torch.device("cuda:0"), torch.device("cuda:1")])
    restored_opt = build_optimizer(restored, 0.001)
    load_checkpoint(checkpoint, restored, restored_opt)
    train_step(sharded, opt_b, ids, ids, use_amp=False)
    train_step(restored, restored_opt, ids, ids, use_amp=False)
    for a, b in zip(sharded.parameters(), restored.parameters()):
        torch.testing.assert_close(a.cpu(), b.cpu())
