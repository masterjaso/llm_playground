"""Sparse CPU memory must survive checkpoint/resume with identical updates."""

import copy

import numpy as np
import pytest
import torch

from flashmini.checkpoint import load_checkpoint, save_checkpoint
from flashmini.config import FlashMiniConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel
from flashmini.optim import build_optimizer
from flashmini.training import train


class Data:
    def __len__(self):
        return 8

    def get_batch(self, indices):
        x = np.arange(32, dtype=np.int64).reshape(8, 4)[indices] % 17
        return x, (x + 1) % 17


def make():
    cfg = FlashMiniConfig(vocab_size=17, d_model=8, num_layers=2, num_heads=1,
                          head_dim=8, max_seq_len=4, gdn_per_attention=0, use_ple=True,
                          moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=8),
                          ple=PLEConfig(num_heads=2, head_dim=4, table_size=31, offload="cpu"))
    torch.manual_seed(123)
    model = FlashMiniModel(cfg)
    return cfg, model, build_optimizer(model, 0.001)


def test_real_ple_sparse_resume_exact(tmp_path):
    cfg, full, optimizer = make()
    common = {"seq_len": 4, "device": torch.device("cpu"), "batch_size": 2,
              "seed": 19, "use_amp": False, "ckpt_every_tokens": 16, "warmup_tokens": 16}
    train(full, optimizer, Data(), cfg, tmp_path / "full", total_tokens=32, **common)
    cfg, partial, optimizer = make()
    train(partial, optimizer, Data(), cfg, tmp_path / "resume", total_tokens=16, **common)
    cfg, resumed, optimizer = make()
    train(resumed, optimizer, Data(), cfg, tmp_path / "resume", total_tokens=32,
          resume_from=tmp_path / "resume/checkpoints/step_2.pt", **common)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    state = optimizer.optimizers[-1].state[resumed.ple.value_embed.weight]
    assert state["exp_avg"].device.type == "cpu"


def test_checkpoint_rejects_same_shape_different_memory_semantics(tmp_path):
    cfg, model, optimizer = make()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 1, cfg)
    other = copy.deepcopy(cfg)
    other.ple.eos_id = 16
    with pytest.raises(ValueError, match="ple.eos_id"):
        load_checkpoint(path, FlashMiniModel(other))
    other = copy.deepcopy(cfg)
    other.ple.offload = "gpu"
    load_checkpoint(path, FlashMiniModel(other))
