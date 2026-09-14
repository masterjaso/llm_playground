"""PLE must retrieve causal context, remain trainable, and offload its table."""

import copy

import pytest
import torch

from flashmini.config import FlashMiniConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel
from flashmini.models.ple import PLE


def test_same_final_token_different_context_changes_memory():
    memory = PLE(PLEConfig(vocab_size=31, d_model=16, num_heads=4, head_dim=4))
    ids = torch.tensor([[1, 2, 7], [4, 5, 7]])
    assert not torch.equal(memory._ngram_keys(ids)[0, -1], memory._ngram_keys(ids)[1, -1])


def test_memory_keys_are_causal_and_reset_at_document_boundary():
    memory = PLE(PLEConfig(vocab_size=31, d_model=16, num_heads=4, head_dim=4,
                           eos_id=30, table_size=101))
    ids = torch.tensor([[1, 2, 30, 7, 8, 9], [5, 6, 30, 7, 8, 4]])
    keys = memory._ngram_keys(ids)
    assert torch.equal(keys[0, 2:5], keys[1, 2:5])
    assert torch.equal(memory._ngram_keys(ids[:, :4]), keys[:, :4])
    assert ((keys >= 0) & (keys < memory.value_embed.num_embeddings)).all()


def test_cpu_memory_sparse_update_and_ablation():
    from flashmini.optim import build_optimizer, clip_gradients

    config = FlashMiniConfig(vocab_size=31, d_model=16, num_layers=2, num_heads=2,
                             head_dim=8, max_seq_len=8, use_ple=True,
                             gdn_per_attention=0,
                             moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
                             ple=PLEConfig(offload="cpu", num_heads=4, head_dim=4,
                                           table_size=101, injection_layer=1))
    model = FlashMiniModel(config)
    optimizer = build_optimizer(model, lr=0.01)
    ids = torch.tensor([[1, 2, 7, 4, 5, 7, 8, 9]])
    before = model.ple.value_embed.weight.detach().clone()
    model(ids, labels=ids.roll(-1, dims=1))["loss"].backward()
    grad = model.ple.value_embed.weight.grad
    assert grad.is_sparse
    assert torch.isfinite(grad.coalesce().values()).all()
    assert grad.coalesce().values().abs().sum() > 0
    assert torch.isfinite(clip_gradients(model, 1.0))
    optimizer.step()
    assert not torch.equal(before, model.ple.value_embed.weight)
    touched = torch.unique(model.ple._ngram_keys(ids))
    untouched = torch.ones(before.shape[0], dtype=torch.bool)
    untouched[touched] = False
    torch.testing.assert_close(before[untouched], model.ple.value_embed.weight[untouched])
    assert model.ple.value_embed.weight.device.type == "cpu"
    model.eval()
    with torch.no_grad():
        disabled = model(ids, ple_enabled=False)["logits"]
        saved = model.ple
        model.ple = None
        torch.testing.assert_close(disabled, model(ids)["logits"])
        model.ple = saved


def test_ple_construction_preserves_matched_backbone_initialization():
    cfg = FlashMiniConfig(vocab_size=31, d_model=16, num_layers=2, num_heads=2,
                          head_dim=8, max_seq_len=8)
    torch.manual_seed(42)
    base = FlashMiniModel(cfg)
    cfg = copy.deepcopy(cfg)
    cfg.use_ple = True
    torch.manual_seed(42)
    augmented = FlashMiniModel(cfg)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, augmented.state_dict()[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires hardware")
def test_offload_device_forward_backward_and_roundtrip():
    from flashmini.optim import build_optimizer, clip_gradients

    cfg = PLEConfig(vocab_size=31, d_model=16, num_heads=4, head_dim=4,
                    table_size=101, offload="cpu")
    cpu = PLE(cfg).cuda()
    cfg = copy.deepcopy(cfg)
    cfg.offload = "gpu"
    gpu = PLE(cfg).cuda()
    gpu.load_state_dict(cpu.state_dict())
    assert cpu.value_embed.weight.device.type == "cpu"
    assert gpu.value_embed.weight.device.type == "cuda"
    ids = torch.tensor([[1, 2, 7, 4, 5, 7]], device="cuda")
    hidden = torch.randn(1, 6, 16, device="cuda")
    cpu_opt, gpu_opt = build_optimizer(cpu, 0.001), build_optimizer(gpu, 0.001)
    torch.testing.assert_close(cpu(ids, hidden), gpu(ids, hidden), rtol=1e-5, atol=1e-6)
    cpu(ids, hidden).square().mean().backward()
    gpu(ids, hidden).square().mean().backward()
    assert cpu.value_embed.weight.grad.device.type == "cpu"
    assert cpu.value_embed.weight.grad.is_sparse
    torch.testing.assert_close(cpu.value_embed.weight.grad.coalesce().to_dense(),
                               gpu.value_embed.weight.grad.coalesce().to_dense().cpu(),
                               rtol=1e-4, atol=1e-7)
    clip_gradients(cpu, 1.0)
    clip_gradients(gpu, 1.0)
    cpu_opt.step()
    gpu_opt.step()
    torch.testing.assert_close(cpu.value_embed.weight, gpu.value_embed.weight.cpu(),
                               rtol=1e-4, atol=1e-6)
    sparse_state = cpu_opt.optimizers[-1].state[cpu.value_embed.weight]
    assert sparse_state["exp_avg"].device.type == "cpu"
    assert sparse_state["exp_avg_sq"].device.type == "cpu"
    # dtype/device conversion must never temporarily copy the entire table to CUDA.
    cpu.bfloat16()
    assert cpu.value_embed.weight.device.type == "cpu"
    assert cpu.value_embed.weight.dtype == torch.bfloat16
