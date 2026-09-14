"""Externally observable v3 architecture invariants."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from flashmini.checkpoint import load_checkpoint, save_checkpoint
from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel
from flashmini.models.gated_delta_net import GatedDeltaNet
from flashmini.models.hyperconnection import GatedResidual
from flashmini.models.ple import PLEV3


def config(variant="c", seq_len=16):
    return FlashMiniConfig(architecture_version=3, vocab_size=32, d_model=16,
        num_layers=4, num_heads=2, head_dim=8, max_seq_len=seq_len,
        gdn_per_attention=0 if variant == "a" else 3, use_ple=variant == "c",
        moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
        gdn=GatedDeltaNetConfig(d_state=4, chunk_size=4),
        ple=PLEConfig(ngram_vocab_size_base=37, heads_per_ngram=2,
                      embed_dim=16, eos_id=31, offload="cpu"))


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_full_model_causality_and_gradients(variant):
    torch.manual_seed(17)
    model = FlashMiniModel(config(variant))
    x = torch.randint(0, 31, (2, 16))
    y = x.clone()
    y[:, 7:] = (y[:, 7:] + 3) % 31
    a, b = model(x)["logits"], model(y)["logits"]
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=0, atol=1e-7)
    a.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        grad = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
        assert torch.isfinite(grad).all(), name


def test_b_c_initialization_and_off_equivalence():
    torch.manual_seed(17)
    b = FlashMiniModel(config("b"))
    torch.manual_seed(17)
    c = FlashMiniModel(config("c"))
    for key, value in b.state_dict().items():
        assert torch.equal(value, c.state_dict()[key]), key
    x = torch.randint(0, 31, (2, 16))
    assert torch.equal(b(x)["logits"], c(x, ple_enabled=False)["logits"])
    torch.manual_seed(17)
    a = FlashMiniModel(config("a"))
    for key, value in a.state_dict().items():
        if key in b.state_dict() and value.shape == b.state_dict()[key].shape:
            assert torch.equal(value, b.state_dict()[key]), key


def test_gated_residual_four_stream_dynamic_read_write():
    torch.manual_seed(1)
    gr = GatedResidual(8, hc_lowrank=2)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    mixed, residual, gate = gr(hidden)
    assert mixed.shape == (2, 5, 8) and gate.shape == (2, 5, 4)
    output = gr.write(residual, mixed, gate)
    assert output.shape == hidden.shape
    assert not torch.equal(gate[:, 0], gate[:, 1])
    assert not torch.equal(output[..., :8], output[..., 8:16])
    output.square().mean().backward()
    assert hidden.grad is not None
    for p in gr.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    with torch.no_grad():
        gr.input_mix_weight_down.weight.zero_()
        gr.block_inject_weight.weight.zero_()
    mixed, residual, gate = gr(hidden.detach())
    streams = hidden.detach().reshape(2, 5, 4, 8)
    normalized = streams / torch.sqrt(streams.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(mixed, 0.5 * normalized.mean(-2))
    assert torch.equal(gate, torch.ones_like(gate))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ple_allocation_does_not_shift_shared_dropout_rng(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA-default construction requires CUDA")
    with torch.device(device):
        x = torch.arange(16).reshape(1, 16)
        outputs, states = [], []
        for variant in ("b", "c"):
            cfg = config(variant)
            cfg.dropout = 0.5
            torch.manual_seed(17)
            model = FlashMiniModel(cfg).train()
            states.append(torch.cuda.get_rng_state(0) if device.startswith("cuda") else torch.get_rng_state())
            outputs.append(model(x, ple_enabled=False)["logits"])
        assert torch.equal(states[0], states[1])
        assert torch.equal(outputs[0], outputs[1])


def test_gdn_zero_mixer_has_no_internal_residual():
    gdn = GatedDeltaNet(16, GatedDeltaNetConfig(d_state=4), residual_in_mixer=False)
    with torch.no_grad():
        gdn.w_out.weight.zero_()
    assert torch.count_nonzero(gdn(torch.randn(2, 16, 16))) == 0


def test_ple_nonzero_convolution_causality_and_eos_isolation():
    ple = PLEV3(config().ple)
    with torch.no_grad():
        ple.conv1d.weight.fill_(0.3)
    x = torch.tensor([[1, 2, 3, 31, 4, 5, 6, 7, 8, 9, 10]])
    hidden = torch.randn(1, x.shape[1], 64)
    changed = x.clone()
    changed[:, :3] = torch.tensor([9, 8, 7])
    changed_hidden = hidden.clone()
    changed_hidden[:, :4] += 20
    a = ple(x, hidden)
    b = ple(changed, changed_hidden)
    torch.testing.assert_close(a[:, 4:], b[:, 4:], atol=1e-6, rtol=1e-6)
    changed = x.clone()
    changed[:, 7:] = 20
    torch.testing.assert_close(a[:, :7], ple(changed, hidden)[:, :7], atol=1e-6, rtol=1e-6)
    a.square().mean().backward()
    assert ple.value_embed.weight.grad.is_sparse
    assert ple.conv1d.weight.grad.abs().sum() > 0
    assert ple.key_proj.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(ple.forward_with_ablation(x, False, hidden)) == 0
    assert not torch.equal(ple._ngram_keys(torch.tensor([[1, 2, 7]]))[:, -1],
                           ple._ngram_keys(torch.tensor([[4, 5, 7]]))[:, -1])


def test_official_configs_matched_and_v2_refused(tmp_path):
    root = Path(__file__).resolve().parents[2]
    configs = {v: FlashMiniConfig.from_dict(yaml.safe_load(
        (root / f"configs/flashmini/poc_{v}_v3.yaml").read_text())).to_dict() for v in "abc"}
    b, c = copy.deepcopy(configs["b"]), copy.deepcopy(configs["c"])
    for values in (b, c):
        values.pop("use_ple")
        values["ple"].pop("enabled")
    assert b == c
    a = copy.deepcopy(configs["a"])
    a.pop("use_ple")
    a["ple"].pop("enabled")
    for values in (a, b):
        values.pop("gdn_per_attention")
        values.pop("attention_layers")
    assert a == b
    cfg = config("b")
    old = config("b").to_dict()
    old["architecture_version"] = 2
    old_cfg = FlashMiniConfig.from_dict(old)
    old_model = FlashMiniModel(old_cfg)
    path = tmp_path / "v2.pt"
    save_checkpoint(path, old_model, None, 1, old_cfg)
    with pytest.raises(ValueError, match="fresh v3"):
        load_checkpoint(path, FlashMiniModel(cfg))


def test_v3_accounting_works_without_allocating_full_weights():
    from flashmini.accounting import ParameterBudget, count_parameters, estimate_flops_per_token
    cfg = config("c")
    counts = count_parameters(FlashMiniModel(cfg), cfg)
    assert ParameterBudget.from_config(cfg).total_params() == counts.total
    assert counts.active_per_token < counts.total
    assert estimate_flops_per_token(cfg) > estimate_flops_per_token(config("b"))
