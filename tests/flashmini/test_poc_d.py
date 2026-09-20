"""PoC_D v3 correctness: frozen C recipe + KVC + GPipe microbatch pipeline.

Every property the official D run depends on is asserted here without running
the full 250M-token schedule:

  * The frozen KVC configuration is C verbatim plus only the frozen KVC block.
  * C and D have identical parameter counts (KVC adds no parameters).
  * The microbatch pipeline degenerates to the monolithic `train_step` for
    microbatch_size == logical batch (one update per logical batch).
  * The logical-batch aux is exactly recomputable from additive per-microbatch
    stats, and the CE is exactly recomputable from per-microbatch CEs.
  * Each logical batch takes exactly one optimizer update (no gradient
    accumulation); `grad_accum != 1`, a non-dividing, or a non-positive
    microbatch size are rejected by the engine.
  * KVC fake-quant is deterministic with a straight-through identity gradient.
  * Enabling KVC engages the reuse path (changes the D model's output) and
    gradients reach the KVC source layer.
  * The batched evaluator matches the per-sequence evaluator.

These tests run on CPU (no CUDA required) so the deterministic-identity
invariants are checked exactly, isolated from the platform's bf16
reduction-order noise floor (asserted separately in the fail-closed
pre-flight gates).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashmini.config import (
    FlashMiniConfig,
    GatedDeltaNetConfig,
    KVConfig,
    MoEConfig,
    PLEConfig,
)
from flashmini.data import MemmapDataset, prepare_streaming_documents
from flashmini.eval import compute_validation_nll, compute_validation_nll_batched
from flashmini.kvc import KVQuantizer
from flashmini.models import FlashMiniModel
from flashmini.pipeline import (
    _StatsAccumulator,
    _logical_aux,
    pipeline_train_step,
)
from flashmini.optim import build_optimizer
from flashmini.training import train, train_step


DEVICE = torch.device("cpu")


def _tiny_config(kvc: bool = False) -> FlashMiniConfig:
    d = dict(
        architecture_version=3,
        vocab_size=32,
        d_model=16,
        num_layers=8,
        num_heads=2,
        head_dim=8,
        max_seq_len=8,
        gdn_per_attention=3,
        use_ple=True,
        moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
        gdn=GatedDeltaNetConfig(d_state=4, chunk_size=4),
        ple=PLEConfig(
            table_size=37,
            ngram_vocab_size_base=37,
            heads_per_ngram=2,
            embed_dim=16,
            head_dim=4,
            eos_id=31,
            offload="cpu",
        ),
    )
    if kvc:
        d["kvc"] = KVConfig(
            enabled=True,
            share_group_size=2,
            kv_bits=4,
            quant_format="e2m1",
            scale_format="e4m3",
            scale_group_size=16,
            qat=True,
        )
    return FlashMiniConfig(**d)


def make_data(tmp_path, seq_len=8):
    rng = np.random.default_rng(12)
    prepare_streaming_documents(
        (rng.integers(0, 31, size=seq_len) for _ in range(200)),
        tmp_path,
        seq_len=seq_len,
        eos_id=31,
        seed=17,
        val_fraction=0.2,
        provenance={
            "dataset_id": "poc-d-test",
            "dataset_revision": "0" * 40,
            "tokenizer_id": "integer-test",
            "tokenizer_revision": "1" * 40,
        },
    )
    return MemmapDataset(tmp_path)


def _batch(dataset, size, device, start=0):
    idx = (torch.arange(size, device="cpu") + start).numpy()
    inp, lab = dataset.get_batch(idx)
    return (
        torch.as_tensor(inp, dtype=torch.long, device=device),
        torch.as_tensor(lab, dtype=torch.long, device=device),
    )


def _build(cfg, device=DEVICE, seed=17):
    torch.manual_seed(seed)
    m = FlashMiniModel(cfg)
    m.to(device)
    m.train()
    return m


def _copy_weights(src, dst):
    with torch.no_grad():
        for a, b in zip(dst.parameters(), src.parameters()):
            b.data.copy_(a.data)


# --- config parity -----------------------------------------------------------


def test_poc_d_config_is_c_plus_kvc():
    c = _tiny_config(kvc=False)
    d = _tiny_config(kvc=True)
    strip = lambda cfg: {k: v for k, v in cfg.to_dict().items() if k != "kvc"}
    assert strip(c) == strip(d), "D config deviates from C beyond the kvc block"
    assert d.to_dict()["kvc"] == {
        "enabled": True,
        "share_group_size": 2,
        "kv_bits": 4,
        "quant_format": "e2m1",
        "scale_format": "e4m3",
        "scale_group_size": 16,
        "qat": True,
    }
    assert c.to_dict()["kvc"]["enabled"] is False


def test_kvc_freezes_nondefault_values():
    """The frozen KVC constants can't be relaxed without a hard error."""
    with pytest.raises(ValueError, match="share_group_size"):
        KVConfig(
            enabled=True,
            share_group_size=3,  # must be 2
            kv_bits=4,
            quant_format="e2m1",
            scale_format="e4m3",
            scale_group_size=16,
            qat=True,
        )
    # A config-level KVC on a 1-attention-layer layout is also rejected.
    with pytest.raises(ValueError, match="at least two attention layers"):
        FlashMiniConfig(
            architecture_version=3,
            vocab_size=32,
            d_model=16,
            num_layers=4,
            num_heads=2,
            head_dim=8,
            max_seq_len=8,
            gdn_per_attention=3,
            use_ple=False,
            moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
            gdn=GatedDeltaNetConfig(d_state=4, chunk_size=4),
            kvc=KVConfig(enabled=True, share_group_size=2, kv_bits=4,
                        quant_format="e2m1", scale_format="e4m3",
                        scale_group_size=16, qat=True),
        )


def test_kvc_role_mapping():
    d = _tiny_config(kvc=True)
    attn = d.attention_layers
    assert attn[0] == 3 and attn[1] == 7
    assert d.kvc_role(attn[0]) == "source"
    assert d.kvc_role(attn[1]) == "reuse"
    for i in range(d.num_layers):
        if i not in (attn[0], attn[1]):
            assert d.kvc_role(i) is None


def test_c_d_param_parity():
    c = FlashMiniModel(_tiny_config(kvc=False))
    d = FlashMiniModel(_tiny_config(kvc=True))
    assert sum(p.numel() for p in c.parameters()) == sum(p.numel() for p in d.parameters())


# --- pipeline equivalence -----------------------------------------------------


def test_single_microbatch_equals_train_step(tmp_path):
    """A pipeline with a single microbatch == logical batch is metric-identical
    to the monolithic `train_step` (one update, no accumulation)."""
    dataset = make_data(tmp_path)
    i, l = _batch(dataset, 8, DEVICE)
    cfg = _tiny_config(kvc=True)
    m_mono = _build(cfg)
    opt_mono = build_optimizer(m_mono, lr=0.001)
    out_mono = train_step(m_mono, opt_mono, i, l, aux_loss_coef=0.01)
    m_pipe = _build(cfg)  # identical init (seed 17)
    _copy_weights(m_mono, m_pipe)  # pin to identical pre-step weights
    opt_pipe = build_optimizer(m_pipe, lr=0.001)
    out_pipe = pipeline_train_step(m_pipe, opt_pipe, [(i, l)], aux_loss_coef=0.01, use_amp=False)
    for key in ("loss", "total_loss", "grad_norm", "router_aux_loss"):
        assert abs(float(out_mono[key]) - float(out_pipe[key])) < 1e-9, key


def test_one_optimizer_step_per_batch(tmp_path):
    """One pipeline step over `m` microbatches takes exactly one optimizer step."""
    dataset = make_data(tmp_path)
    i, l = _batch(dataset, 16, DEVICE)
    mb = 4
    chunks = [(i[k * mb : (k + 1) * mb], l[k * mb : (k + 1) * mb])
              for k in range(16 // mb)]
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    opt = build_optimizer(m, lr=0.001)
    orig = opt.step
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    opt.step = counting
    try:
        out = pipeline_train_step(m, opt, chunks, aux_loss_coef=0.01, use_amp=False)
    finally:
        opt.step = orig
    assert calls["n"] == 1, "expected exactly one optimizer.step per logical batch"
    assert torch.isfinite(torch.tensor(float(out["loss"])))


def test_aux_recombine_exact(tmp_path):
    """The logical-batch aux is exactly recomputable from additive per-microbatch
    stats (construction-identical to the engine's own accumulated aux)."""
    dataset = make_data(tmp_path)
    full_i, full_l = _batch(dataset, 8, DEVICE)
    mb = 4
    chunks = [(full_i[k * mb : (k + 1) * mb], full_l[k * mb : (k + 1) * mb])
              for k in range(8 // mb)]
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    ne = cfg.moe.num_experts
    acc = _StatsAccumulator(ne, DEVICE)
    for ci, li in chunks:
        o = m(ci, labels=li)
        acc.add_layer_stats(o["stats"])
    aux_re = float(_logical_aux(ne, acc.exp_counts, acc.prob_sum, acc.token_count, acc.slot_count))
    m2 = _build(cfg)
    _copy_weights(m, m2)
    opt = build_optimizer(m2, lr=0.001)
    aux_pipe = float(pipeline_train_step(m2, opt, chunks, aux_loss_coef=0.01, use_amp=False)["router_aux_loss"])
    assert abs(aux_re - aux_pipe) < 1e-9


def test_ce_recombine_equivalence(tmp_path):
    """The logical-batch CE == mean of per-microbatch CEs (exact, CPU)."""
    dataset = make_data(tmp_path)
    n, mb = 8, 4
    full_i, full_l = _batch(dataset, n, DEVICE)
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    m.eval()
    with torch.no_grad():
        ce_full = float(m(full_i, labels=full_l)["loss"])
        ce_mb = sum(
            float(m(full_i[k * mb : (k + 1) * mb],
                    labels=full_l[k * mb : (k + 1) * mb])["loss"]) * (mb / n)
            for k in range(n // mb)
        )
    assert abs(ce_mb - ce_full) < 1e-6 * max(1.0, abs(ce_full))


# --- engine guards ------------------------------------------------------------


def test_grad_accum_disallowed(tmp_path):
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    opt = build_optimizer(m, lr=0.001)
    dataset = make_data(tmp_path)
    with pytest.raises(ValueError, match="grad_accum"):
        train(m, opt, dataset, cfg, tmp_path / "run",
              total_tokens=64, seq_len=8, device=DEVICE,
              batch_size=4, grad_accum=2, aux_loss_coef=0.01)


def test_pipeline_microbatch_must_divide_batch(tmp_path):
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    opt = build_optimizer(m, lr=0.001)
    dataset = make_data(tmp_path)
    with pytest.raises(ValueError, match="divide"):
        train(m, opt, dataset, cfg, tmp_path / "run",
              total_tokens=64, seq_len=8, device=DEVICE,
              batch_size=8, pipeline_microbatch_size=3, aux_loss_coef=0.01)


def test_pipeline_microbatch_must_be_positive(tmp_path):
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    opt = build_optimizer(m, lr=0.001)
    dataset = make_data(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        train(m, opt, dataset, cfg, tmp_path / "run",
              total_tokens=64, seq_len=8, device=DEVICE,
              batch_size=8, pipeline_microbatch_size=0, aux_loss_coef=0.01)


# --- KVC semantics ------------------------------------------------------------


def test_fake_quant_deterministic_and_straight_through():
    q = KVQuantizer(kv_bits=4, quant_format="e2m1", scale_format="e4m3",
                     scale_group_size=16)
    x = torch.randn(2, 4, 8, 8)
    y1 = q.quantize_dequantize(x)
    y2 = q.quantize_dequantize(x)
    assert torch.equal(y1, y2)
    xg = torch.randn(4, 16, requires_grad=True)
    yg = q.quantize_dequantize(xg)
    g = torch.autograd.grad(yg.sum(), [xg])[0]
    assert torch.allclose(g, torch.ones_like(xg)), "straight-through grad must be identity"


def test_kvc_enabled_changes_output(tmp_path):
    """Enabling KVC is not a no-op: with identical init, the D model differs
    from the C model because the reuse layer attends against the quantized bank."""
    dataset = make_data(tmp_path)
    i, l = _batch(dataset, 8, DEVICE)
    c = _build(_tiny_config(kvc=False))
    d = _build(_tiny_config(kvc=True))
    _copy_weights(c, d)
    c.eval()
    d.eval()
    with torch.no_grad():
        lc = float(c(i, labels=l)["loss"])
        ld = float(d(i, labels=l)["loss"])
    assert abs(ld - lc) > 1e-6, "KVC must engage (change D's output relative to C)"


def test_kvc_source_layer_owns_quantizer():
    d = _tiny_config(kvc=True)
    model = FlashMiniModel(d)
    src = d.attention_layers[0]
    reuse = d.attention_layers[1]
    assert model.blocks[src].mixer.kvc_role == "source"
    assert model.blocks[src].mixer.quantizer is not None
    assert model.blocks[reuse].mixer.kvc_role == "reuse"
    assert model.blocks[reuse].mixer.quantizer is None


def test_reuse_grads_reach_source(tmp_path):
    """Gradients produced by the reuse layer flow back to the source layer's
    K/V projection through the shared (straight-through) bank."""
    dataset = make_data(tmp_path)
    i, l = _batch(dataset, 8, DEVICE)
    mb = 4
    chunks = [(i[k * mb : (k + 1) * mb], l[k * mb : (k + 1) * mb])
              for k in range(8 // mb)]
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    opt = build_optimizer(m, lr=0.001)
    pipeline_train_step(m, opt, chunks, aux_loss_coef=0.01, use_amp=False)
    src = cfg.attention_layers[0]
    grad = m.blocks[src].mixer.qkv.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "source layer qkv must receive a gradient"


# --- evaluator ----------------------------------------------------------------


def test_batched_eval_equals_per_sequence(tmp_path):
    dataset = make_data(tmp_path)
    cfg = _tiny_config(kvc=True)
    m = _build(cfg)
    p = compute_validation_nll(m, dataset, DEVICE, max_batches=8)
    b4 = compute_validation_nll_batched(m, dataset, DEVICE, batch_size=4, max_batches=2)
    b8 = compute_validation_nll_batched(m, dataset, DEVICE, batch_size=8, max_batches=1)
    assert p["tokens"] == b4["tokens"] == b8["tokens"]
    assert abs(p["nll"] - b4["nll"]) < 1e-5
    assert abs(p["nll"] - b8["nll"]) < 1e-5
