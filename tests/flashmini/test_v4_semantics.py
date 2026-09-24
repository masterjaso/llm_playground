"""FlashMini-50B v4 semantic tests: GDN, PLE, MTP, KVC, HC, InitSpec, architecture identity."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import yaml

from flashmini import v4_init
from flashmini.base_init_bundle import materialize_tensor
from flashmini.base_init_config import DEFAULT_CONFIG_PATH, architecture_fingerprint_payload, canonical_sha256, load_config
from flashmini.base_init_model import (
    MTP_MAX_RECURSIVE_STEPS,
    AttentionV4,
    FlashMini50BBaseInit,
    GatedDeltaNetV4,
    chunk_gated_delta_rule,
    gdn_log_decay,
    mtp_recursive_steps,
    mtp_teacher_and_targets,
    recurrent_gated_delta_rule,
    surrogate_config,
)

EOS = 0


@pytest.fixture(scope="module")
def config():
    return surrogate_config()


@pytest.fixture()
def model(config):
    torch.manual_seed(0)
    return FlashMini50BBaseInit(config)


# -- GDN -----------------------------------------------------------------------------

def test_gdn_decay_is_exp_of_nonpositive_log_decay_in_unit_interval():
    A_log = v4_init.materialize("blocks.0.mixer.A_log", (4096,)).float()
    assert float(A_log.exp().min()) >= 0.01 - 1e-4 and float(A_log.exp().max()) <= 16.0 + 1e-2
    a = torch.linspace(-12.0, 3.0, 4096)
    g = gdn_log_decay(A_log, a, torch.ones(4096))
    decay = g.exp()
    assert bool((g <= 0).all())
    assert bool((decay > 0).all()) and bool((decay <= 1).all())
    expected = torch.exp(-A_log.exp() * torch.nn.functional.softplus(a + 1.0))
    torch.testing.assert_close(decay, expected, rtol=0, atol=0)


def test_gdn_module_decay_factors_bounded(config):
    torch.manual_seed(1)
    gdn = GatedDeltaNetV4(config)
    for name, parameter in gdn.named_parameters():
        v4_init.fill_(parameter, f"blocks.0.mixer.{name}")
    decay = gdn.decay_factors(torch.randn(2, 9, config.d_model) * 5)
    assert decay.shape == (2, 9, gdn.num_v_heads)
    assert bool((decay > 0).all()) and bool((decay <= 1).all())


def _manual_inputs(g_value: float):
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 1, 2)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 1, 2)
    value = torch.tensor([[2.0], [3.0]]).reshape(1, 2, 1, 1)
    g = torch.full((1, 2, 1), math.log(g_value)) if g_value > 0 else torch.zeros(1, 2, 1)
    beta = torch.ones(1, 2, 1)
    return query, key, value, g, beta


def test_gdn_manual_two_step_case():
    """Hand-computed: decay 0.5 halves the stored memory before the second write.

    t=1: S = k1 (2)^T = [[2],[0]], o1 = S^T q1 / sqrt(2) = sqrt(2)
    t=2: S = 0.5 S = [[1],[0]]; k2^T S = 0; S += k2 (3)^T = [[1],[3]]; o2 = 1/sqrt(2)
    """
    for kernel in (recurrent_gated_delta_rule, lambda *args: chunk_gated_delta_rule(*args, chunk_size=4)):
        output, state = kernel(*_manual_inputs(0.5))
        torch.testing.assert_close(output.reshape(-1), torch.tensor([math.sqrt(2.0), 1 / math.sqrt(2.0)]), atol=1e-5, rtol=0)
        torch.testing.assert_close(state.reshape(2, 1), torch.tensor([[1.0], [3.0]]), atol=1e-5, rtol=0)
    output, _ = recurrent_gated_delta_rule(*_manual_inputs(1.0))
    torch.testing.assert_close(output.reshape(-1), torch.tensor([math.sqrt(2.0), math.sqrt(2.0)]), atol=1e-5, rtol=0)


def _numpy_reference(query, key, value, g, beta):
    q, k, v, gg, bb = (x.double().numpy() for x in (query, key, value, g, beta))
    batch, steps, heads, dk = q.shape
    out = np.zeros((batch, steps, heads, v.shape[-1]))
    for b in range(batch):
        for h in range(heads):
            state = np.zeros((dk, v.shape[-1]))
            for t in range(steps):
                qt = q[b, t, h] / np.sqrt((q[b, t, h] ** 2).sum() + 1e-6)
                kt = k[b, t, h] / np.sqrt((k[b, t, h] ** 2).sum() + 1e-6)
                state = np.exp(gg[b, t, h]) * state
                state = state + np.outer(kt, bb[b, t, h] * (v[b, t, h] - state.T @ kt))
                out[b, t, h] = state.T @ qt / np.sqrt(dk)
    return torch.from_numpy(out)


@pytest.mark.parametrize("steps,chunk", [(37, 8), (16, 16), (5, 64)])
def test_gdn_chunked_and_recurrent_match_float64_reference(steps, chunk):
    torch.manual_seed(steps)
    query, key = torch.randn(2, steps, 3, 8), torch.randn(2, steps, 3, 8)
    value = torch.randn(2, steps, 3, 6)
    g = gdn_log_decay(torch.randn(3) * 0.5, torch.randn(2, steps, 3), torch.ones(3))
    beta = torch.rand(2, steps, 3)
    reference = _numpy_reference(query, key, value, g, beta).float()
    recurrent, recurrent_state = recurrent_gated_delta_rule(query, key, value, g, beta)
    chunked, chunked_state = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=chunk)
    torch.testing.assert_close(recurrent, reference, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(chunked, reference, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(chunked_state, recurrent_state, atol=2e-5, rtol=1e-4)


def test_gdn_gradients_are_finite_and_recurrence_does_not_sign_flip(config):
    torch.manual_seed(4)
    gdn = GatedDeltaNetV4(config, kernel="recurrent")
    for name, parameter in gdn.named_parameters():
        v4_init.fill_(parameter, f"blocks.0.mixer.{name}")
    hidden = torch.randn(2, 7, config.d_model, requires_grad=True)
    output = gdn(hidden)
    assert bool(torch.isfinite(output).all())
    output.square().mean().backward()
    assert bool(torch.isfinite(hidden.grad).all())
    for name, parameter in gdn.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
    decay = gdn.decay_factors(hidden.detach())
    assert bool((decay > 0).all()) and bool((decay <= 1).all())


def test_gdn_module_kernels_agree(config):
    torch.manual_seed(2)
    chunked = GatedDeltaNetV4(config, chunk_size=4, kernel="chunked")
    for name, parameter in chunked.named_parameters():
        v4_init.fill_(parameter, f"blocks.0.mixer.{name}")
    recurrent = copy.deepcopy(chunked)
    recurrent.kernel = "recurrent"
    hidden = torch.randn(2, 11, config.d_model)
    torch.testing.assert_close(chunked(hidden), recurrent(hidden), atol=1e-5, rtol=1e-4)


def test_gdn_short_conv_is_qwen_silu_without_additive_residual(config):
    gdn = GatedDeltaNetV4(config, kernel="recurrent")
    hidden = torch.randn(2, 5, config.d_model)
    with torch.no_grad():
        gdn.conv1d.weight.zero_()
    query, key, value, _, _, _ = gdn._project(hidden)
    assert torch.count_nonzero(query) == 0
    assert torch.count_nonzero(key) == 0
    assert torch.count_nonzero(value) == 0


# -- PLE -----------------------------------------------------------------------------

def _python_keys(ple, components):
    m0, m1, m2 = ple.multipliers
    bigram = (components[0] * m0) ^ (components[1] * m1)
    trigram = bigram ^ (components[2] * m2)
    per_order = ple.heads_per_order
    return [bigram % rows for rows in ple.rows[:per_order]] + [trigram % rows for rows in ple.rows[per_order:]]


def test_ple_literal_bigram_trigram_preimages(model):
    ple = model.ple
    ids = torch.tensor([[11, 22, 33, 44, 55]])
    expected = [(11, EOS, EOS), (22, 11, EOS), (33, 22, 11), (44, 33, 22), (55, 44, 33)]
    assert ple.ngram_components(ids)[0].tolist() == [list(item) for item in expected]
    keys = ple.keys(ids)[0].tolist()
    for position, components in enumerate(expected):
        assert keys[position] == _python_keys(ple, components), position
    # The previous-token slot is x_{t-1}, never x_t itself.
    assert keys[1] != _python_keys(ple, (22, 22, EOS))


def test_ple_eos_isolation_of_ngrams(model):
    ple = model.ple
    ids = torch.tensor([[11, 22, EOS, 33, 44]])
    assert ple.ngram_components(ids)[0].tolist() == [[11, 0, 0], [22, 11, 0], [0, 22, 11], [33, 0, 0], [44, 33, 0]]
    fresh = ple.keys(torch.tensor([[33, 44]]))[0]
    assert torch.equal(ple.keys(ids)[0, 3:], fresh)
    other_prefix = ple.keys(torch.tensor([[50, 51, EOS, 33, 44]]))[0, 3:]
    assert torch.equal(other_prefix, fresh)


def test_ple_segment_conv_does_not_cross_eos(model):
    ple = model.ple
    with torch.no_grad():
        ple.conv1d.weight.normal_()
    ids = torch.tensor([[11, 22, EOS, 33, 44, 55]])
    values = torch.randn(1, 6, ple.conv1d.in_channels)
    base = ple.segment_conv(values, ids)
    perturbed = values.clone()
    perturbed[:, :3] += 10.0
    after = ple.segment_conv(perturbed, ids)
    torch.testing.assert_close(after[:, 3:], base[:, 3:], atol=0, rtol=0)
    assert not torch.allclose(after[:, :3], base[:, :3])


def test_ple_forward_eos_isolation_and_causality(model, config):
    ple = model.ple
    with torch.no_grad():
        ple.conv1d.weight.normal_(std=0.3)
    ple.eval()
    hidden = torch.randn(1, 6, 4 * config.d_model)
    first = ple(torch.tensor([[11, 22, EOS, 33, 44, 55]]), hidden)
    second = ple(torch.tensor([[40, 41, EOS, 33, 44, 55]]), hidden)
    torch.testing.assert_close(first[:, 3:], second[:, 3:], atol=0, rtol=0)
    changed_future = ple(torch.tensor([[11, 22, EOS, 33, 44, 9]]), hidden)
    torch.testing.assert_close(first[:, :5], changed_future[:, :5], atol=0, rtol=0)


def test_ple_and_gdn_conv_initialization(model):
    assert torch.count_nonzero(model.ple.conv1d.weight) == 0
    gdn = next(block.mixer for block in model.blocks if isinstance(block.mixer, GatedDeltaNetV4))
    weight = gdn.conv1d.weight.detach()
    assert torch.count_nonzero(weight) == weight.numel()
    assert abs(float(weight.std()) - 0.02) < 0.004


def test_ple_tables_are_host_tensors_not_parameters(model):
    names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("ple.tables.") for name in names)
    tables = dict(model.ple.store.named_tables())
    assert len(tables) == 16 and all(table.device.type == "cpu" and table.dtype == torch.bfloat16 for table in tables.values())


# -- MTP -----------------------------------------------------------------------------

def test_mtp_window_semantics():
    assert [mtp_recursive_steps(w) for w in range(5)] == [0, 0, 1, 2, 3]
    assert MTP_MAX_RECURSIVE_STEPS == 3
    for bad in (5, -1, True, 2.0):
        with pytest.raises(ValueError):
            mtp_recursive_steps(bad)


def test_mtp_forward_depth_counts(model, config):
    ids = torch.randint(2, config.vocab_size, (1, 8))
    labels = ids.roll(-1, 1)
    for window, depths in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3)):
        out = model(ids, labels=labels, mtp_window=window)
        assert len(out.get("mtp_logits", [])) == depths
    with pytest.raises(ValueError):
        model(ids, labels=labels, mtp_window=5)
    with pytest.raises(ValueError):
        model(ids, mtp_window=2)


def test_mtp_teacher_forcing_alignment_literal():
    # tokens x_0..x_8 = 100..108; input_ids = x_0..x_7, labels[t] = x_{t+1}.
    labels = torch.arange(101, 109).unsqueeze(0)
    table = mtp_teacher_and_targets(labels, 3, pad_id=1)
    ignore = -100
    # depth 1 predicts t+2 from teacher x_{t+1}
    assert table[0][0].tolist() == [[101, 102, 103, 104, 105, 106, 107, 1]]
    assert table[0][1].tolist() == [[102, 103, 104, 105, 106, 107, 108, ignore]]
    # depth 2 predicts t+3 from teacher x_{t+2}
    assert table[1][0].tolist() == [[102, 103, 104, 105, 106, 107, 1, 1]]
    assert table[1][1].tolist() == [[103, 104, 105, 106, 107, 108, ignore, ignore]]
    # depth 3 predicts t+4 from teacher x_{t+3}
    assert table[2][0].tolist() == [[103, 104, 105, 106, 107, 1, 1, 1]]
    assert table[2][1].tolist() == [[104, 105, 106, 107, 108, ignore, ignore, ignore]]
    for depth, (_, target) in enumerate(table, start=1):
        for t in range(8 - depth):
            assert int(target[0, t]) == 100 + t + depth + 1  # x_{t+depth+1}: t+2, t+3, t+4


def test_mtp_losses_use_aligned_targets_and_recursion(model, config):
    torch.manual_seed(3)
    ids = torch.randint(2, config.vocab_size, (2, 10))
    labels = ids.roll(-1, 1)
    labels[:, -1] = -100
    out = model(ids, labels=labels, mtp_window=4)
    for depth, (_, target) in enumerate(mtp_teacher_and_targets(labels, 3, model.pad_id)):
        logits = out["mtp_logits"][depth]
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), ignore_index=-100, reduction="sum")
        torch.testing.assert_close(out["mtp_loss_sums"][depth], expected)
        assert out["mtp_loss_counts"][depth] == int((target != -100).sum())
    changed = labels.clone()
    changed[0, 0] = (changed[0, 0] + 1) % config.vocab_size
    changed_out = model(ids, labels=changed, mtp_window=4)
    # depth-1 teacher at t=0 changed -> depth 1 and (recursively) depth 2 logits change at t=0
    assert not torch.allclose(changed_out["mtp_logits"][0][0, 0], out["mtp_logits"][0][0, 0])
    assert not torch.allclose(changed_out["mtp_logits"][1][0, 0], out["mtp_logits"][1][0, 0])


# -- KVC / HC ------------------------------------------------------------------------

def test_kvc_reuse_layers_have_q_and_o_but_no_kv(model, config):
    pairs = [pair for pair in config.kvc_pairs if pair[1] < config.num_layers]
    assert pairs
    for source, reuse in pairs:
        reuse_mixer, source_mixer = model.blocks[reuse].mixer, model.blocks[source].mixer
        assert isinstance(reuse_mixer, AttentionV4) and reuse_mixer.role == "reuse"
        assert source_mixer.role == "source"
        names = {name for name, _ in reuse_mixer.named_parameters()}
        assert names == {"q_proj.weight", "q_norm.offset", "o_proj.weight"}
        assert {"k_proj.weight", "v_proj.weight"} <= {name for name, _ in source_mixer.named_parameters()}


def test_kvc_reuse_attends_to_source_bank(model, config):
    source, reuse = config.kvc_pairs[0]
    ids = torch.randint(2, config.vocab_size, (1, 8))
    captured = {}
    handle = model.blocks[reuse].mixer.register_forward_hook(lambda module, args, output: captured.__setitem__("out", output[0]))
    model(ids)
    base = captured["out"].detach().clone()
    with torch.no_grad():
        model.blocks[source].mixer.v_proj.weight.mul_(3.0)
    model(ids)
    handle.remove()
    assert not torch.allclose(captured["out"], base)


def test_hc_read_write_is_input_dependent(model, config):
    hc = model.blocks[0].mixer_hc
    hidden = torch.randn(2, 5, 4 * config.d_model)
    _, _, gates = hc.read_write(hidden)
    assert gates.shape == (2, 5, 4) and bool((gates > 0).all()) and bool((gates < 2).all())
    weights = hc._read_weights(hc.normalize(hidden))
    assert weights.shape == (2, 5, 4, config.d_model)
    assert not torch.allclose(weights[:, 0], weights[:, 1])
    assert not torch.allclose(gates[:, 0], gates[:, 1])
    mixed, residual, _ = hc.read_write(hidden)
    output = torch.randn(2, 5, config.d_model)
    written = hc.write(residual, output, gates)
    expected = hidden + (output.unsqueeze(-2) * gates.unsqueeze(-1)).reshape_as(hidden)
    torch.testing.assert_close(written, expected)


# -- InitSpec: direct construction == materializer ------------------------------------

def test_direct_init_is_bit_identical_to_materializer_for_every_rule(model):
    covered = set()
    for name, tensor in model.named_logical_tensors():
        law = v4_init.init_law(name)
        item = {"name": name, "shape": list(tensor.shape), "dtype": "bfloat16", "init_law": law}
        materialized = materialize_tensor(item)
        direct = tensor.detach().to(torch.bfloat16)
        assert torch.equal(direct.view(torch.int16), materialized.view(torch.int16)), name
        covered.add(law["rule"])
    assert covered == {rule for rule, _, _ in v4_init.INIT_RULES}


def test_special_initializer_values(model):
    tensors = dict(model.named_logical_tensors())
    for name, tensor in tensors.items():
        rule = v4_init.init_law(name)["rule"]
        if rule == "gdn_A_log_log_uniform":
            values = tensor.float().exp()
            assert bool((values >= 0.0099).all()) and bool((values <= 16.1).all())
        elif rule == "gdn_dt_bias_ones":
            assert bool((tensor == 1).all())
        elif rule.endswith("_zero"):
            assert torch.count_nonzero(tensor) == 0, name
    assert v4_init.init_law("ple.conv1d.weight")["kind"] == "zeros"
    assert v4_init.init_law("blocks.0.mixer.conv1d.weight")["kind"] == "normal"


def test_init_law_fails_closed_on_unknown_names():
    with pytest.raises(KeyError):
        v4_init.init_law("blocks.0.mixer.unknown.weight")


def test_initialization_independent_of_construction_order_and_dtype(config):
    first = FlashMini50BBaseInit(config)
    second = FlashMini50BBaseInit(config, dtype=torch.bfloat16)
    for (name, a), (_, b) in zip(first.named_logical_tensors(), second.named_logical_tensors()):
        assert torch.equal(a.detach().to(torch.bfloat16), b.detach().to(torch.bfloat16)), name


# -- required gradient families ------------------------------------------------------

def test_required_parameter_families_receive_finite_gradients(model, config):
    from flashmini.v4_balance import RouterBalance

    model.train()
    torch.manual_seed(5)
    ids = torch.randint(2, config.vocab_size, (2, 8))
    labels = ids.roll(-1, 1)
    labels[:, -1] = -100
    out = model(ids, labels=labels, mtp_window=4)
    balance = RouterBalance(config.section("moe")["routed_experts"], config.section("moe")["top_k"], mode="ga_buffer")
    balance.set_logical_tokens(ids.numel())
    aux = balance.microbatch_loss(out["stats"], ids.numel())
    (out["loss"] + out["mtp_loss"] + 0.01 * aux).backward()
    model.ple.store.collect_gradients()
    named = dict(model.named_parameters())
    source, reuse = next(pair for pair in config.kvc_pairs if pair[1] < config.num_layers)
    required = {
        "token_embeddings": "embed_tokens.weight",
        "lm_head": "lm_head.weight",
        "attention_q": f"blocks.{source}.mixer.q_proj.weight",
        "attention_kv_source": f"blocks.{source}.mixer.k_proj.weight",
        "attention_reuse_q": f"blocks.{reuse}.mixer.q_proj.weight",
        "gdn": "blocks.0.mixer.in_proj_qkvz.weight",
        "hc_read": "blocks.0.mixer_hc.read_down.weight",
        "hc_write": "blocks.0.mixer_hc.write_gate.weight",
        "moe_router": "blocks.0.moe.router.weight",
        "routed_expert": "blocks.0.moe.experts.0.gate_proj.weight",
        "shared_expert": "blocks.0.moe.shared_expert.down_proj.weight",
        "shared_gate": "blocks.0.moe.shared_gate.weight",
        "ple_dense": "ple.key_proj.weight",
        "mtp_fusion": "mtp.fusion.fc_hidden.weight",
        "mtp_attention": "mtp.block.mixer.q_proj.weight",
        "mtp_moe": "mtp.block.moe.router.weight",
    }
    for label, name in required.items():
        grad = named[name].grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0, label
    assert f"blocks.{reuse}.mixer.k_proj.weight" not in named
    assert f"blocks.{reuse}.mixer.v_proj.weight" not in named
    assert model.ple.store._grads, "touched PLE table rows must receive gradients"
    for head, parts in model.ple.store._grads.items():
        for rows, grads in parts:
            assert bool(torch.isfinite(grads).all()) and grads.shape[0] == rows.numel()
    assert named["blocks.0.moe.router.weight"].grad.abs().sum() > 0


# -- architecture identity -----------------------------------------------------------

def _raw():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def test_architecture_sha_covers_geometry_not_training_metadata():
    raw = _raw()
    base = canonical_sha256(architecture_fingerprint_payload(raw))
    assert base == load_config().architecture_sha256
    geometry_edits = [
        ("architecture", "d_model", 2304), ("attention", "head_dim", 128), ("moe", "top_k", 8),
        ("moe", "routed_experts", 96), ("hyperconnections", "lowrank", 128), ("ple", "head_dim", 64),
        ("tokenizer", "vocab_size", 65536), ("mtp", "hidden_layers", 2),
    ]
    for section, key, value in geometry_edits:
        edited = copy.deepcopy(raw)
        assert key in edited[section], (section, key)
        edited[section][key] = value
        assert canonical_sha256(architecture_fingerprint_payload(edited)) != base, (section, key)
    edited = copy.deepcopy(raw)
    edited["mixer"]["attention_layers_one_based"] = [5, 8, 12, 16, 20, 24, 29, 34, 39, 44]
    assert canonical_sha256(architecture_fingerprint_payload(edited)) != base
    edited = copy.deepcopy(raw)
    edited["mtp"]["prediction_window"]["maximum"] = 5
    assert canonical_sha256(architecture_fingerprint_payload(edited)) != base
    edited = copy.deepcopy(raw)
    edited["kvc"]["pairs_zero_based"][0] = [3, 11]
    assert canonical_sha256(architecture_fingerprint_payload(edited)) != base
    for section in ("optimizer", "storage", "initialization"):
        edited = copy.deepcopy(raw)
        edited[section]["flashmini_test_marker"] = 1
        assert canonical_sha256(architecture_fingerprint_payload(edited)) == base, section
        assert canonical_sha256(edited) != canonical_sha256(raw)
