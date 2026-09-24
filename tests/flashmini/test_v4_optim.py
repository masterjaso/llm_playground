"""v4 optimizer taxonomy, logical Muon, AdamW classes, PLE sparse Adam, and global router balancing."""

from __future__ import annotations

import math

import pytest
import torch

from flashmini.base_init_accounting import build_meta_model
from flashmini.base_init_config import load_config
from flashmini.base_init_model import FlashMini50BBaseInit, surrogate_config
from flashmini.base_init_optimizer import ADAMW, MUON, PLE_ADAM, OptimizerTaxonomy
from flashmini.v4_balance import RouterBalance
from flashmini.v4_optim import LogicalAdamW, LogicalMuon, OptimizerSettings, OptimizerStack, newton_schulz
from flashmini.v4_ple_store import PLESparseAdam, PLETableStore

SETTINGS = {
    "muon": {"lr": 0.02, "weight_decay": 0.1, "ns_dtype": "float32"},
    "adamw": {"lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8,
              "weight_decay": {"embedding": 0.05, "control_matrix": 0.1, "no_decay": 0.0}},
    "ple": {"lr": 1e-2, "betas": [0.9, 0.95], "eps": 1e-8},
    "grad_clip": 1.0,
}


@pytest.fixture(scope="module")
def config():
    return surrogate_config()


def _assert_partition(cls):
    rows = torch.cat([item.row_index() for item in cls.slices])
    assert torch.equal(rows.sort().values, torch.arange(cls.shape[0]))


def test_every_production_tensor_is_classified_and_partitioned():
    config = load_config()
    taxonomy = OptimizerTaxonomy(config)
    families = {MUON: 0, ADAMW: 0, PLE_ADAM: 0}
    for name, tensor in build_meta_model(config).named_logical_tensors():
        cls = taxonomy.classify(name, tuple(tensor.shape))
        _assert_partition(cls)
        for family in {item.family for item in cls.slices}:
            families[family] += 1
    assert families[PLE_ADAM] == 16 and families[MUON] > 10_000 and families[ADAMW] > 500


def test_fused_projections_split_into_logical_operators(config):
    taxonomy = OptimizerTaxonomy(config)
    attention = config.section("attention")
    heads, dim = attention["query_heads"], attention["head_dim"]
    q = taxonomy.classify("blocks.3.mixer.q_proj.weight", (heads * dim * 2, config.d_model))
    assert [s.logical_operator for s in q.slices] == ["attention_query", "attention_output_gate"]
    assert [s.family for s in q.slices] == [MUON, ADAMW]
    query_rows = q.slices[0].row_index()
    assert query_rows[:dim].tolist() == list(range(dim))
    assert query_rows[dim:2 * dim].tolist() == list(range(2 * dim, 3 * dim))
    gdn = config.section("gdn")
    ratio = gdn["value_heads"] // gdn["key_query_heads"]
    widths = [gdn["key_head_dim"], gdn["key_head_dim"], ratio * gdn["value_head_dim"], ratio * gdn["value_head_dim"]]
    qkvz = taxonomy.classify("blocks.0.mixer.in_proj_qkvz.weight", (gdn["key_query_heads"] * sum(widths), config.d_model))
    assert [s.logical_operator for s in qkvz.slices] == ["gdn_query", "gdn_key", "gdn_value", "gdn_output_gate_z"]
    assert [s.width for s in qkvz.slices] == widths
    assert [s.family for s in qkvz.slices] == [MUON, MUON, MUON, ADAMW]
    ba = taxonomy.classify("blocks.0.mixer.in_proj_ba.weight", (gdn["value_heads"] * 2, config.d_model))
    assert [s.logical_operator for s in ba.slices] == ["gdn_beta", "gdn_decay_a"] and ba.family == ADAMW
    mtp_q = taxonomy.classify("mtp.block.mixer.q_proj.weight", (heads * dim * 2, config.d_model))
    assert mtp_q.slices[0].logical_operator == "mtp_attention_query"


def test_classification_fails_closed(config):
    taxonomy = OptimizerTaxonomy(config)
    with pytest.raises(KeyError):
        taxonomy.classify("blocks.0.mixer.surprise.weight", (8, 8))
    with pytest.raises(KeyError):
        taxonomy.classify("blocks.0.moe.experts.0.fused_proj.weight", (8, 8))
    with pytest.raises(ValueError):
        taxonomy.classify("blocks.3.mixer.q_proj.weight", (7, config.d_model))


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    matrix = torch.randn(24, 40)
    singular = torch.linalg.svdvals(newton_schulz(matrix))
    assert float(singular.min()) > 0.5 and float(singular.max()) < 1.3


def test_muon_orthogonalizes_each_logical_slice_separately(config):
    taxonomy = OptimizerTaxonomy(config)
    attention = config.section("attention")
    shape = (attention["query_heads"] * attention["head_dim"] * 2, config.d_model)
    cls = taxonomy.classify("blocks.3.mixer.q_proj.weight", shape)
    param = torch.nn.Parameter(torch.zeros(shape))
    torch.manual_seed(1)
    param.grad = torch.randn(shape)
    muon = LogicalMuon([("blocks.3.mixer.q_proj.weight", param, cls)], lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False)
    muon.step()
    expected = torch.zeros(shape)
    for item in cls.slices:
        if item.family != MUON:
            continue
        rows = item.row_index()
        expected[rows] = -newton_schulz(param.grad[rows]) * 0.2 * math.sqrt(max(item.rows, item.columns))
    torch.testing.assert_close(param.detach(), expected, atol=1e-5, rtol=1e-4)
    whole = -newton_schulz(param.grad) * 0.2 * math.sqrt(max(shape))
    assert not torch.allclose(param.detach(), whole, atol=1e-3)


def test_optimizer_stack_families_and_decay_classes(config):
    model = FlashMini50BBaseInit(config)
    stack = OptimizerStack(model, OptimizerTaxonomy(config), OptimizerSettings.from_mapping(SETTINGS))
    assert isinstance(stack.adamw, LogicalAdamW)
    assert stack.adamw.weight_decay == {"embedding": 0.05, "control_matrix": 0.1, "no_decay": 0.0}
    adamw_ids = {id(p) for p in stack.adamw.param_groups[0]["params"]}
    assert id(model.embed_tokens.weight) in adamw_ids and id(model.lm_head.weight) in adamw_ids
    muon_ids = {id(p) for p in stack.muon.param_groups[0]["params"]}
    assert id(model.blocks[0].moe.experts[0].gate_proj.weight) in muon_ids
    assert id(model.blocks[0].moe.router.weight) not in muon_ids
    mixed = model.blocks[3].mixer.q_proj.weight
    assert id(mixed) in muon_ids and id(mixed) in adamw_ids
    assert stack.ple.lr == 1e-2
    covered = muon_ids | adamw_ids
    assert covered == {id(p) for p in model.parameters()}


def test_mixed_attention_projection_uses_muon_for_query_and_adamw_for_gate(config):
    taxonomy = OptimizerTaxonomy(config)
    attention = config.section("attention")
    shape = (attention["query_heads"] * attention["head_dim"] * 2, config.d_model)
    cls = taxonomy.classify("blocks.3.mixer.q_proj.weight", shape)
    param = torch.nn.Parameter(torch.zeros(shape))
    param.grad = torch.ones_like(param)

    muon = LogicalMuon(
        [("blocks.3.mixer.q_proj.weight", param, cls)],
        lr=0.1, weight_decay=0.0, momentum=0.0, nesterov=False,
    )
    adamw = LogicalAdamW(
        [("blocks.3.mixer.q_proj.weight", param, cls)],
        lr=0.01, betas=(0.0, 0.0), eps=1e-8,
        weight_decay={"embedding": 0.0, "control_matrix": 0.0, "no_decay": 0.0},
    )
    muon.step()
    after_muon = param.detach().clone()
    query_rows = cls.slices[0].row_index()
    gate_rows = cls.slices[1].row_index()
    assert torch.count_nonzero(after_muon.index_select(0, query_rows)) > 0
    assert torch.count_nonzero(after_muon.index_select(0, gate_rows)) == 0

    adamw.step()
    assert torch.equal(param.detach().index_select(0, query_rows), after_muon.index_select(0, query_rows))
    assert torch.count_nonzero(param.detach().index_select(0, gate_rows)) > 0


def test_optimizer_settings_require_explicit_values():
    broken = {**SETTINGS, "adamw": {**SETTINGS["adamw"], "weight_decay": {"embedding": 0.1}}}
    with pytest.raises(ValueError):
        OptimizerSettings.from_mapping(broken)
    with pytest.raises((KeyError, ValueError, TypeError)):
        OptimizerSettings.from_mapping({**SETTINGS, "muon": {"lr": None, "weight_decay": 0.1, "ns_dtype": "float32"}})


def test_non_finite_gradient_refuses_step(config):
    model = FlashMini50BBaseInit(config)
    stack = OptimizerStack(model, OptimizerTaxonomy(config), OptimizerSettings.from_mapping(SETTINGS))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    model.blocks[0].moe.router.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        stack.step()
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), name


# -- PLE sparse Adam -----------------------------------------------------------------

def _dense_adam_reference(weight, grads, lr, betas, eps):
    m = torch.zeros_like(weight)
    v = torch.zeros_like(weight)
    for step, grad in enumerate(grads, start=1):
        m = betas[0] * m + (1 - betas[0]) * grad
        v = betas[1] * v + (1 - betas[1]) * grad * grad
        weight = (weight - lr * (m / (1 - betas[0] ** step)) / ((v / (1 - betas[1] ** step)).sqrt() + eps)).to(torch.bfloat16).float()
    return weight


def test_ple_sparse_adam_updates_only_touched_rows_with_row_local_steps():
    store = PLETableStore([11, 13], 4, device="cpu")
    store.allocate(device="cpu")
    torch.manual_seed(2)
    for table in store.tables:
        table.copy_(torch.randn_like(table, dtype=torch.float32) * 0.02)
    initial = [table.clone() for table in store.tables]
    optimizer = PLESparseAdam(store, lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    # step 1 touches rows 2 and 5 of head 0 (row 2 twice) and row 7 of head 1; step 2 touches row 2 only.
    grads_row2 = [torch.randn(4), torch.randn(4)]
    second_hit = torch.randn(4)
    grad_row5, grad_row7 = torch.randn(4), torch.randn(4)
    store._grads = {0: [(torch.tensor([2, 5, 2]), torch.stack([grads_row2[0], grad_row5, second_hit]))], 1: [(torch.tensor([7]), grad_row7[None])]}
    optimizer.step(optimizer.reduce_gradients())
    store._grads = {0: [(torch.tensor([2]), grads_row2[1][None])]}
    optimizer.step(optimizer.reduce_gradients())
    expected_row2 = _dense_adam_reference(initial[0][2].float(), [grads_row2[0] + second_hit, grads_row2[1]], 1e-2, (0.9, 0.95), 1e-8)
    expected_row5 = _dense_adam_reference(initial[0][5].float(), [grad_row5], 1e-2, (0.9, 0.95), 1e-8)
    torch.testing.assert_close(store.tables[0][2].float(), expected_row2, atol=1e-6, rtol=0)
    torch.testing.assert_close(store.tables[0][5].float(), expected_row5, atol=1e-6, rtol=0)
    untouched = [row for row in range(11) if row not in (2, 5)]
    assert torch.equal(store.tables[0][untouched], initial[0][untouched])
    assert torch.equal(store.tables[1][[r for r in range(13) if r != 7]], initial[1][[r for r in range(13) if r != 7]])
    assert optimizer.state[0].step.tolist()[2] == 2 and optimizer.state[0].step.tolist()[5] == 1
    assert all(table.dtype == torch.bfloat16 for table in store.tables)


def test_ple_sparse_adam_checkpoint_roundtrip(tmp_path):
    store = PLETableStore([5, 7], 3, device="cpu")
    store.allocate(device="cpu")
    for table in store.tables:
        table.normal_()
    optimizer = PLESparseAdam(store, lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    store._grads = {1: [(torch.tensor([3]), torch.ones(1, 3))]}
    optimizer.step(optimizer.reduce_gradients())
    optimizer.save(tmp_path)
    restored_store = PLETableStore([5, 7], 3, device="cpu")
    restored_store.allocate(device="cpu")
    restored = PLESparseAdam(restored_store, lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    restored.load(tmp_path)
    for a, b in zip(store.tables, restored_store.tables):
        assert torch.equal(a, b)
    assert torch.equal(restored.state[1].exp_avg, optimizer.state[1].exp_avg)
    assert torch.equal(restored.state[1].step, optimizer.state[1].step)


def test_ple_lookup_transfers_only_touched_rows(config):
    model = FlashMini50BBaseInit(config)
    model.train()
    ids = torch.tensor([[5, 6, 5, 6, 5, 6]])
    model.ple.store.stats.snapshot_and_reset()
    model(ids)
    stats = model.ple.store.stats
    assert stats.lookups == 16 * ids.numel()
    assert stats.max_working_set_rows <= ids.numel() < min(model.ple.store.rows)
    assert stats.transfer_bytes == stats.unique_rows * model.ple.store.head_dim * 2


# -- global router balancing ---------------------------------------------------------

def _stats(counts, probs, layer="blocks.0", depth=None):
    stat = {"layer": layer, "expert_counts": torch.tensor(counts, dtype=torch.float64),
            "router_prob_sum": probs, "tokens": int(sum(counts) // 2), "top_k": 2}
    if depth is not None:
        stat["depth"] = depth
    return stat


def _logical_objective(all_counts, all_probs, experts=4, top_k=2):
    counts = torch.stack(all_counts).sum(0)
    tokens = counts.sum() / top_k
    probs = torch.stack(all_probs).sum(0) / tokens
    return experts * ((counts / (tokens * top_k)).float() * probs).sum()


def _microbatches(seed, sizes):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for tokens in sizes:
        logits = torch.randn(tokens, 4, generator=generator, requires_grad=True)
        probs = logits.softmax(-1)
        top = probs.topk(2, dim=-1).indices
        counts = torch.bincount(top.reshape(-1), minlength=4).to(torch.float64)
        batches.append((logits, probs, counts))
    return batches


def test_uniform_routing_gives_unit_balance_loss():
    balance = RouterBalance(4, 2, mode="ga_buffer")
    balance.set_logical_tokens(8)
    loss = balance.microbatch_loss([_stats([4, 4, 4, 4], torch.full((4,), 2.0))], positions=8)
    assert float(loss) == pytest.approx(1.0)
    assert balance.finalize()["router_aux_logical"] == pytest.approx(1.0)


def test_exact_prepass_gradient_equals_logical_batch_gradient():
    sizes = [5, 9, 3]
    batches = _microbatches(0, sizes)
    reference = _logical_objective([c for _, _, c in batches], [p.sum(0) for _, p, _ in batches])
    reference_grads = torch.autograd.grad(reference, [logits for logits, _, _ in batches])
    batches = _microbatches(0, sizes)
    balance = RouterBalance(4, 2, mode="exact_prepass")
    balance.set_logical_tokens(sum(sizes))
    balance.add_prepass([_stats(c.tolist(), p.detach().sum(0)) for _, p, c in batches])
    total = sum(balance.microbatch_loss([_stats(c.tolist(), p.sum(0))], positions=n) for (_, p, c), n in zip(batches, sizes))
    grads = torch.autograd.grad(total, [logits for logits, _, _ in batches])
    torch.testing.assert_close(total, reference)
    for a, b in zip(grads, reference_grads):
        torch.testing.assert_close(a, b)
    assert balance.finalize()["router_aux_logical"] == pytest.approx(float(reference), rel=1e-6)


def test_logical_metric_is_partition_invariant_and_ga_buffer_final_microbatch_is_exact():
    sizes = [4, 6, 2, 8]
    batches = _microbatches(1, sizes)
    reference = float(_logical_objective([c for _, _, c in batches], [p.sum(0) for _, p, _ in batches]))
    for grouping in ([[0, 1, 2, 3]], [[0], [1], [2], [3]], [[0, 1], [2, 3]]):
        balance = RouterBalance(4, 2, mode="ga_buffer")
        balance.set_logical_tokens(sum(sizes))
        for group in grouping:
            counts = sum(batches[i][2] for i in group)
            probs = sum(batches[i][1].detach().sum(0) for i in group)
            balance.microbatch_loss([_stats(counts.tolist(), probs)], positions=sum(sizes[i] for i in group))
        assert balance.finalize()["router_aux_logical"] == pytest.approx(reference, rel=1e-6)


def test_balance_is_invariant_to_batch_scale():
    batches = _microbatches(2, [6])
    counts, probs = batches[0][2], batches[0][1].detach().sum(0)
    single = RouterBalance(4, 2, mode="ga_buffer")
    single.set_logical_tokens(6)
    single.microbatch_loss([_stats(counts.tolist(), probs)], positions=6)
    doubled = RouterBalance(4, 2, mode="ga_buffer")
    doubled.set_logical_tokens(12)
    doubled.microbatch_loss([_stats((2 * counts).tolist(), 2 * probs)], positions=12)
    assert single.finalize()["router_aux_logical"] == pytest.approx(doubled.finalize()["router_aux_logical"], rel=1e-9)


def test_mtp_depths_form_one_logical_layer():
    balance = RouterBalance(4, 2, mode="ga_buffer")
    balance.set_logical_tokens(4)
    stats = [_stats([2, 2, 2, 2], torch.full((4,), 1.0), "blocks.0")]
    stats += [_stats([4, 0, 2, 2], torch.tensor([1.5, 0.5, 1.0, 1.0]), "mtp.block", depth) for depth in (1, 2, 3)]
    balance.microbatch_loss(stats, positions=4)
    layers = balance.finalize()["layers"]
    assert set(layers) == {"blocks.0", "mtp.block"}
    assert layers["mtp.block"]["tokens"] == 12


def test_balance_mode_is_validated():
    with pytest.raises(ValueError):
        RouterBalance(4, 2, mode="microbatch_local")
    balance = RouterBalance(4, 2, mode="exact_prepass")
    balance.set_logical_tokens(4)
    with pytest.raises(RuntimeError):
        balance.microbatch_loss([_stats([2, 2, 2, 2], torch.ones(4))], positions=4)
