"""v4 runner: config contract, data stream, loss schedule, trainability, NaN guard, resume equivalence."""

from __future__ import annotations

import json
import math

import pytest
import torch

import v4_testlib as lib
from flashmini.v4_data import DataExhausted, PackedTokenStream, load_manifest, write_packed
from flashmini.v4_train import (
    MTP_COEFFICIENT_EARLY,
    MTP_COEFFICIENT_LATE,
    Runner,
    TrainConfig,
    lr_multiplier,
    mtp_coefficient,
)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return lib.write_synthetic_data(tmp_path_factory.mktemp("v4data"))


# -- configuration contract ----------------------------------------------------------

def test_unresolved_donor_settings_fail_startup(tmp_path, manifest):
    config = lib.train_config(tmp_path, manifest, steps_a=2, steps_b=2)
    for path in ("loss.router_aux_coefficient", "distributed.nodes", "optimizer.muon.lr", "schedule.planned_total_tokens", "data.teacher_mixture"):
        section, *rest = path.split(".")
        broken = json.loads(json.dumps(config))
        target = broken[section]
        for key in rest[:-1]:
            target = target[key]
        target[rest[-1]] = None
        with pytest.raises(ValueError, match="unresolved mandatory training settings"):
            TrainConfig(broken)
    broken = json.loads(json.dumps(config))
    broken["data"]["phases"][0]["gradient_accumulation"] = None
    with pytest.raises(ValueError, match=r"data.phases\[0\].gradient_accumulation"):
        TrainConfig(broken)


def test_frozen_loss_contract_cannot_be_overridden(tmp_path, manifest):
    config = lib.train_config(tmp_path, manifest, steps_a=2, steps_b=2)
    for key, value in (("mtp_coefficient", 0.5), ("main_weight", 2.0), ("mtp_window", 2)):
        with pytest.raises(ValueError, match="frozen"):
            TrainConfig(lib.with_overrides(config, loss={key: value}))
    with pytest.raises(ValueError, match="router_aux_coefficient"):
        TrainConfig(lib.with_overrides(config, loss={"router_aux_coefficient": "0.01"}))
    with pytest.raises(ValueError):
        TrainConfig(lib.with_overrides(config, run={"purpose": "production"}))


def test_mtp_coefficient_and_lr_schedules():
    planned = 1000
    assert mtp_coefficient(0, planned) == (MTP_COEFFICIENT_EARLY, "early") == (0.30, "early")
    assert mtp_coefficient(699, planned)[0] == 0.30
    assert mtp_coefficient(700, planned) == (MTP_COEFFICIENT_LATE, "late") == (0.10, "late")
    cosine = {"kind": "cosine", "warmup_tokens": 100, "min_lr_ratio": 0.1}
    assert lr_multiplier(cosine, 50, planned) == pytest.approx(0.5)
    assert lr_multiplier(cosine, 100, planned) == pytest.approx(1.0)
    assert lr_multiplier(cosine, 1000, planned) == pytest.approx(0.1)
    wsd = {"kind": "wsd", "warmup_tokens": 100, "min_lr_ratio": 0.0, "decay_start_fraction": 0.8}
    assert lr_multiplier(wsd, 800, planned) == 1.0
    assert lr_multiplier(wsd, 900, planned) == pytest.approx(0.5)


# -- data -----------------------------------------------------------------------------

def test_packed_data_identity_and_windows(tmp_path):
    write_packed([[5, 6, 7], [8, 9]], tmp_path, tokenizer_fingerprint=lib.FINGERPRINT, vocab_size=64, eos_id=0, shard_tokens=4)
    path = tmp_path / "packed_manifest.json"
    with pytest.raises(ValueError, match="fingerprint"):
        load_manifest(path, expected_fingerprint="1" * 64, vocab_size=64)
    with pytest.raises(ValueError, match="vocab"):
        load_manifest(path, expected_fingerprint=lib.FINGERPRINT, vocab_size=128)
    manifest = load_manifest(path, expected_fingerprint=lib.FINGERPRINT, vocab_size=64, verify_hashes=True)
    assert len(manifest["shards"]) == 2
    stream = PackedTokenStream(manifest, seq_len=3)
    assert stream.window(0).tolist() == [5, 6, 7, 0]
    assert stream.window(1).tolist() == [0, 8, 9, 0]
    batch = stream.microbatch(0, micro_index=0, rank=0, world=1, micro_batch=2)
    assert torch.equal(batch.labels[:, :-1], batch.input_ids[:, 1:])
    with pytest.raises(DataExhausted):
        stream.window(2)


def test_logical_batch_is_independent_of_rank_and_accumulation_split(manifest):
    stream = PackedTokenStream(load_manifest(manifest, expected_fingerprint=lib.FINGERPRINT, vocab_size=64), seq_len=16)

    def windows(world, accumulation, micro):
        return sorted(i for m in range(accumulation) for r in range(world)
                      for i in stream.microbatch(10, micro_index=m, rank=r, world=world, micro_batch=micro).window_indices)

    assert windows(1, 4, 2) == windows(2, 2, 2) == windows(4, 1, 2) == windows(1, 1, 8) == list(range(10, 18))


# -- training --------------------------------------------------------------------------

def test_surrogate_trains_and_all_parameter_classes_update(tmp_path, manifest):
    config = lib.train_config(tmp_path, manifest, steps_a=10, steps_b=30, micro=(2, 4), accumulation=(2, 2), seq=(32, 16), mode="ga_buffer")
    path = lib.write_config(tmp_path / "train.yaml", config)
    runner = Runner(TrainConfig.load(path))
    runner.setup()
    initial = lib.full_state(runner)
    runner.run()
    runner.close()
    final = lib.full_state(runner)
    records = [r for r in lib.metrics(tmp_path / "metrics.jsonl") if r["event"] == "step"]
    assert len(records) == 40
    for record in records:
        for key in ("loss_main", "loss_mtp", "router_aux_logical", "grad_norm_total", "loss_total"):
            assert math.isfinite(record[key]), key
        assert set(record["loss_mtp_by_depth"]) == {"t+2", "t+3", "t+4"}
    first, last = records[0], records[-1]
    assert last["loss_main"] < first["loss_main"] - 1.5
    assert last["loss_mtp"] < first["loss_mtp"] - 1.5
    planned = config["schedule"]["planned_total_tokens"]
    previous = 0
    for record in records:
        assert record["mtp_coefficient"] == (0.30 if previous < 0.7 * planned else 0.10)
        previous = record["tokens_consumed"]
    assert records[-1]["mtp_coefficient"] == 0.10
    assert min(r["expert_load_entropy_normalized_min"] for r in records) > 0.6
    assert max(r["expert_max_over_mean_load"] for r in records) < 2.5
    assert all(r["ple"]["rows_updated_owned"] > 0 for r in records)
    assert {r["phase"] for r in records} == {"short", "shorter"}
    unchanged = [name for name in final if name.startswith("param/") and torch.equal(final[name], initial[name])]
    assert unchanged == []
    for head in range(16):
        assert not torch.equal(final[f"ple/ple.tables.{head}.weight"], initial[f"ple/ple.tables.{head}.weight"])
    assert (tmp_path / "checkpoints" / "latest").exists()


def test_non_finite_loss_fails_the_step_without_update(tmp_path, manifest):
    config = lib.train_config(tmp_path, manifest, steps_a=2, steps_b=2)
    runner = Runner(TrainConfig(config))
    runner.setup()
    before = lib.full_state(runner)
    with torch.no_grad():
        runner.model.lm_head.weight[0, 0] = float("inf")
    poisoned = lib.full_state(runner)
    with pytest.raises(FloatingPointError, match="non-finite"):
        runner.run()
    runner.close()
    after = lib.full_state(runner)
    for name in before:
        assert torch.equal(after[name], poisoned[name]), name
    assert runner.state["optimizer_step"] == 0
    events = [r["event"] for r in lib.metrics(tmp_path / "metrics.jsonl")]
    assert events[-1] == "failed"


@pytest.mark.parametrize("mode", ["exact_prepass", "ga_buffer"])
def test_checkpoint_resume_is_bit_identical_to_continuous_training(tmp_path, manifest, mode):
    continuous_root, resumed_root = tmp_path / "continuous", tmp_path / "resumed"
    base = lib.train_config(continuous_root, manifest, steps_a=3, steps_b=3, mode=mode)
    continuous = lib.run(lib.write_config(continuous_root / "train.yaml", base), keep_runner=True)
    split = lib.train_config(resumed_root, manifest, steps_a=3, steps_b=3, mode=mode)
    split["checkpoint"]["interval_optimizer_steps"] = 4
    split["run"]["stop_after_optimizer_steps"] = 4
    lib.run(lib.write_config(resumed_root / "part1.yaml", split))
    assert (resumed_root / "checkpoints" / "step_00000004" / "COMPLETE.json").exists()
    split["run"]["stop_after_optimizer_steps"] = None
    resumed = lib.run(lib.write_config(resumed_root / "part2.yaml", split), keep_runner=True)
    assert continuous.state == resumed.state
    assert continuous.state["optimizer_step"] == 6 and continuous.state["mtp_phase"] == "late"
    a, b = lib.full_state(continuous), lib.full_state(resumed)
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    steps_a = {r["optimizer_step"]: r for r in lib.metrics(continuous_root / "metrics.jsonl") if r["event"] == "step"}
    steps_b = {r["optimizer_step"]: r for r in lib.metrics(resumed_root / "metrics.jsonl") if r["event"] == "step"}
    for step in (5, 6):
        for key in ("loss_main", "loss_mtp", "router_aux_logical", "grad_norm_total", "lr_multiplier"):
            assert steps_a[step][key] == steps_b[step][key], (step, key)
    assert any(r["event"] == "resumed" for r in lib.metrics(resumed_root / "metrics.jsonl"))


def test_resume_rejects_changed_training_semantics(tmp_path, manifest):
    config = lib.train_config(tmp_path, manifest, steps_a=2, steps_b=1)
    config["run"]["stop_after_optimizer_steps"] = 1
    config["checkpoint"]["interval_optimizer_steps"] = 1
    lib.run(lib.write_config(tmp_path / "a.yaml", config))
    changed = lib.with_overrides(config, loss={"router_aux_coefficient": 0.02, "router_balance_mode": "exact_prepass"})
    changed["run"]["stop_after_optimizer_steps"] = 2
    with pytest.raises(ValueError, match="train_config_sha256"):
        lib.run(lib.write_config(tmp_path / "b.yaml", changed))
