from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, KVConfig, MoEConfig, PLEConfig
from flashmini.data_v4.source import SourceCursor
from flashmini.data_v4.virtual import VirtualBatchStream, VirtualCorpus
from flashmini.models import FlashMiniModel
from flashmini.observability import StatusLogger, Watchdog
from flashmini.optim import build_optimizer
from flashmini.production import (
    CHECKPOINT_INTERVAL_TOKENS,
    PREVIEW_PAUSE_TOKENS,
    TOTAL_TRAINING_TOKENS,
    checkpoint_boundary,
    freeze_manifest,
    load_production_config,
    parameter_report,
    validate_freeze_manifest,
)
from flashmini.production_checkpoint import (
    DurableCheckpointManager,
    FilesystemRemoteBackend,
    save_full_checkpoint,
    verify_checkpoint,
)
from flashmini.production_metrics import MetricsLedger
from flashmini.tpu_backend import SessionBudget


def test_frozen_one_b_config_and_report():
    config = load_production_config()
    report = parameter_report(config)
    assert 950_000_000 <= report["total_learned_parameters"] <= 1_050_000_000
    assert report["kvc_parameter_contribution"] == 0
    assert report["ple_parameters"] > 0
    assert report["moe_expert_parameters"] > report["attention_parameters"]
    assert report["categories_sum"] == report["total_learned_parameters"]


def test_freeze_manifest_declares_long_trajectory(tmp_path):
    path = tmp_path / "freeze.json"
    manifest = freeze_manifest(path)
    validate_freeze_manifest(manifest)
    assert manifest["trajectory"]["total_training_tokens"] == TOTAL_TRAINING_TOKENS
    assert manifest["trajectory"]["preview_pause_tokens"] == PREVIEW_PAUSE_TOKENS
    assert manifest["data_view"]["full_corpus_materialization_required"] is False
    assert json.loads(path.read_text())["model_config_sha256"] == manifest["model_config_sha256"]


def test_checkpoint_boundary_is_first_crossed_threshold():
    assert checkpoint_boundary(CHECKPOINT_INTERVAL_TOKENS - 10, 20) == {
        "checkpoint_threshold_tokens": CHECKPOINT_INTERVAL_TOKENS,
        "actual_tokens_seen": CHECKPOINT_INTERVAL_TOKENS + 10,
        "overshoot_tokens": 10,
    }
    assert checkpoint_boundary(0, CHECKPOINT_INTERVAL_TOKENS - 1) is None


class _Tok:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 8 for char in text]}


def test_virtual_cursor_is_deterministic_and_resumable():
    recipe = {"name": "r", "target_tokens": 8, "seed": 17,
              "domains": {"code": {"weight": 1.0, "sources": ["s"]}}}
    source = {"s": {"dataset_id": "owner/data", "revision": "a" * 40,
                      "text_field": "text", "domain": "code"}}
    a = VirtualCorpus(recipe, source, tokenizer_identity="gpt2@" + "b" * 40, seq_len=8)
    rows = a.accept_window("code", a.source_for("code"),
                           [{"document_id": "b", "text": "abcd"}, {"document_id": "a", "text": "abcd"}],
                           tokenizer=_Tok())
    assert rows
    state = a.state()
    b = VirtualCorpus(recipe, source, tokenizer_identity="gpt2@" + "b" * 40, seq_len=8)
    b.restore(state)
    assert b.state()["cursor"] == state["cursor"]
    assert b.scheduler.actual == a.scheduler.actual


def test_virtual_batch_stream_preserves_pending_tokens_on_restore():
    recipe = {"name": "r", "target_tokens": 64, "seed": 17,
              "domains": {"code": {"weight": 1.0, "sources": ["s"]}}}
    source = {"s": {"dataset_id": "owner/data", "revision": "a" * 40,
                      "text_field": "text", "domain": "code"}}
    corpus = VirtualCorpus(recipe, source, tokenizer_identity="gpt2@" + "b" * 40, seq_len=4)

    class Tokenizer(_Tok):
        eos_token_id = 9

    def reader(virtual_source, cursor, limit):
        del virtual_source, limit
        start = int(cursor.offset)
        rows = [{"document_id": str(start), "text": "abcdefgh"}]
        return rows, SourceCursor(offset=start + 1)

    stream = VirtualBatchStream(corpus, tokenizer=Tokenizer(), reader=reader, window_documents=1)
    first_inputs, first_labels = stream.next_batch(1)
    assert first_inputs.shape == (1, 4)
    state = stream.state()
    restored = VirtualBatchStream(
        VirtualCorpus(recipe, source, tokenizer_identity="gpt2@" + "b" * 40, seq_len=4),
        tokenizer=Tokenizer(), reader=reader, window_documents=1,
    )
    restored.restore(state)
    next_a = stream.next_batch(1)
    next_b = restored.next_batch(1)
    assert first_labels.shape == (1, 4)
    assert (next_a[0] == next_b[0]).all()
    assert (next_a[1] == next_b[1]).all()


def test_status_files_are_atomic_and_watchdog_distinguishes_compile(tmp_path):
    status = StatusLogger(tmp_path, run_id="r", heartbeat_interval=0.1)
    status.start()
    status.update(progress=True, phase="training", global_step=1, global_exact_tokens=32, recent_loss=2.0)
    time.sleep(0.12)
    status.stop()
    assert json.loads((tmp_path / "status/heartbeat.json").read_text())["global_step"] == 1
    assert json.loads((tmp_path / "run_status.json").read_text())["tokens_seen"] == 32

    clock = iter([0.0, 10.0, 20.0])
    watchdog = Watchdog(compile_timeout_seconds=15, now=lambda: next(clock))
    watchdog.set_phase("xla_compile")
    assert watchdog.poll() is None
    assert watchdog.poll()["event"] == "XLA_COMPILE_TIMEOUT"


def test_full_checkpoint_remote_replacement(tmp_path):
    config = FlashMiniConfig(vocab_size=17, d_model=8, num_layers=1, num_heads=1,
                             head_dim=8, max_seq_len=4)
    model = FlashMiniModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    identity = {"model_config_sha256": __import__("flashmini.production", fromlist=["model_config_sha256"]).model_config_sha256(config),
                "data_view_fingerprint": "d", "tokenizer_fingerprint": "t"}
    first = tmp_path / "first"
    manifest = save_full_checkpoint(first, model=model, optimizer=optimizer, step=1,
                                    exact_tokens=8, config=config, identity=identity,
                                    data_cursor={"offset": 1})
    assert verify_checkpoint(first)["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    remote = FilesystemRemoteBackend(tmp_path / "remote")
    manager = DurableCheckpointManager(tmp_path / "local", remote)
    result = manager.save_and_sync(step=2, model=model, optimizer=optimizer, exact_tokens=16,
                                   config=config, identity=identity, data_cursor={"offset": 2})
    assert Path(result["local_path"]).is_dir()
    assert (tmp_path / "remote/LATEST.json").is_file()


def test_remote_failure_keeps_previous_verified_checkpoint(tmp_path):
    config = FlashMiniConfig(vocab_size=17, d_model=8, num_layers=1, num_heads=1,
                             head_dim=8, max_seq_len=4)
    model = FlashMiniModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    identity = {"model_config_sha256": __import__("flashmini.production", fromlist=["model_config_sha256"]).model_config_sha256(config)}
    remote = FilesystemRemoteBackend(tmp_path / "remote")
    manager = DurableCheckpointManager(tmp_path / "local", remote)
    manager.save_and_sync(step=1, model=model, optimizer=optimizer, exact_tokens=8,
                          config=config, identity=identity, data_cursor={"offset": 1})

    class FailingRemote(FilesystemRemoteBackend):
        def publish(self, local_checkpoint, manifest):
            raise OSError("injected upload failure")

    broken = DurableCheckpointManager(tmp_path / "local", FailingRemote(tmp_path / "remote"))
    try:
        broken.save_and_sync(step=2, model=model, optimizer=optimizer, exact_tokens=16,
                             config=config, identity=identity, data_cursor={"offset": 2})
    except OSError:
        pass
    else:
        raise AssertionError("injected remote failure unexpectedly succeeded")
    assert verify_checkpoint(tmp_path / "remote/checkpoint_step_1")["step"] == 1
    assert (tmp_path / "local/checkpoint_step_1").is_dir()


def test_session_budget_reserves_checkpoint_window():
    budget = SessionBudget(hard_limit_seconds=100, reserve_seconds=25, started_at=10)
    assert budget.soft_limit_seconds == 75
    assert budget.should_soft_stop(now=85)
    assert budget.seconds_remaining(now=85) == 0


def test_reduced_resume_matches_uninterrupted_update(tmp_path):
    config = FlashMiniConfig(
        architecture_version=3,
        experiment_mode="screening",
        vocab_size=31,
        d_model=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        gdn_per_attention=1,
        attention_layers=[0, 1],
        max_seq_len=8,
        use_ple=True,
        moe=MoEConfig(num_experts=2, top_k=1, shared_experts=1, expert_intermediate=16),
        gdn=GatedDeltaNetConfig(d_state=8, chunk_size=4, residual_in_mixer=False),
        ple=PLEConfig(
            ngram=3, ngram_vocab_size_base=31, heads_per_ngram=2, embed_dim=32,
            injection_layer=0, eos_id=30, sparse=False,
        ),
        kvc=KVConfig(enabled=True),
    )
    torch.manual_seed(123)
    uninterrupted = FlashMiniModel(config)
    uninterrupted_optimizer = build_optimizer(uninterrupted, 1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    labels = torch.roll(input_ids, -1, dims=1)

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=labels)["loss"]
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    update(uninterrupted, uninterrupted_optimizer)
    identity = {"model_config_sha256": __import__("flashmini.production", fromlist=["model_config_sha256"]).model_config_sha256(config)}
    checkpoint = save_full_checkpoint(
        tmp_path / "resume",
        model=uninterrupted,
        optimizer=uninterrupted_optimizer,
        step=1,
        exact_tokens=16,
        config=config,
        identity=identity,
        data_cursor={"sequence": 1},
    )
    del checkpoint
    resumed = FlashMiniModel(config)
    resumed_optimizer = build_optimizer(resumed, 1e-3)
    payload = torch.load(tmp_path / "resume/state.pt", map_location="cpu", weights_only=False)
    resumed.load_state_dict(payload["model_state_dict"])
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert update(uninterrupted, uninterrupted_optimizer) == update(resumed, resumed_optimizer)
    assert all(torch.equal(left, right) for left, right in zip(uninterrupted.parameters(), resumed.parameters()))


def test_metrics_milestones_are_cumulative_and_append_only(tmp_path):
    ledger = MetricsLedger(tmp_path)
    ledger.append("train_status", global_step=1, global_exact_tokens=32)
    ledger.append_milestone(checkpoint_threshold_tokens=25_000_000,
                            actual_tokens_seen=25_000_032, global_step=2,
                            checkpoint_sha256="a")
    ledger.append_milestone(checkpoint_threshold_tokens=50_000_000,
                            actual_tokens_seen=50_000_032, global_step=3,
                            checkpoint_sha256="b")
    assert [row["checkpoint_sha256"] for row in ledger.milestones()] == ["a", "b"]
    assert len(ledger.tail(10)) == 3
