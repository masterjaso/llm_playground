"""Fresh v3 training, restart trajectory, and persisted comparison evidence."""

import copy
import json

import numpy as np
import pytest
import torch

from flashmini.comparison import validate_ple_pair
from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, MoEConfig, PLEConfig
from flashmini.data import MemmapDataset, prepare_streaming_documents, sha256_file
from flashmini.models import FlashMiniModel
from flashmini.optim import build_optimizer
from flashmini.training import train


def tiny_config(ple=False):
    return FlashMiniConfig(architecture_version=3, vocab_size=32, d_model=16,
        num_layers=4, num_heads=2, head_dim=8, max_seq_len=8, use_ple=ple,
        moe=MoEConfig(num_experts=2, top_k=1, expert_intermediate=16),
        gdn=GatedDeltaNetConfig(d_state=4, chunk_size=4),
        ple=PLEConfig(table_size=37, ngram_vocab_size_base=37, heads_per_ngram=2,
                      embed_dim=16, head_dim=4, eos_id=31, offload="cpu"))


def make_data(path):
    rng = np.random.default_rng(12)
    prepare_streaming_documents((rng.integers(0, 31, size=15) for _ in range(100)),
        path, seq_len=8, eos_id=31, seed=17, val_fraction=0.2,
        provenance={"dataset_id": "synthetic-test", "dataset_revision": "0" * 40,
                    "tokenizer_id": "integer-test", "tokenizer_revision": "1" * 40})
    return MemmapDataset(path)


def _test_fingerprint():
    # A minimal, consistent execution fingerprint for the test fixtures. The
    # same dict is used for every run so resume fingerprint matching passes.
    return {
        "git_commit": "test-commit",
        "git_dirty": False,
        "source_sha256": "test-source",
        "source_files": {},
        "config_sha256": "test-config",
        "data_manifest_sha256": "test-data",
        "python_version": "3.13",
        "platform": "test",
        "pyproject_sha256": None,
        "uv_lock_sha256": None,
        "requirements_sha256": None,
        "nvidia_driver_version": "",
        "torch": {"torch_version": "test", "cuda_available": False,
                  "cuda_version": None, "device_count": 0, "devices": []},
        "fingerprint_sha256": "test-fingerprint",
    }


def fit(config, dataset, directory, *, resume=None, stop=None, aux_loss_coef=None, eval_prefix=None,
        ple_lr_multiplier=5):
    torch.manual_seed(17)
    model = FlashMiniModel(copy.deepcopy(config))
    optimizer = build_optimizer(model, 0.001, ple_lr_multiplier=ple_lr_multiplier)
    summary = train(model, optimizer, dataset, config, directory, total_tokens=64,
        seq_len=8, device=torch.device("cpu"), batch_size=2, seed=17,
        log_every=1, ckpt_every_tokens=16, warmup_tokens=16, cosine_decay=True,
        min_lr_ratio=0.1, resume_from=resume, stop_after_tokens=stop,
        use_amp=False, aux_loss_coef=aux_loss_coef,
        run_metadata={"source_sha256": "test-source",
                      "execution_fingerprint": _test_fingerprint()},
        val_dataset=MemmapDataset(dataset.data_dir, "val") if eval_prefix is not None else None,
        eval_every_tokens=32 if eval_prefix is not None else 0, val_max_batches=eval_prefix)
    return model, summary


def test_v3_restart_is_exact_and_keeps_single_checkpoint(tmp_path):
    data = make_data(tmp_path / "data")
    cfg = tiny_config(True)
    full, summary = fit(cfg, data, tmp_path / "full")
    _, paused = fit(cfg, data, tmp_path / "resumed", stop=32)
    assert paused["status"] == "paused"
    resumed, result = fit(cfg, data, tmp_path / "resumed",
        resume=tmp_path / "resumed/checkpoints/step_2.pt")
    assert result["seq_len"] == data.seq_len == 8
    assert result["tokens_seen"] == summary["tokens_seen"] == 64
    assert result["clipping_counts"] == summary["clipping_counts"]
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    assert len(list((tmp_path / "resumed/checkpoints").glob("step_*.pt"))) == 1
    rows = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert all("grad_clip_fraction_shared" in row for row in rows)


def test_real_b3_c3_checkpoint_metadata_is_comparable(tmp_path, monkeypatch):
    data = make_data(tmp_path / "data")
    for name, ple in (("b", False), ("c", True)):
        fit(tiny_config(ple), data, tmp_path / name, eval_prefix=2)
    b, c = [torch.load(tmp_path / name / "checkpoints/step_4.pt", weights_only=False)
            for name in ("b", "c")]
    result = validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))
    assert result["training_seed"] == 17
    assert result["data_contract"]["sampling"] == "seeded_epoch_permutation_v1"
    import sys

    from flashmini.compare_ple import main
    monkeypatch.setattr(sys, "argv", ["compare_ple", "--baseline", str(tmp_path / "b"),
        "--candidate", str(tmp_path / "c"), "--data-dir", str(tmp_path / "data"),
        "--out", str(tmp_path / "comparison.json")])
    main()
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["baseline"] == "baseline"
    assert "candidate" in report["comparisons"]
    assert report["skipped_tuning_sequences"] == 2
    monkeypatch.setattr(sys, "argv", ["compare_ple", "--baseline", str(tmp_path / "b"),
        "--candidate", str(tmp_path / "c"), "--data-dir", str(tmp_path / "data"),
        "--skip-sequences", "0", "--out", str(tmp_path / "bad.json")])
    with pytest.raises(ValueError, match="holdout overlaps"):
        main()


def test_comparison_rejects_different_ple_optimizer_recipe(tmp_path):
    data = make_data(tmp_path / "data")
    fit(tiny_config(), data, tmp_path / "b", ple_lr_multiplier=1)
    fit(tiny_config(True), data, tmp_path / "c", ple_lr_multiplier=5)
    b, c = [torch.load(tmp_path / name / "checkpoints/step_4.pt", weights_only=False)
            for name in ("b", "c")]
    with pytest.raises(ValueError, match="execution_policy"):
        validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))
    for saved in (b, c):
        policy = saved["extra"]["training"]["run_metadata"]["execution_policy"]
        policy["optimizer_recipe"] = dict.fromkeys(policy["optimizer_recipe"])
    with pytest.raises(ValueError, match="optimizer recipe contains invalid"):
        validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))


def test_objective_mismatch_rejected_on_resume_and_comparison(tmp_path):
    data = make_data(tmp_path / "data")
    fit(tiny_config(), data, tmp_path / "b", aux_loss_coef=0)
    fit(tiny_config(True), data, tmp_path / "c", aux_loss_coef=10)
    b, c = [torch.load(tmp_path / name / "checkpoints/step_4.pt", weights_only=False)
            for name in ("b", "c")]
    with pytest.raises(ValueError, match="execution_policy"):
        validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))
    with pytest.raises(ValueError, match="execution_policy"):
        fit(tiny_config(), data, tmp_path / "b", aux_loss_coef=10,
            resume=tmp_path / "b/checkpoints/step_4.pt")
    c["extra"]["training"]["run_metadata"]["execution_policy"] = copy.deepcopy(
        b["extra"]["training"]["run_metadata"]["execution_policy"])
    c["extra"]["training"]["run_metadata"]["execution_policy"]["precision"] = "cuda_bfloat16_autocast"
    with pytest.raises(ValueError, match="execution_policy"):
        validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))


def test_training_rejects_wrong_length_before_writing(tmp_path):
    data = make_data(tmp_path / "data")
    cfg = tiny_config()
    cfg.max_seq_len = 2048
    with pytest.raises(ValueError, match="sequence length"):
        fit(cfg, data, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_dataset_eos_must_match_config(tmp_path):
    data = make_data(tmp_path / "data")
    config = tiny_config(True)
    config.ple.eos_id = 30
    with pytest.raises(ValueError, match="dataset EOS differs"):
        fit(config, data, tmp_path / "bad")


def test_train_and_validation_split_contracts(tmp_path):
    data = make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="train split"):
        fit(tiny_config(), MemmapDataset(data.data_dir, "val"), tmp_path / "bad")
    wrong = make_data(tmp_path / "wrong")
    manifest_path = wrong.data_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["eos_id"] = 30
    manifest_path.write_text(json.dumps(manifest))
    config = tiny_config()
    model = FlashMiniModel(config)
    kwargs = {"total_tokens": 64, "seq_len": 8, "device": torch.device("cpu"),
              "batch_size": 2, "eval_every_tokens": 32, "val_max_batches": 2}
    with pytest.raises(ValueError, match="dataset EOS differs"):
        train(model, build_optimizer(model, 0.001), data, config, tmp_path / "badval",
              val_dataset=MemmapDataset(wrong.data_dir, "val"), **kwargs)
    with pytest.raises(ValueError, match="val split"):
        train(model, build_optimizer(model, 0.001), data, config, tmp_path / "train_as_val",
              val_dataset=data, **kwargs)


@pytest.mark.parametrize("field,value", [("tokens_seen", 80), ("real_tokens_seen", 80),
                                         ("step", 99), ("clipping_counts.shared", 99),
                                         ("both_token_counters", 24)])
def test_resume_counter_consistency(tmp_path, field, value):
    data = make_data(tmp_path / "data")
    config = tiny_config()
    fit(config, data, tmp_path / "run", stop=32)
    path = tmp_path / "run/checkpoints/step_2.pt"
    saved = torch.load(path, weights_only=False)
    if field == "step":
        saved["step"] = value
    elif field == "both_token_counters":
        saved["extra"]["tokens_seen"] = saved["extra"]["real_tokens_seen"] = value
    elif field == "clipping_counts.shared":
        saved["extra"]["clipping_counts"]["shared"] = value
    else:
        saved["extra"][field] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match="v3 checkpoint|v3 clipping"):
        fit(config, data, tmp_path / "run", resume=path)


def test_short_epoch_end_checkpoint_remains_resumable(tmp_path):
    prepare_streaming_documents([range(1, 9), range(9, 17)], tmp_path / "data",
        seq_len=8, eos_id=31, seed=0, val_fraction=0.5)
    data = MemmapDataset(tmp_path / "data")
    config = tiny_config()
    kwargs = {"total_tokens": 32, "seq_len": 8, "device": torch.device("cpu"),
              "batch_size": 2, "seed": 17, "allow_repeated_corpus": True,
              "ckpt_every_tokens": 8, "use_amp": False,
              "run_metadata": {"source_sha256": "test-source",
                               "execution_fingerprint": _test_fingerprint()}}
    torch.manual_seed(17)
    model = FlashMiniModel(config)
    train(model, build_optimizer(model, 0.001), data, config, tmp_path / "run",
          stop_after_tokens=8, **kwargs)
    model = FlashMiniModel(config)
    result = train(model, build_optimizer(model, 0.001), data, config, tmp_path / "run",
        resume_from=tmp_path / "run/checkpoints/step_1.pt", **kwargs)
    assert result["steps"] == 4 and result["tokens_seen"] == 32


@pytest.mark.parametrize("field,value", [("eos_id", 30), ("hash_seed", 9), ("conv_dilation", 2)])
def test_comparison_rejects_unmatched_ple_spec(tmp_path, field, value):
    data = make_data(tmp_path / "data")
    for name, ple in (("b", False), ("c", True)):
        fit(tiny_config(ple), data, tmp_path / name)
    b, c = [torch.load(tmp_path / name / "checkpoints/step_4.pt", weights_only=False)
            for name in ("b", "c")]
    c["config"]["ple"][field] = value
    with pytest.raises(ValueError, match="config mismatch"):
        validate_ple_pair(b, c, sha256_file(tmp_path / "data/data_manifest.json"))


def test_v3_resume_rejects_missing_optimizer(tmp_path):
    from flashmini.checkpoint import load_checkpoint
    data = make_data(tmp_path / "data")
    config = tiny_config()
    model, _ = fit(config, data, tmp_path / "run")
    saved = torch.load(tmp_path / "run/checkpoints/step_4.pt", weights_only=False)
    saved["optimizer_state_dict"] = None
    broken = tmp_path / "missing_optimizer.pt"
    torch.save(saved, broken)
    with pytest.raises(ValueError, match="requires optimizer state"):
        load_checkpoint(broken, model, build_optimizer(model, 0.001))
    saved["config"] = None
    torch.save(saved, broken)
    with pytest.raises(ValueError, match="complete versioned config"):
        load_checkpoint(broken, model)


def test_pause_gate_cannot_split_optimizer_batch(tmp_path):
    data = make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="pause gates must align"):
        fit(tiny_config(), data, tmp_path / "bad", stop=24)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("field,value", [
    ("training", {}), ("rng_state", {}), ("rng_state", None), ("clipping_counts", {}),
    ("training.schedule", {}), ("training.run_metadata", {}), ("training.base_lrs", None),
])
def test_resume_rejects_incomplete_nested_metadata(tmp_path, field, value):
    data = make_data(tmp_path / "data")
    config = tiny_config()
    fit(config, data, tmp_path / "run", stop=32)
    path = tmp_path / "run/checkpoints/step_2.pt"
    saved = torch.load(path, weights_only=False)
    parent = saved["extra"]
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match="v3 resume requires"):
        fit(config, data, tmp_path / "run", resume=path)


def test_collision_audit_resets_context_after_eos(tmp_path, monkeypatch):
    import sys

    import yaml

    from flashmini.audit_ple import main
    data = make_data(tmp_path / "data")
    config = tiny_config(True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    monkeypatch.setattr(sys, "argv", ["audit_ple", "--config", str(config_path),
        "--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "audit.json"),
        "--sequences", "2048"])
    main()
    report = json.loads((tmp_path / "audit.json").read_text())
    val = MemmapDataset(data.data_dir, "val")
    expected = {2: set(), 3: set()}
    for row in val.input:
        context = [31, 31]
        for token in row:
            context.append(int(token))
            for order, contexts in expected.items():
                contexts.add(tuple(context[-order:]))
            if token == 31:
                context = [31, 31]
    for head in report["heads"]:
        assert head["distinct_contexts"] == len(expected[head["ngram"]])


def test_executor_transition_requires_authorization_and_is_recorded(tmp_path):
    """A recorded executor change is refused without the explicit flag, allowed
    with it, and written into the resumed run's checkpoint provenance."""
    import torch

    from flashmini.models import FlashMiniModel
    from flashmini.optim import build_optimizer

    data = make_data(tmp_path / "data")
    cfg = tiny_config(True)
    run = tmp_path / "run"

    def fit(schedule, **kwargs):
        microbatch = None if schedule == "monolithic" else 2
        torch.manual_seed(17)
        model = FlashMiniModel(copy.deepcopy(cfg))
        optimizer = build_optimizer(model, 0.001, ple_lr_multiplier=5)
        return train(model, optimizer, data, cfg, run, total_tokens=64, seq_len=8,
            device=torch.device("cpu"), batch_size=2, seed=17, log_every=1,
            ckpt_every_tokens=16, warmup_tokens=16, cosine_decay=True, min_lr_ratio=0.1,
            use_amp=False, pipeline_schedule=schedule, pipeline_microbatch_size=microbatch,
            run_metadata={"source_sha256": "test-source",
                          "execution_fingerprint": _test_fingerprint()}, **kwargs)

    paused = fit("serial_microbatch_v1", stop_after_tokens=32)
    assert paused["status"] == "paused"
    checkpoint = run / "checkpoints" / "step_2.pt"
    assert checkpoint.is_file()

    # Without the authorization the executor change is refused.
    with pytest.raises(ValueError, match="execution_policy"):
        fit("monolithic", resume_from=checkpoint)

    # With it, the run continues and the transition is recorded.
    resumed = fit(
        "monolithic", resume_from=checkpoint,
        allow_pipeline_policy_transition=True,
    )
    assert resumed["tokens_seen"] == 64
    saved = torch.load(run / "checkpoints" / "step_4.pt", map_location="cpu", weights_only=False)
    transition = saved["extra"]["training"]["run_metadata"]["execution_transition"]
    assert transition["from"] == "serial_microbatch_v1"
    assert transition["to"] == "monolithic"
    assert transition["reason"] == "performance optimization"
    assert transition["architecture_changed"] is False
    assert transition["logical_batch_changed"] is False
    assert transition["optimizer_changed"] is False
    assert transition["parent_tokens_seen"] == 32
    assert transition["parent_checkpoint"].endswith("step_2.pt")

