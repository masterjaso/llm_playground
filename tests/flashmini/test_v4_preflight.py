"""Donor preflight fail-closed checks and representative materialization integrity."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from flashmini import base_init_bundle as bundle_mod
from flashmini.base_init_accounting import parameter_report
from flashmini.base_init_bundle import materialize_tensor, write_shard
from flashmini.base_init_config import load_config
from flashmini.v4_preflight import (
    FROZEN_PARAMETER_COUNTS,
    check_environment,
    check_identity,
    check_optimizer_classification,
    check_parameters,
    check_ple_storage,
    check_tokenizer,
    check_topology,
    check_train_config,
    run,
)


def test_frozen_parameter_counts_match_meta_accountant():
    report = parameter_report(load_config())
    assert {
        "base": report["base_total_learned_parameters"],
        "mtp": report["mtp_total_learned_parameters"],
        "total": report["checkpoint_total_learned_parameters"],
    } == FROZEN_PARAMETER_COUNTS


def test_tokenizer_and_identity_checks_pass_on_current_artifacts(tmp_path):
    config = load_config()
    assert check_tokenizer(config) == []
    written = bundle_mod.write_bundle(tmp_path / "bundle")
    bundle = Path(written["bundle"])
    assert check_identity(bundle, config) == []
    assert check_parameters(bundle, config) == []
    assert check_optimizer_classification(bundle, config) == []
    assert check_environment(bundle) == []


def test_train_config_and_topology_fail_closed(tmp_path):
    config = load_config()
    problems, train = check_train_config(None, config)
    assert train is None and problems
    example = Path("configs/flashmini/v4_train_example.yaml")
    problems, _ = check_train_config(example, config)
    assert any("unresolved" in item or "must be" in item for item in problems)
    missing = check_topology(
        type("T", (), {"get": staticmethod(lambda key: {"distributed.nodes": 1, "distributed.gpus_per_node": 8, "distributed.strategy": "fsdp2"}[key])})(),
        config, cuda_devices=0, device_memory_bytes=80 * 1024**3, world_env=8,
    )
    assert any("no CUDA" in item for item in missing)
    too_small = check_topology(
        type("T", (), {"get": staticmethod(lambda key: {"distributed.nodes": 1, "distributed.gpus_per_node": 8, "distributed.strategy": "single"}[key])})(),
        config, cuda_devices=8, device_memory_bytes=80 * 1024**3, world_env=8,
    )
    assert any("fsdp2" in item for item in too_small)


def test_ple_storage_refuses_insufficient_host_memory():
    config = load_config()
    train = type("T", (), {"get": staticmethod(lambda key: {
        "distributed.nodes": 1, "distributed.gpus_per_node": 8, "distributed.ple_backing": "process",
    }[key])})()
    problems = check_ple_storage(train, config, available_bytes=1024)
    assert any("PLE storage unavailable" in item for item in problems)
    ok = check_ple_storage(train, config, available_bytes=10**15)
    assert ok == []


def test_preflight_fail_line_when_checkpoint_missing(tmp_path):
    ok, checks = run(Path("flashmini_50b_base_init_v1"), None, None, scope="artifacts")
    assert ok is False
    names = {check.name: check for check in checks}
    assert names["checkpoint_shards"].problems
    assert names["training_hyperparameters"].problems


def test_representative_shards_are_readable_unique_and_reproducible(tmp_path):
    config = load_config()
    report = {item["name"]: item for item in parameter_report(config)["tensors"]}
    # Small special-init, MTP, and attention tensors; a prefix of a PLE table (not the 2 GiB head).
    names = [
        "blocks.0.mixer.A_log",
        "blocks.0.mixer.dt_bias",
        "blocks.0.mixer.conv1d.weight",
        "ple.conv1d.weight",
        "mtp.fusion.fc_hidden.weight",
        "blocks.3.mixer.q_proj.weight",
    ]
    items = []
    for name in names:
        law = report[name]
        items.append({
            "name": name, "shape": law["shape"], "dtype": "bfloat16", "numel": law["numel"],
            "init_law": __import__("flashmini.v4_init", fromlist=["init_law"]).init_law(name),
        })
    path = tmp_path / "representative.safetensors"
    digest = write_shard(path, items, metadata={"format": "pt", "kind": "representative"})
    again = write_shard(tmp_path / "representative-2.safetensors", items, metadata={"format": "pt", "kind": "representative"})
    assert digest == again
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        assert keys == names or sorted(keys) == sorted(names)
        assert len(set(keys)) == len(keys)
        for name in names:
            tensor = handle.get_tensor(name)
            assert list(tensor.shape) == report[name]["shape"]
            assert tensor.dtype == torch.bfloat16
            torch.testing.assert_close(tensor.view(torch.int16), materialize_tensor(items[names.index(name)]).view(torch.int16), atol=0, rtol=0)
    assert len(digest) == 64
