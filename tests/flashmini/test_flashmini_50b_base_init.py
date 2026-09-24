from __future__ import annotations

import json
from pathlib import Path

import torch

from flashmini.base_init_accounting import parameter_report, compare_target
from flashmini.base_init_bundle import build_manifest, materialize_tensor, plan_shards, write_bundle
from flashmini.base_init_config import load_config
from flashmini.base_init_model import FlashMini50BBaseInit, surrogate_config
from flashmini.base_init_optimizer import classify_parameters


def test_frozen_config_and_exact_meta_accounting():
    config = load_config()
    assert config.architecture_version == 4
    assert config.attention_layers == [3, 7, 11, 15, 19, 23, 28, 33, 38, 43]
    assert config.kvc_pairs == [[3, 7], [11, 15], [19, 23], [28, 33], [38, 43]]
    assert config.ple_table_sizes[0] > 8_388_608
    report = parameter_report(config)
    assert report["base_total_learned_parameters"] == 50_276_673_408
    assert report["mtp_total_learned_parameters"] == 685_511_168
    assert report["checkpoint_total_learned_parameters"] == 50_962_184_576
    assert report["categories_sum"] == report["checkpoint_total_learned_parameters"]
    assert compare_target(report)["within_tolerance"] is True
    assert not any(name.startswith("mtp_") for name in report["categories"] if report["categories"][name] and name in {"input_embeddings", "lm_head", "ple_table"})


def test_meta_model_topology_and_untied_head():
    config = load_config()
    with torch.device("meta"):
        model = FlashMini50BBaseInit(config)
    names = dict(model.named_parameters())
    assert names["embed_tokens.weight"].shape == (131_072, 2048)
    assert names["lm_head.weight"].shape == (2048, 131_072)
    assert names["embed_tokens.weight"] is not names["lm_head.weight"]
    assert sum(name.startswith("mtp.") for name in names) > 0
    for source, reuse in config.kvc_pairs:
        source_names = [name for name in names if name.startswith(f"blocks.{source}.mixer.")]
        reuse_names = [name for name in names if name.startswith(f"blocks.{reuse}.mixer.")]
        assert any("q_proj" in name for name in source_names)
        assert not any("k_proj" in name or "v_proj" in name for name in reuse_names)


def test_surrogate_forward_backward_mtp_and_kvc():
    config = surrogate_config()
    model = FlashMini50BBaseInit(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    labels = torch.roll(input_ids, -1, dims=1)
    result = model(input_ids, labels=labels, mtp_window=3)
    assert result["logits"].shape == (1, 8, config.vocab_size)
    assert len(result["mtp_logits"]) == 3
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["mtp_loss"])
    (result["loss"] + result["mtp_loss"]).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_initialization_is_independent_of_construction_order():
    config = surrogate_config()
    first = FlashMini50BBaseInit(config)
    second = FlashMini50BBaseInit(config)
    first_values = {name: parameter.detach().clone() for name, parameter in first.named_parameters()}
    for name, parameter in second.named_parameters():
        assert torch.equal(parameter, first_values[name]), name

    reordered = FlashMini50BBaseInit(config)
    state = {name: parameter.detach().clone() for name, parameter in reordered.named_parameters()}
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, state[name]), name


def test_manifest_is_complete_and_reproducible(tmp_path):
    config = load_config()
    manifest = build_manifest(config)
    assert manifest["tensor_count"] > 10_000
    assert len({item["name"] for item in manifest["tensors"]}) == manifest["tensor_count"]
    assert sum(item["numel"] for item in manifest["tensors"]) == manifest["parameter_counts"]["total"]
    assert all(item["dtype"] == "bfloat16" for item in manifest["tensors"])
    plan = plan_shards(manifest)
    assert plan["shard_count"] > 20
    assert all(shard["sha256"] is None for shard in plan["shards"])
    sample = next(item for item in manifest["tensors"] if item["name"] == "blocks.0.moe.router.weight")
    first, second = materialize_tensor(sample), materialize_tensor(sample)
    assert torch.equal(first, second)
    assert first.dtype == torch.bfloat16


def test_donor_bundle_contains_handoff_contract(tmp_path):
    result = write_bundle(tmp_path / "bundle")
    bundle = Path(result["bundle"])
    required = {
        "flashmini_50b_base_init_v1.yaml", "init_spec.json", "parameter_report.json",
        "tensor_manifest.json", "shard_plan.json", "architecture.md", "training_recipe.md",
        "donor_handoff.md", "source_git_sha.txt", "tokenizer_manifest.json", "environment.lock",
        "materialization_command.txt", "donor_smoke_test_command.txt",
    }
    assert required <= {path.name for path in bundle.iterdir()}
    handoff = (bundle / "donor_handoff.md").read_text()
    assert "BASE PARAMETER COUNT" in handoff
    assert "MTP PARAMETER COUNT" in handoff
    assert "TOTAL CHECKPOINT PARAMS" in handoff
    spec = json.loads((bundle / "init_spec.json").read_text())
    assert spec["parameter_counts"]["base"] == 50_276_673_408
    assert spec["parameter_counts"]["mtp"] == 685_511_168
    assert spec["source_git_sha"]
