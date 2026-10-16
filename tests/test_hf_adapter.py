from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from flashmini.hf_adapter import FlashMiniConfig, FlashMiniForCausalLM
from scripts.flashmini_hf_export import main as export_hf_main


def tiny_config(*, architecture_version: int = 2, use_ple: bool = False, sparse: bool = False):
    return FlashMiniConfig(
        architecture_version=architecture_version,
        vocab_size=31,
        d_model=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        gdn_per_attention=1,
        max_seq_len=8,
        use_ple=use_ple,
        moe={"num_experts": 2, "top_k": 1, "expert_intermediate": 16},
        gdn={"d_state": 4, "chunk_size": 4},
        ple={
            "table_size": 37,
            "ngram_vocab_size_base": 37,
            "heads_per_ngram": 2,
            "embed_dim": 16,
            "head_dim": 4,
            "eos_id": 30,
            "offload": "cpu" if sparse else "gpu",
            "sparse": sparse,
        },
    )


@pytest.mark.parametrize("architecture_version", [2, 3])
def test_shifted_loss_and_native_router_aux(architecture_version):
    torch.manual_seed(4)
    model = FlashMiniForCausalLM(tiny_config(architecture_version=architecture_version))
    input_ids = torch.randint(0, 31, (2, 8))
    labels = input_ids.roll(-1, dims=1)
    labels[0, 3] = -100

    output = model(input_ids, labels=labels)
    shifted = output.logits[:, :-1].reshape(-1, 31)
    shifted_labels = labels[:, 1:].reshape(-1)
    ce = F.cross_entropy(shifted, shifted_labels, ignore_index=-100)
    expected = ce + model.config.moe["aux_loss_coef"] * output.router_aux_loss
    assert output.logits.shape == (2, 8, 31)
    assert torch.allclose(output.loss, expected)

    ignored = model(input_ids, labels=torch.full_like(labels, -100))
    assert torch.allclose(ignored.loss, model.config.moe["aux_loss_coef"] * ignored.router_aux_loss)


def test_right_padding_trims_each_row_and_rejects_other_masks():
    torch.manual_seed(5)
    model = FlashMiniForCausalLM(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 31, (2, 8))
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1] * 8])
    output = model(input_ids, attention_mask=mask)
    expected_first = model.model(input_ids[:1, :5])["logits"][0]
    expected_second = model.model(input_ids[1:2])["logits"][0]
    assert torch.allclose(output.logits[0, :5], expected_first)
    assert torch.allclose(output.logits[1], expected_second)

    with pytest.raises(ValueError, match="right padding"):
        model(input_ids, attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1], [1] * 8]))
    with pytest.raises(ValueError, match="right padding"):
        model(input_ids, attention_mask=torch.tensor([[1, 1, 0, 1, 1, 0, 0, 0], [1] * 8]))


def test_save_load_auto_and_native_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = FlashMiniForCausalLM(tiny_config(architecture_version=3))
    model.eval()
    input_ids = torch.randint(0, 31, (1, 8))
    expected = model(input_ids).logits

    folder = tmp_path / "hf"
    model.save_pretrained(folder)
    saved_config = json.loads((folder / "config.json").read_text())
    assert saved_config["model_type"] == "flashmini"
    assert saved_config["auto_map"]["AutoConfig"].endswith("FlashMiniConfig")
    loaded = FlashMiniForCausalLM.from_pretrained(folder)
    assert torch.allclose(loaded(input_ids).logits, expected)

    native_path = tmp_path / "native.pt"
    model.save_native_checkpoint(native_path, step=3)
    native_loaded = FlashMiniForCausalLM.from_native_checkpoint(native_path)
    assert torch.allclose(native_loaded(input_ids).logits, expected)

    mismatched = tiny_config(architecture_version=3)
    mismatched.d_model = 24
    with pytest.raises(ValueError, match="config mismatch"):
        FlashMiniForCausalLM.from_native_checkpoint(native_path, config=mismatched)


@pytest.mark.parametrize("architecture_version", [2, 3])
def test_ple_save_load_rebuilds_runtime_buffers_for_auto_model(tmp_path, architecture_version):
    torch.manual_seed(8 + architecture_version)
    model = FlashMiniForCausalLM(tiny_config(architecture_version=architecture_version, use_ple=True))
    model.eval()
    input_ids = torch.randint(0, 31, (1, 8))
    expected = model(input_ids).logits

    folder = tmp_path / f"ple-v{architecture_version}"
    model.save_pretrained(folder)
    loaded = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
    loaded.eval()
    assert torch.allclose(loaded(input_ids).logits, expected)

    native_path = tmp_path / f"ple-v{architecture_version}.pt"
    model.save_native_checkpoint(native_path)
    native_loaded = FlashMiniForCausalLM.from_native_checkpoint(native_path)
    native_loaded.eval()
    assert torch.allclose(native_loaded(input_ids).logits, expected)


def test_sparse_ple_is_rejected_for_hf_adamw():
    with pytest.raises(ValueError, match="dense AdamW.*sparse PLE"):
        FlashMiniForCausalLM(tiny_config(architecture_version=3, use_ple=True, sparse=True))


def test_generation_cache_is_explicitly_unsupported():
    model = FlashMiniForCausalLM(tiny_config())
    input_ids = torch.randint(0, 31, (1, 4))
    with pytest.raises(ValueError, match="KV-cache"):
        model(input_ids, use_cache=True)
    with pytest.raises(ValueError, match="KV-cache"):
        model(input_ids, past_key_values=())


def test_bfloat16_forward_backward_covers_moe_ple_and_kvc(tmp_path):
    """HF full finetuning keeps the native sparse-routing path dtype-safe."""
    torch.manual_seed(11)
    config = FlashMiniConfig(
        architecture_version=3,
        vocab_size=31,
        d_model=16,
        num_layers=4,
        num_heads=2,
        head_dim=8,
        gdn_per_attention=0,
        max_seq_len=8,
        use_ple=True,
        moe={"num_experts": 2, "top_k": 1, "expert_intermediate": 16},
        gdn={"d_state": 4, "chunk_size": 4},
        ple={
            "table_size": 37,
            "ngram_vocab_size_base": 37,
            "heads_per_ngram": 2,
            "embed_dim": 16,
            "head_dim": 4,
            "eos_id": 30,
            "offload": "gpu",
            "sparse": False,
        },
        kvc={"enabled": True},
    )
    model = FlashMiniForCausalLM(config).to(dtype=torch.bfloat16)
    input_ids = torch.randint(0, 31, (1, 8))
    output = model(input_ids, labels=input_ids)
    assert torch.isfinite(output.loss)
    folder = tmp_path / "bf16"
    model.save_pretrained(folder)
    reloaded = FlashMiniForCausalLM.from_pretrained(folder)
    assert torch.allclose(model(input_ids).logits, reloaded(input_ids).logits, atol=1e-2, rtol=1e-2)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_desktop_pad_export_clones_config_and_refuses_overwrite(tmp_path):
    config_values = {
        "architecture_version": 3,
        "vocab_size": 4,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 8,
        "gdn_per_attention": 1,
        "max_seq_len": 8,
        "use_ple": False,
        "moe": {"num_experts": 2, "top_k": 1, "expert_intermediate": 16},
        "gdn": {"d_state": 4, "chunk_size": 4},
    }
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(config_values), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = Tokenizer(WordLevel({"a": 0, "b": 1, "<eos>": 2, "<unk>": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>")
    fast.save_pretrained(tokenizer_path)

    bad_config_path = tmp_path / "bad.json"
    bad_config_path.write_text(json.dumps({**config_values, "vocab_size": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="source config vocab_size"):
        export_hf_main(
            [
                "--config",
                str(bad_config_path),
                "--tokenizer",
                str(tokenizer_path),
                "--desktop-pad-token",
                "--output",
                str(tmp_path / "bad-export"),
            ]
        )
    with pytest.raises(SystemExit, match="only for fresh"):
        export_hf_main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--tokenizer",
                str(tokenizer_path),
                "--desktop-pad-token",
                "--output",
                str(tmp_path / "checkpoint-export"),
            ]
        )

    output = tmp_path / "desktop"
    args = [
        "--config",
        str(config_path),
        "--tokenizer",
        str(tokenizer_path),
        "--desktop-pad-token",
        "--output",
        str(output),
    ]
    assert export_hf_main(args) == 0
    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert saved["vocab_size"] == 5
    assert saved["pad_token_id"] != saved["eos_token_id"]
    assert json.loads(config_path.read_text(encoding="utf-8")) == config_values

    before = (output / "config.json").read_bytes()
    with pytest.raises(SystemExit, match="non-empty"):
        export_hf_main(args)
    assert (output / "config.json").read_bytes() == before
