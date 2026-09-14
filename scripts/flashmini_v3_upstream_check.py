"""Compare GR, PLE hashing and one-document convolution to pinned upstream code.

Download the pinned Transformers source separately; it is never vendored here.
Only the named standalone mechanisms are evaluated after verifying its SHA-256.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from flashmini.config import FlashMiniConfig, PLEConfig
from flashmini.models.hyperconnection import GatedResidual
from flashmini.models.ple import PLEV3

UPSTREAM_COMMIT = "bd15bc95a89e728bbc1224084eb3b5829428c353"
UPSTREAM_SHA256 = "2a44aeadb215acbb5c75939fcc97e9f14bccff5a51c232826427594993f6a760"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = Path(args.source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("upstream source hash differs from reviewed commit")
    names = {"Qwen4ExpTextRMSNorm", "Qwen4ExpTextGatedResidual", "Qwen4ExpTextNGramEmbedding",
             "Qwen4ExpTextPLELayer", "_splitmix64", "_build_layer_multipliers", "_is_prime", "_find_nth_prime_after"}
    nodes = [node for node in ast.parse(raw).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math,
        "_MASK64": (1 << 64) - 1, "_SPLITMIX_GAMMA": 0x9E3779B97F4A7C15,
        "_SPLITMIX_M1": 0xBF58476D1CE4E5B9, "_SPLITMIX_M2": 0x94D049BB133111EB, "_PRIME_1": 10007}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), args.source, "exec"), namespace)  # noqa: S102 -- allowlisted upstream definitions, pinned SHA-256 checked above
    cfg = SimpleNamespace(hc_count=4, hidden_size=16, hc_lowrank=4, rms_norm_eps=1e-6,
        ngram_size=3, heads_per_ngram=2, vocab_size=32, ngram_vocab_size_base=101,
        seed=1234, eos_token_id=31, make_ngram_vocab_size_divisible_by=1,
        ple_embed_dim=16, ple_conv_kernel_size=4)
    torch.manual_seed(17)
    upstream_gr = namespace["Qwen4ExpTextGatedResidual"](cfg)
    local_gr = GatedResidual(16, hc_lowrank=4)
    local_gr.load_state_dict({("norm_weight" if key == "hc_norm.weight" else key): value
                             for key, value in upstream_gr.state_dict().items()})
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, _, expected_gate = upstream_gr(x)
    actual, _, actual_gate = local_gr(y)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=1e-6, atol=1e-6)
    (expected.square().sum() + expected_gate.square().sum()).backward()
    (actual.square().sum() + actual_gate.square().sum()).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    upstream = namespace["Qwen4ExpTextPLELayer"](cfg, 1, 0)
    pcfg = FlashMiniConfig(architecture_version=3, vocab_size=32, d_model=16,
        ple=PLEConfig(ngram_vocab_size_base=101, heads_per_ngram=2, embed_dim=16,
                      eos_id=31, hash_seed=1234)).ple
    local = PLEV3(pcfg)
    mappings = {"value_embed.weight": "ple_embedding.ngram_embedding.weight"}
    upstream_state = upstream.state_dict()
    local.load_state_dict({key: upstream_state[mappings.get(key, key)] for key in local.state_dict()})
    ids = torch.randint(0, 32, (2, 16))
    seen_keys = []
    handle = upstream.ple_embedding.ngram_embedding.register_forward_pre_hook(
        lambda module, inputs: seen_keys.append(inputs[0].detach()))
    upstream.ple_embedding(ids, None)
    assert torch.equal(local._ngram_keys(ids), seen_keys[0])
    handle.remove()
    # Upstream convolution carries across EOS; FlashMini deliberately resets it.
    # Compare the common one-document semantics with nonzero convolution weights.
    ids[ids == 31] = 30
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = upstream(x, ids, None), local(ids, y)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    report = {"status": "PASS", "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SHA256,
        "checks": ["GR_read_write_gates_and_input_gradients", "PLE_hash_keys_including_EOS",
                   "PLE_one_document_nonzero_convolution_output_and_input_gradients"],
        "ple_max_abs_difference": float((actual - expected).abs().max().detach()),
        "intentional_deviation": "PLE convolution reset after EOS; upstream carries convolution across EOS"}
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
