#!/usr/bin/env python3
"""Bounded CPU/XLA semantic probe for the accepted FlashMini-D path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flashmini.config import FlashMiniConfig, GatedDeltaNetConfig, KVConfig, MoEConfig, PLEConfig
from flashmini.models import FlashMiniModel
from flashmini.production import canonical_sha256
from flashmini.tpu_backend import discover_topology


def reduced_config() -> FlashMiniConfig:
    """A shape-reduced probe that preserves every D mechanism and KVC pair."""
    return FlashMiniConfig(
        architecture_version=3, experiment_mode="screening", vocab_size=257,
        d_model=32, num_layers=4, num_heads=2, head_dim=16, gdn_per_attention=1,
        attention_layers=[1, 3], max_seq_len=16, use_hyperconnection=True,
        hc_count=4, hc_lowrank=4, use_ple=True,
        moe=MoEConfig(num_experts=2, top_k=1, shared_experts=1, expert_intermediate=32),
        gdn=GatedDeltaNetConfig(d_state=16, chunk_size=8, residual_in_mixer=False),
        ple=PLEConfig(ngram=3, ngram_vocab_size_base=257, heads_per_ngram=2,
                      embed_dim=64, injection_layer=0, eos_id=256, offload="gpu", sparse=False),
        kvc=KVConfig(enabled=True),
    )


def _probe(model: torch.nn.Module, ids: torch.Tensor, labels: torch.Tensor) -> dict:
    model.zero_grad(set_to_none=True)
    out = model(ids, labels=labels)
    loss = out["loss"]
    loss.backward()
    norm = torch.sqrt(sum(p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None))
    aux_value = out["stats"].get("router_aux_loss", 0.0)
    if isinstance(aux_value, (list, tuple)):
        aux_value = torch.stack([
            value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for value in aux_value
        ]).mean()
    elif not isinstance(aux_value, torch.Tensor):
        aux_value = torch.as_tensor(aux_value)
    return {"loss": float(loss.detach().float().item()), "grad_norm": float(norm.item()),
            "logits_shape": list(out["logits"].shape),
            "router_aux_loss": float(aux_value.detach().float().mean().item())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/flashmini/1b_kaggle/equivalence_probe.json")
    parser.add_argument("--allow-unavailable", action="store_true")
    args = parser.parse_args(argv)
    torch.manual_seed(17)
    cfg = reduced_config()
    cpu = FlashMiniModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    labels = torch.roll(ids, -1, dims=1)
    cpu_result = _probe(cpu, ids, labels)
    topology = discover_topology()
    result = {"status": "XLA_UNAVAILABLE", "config": cfg.to_dict(),
              "config_sha256": canonical_sha256(cfg.to_dict()), "cpu": cpu_result,
              "topology": topology, "tolerance": {"loss_abs": 0.05, "grad_norm_rel": 0.10}}
    if topology.get("available"):
        try:
            import torch_xla.core.xla_model as xm
            xla = FlashMiniModel(cfg).to(xm.xla_device())
            xla.load_state_dict(cpu.state_dict())
            xla_result = _probe(xla, ids.to(xm.xla_device()), labels.to(xm.xla_device()))
            xm.mark_step()
            loss_delta = abs(cpu_result["loss"] - xla_result["loss"])
            grad_delta = abs(cpu_result["grad_norm"] - xla_result["grad_norm"]) / max(cpu_result["grad_norm"], 1e-8)
            result.update({"status": "PASS" if loss_delta <= 0.05 and grad_delta <= 0.10 else "FAIL",
                           "xla": xla_result, "loss_abs_delta": loss_delta,
                           "grad_norm_relative_delta": grad_delta})
        except Exception as exc:  # noqa: BLE001 - probe records runtime failure
            result.update({"status": "FAIL", "xla_error": f"{type(exc).__name__}: {exc}"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "XLA_UNAVAILABLE" and args.allow_unavailable:
        return 0
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
