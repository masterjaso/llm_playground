"""Mandatory micro-overfit test.

Trains a tiny Flash+PLE configuration on a tiny fixed batch and verifies:
- forward and backward complete
- all expected trainable blocks receive finite non-zero gradients
- PLE receives gradients and updates
- MoE experts/router receive gradients
- loss falls by at least ~40% within a bounded number of steps
- save/resume continues the trajectory without corruption
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import FlashMiniConfig
from .models import FlashMiniModel
from .optim import build_optimizer, clip_gradient_groups, clip_gradients


def _finite_nonzero_grads(model: torch.nn.Module) -> dict[str, bool]:
    """Check that all trainable params have finite non-zero gradients."""
    result: dict[str, bool] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                result[name] = False
            else:
                grad = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                result[name] = bool(torch.isfinite(grad).all() and grad.abs().sum() > 0)
    return result


def run_overfit_test(
    config: FlashMiniConfig,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-3,
) -> dict:
    """Run the micro-overfit test. Returns a dict with pass/fail and evidence."""
    model = FlashMiniModel(config).to(device)
    optimizer = build_optimizer(model, lr=lr)

    # Tiny fixed batch
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (4, config.max_seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (4, config.max_seq_len), device=device)

    losses = []
    clip = clip_gradient_groups if config.architecture_version >= 3 else clip_gradients
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        clip(model, 1.0)
        optimizer.step()
        losses.append(loss.item())

    # Check gradients
    grads = _finite_nonzero_grads(model)
    all_finite = all(grads.values())
    n_bad = sum(1 for v in grads.values() if not v)

    # Check PLE grads
    ple_grads = {k: v for k, v in grads.items() if k.startswith("ple")}
    ple_active = bool(ple_grads) and all(ple_grads.values())

    # Check MoE grads
    moe_grads = {k: v for k, v in grads.items() if "moe" in k}
    moe_active = bool(moe_grads) and all(moe_grads.values())

    # Loss drop
    loss_drop = (losses[0] - losses[-1]) / losses[0] if losses else 0.0
    loss_ok = loss_drop >= 0.40

    # Save/resume continuity
    resume_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        ckpt1 = Path(tmp) / "ckpt1.pt"
        ckpt2 = Path(tmp) / "ckpt2.pt"
        save_checkpoint(ckpt1, model, optimizer, step=steps, config=config)
        # Continue training a few more steps
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels)
            out["loss"].backward()
            clip(model, 1.0)
            optimizer.step()
        save_checkpoint(ckpt2, model, optimizer, step=steps + 5, config=config)
        # Reload ckpt1 into a fresh model and verify it matches
        model2 = FlashMiniModel(config).to(device)
        optimizer2 = build_optimizer(model2, lr=lr)
        meta = load_checkpoint(ckpt1, model2, optimizer2)
        for _ in range(5):
            optimizer2.zero_grad(set_to_none=True)
            model2(input_ids, labels=labels)["loss"].backward()
            clip(model2, 1.0)
            optimizer2.step()
        resume_ok = meta["step"] == steps and all(
            torch.equal(v, model2.state_dict()[k]) for k, v in model.state_dict().items())

    result = {
        "pass": all_finite and loss_ok and (ple_active or not config.use_ple) and moe_active and resume_ok,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_drop": loss_drop,
        "loss_ok": loss_ok,
        "all_grads_finite": all_finite,
        "n_bad_grads": n_bad,
        "ple_active": ple_active,
        "moe_active": moe_active,
        "resume_ok": resume_ok,
        "steps": steps,
    }
    return result
