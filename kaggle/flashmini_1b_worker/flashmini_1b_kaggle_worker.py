#!/usr/bin/env python3
"""Thin Kaggle worker entrypoint for FlashMini-1B.

The notebook only invokes this file; model/training/checkpoint behavior lives
in ``flashmini`` modules so the same smoke and resume paths are testable off
Kaggle.  A production worker fails closed until a verified virtual data view
and a supported XLA sharding runtime are present.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path


def _find_repo() -> Path:
    candidates = [os.environ.get("FLASHMINI_SOURCE_DIR", ""), "/kaggle/input/flashmini-source",
                  str(Path(__file__).resolve().parents[1])]
    for candidate in candidates:
        if candidate and (Path(candidate) / "src/flashmini").is_dir():
            return Path(candidate)
    return Path(__file__).resolve().parents[1]


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))


def _run_reduced_smoke(*, topology: dict, allow_local_cpu: bool) -> dict:
    """Exercise the accepted model path with a fixed, shape-reduced update.

    The full 1B worker is never replaced by this probe.  The probe exists so a
    Kaggle smoke kernel checks model forward/backward, router statistics, PLE,
    KVC construction, and one optimizer update before an official run is
    allowed to consume a quota window.
    """
    import torch

    from flashmini.config import (
        FlashMiniConfig,
        GatedDeltaNetConfig,
        KVConfig,
        MoEConfig,
        PLEConfig,
    )
    from flashmini.models import FlashMiniModel
    from flashmini.optim import build_optimizer

    config = FlashMiniConfig(
        architecture_version=3,
        experiment_mode="screening",
        vocab_size=257,
        d_model=32,
        num_layers=4,
        num_heads=2,
        head_dim=16,
        gdn_per_attention=1,
        attention_layers=[1, 3],
        max_seq_len=16,
        use_hyperconnection=True,
        hc_count=4,
        hc_lowrank=4,
        use_ple=True,
        moe=MoEConfig(num_experts=2, top_k=1, shared_experts=1, expert_intermediate=32),
        gdn=GatedDeltaNetConfig(d_state=16, chunk_size=8, residual_in_mixer=False),
        ple=PLEConfig(
            ngram=3,
            ngram_vocab_size_base=257,
            heads_per_ngram=2,
            embed_dim=64,
            injection_layer=0,
            eos_id=256,
            offload="gpu",
            sparse=False,
        ),
        kvc=KVConfig(enabled=True),
    )
    torch.manual_seed(17)
    use_xla = bool(topology.get("available"))
    if use_xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        if not allow_local_cpu:
            raise RuntimeError("reduced smoke requires XLA unless --allow-local-cpu is set")
        device = torch.device("cpu")
    model = FlashMiniModel(config).to(device)
    optimizer = build_optimizer(model, lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)
    labels = torch.roll(input_ids, -1, dims=1)
    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids, labels=labels)
        loss = result["loss"]
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("reduced smoke produced a non-finite loss")
        loss.backward()
        finite_grads = True
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
            finite_grads = finite_grads and bool(torch.isfinite(gradient.detach()).all().item())
        if not finite_grads:
            raise FloatingPointError("reduced smoke produced a non-finite gradient")
        if use_xla:
            xm.optimizer_step(optimizer, barrier=True)
            xm.mark_step()
        else:
            optimizer.step()
        losses.append(float(loss.detach().float().item()))
    aux_value = result["stats"].get("router_aux_loss", 0.0)
    if isinstance(aux_value, (list, tuple)):
        aux_value = torch.stack([
            value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
            for value in aux_value
        ]).mean()
    elif not isinstance(aux_value, torch.Tensor):
        aux_value = torch.as_tensor(aux_value, device=device)
    return {
        "sequence_length": config.max_seq_len,
        "batch_size": 2,
        "losses": losses,
        "loss_finite": True,
        "router_aux_loss": float(aux_value.detach().float().mean().item()),
        "kvc_enabled": config.kvc.enabled,
        "ple_enabled": config.use_ple,
        "optimizer_updates": 2,
        "device": str(device),
    }


def _build_virtual_stream(manifest: dict, run_dir: Path):
    """Construct the pinned HF-streaming view embedded in the freeze."""
    from flashmini.data_v4.source import stream_source_window
    from flashmini.data_v4.tokenizer import load_tokenizer
    from flashmini.data_v4.virtual import VirtualBatchStream, VirtualCorpus

    recipe = manifest["training_recipe"]["foundation_recipe"]
    sources = manifest["data_view"]["sources"]
    tokenizer_meta = manifest["tokenizer"]
    tokenizer = load_tokenizer(
        tokenizer_meta["tokenizer_id"], tokenizer_meta["revision"], production=True
    )
    tokenizer_identity = f"{tokenizer_meta['tokenizer_id']}@{tokenizer_meta['revision']}"
    corpus = VirtualCorpus(
        recipe,
        sources,
        tokenizer_identity=tokenizer_identity,
        seq_len=int(manifest["data_view"]["sequence_length"]),
        seed=int(recipe["seed"]),
    )
    cache_dir = run_dir / "hf_cache"

    def reader(source, cursor, limit):
        source_mapping = source.as_dict()
        source_mapping["_cache_dir"] = str(cache_dir)
        result = stream_source_window(source_mapping, limit=int(limit), cursor=cursor)
        if not result.ok:
            raise RuntimeError(
                f"virtual source {source.source_id} unavailable: {result.status}:{result.reason}"
            )
        return result.records, result.cursor

    return VirtualBatchStream(
        corpus,
        tokenizer=tokenizer,
        reader=reader,
        window_documents=int(os.environ.get("FLASHMINI_WINDOW_DOCUMENTS", "64")),
    )


def _capture_rng_state() -> dict:
    import random

    import numpy as np
    import torch

    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _run_official(manifest: dict, run_dir: Path, status, topology: dict) -> dict:
    """Run the resumable XLA loop once a runnable freeze and remote are present."""
    import torch

    from flashmini.config import FlashMiniConfig
    from flashmini.models import FlashMiniModel
    from flashmini.observability import atomic_write_json
    from flashmini.optim import build_optimizer
    from flashmini.production import model_config_sha256
    from flashmini.production_checkpoint import (
        DurableCheckpointManager,
        FilesystemRemoteBackend,
        verify_checkpoint,
    )
    from flashmini.production_metrics import MetricsLedger, reconcile_metrics_history
    from flashmini.tpu_backend import SessionBudget, TPUBackend, TPUBackendConfig
    from flashmini.xla_training import XLATrainingConfig, train_xla

    remote_root = os.environ.get("FLASHMINI_REMOTE_CHECKPOINT_DIR", "")
    if not remote_root:
        raise RuntimeError(
            "FLASHMINI_REMOTE_CHECKPOINT_DIR is required; local Kaggle scratch "
            "cannot be the only durable recovery copy"
        )
    config = FlashMiniConfig.from_dict(manifest["architecture"])
    identity = {
        "run_id": manifest["run_id"],
        "git_commit": manifest.get("git_commit"),
        "source_fingerprint": manifest.get("source_fingerprint"),
        "model_config_sha256": manifest["model_config_sha256"],
        "architecture_version": manifest["architecture_version"],
        "parameter_report_fingerprint": model_config_sha256(config),
        "data_view_fingerprint": manifest["data_view"]["fingerprint"],
        "source_lock_sha256": manifest["data_view"]["source_lock_sha256"],
        "tokenizer_fingerprint": manifest["tokenizer"].get("fingerprint"),
        "checkpoint_schema_version": manifest["checkpoint_schema_version"],
        "topology": topology.get("topology"),
        "torch_version": topology.get("torch_version"),
        "torch_xla_version": topology.get("torch_xla_version"),
    }
    local_root = run_dir / "checkpoints"
    remote_root_path = Path(remote_root)
    # Restore cumulative metrics before reconciling a downloaded checkpoint;
    # checkpoint retention is bounded, but metrics history is not.
    for relative in (Path("metrics.jsonl"), Path("metrics/checkpoints.jsonl"), Path("metrics/metrics_checkpoint.json")):
        durable = remote_root_path / relative
        local = run_dir / relative
        if durable.is_file() and not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(durable, local)
    remote = FilesystemRemoteBackend(remote_root_path)
    manager = DurableCheckpointManager(local_root, remote)
    backend = TPUBackend(
        TPUBackendConfig(
            sequence_length=int(manifest["data_view"]["sequence_length"]),
            cache_dir=str(run_dir / "xla_cache"),
        )
    )
    backend.initialize(cache_identity=identity)
    model = backend.shard_model(FlashMiniModel(config))
    optimizer = build_optimizer(model, lr=float(os.environ.get("FLASHMINI_LR", "0.0003")))
    steps_per_trajectory = max(1, int(manifest["trajectory"]["total_training_tokens"]) //
                               (int(os.environ.get("FLASHMINI_BATCH_SIZE", "8")) *
                                int(manifest["data_view"]["sequence_length"])))
    warmup_steps = max(1, int(steps_per_trajectory * 0.01))
    total_steps = steps_per_trajectory
    import math

    def lr_scale(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    stream = _build_virtual_stream(manifest, run_dir)
    initial_step, initial_tokens = 0, 0
    latest = remote.latest_path()
    if latest is not None:
        checkpoint_manifest = verify_checkpoint(latest, expected_identity=identity)
        payload = torch.load(Path(latest) / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        stream.restore(payload["data_cursor"])
        initial_step, initial_tokens = int(payload["step"]), int(payload["exact_tokens"])
        reconcile_metrics_history(run_dir / "metrics.jsonl", max_step=initial_step, max_tokens=initial_tokens)
        status.update(phase="training", event="resume_verified", global_step=initial_step,
                      global_exact_tokens=initial_tokens,
                      parent_checkpoint_sha256=checkpoint_manifest.get("checkpoint_sha256"))
    ledger = MetricsLedger(run_dir)

    def sync_metrics() -> None:
        for relative in (Path("metrics.jsonl"), Path("metrics/checkpoints.jsonl"), Path("metrics/metrics_checkpoint.json")):
            source = run_dir / relative
            if not source.is_file():
                continue
            target = remote_root_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)

    def checkpoint(row: dict) -> dict:
        threshold = row.get("checkpoint_threshold_tokens")
        result = manager.save_and_sync(
            step=int(row["step"]),
            model=model,
            optimizer=optimizer,
            exact_tokens=int(row["exact_tokens"]),
            config=config,
            identity=identity,
            data_cursor=dict(row["data_cursor"] if "data_cursor" in row else row["sampler_state"]),
            scheduler_state=scheduler.state_dict(),
            rng_state=_capture_rng_state(),
            metrics_state={"last": dict(row)},
            session_lineage={"session_id": status.session_id, "parent_checkpoint_sha256": row.get("parent_checkpoint_sha256")},
            parent_checkpoint_sha256=row.get("parent_checkpoint_sha256"),
            extra={"checkpoint_reason": row.get("checkpoint_reason", "threshold")},
        )
        saved_manifest = result["manifest"]
        if threshold is not None:
            ledger.append_milestone(
                checkpoint_threshold_tokens=int(threshold),
                actual_tokens_seen=int(row["actual_tokens_seen"]),
                global_step=int(row["global_step"]),
                loss=row.get("loss"),
                ema_loss=row.get("ema_loss"),
                recent_tokens_per_sec=row.get("recent_tokens_per_sec"),
                checkpoint_sha256=saved_manifest.get("checkpoint_sha256"),
                session_id=status.session_id,
            )
        sync_metrics()
        return {
            "latest_checkpoint": result["local_path"],
            "checkpoint_sha256": saved_manifest.get("checkpoint_sha256"),
        }

    batch_size = int(os.environ.get("FLASHMINI_BATCH_SIZE", "8"))
    stop_after = int(os.environ.get(
        "FLASHMINI_STOP_AFTER_TOKENS", manifest["trajectory"]["preview_pause_tokens"]
    ))
    training = train_xla(
        model,
        optimizer,
        stream,
        backend=backend,
        config=XLATrainingConfig(
            batch_size=batch_size,
            sequence_length=int(manifest["data_view"]["sequence_length"]),
            target_tokens=int(manifest["trajectory"]["total_training_tokens"]),
            stop_after_tokens=stop_after,
            checkpoint_interval_tokens=int(manifest["trajectory"]["checkpoint_interval_tokens"]),
        ),
        status=status,
        checkpoint=checkpoint,
        scheduler=scheduler,
        session_budget=SessionBudget(),
        initial_step=initial_step,
        initial_tokens=initial_tokens,
    )
    trajectory_complete = training["exact_tokens"] >= int(manifest["trajectory"]["total_training_tokens"])
    result = {
        # Reaching the 10B preview stop is a resumable pause, not completion of
        # the declared 100B trajectory.
        "status": "COMPLETE" if trajectory_complete else "TRAINING_IN_PROGRESS",
        "run_id": manifest["run_id"],
        "session_id": status.session_id,
        "pause_reason": None if trajectory_complete else (
            "preview_milestone" if training["exact_tokens"] >= int(manifest["trajectory"]["preview_pause_tokens"])
            else "session_budget"
        ),
        "training": training,
        "latest_remote": str(remote.latest_path()) if remote.latest_path() else None,
    }
    atomic_write_json(run_dir / "result_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.environ.get("FLASHMINI_RUN_DIR", "/kaggle/working/flashmini_run"))
    parser.add_argument("--freeze-manifest", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-local-cpu", action="store_true")
    args = parser.parse_args(argv)

    from flashmini.observability import StatusLogger, atomic_write_json
    from flashmini.production import validate_freeze_manifest
    from flashmini.tpu_backend import XLACompileCache, discover_topology, validate_topology

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.freeze_manifest or os.environ.get("FLASHMINI_FREEZE_MANIFEST", run_dir / "freeze_manifest.json"))
    if not manifest_path.is_file():
        raise SystemExit(f"freeze manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    validate_freeze_manifest(manifest, strict_files=False)
    status = StatusLogger(run_dir, run_id=manifest["run_id"],
                          heartbeat_interval=float(os.environ.get("FLASHMINI_HEARTBEAT_SECONDS", "60")))
    status.start()
    status.update(phase="xla_compile", event="worker_start", global_exact_tokens=0,
                  topology=discover_topology(), backend="pytorch_xla")
    try:
        topology = validate_topology(allow_local_cpu=args.allow_local_cpu or args.smoke)
        identity = {
            "git_commit": manifest.get("git_commit"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "model_config_sha256": manifest["model_config_sha256"],
            "data_view_fingerprint": manifest["data_view"]["fingerprint"],
            "tokenizer_fingerprint": manifest["tokenizer"].get("fingerprint"),
            "xla_topology": topology.get("topology"),
            "torch_version": topology.get("torch_version"),
            "torch_xla_version": topology.get("torch_xla_version"),
            "static_sequence_length": manifest["data_view"]["sequence_length"],
        }
        cache = XLACompileCache(run_dir / "xla_cache", identity=identity).initialize()
        if not args.smoke and manifest.get("state") != "INFRASTRUCTURE_READY":
            raise RuntimeError(
                f"freeze manifest is not runnable: {manifest.get('state')}; "
                "repair the pinned source audit before official training"
            )
        source_manifest_path = Path(
            os.environ.get("FLASHMINI_VIRTUAL_SOURCE_MANIFEST", str(manifest_path))
        )
        if not args.smoke:
            if not source_manifest_path.is_file():
                raise RuntimeError(
                    f"pinned virtual source manifest is missing: {source_manifest_path}"
                )
            source_manifest = json.loads(source_manifest_path.read_text())
            if source_manifest.get("data_view", {}).get("fingerprint") != manifest["data_view"]["fingerprint"]:
                raise RuntimeError("virtual source manifest fingerprint differs from the frozen run")
        # The full worker is intentionally gated on the runtime's FSDP/SPMD
        # availability.  A plain replicated 1B model would violate the 16 GiB
        # per-chip memory contract, so it must never be launched accidentally.
        if not args.smoke:
            try:
                import torch_xla.distributed.fsdp  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("PyTorch/XLA FSDP is unavailable; refusing unsharded 1B launch") from exc
        smoke = _run_reduced_smoke(topology=topology,
                                   allow_local_cpu=args.allow_local_cpu or args.smoke) if args.smoke else None
        if args.smoke:
            result = {
                "status": "SMOKE_READY",
                "run_id": manifest["run_id"], "session_id": status.session_id,
                "topology": topology, "xla_cache": cache, "smoke": smoke,
                "manifest": str(manifest_path),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "python_version": platform.python_version(),
            }
        else:
            result = _run_official(manifest, run_dir, status, topology)
        atomic_write_json(run_dir / "result_manifest.json", result)
        status.update(phase="complete", event="worker_complete",
                      global_exact_tokens=result.get("training", {}).get("exact_tokens", 0))
        return 0
    except Exception as exc:  # noqa: BLE001 - convert worker failure to manifest
        result = {"status": "BLOCKED_" + type(exc).__name__.upper(), "error": str(exc),
                  "run_id": manifest.get("run_id"), "session_id": status.session_id}
        atomic_write_json(run_dir / "result_manifest.json", result)
        status.update(phase="failed", event="worker_failed", error=str(exc))
        print(f"FLASHMINI_WORKER_ERROR {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        status.stop()


if __name__ == "__main__":
    raise SystemExit(main())
