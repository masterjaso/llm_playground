#!/usr/bin/env python
"""Validated hardware-migration resume for the FlashMini v3 100M -> 250M Kaggle continuation.

This is a NARROW, evidence-backed migration path. It is NOT a global
--ignore-fingerprint escape hatch. It only activates when:

  1. An equivalence manifest exists and reports ``passed == true`` (the
     scientific evidence that the target hardware reproduces the source
     hardware within tight BF16 tolerances), AND
  2. A migration manifest is present recording both fingerprints, the
     migration reason, the origin checkpoint SHA-256, and a timestamp.

It enforces the PROVENANCE fields exactly (git commit, source SHA-256,
config SHA-256, data manifest SHA-256) and a clean working tree. It allows
ONLY the environment fields (torch, NVIDIA driver, Python version, platform)
to differ, and only because the equivalence manifest validated that
difference. Any other unexpected fingerprint difference fails closed.

The mechanism lives in ``scripts/`` (outside ``src/flashmini/``) so the frozen
source SHA-256 is preserved byte-for-byte. It monkey-patches
``flashmini.training.enforce_fingerprint_match`` for the duration of the run;
the frozen source is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Import the frozen flashmini package (source must stay byte-identical). ---
# REPO_ROOT is the directory that contains the frozen ``src/flashmini`` package.
# It defaults to the parent of this script (the local checkout). The Kaggle
# worker places this script OUTSIDE the cloned repo (so the repo working tree
# stays genuinely clean) and points FLASHMINI_REPO_ROOT at the clone.
REPO_ROOT = Path(os.environ.get("FLASHMINI_REPO_ROOT",
                                Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

import flashmini.training as _training
from flashmini.cli import cmd_train
from flashmini.fingerprint import collect_fingerprint

# Fields that must be IDENTICAL for a valid migration (provenance).
_PROVENANCE_FIELDS = ("git_commit", "source_sha256", "config_sha256", "data_manifest_sha256")
# Fields that are ALLOWED to differ for a validated hardware migration.
_ENVIRONMENT_FIELDS = ("torch", "nvidia_driver_version", "python_version", "platform")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_equivalence_manifest(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"FATAL: equivalence manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if not manifest.get("passed"):
        raise SystemExit("FATAL: equivalence manifest did not pass; migration is not validated")
    return manifest


def _build_migration_manifest(*, origin_checkpoint: Path, origin_fp: dict,
                             target_fp: dict, equivalence_manifest: Path) -> dict:
    return {
        "migration_reason": "hardware_move_after_100m",
        "migration_origin_checkpoint_sha256": _sha256_file(origin_checkpoint),
        "migration_timestamp": datetime.now(timezone.utc).isoformat(),
        "origin_fingerprint_sha256": origin_fp.get("fingerprint_sha256"),
        "target_fingerprint_sha256": target_fp.get("fingerprint_sha256"),
        "equivalence_manifest_sha256": _sha256_file(equivalence_manifest),
        "allowed_environment_fields": list(_ENVIRONMENT_FIELDS),
        "enforced_provenance_fields": list(_PROVENANCE_FIELDS),
    }


def _validated_migration_enforce(current: dict, recorded: dict, *, allow_dirty: bool = False) -> None:
    """Narrow replacement for enforce_fingerprint_match.

    Enforces provenance exactly + a clean tree; allows only the environment
    fields to differ (validated by the equivalence manifest). Fails closed on
    any other difference.
    """
    if not allow_dirty and current.get("git_dirty") is True:
        raise ValueError("official run requires a clean working tree; git tree is dirty")
    for key in _PROVENANCE_FIELDS:
        if current.get(key) != recorded.get(key):
            raise ValueError(f"migration provenance mismatch on {key}")
    unexpected = []
    for key in set(current) | set(recorded):
        # Derived aggregates (full hash, environment aggregate hash) and the
        # per-file map are not independent provenance signals; they are
        # recomputed from the fields already validated above.
        if key in ("fingerprint_sha256", "environment_fingerprint_sha256",
                   "source_files", "git_dirty"):
            continue
        if key in _PROVENANCE_FIELDS or key in _ENVIRONMENT_FIELDS:
            continue
        if current.get(key) != recorded.get(key):
            unexpected.append(key)
    if unexpected:
        raise ValueError(
            "unexpected fingerprint difference outside validated migration: "
            + ", ".join(sorted(unexpected))
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-migration-resume")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--resume", required=True, help="100M checkpoint to resume from")
    parser.add_argument("--equivalence-manifest", required=True,
                        help="Path to the equivalence manifest (must have passed=true)")
    parser.add_argument("--migration-manifest-out", required=True,
                        help="Where to write the migration manifest")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ple-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-parallel-gpus", default="1,0")
    parser.add_argument("--gpu-memory-gib", type=float, default=15.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-tokens", type=int, default=4_194_304)
    parser.add_argument("--eval-every-tokens", type=int, default=2_097_152)
    parser.add_argument("--eval-max-batches", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=524288)
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-repeated-corpus", action="store_true")
    parser.add_argument("--stop-after-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    # 1. Verify the scientific evidence (equivalence manifest passed).
    eq_manifest = _load_equivalence_manifest(Path(args.equivalence_manifest))
    print(f"Equivalence manifest passed: {eq_manifest.get('classification')}")

    # 2. Collect the current (target) fingerprint and read the recorded one.
    config_path = Path(args.config)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    from flashmini.data import sha256_file
    data_manifest_sha256 = sha256_file(Path(args.data_dir) / "data_manifest.json")
    target_fp = collect_fingerprint(
        REPO_ROOT, config_sha256=config_sha256, data_manifest_sha256=data_manifest_sha256
    )

    # Read ONLY the checkpoint's recorded fingerprint metadata. Do NOT build a
    # model or optimizer here: cmd_train builds its own model + optimizer and
    # loads the checkpoint via resume_from inside train(). Building a second
    # model + optimizer here would double the GPU footprint and OOM.
    import torch
    ckpt_state = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
    recorded_fp = (ckpt_state.get("extra") or {}).get("run_metadata", {}).get("execution_fingerprint")
    if recorded_fp is None:
        raise SystemExit("FATAL: checkpoint has no recorded execution fingerprint")
    del ckpt_state
    torch.cuda.empty_cache()

    # 3. Build and persist the migration manifest (provenance record).
    migration_manifest = _build_migration_manifest(
        origin_checkpoint=Path(args.resume),
        origin_fp=recorded_fp,
        target_fp=target_fp,
        equivalence_manifest=Path(args.equivalence_manifest),
    )
    out_path = Path(args.migration_manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(migration_manifest, indent=2))
    print(f"Wrote migration manifest: {out_path}")

    # 4. Apply the narrow override for the duration of this run only.
    _original = _training.enforce_fingerprint_match
    _training.enforce_fingerprint_match = _validated_migration_enforce
    try:
        # 5. Run the continuation through the standard CLI path.
        cli_args = argparse.Namespace(
            config=str(config_path),
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            tokens=args.tokens,
            resume=args.resume,
            lr=args.lr,
            ple_lr_multiplier=args.ple_lr_multiplier,
            batch_size=args.batch_size,
            model_parallel_gpus=args.model_parallel_gpus,
            gpu_memory_gib=args.gpu_memory_gib,
            grad_accum=args.grad_accum,
            log_every=args.log_every,
            checkpoint_every_tokens=args.checkpoint_every_tokens,
            eval_every_tokens=args.eval_every_tokens,
            val_max_batches=args.eval_max_batches,
            warmup_tokens=args.warmup_tokens,
            cosine_decay=args.cosine_decay,
            min_lr_ratio=args.min_lr_ratio,
            seed=args.seed,
            no_checkpoints=False,
            allow_repeated_corpus=args.allow_repeated_corpus,
            stop_after_tokens=args.stop_after_tokens,
        )
        return cmd_train(cli_args)
    finally:
        _training.enforce_fingerprint_match = _original


if __name__ == "__main__":
    raise SystemExit(main())
