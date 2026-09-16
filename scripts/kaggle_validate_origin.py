#!/usr/bin/env python
"""Phase 1 — Validate the official local v3 100M A/B/C checkpoints.

Read-only. Loads each checkpoint, verifies architecture/step/tokens/config
SHA/data SHA/optimizer/RNG/clipping/fingerprint, and records SHA-256 + size.
Writes origin_manifest.json into the migration workspace. Never mutates the
official checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from flashmini.config import FlashMiniConfig
from flashmini.data import sha256_file
from flashmini.fingerprint import (
    collect_fingerprint,
    environment_fingerprint_sha256,
)

CONFIG_DIR = REPO_ROOT / "configs" / "flashmini"
DATA_DIR = REPO_ROOT / "data" / "fineweb_v3_2b"
RUN_ROOT = REPO_ROOT / "runs" / "flashmini" / "v3_execution"
MIGRATION_ROOT = REPO_ROOT / "runs" / "flashmini" / "kaggle_continuation"

TREATMENTS = ("A", "B", "C")
TREATMENT_CONFIGS = {
    "A": CONFIG_DIR / "poc_a_v3.yaml",
    "B": CONFIG_DIR / "poc_b_v3.yaml",
    "C": CONFIG_DIR / "poc_c_v3.yaml",
}
EXPECTED_STEP = 24_576
EXPECTED_TOKENS = 100_663_296


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint(treatment: str) -> dict:
    ckpt = RUN_ROOT / f"treatment_{treatment}" / "checkpoints" / "step_24576.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint for {treatment}: {ckpt}")
    ckpt_sha = _sha256(ckpt)
    size = ckpt.stat().st_size
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    training = extra.get("training", {})
    run_metadata = training.get("run_metadata", {})

    arch = int(state.get("architecture_version", -1))
    step = int(state.get("step", -1))
    tokens_seen = extra.get("tokens_seen")
    real_tokens_seen = extra.get("real_tokens_seen")
    data_manifest_sha256 = extra.get("data_manifest_sha256")
    config_sha256 = run_metadata.get("config_sha256")
    fingerprint = run_metadata.get("execution_fingerprint")
    optimizer_state = state.get("optimizer_state_dict")
    rng_state = extra.get("rng_state")
    clipping_counts = extra.get("clipping_counts")

    problems = []
    if arch != 3:
        problems.append(f"architecture_version={arch} (expected 3)")
    if step != EXPECTED_STEP:
        problems.append(f"step={step} (expected {EXPECTED_STEP})")
    if tokens_seen != EXPECTED_TOKENS:
        problems.append(f"tokens_seen={tokens_seen} (expected {EXPECTED_TOKENS})")
    if data_manifest_sha256 != sha256_file(DATA_DIR / "data_manifest.json"):
        problems.append("data_manifest_sha256 mismatch")
    if config_sha256 != sha256_file(TREATMENT_CONFIGS[treatment]):
        problems.append("config_sha256 mismatch")
    if fingerprint is None:
        problems.append("missing execution_fingerprint")
    if optimizer_state is None:
        problems.append("missing optimizer_state_dict")
    if rng_state is None or not isinstance(rng_state, dict):
        problems.append("missing/incomplete rng_state")
    else:
        for key in ("sampling", "torch", "python", "numpy", "cuda"):
            if rng_state.get(key) is None:
                problems.append(f"rng_state missing {key}")
    if clipping_counts is None or not isinstance(clipping_counts, dict):
        problems.append("missing clipping_counts")
    else:
        if set(clipping_counts) != {"shared", "ple_dense", "ple_sparse"}:
            problems.append(f"clipping_counts keys {sorted(clipping_counts)}")

    env_sha = environment_fingerprint_sha256(fingerprint) if fingerprint else None
    # Rebuild the current environment fingerprint for comparison.
    current_fp = collect_fingerprint(
        REPO_ROOT,
        config_sha256=sha256_file(TREATMENT_CONFIGS[treatment]),
        data_manifest_sha256=data_manifest_sha256,
    )
    current_env_sha = environment_fingerprint_sha256(current_fp)

    # Record the original execution fingerprint (the one that produced the run).
    orig_env = {
        "git_commit": fingerprint.get("git_commit") if fingerprint else None,
        "torch_version": (fingerprint or {}).get("torch", {}).get("torch_version"),
        "cuda_version": (fingerprint or {}).get("torch", {}).get("cuda_version"),
        "device_count": (fingerprint or {}).get("torch", {}).get("device_count"),
        "devices": (fingerprint or {}).get("torch", {}).get("devices"),
        "nvidia_driver_version": (fingerprint or {}).get("nvidia_driver_version"),
        "python_version": (fingerprint or {}).get("python_version"),
        "platform": (fingerprint or {}).get("platform"),
        "source_sha256": (fingerprint or {}).get("source_sha256"),
        "data_manifest_sha256": (fingerprint or {}).get("data_manifest_sha256"),
        "environment_fingerprint_sha256": env_sha,
        "fingerprint_sha256": (fingerprint or {}).get("fingerprint_sha256"),
    }

    result = {
        "treatment": treatment,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "size_bytes": size,
        "architecture_version": arch,
        "step": step,
        "tokens_seen": tokens_seen,
        "real_tokens_seen": real_tokens_seen,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "optimizer_state_present": optimizer_state is not None,
        "rng_state_keys": sorted(rng_state.keys()) if isinstance(rng_state, dict) else [],
        "clipping_counts": clipping_counts,
        "original_execution_fingerprint": orig_env,
        "current_environment_fingerprint_sha256": current_env_sha,
        "original_environment_fingerprint_sha256": env_sha,
        "problems": problems,
        "ok": not problems,
    }
    del state
    return result


def main() -> int:
    MIGRATION_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for t in TREATMENTS:
        print(f"validating {t} ...", flush=True)
        r = validate_checkpoint(t)
        results[t] = r
        print(f"  {t}: ok={r['ok']} sha={r['checkpoint_sha256'][:16]} "
              f"size={r['size_bytes']} problems={r['problems']}", flush=True)
        del r  # free the loaded checkpoint state between treatments

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "git_commit": "1cb759783a36d5558a2fba50ebe52c12131690d2",
        "branch": "flash-mini",
        "source_sha256": "de6824c20b35f3e8fb6b5900f26258ad7f02b989bf7f1aab801de65e271bf6fb",
        "data_manifest_sha256": "b06f559f65e6969a9dae36d392873610330b8a2425fe27f47518c4d200d42dab",
        "expected_step": EXPECTED_STEP,
        "expected_tokens": EXPECTED_TOKENS,
        "checkpoints": results,
    }
    out = MIGRATION_ROOT / "origin_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    all_ok = all(r["ok"] for r in results.values())
    print(f"ALL_OK={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
