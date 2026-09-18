#!/usr/bin/env python
"""Kaggle continuation controller for the FlashMini v3 100M -> 250M run.

Local orchestrator that runs ONE treatment for ONE quota window at a time.
It is deliberately narrow and fail-closed:

  * It never modifies the frozen source (``src/flashmini``).
  * It only rewrites the worker kernel notebook (TREATMENT,
    STOP_AFTER_TOKENS) and the worker kernel-metadata.json (dataset sources).
  * It pushes the kernel to Kaggle, polls until COMPLETE or ERROR, downloads
    the result manifest and the latest checkpoint via ``kaggle kernels output``,
    verifies the checkpoint SHA-256, and updates the per-treatment checkpoint
    dataset with a new version.
  * It records per-treatment progress in ``controller_state.json``.

Checkpoint durability: each treatment has its own Kaggle dataset containing
only that treatment's latest checkpoint. After each window the controller
re-uploads that single checkpoint (~6 GB) as a new dataset version, so the
next window resumes from the latest state. The 16 GB corpus dataset is static
and is never re-uploaded.

Usage:
  python scripts/kaggle_controller.py init
  python scripts/kaggle_controller.py status
  python scripts/kaggle_controller.py run-window --treatment A [--short]
  python scripts/kaggle_controller.py run-window --treatment A --stop-after-tokens 200000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_DIR = REPO_ROOT / "runs" / "flashmini" / "kaggle_continuation"
WORKER_DIR = CONTINUATION_DIR / "worker"
STATE_FILE = CONTINUATION_DIR / "controller_state.json"
ORIGIN_MANIFEST = CONTINUATION_DIR / "origin_manifest.json"
KAGGLE_CLI = "/tmp/kaggle_venv/bin/kaggle"

KERNEL_ID = "masterjaso/flashmini-v3-kaggle-continuation-worker"
CORPUS_DATASET = "masterjaso/flashmini-v3-100m-continuation-corpus"
CKPT_DATASET_PREFIX = "masterjaso/flashmini-v3-100m-continuation-ckpt"

TARGET_TOKENS = 250_000_000
SEQ_LEN = 256
BATCH_SIZE = 16
N_SEQS = 7_812_502
CKPT_EVERY_TOKENS = 4_194_304
# Pure-training wall-clock budget per window. The Kaggle hard limit is 21,600 s;
# we reserve ~3,600 s for startup (checkpoint load, model build, migration
# check, data mmap) plus a safety margin, leaving 18,000 s for training.
USABLE_SECONDS = 18_000
# Measured steady-state throughput (tok/s) on 2x T4, model-parallel "1,0",
# BF16 autocast, exact recipe. A is measured from the short validation window
# (4,956,160 tokens / 2,733.7 s = 1813 tok/s); B and C are probe values derated
# ~5% to match the measured-vs-probe gap observed for A.
TOK_PER_SEC = {"A": 1813.0, "B": 1780.0, "C": 1650.0}
# Short validation window (for the first end-to-end test).
SHORT_WINDOW_TOKENS = 5_000_000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kaggle_env() -> dict:
    """Build a subprocess env with KAGGLE_API_TOKEN set from .env (never printed)."""
    env = dict(os.environ)
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    if "KAGGLE_API_KEY" in env:
        env["KAGGLE_API_TOKEN"] = env["KAGGLE_API_KEY"]
    return env


def _run(cmd: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=_kaggle_env())


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if not STATE_FILE.is_file():
        raise SystemExit(f"controller state not found: {STATE_FILE}; run `init` first")
    return json.loads(STATE_FILE.read_text())


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def init_state() -> dict:
    origin = json.loads(ORIGIN_MANIFEST.read_text())
    state: dict = {}
    for treatment in ("A", "B", "C"):
        info = origin["checkpoints"][treatment]
        state[treatment] = {
            "tokens_seen": int(info["tokens_seen"]),
            "latest_checkpoint_local": str(info["checkpoint_path"]),
            "latest_checkpoint_sha256": info["checkpoint_sha256"],
            "windows_completed": 0,
            "status": "pending",
        }
    save_state(state)
    print(f"Initialized controller state: {STATE_FILE}")
    for t in ("A", "B", "C"):
        print(f"  {t}: tokens_seen={state[t]['tokens_seen']}, "
              f"sha={state[t]['latest_checkpoint_sha256'][:16]}...")
    return state


# --------------------------------------------------------------------------- #
# Window computation
# --------------------------------------------------------------------------- #
def _pause_gate_valid(stop: int) -> bool:
    """v3 pause gates must align with full batches (training.py)."""
    if stop >= TARGET_TOKENS:
        return True  # final window: the gate check is skipped
    return (stop % SEQ_LEN == 0) and ((stop // SEQ_LEN) % N_SEQS) % BATCH_SIZE == 0


def compute_stop_after_tokens(treatment: str, tokens_seen: int, *, short: bool = False) -> int:
    remaining = TARGET_TOKENS - tokens_seen
    if remaining <= 0:
        raise SystemExit(f"treatment {treatment} already complete")
    if short:
        window = min(SHORT_WINDOW_TOKENS, remaining)
    else:
        raw = USABLE_SECONDS * TOK_PER_SEC[treatment]
        window = int(raw)
    # Align down to a 4096-token boundary (multiple of seq_len*batch_size).
    window = (window // (SEQ_LEN * BATCH_SIZE)) * (SEQ_LEN * BATCH_SIZE)
    stop = tokens_seen + window
    if stop >= TARGET_TOKENS:
        stop = TARGET_TOKENS
    if not _pause_gate_valid(stop):
        raise SystemExit(f"computed stop {stop} is not pause-gate valid")
    return stop


# --------------------------------------------------------------------------- #
# Kernel rewrite
# --------------------------------------------------------------------------- #
def rewrite_kernel(treatment: str, stop_after_tokens: int) -> None:
    """Rewrite the worker notebook (TREATMENT, STOP_AFTER_TOKENS) and the
    worker kernel-metadata.json (dataset sources) in place. Idempotent: if the
    values are already correct, the notebook is left unchanged but still
    validated."""
    nb_path = WORKER_DIR / "kernel_continuation_worker.ipynb"
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        new_src = re.sub(r'^TREATMENT = "[ABC]"',
                         f'TREATMENT = "{treatment}"', src, count=1, flags=re.M)
        new_src = re.sub(r'^STOP_AFTER_TOKENS = \d[\d_]*',
                         f"STOP_AFTER_TOKENS = {stop_after_tokens}", new_src,
                         count=1, flags=re.M)
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
    # Validate the final state: the target cell must carry the exact values.
    full = "".join("".join(c["source"]) for c in nb["cells"]
                   if c.get("cell_type") == "code")
    if f'TREATMENT = "{treatment}"' not in full:
        raise SystemExit(f"kernel rewrite failed: TREATMENT = {treatment!r} not present")
    if f"STOP_AFTER_TOKENS = {stop_after_tokens}" not in full:
        raise SystemExit(
            f"kernel rewrite failed: STOP_AFTER_TOKENS = {stop_after_tokens} not present")
    # papermill requires a kernelspec to resolve the kernel name; ensure it is
    # present so the push does not fail with "No kernel name found".
    nb.setdefault("metadata", {})
    nb["metadata"].setdefault("kernelspec", {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    })
    nb_path.write_text(json.dumps(nb, indent=1))

    meta_path = WORKER_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["dataset_sources"] = [
        f"{CKPT_DATASET_PREFIX}-{treatment.lower()}",
        CORPUS_DATASET,
    ]
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Rewrote kernel for treatment {treatment}, stop_after_tokens={stop_after_tokens}")
    print(f"  dataset_sources: {meta['dataset_sources']}")


# --------------------------------------------------------------------------- #
# Kaggle push / poll / download
# --------------------------------------------------------------------------- #
def push_kernel() -> None:
    r = _run([KAGGLE_CLI, "kernels", "push", "-p", str(WORKER_DIR)],
             timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"kaggle kernels push failed:\n{r.stderr[-2000:]}")
    print("Kernel pushed (push also starts the run).")


def poll_kernel(kernel_id: str, *, max_wait: int = 22_200) -> str:
    """Poll kernel status until COMPLETE or ERROR. Returns the final status string."""
    deadline = time.monotonic() + max_wait
    last = ""
    while time.monotonic() < deadline:
        r = _run([KAGGLE_CLI, "kernels", "status", kernel_id], timeout=120)
        if r.returncode != 0:
            raise SystemExit(f"kaggle kernels status failed:\n{r.stderr[-1000:]}")
        last = r.stdout.strip()
        m = re.search(r'status\s+"(KernelWorkerStatus\.\w+)"', last)
        status = m.group(1) if m else last
        print(f"  [{datetime.now(timezone.utc).isoformat()}] {status}")
        if status.endswith("COMPLETE"):
            return status
        if status.endswith("ERROR"):
            return status
        time.sleep(30)
    raise SystemExit(f"kernel {kernel_id} did not finish within {max_wait}s; last: {last}")


def download_output(kernel_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    r = _run([KAGGLE_CLI, "kernels", "output", kernel_id, "-p", str(dest)],
             timeout=3600)
    if r.returncode != 0:
        raise SystemExit(f"kaggle kernels output failed:\n{r.stderr[-2000:]}")
    return dest


def fetch_logs(kernel_id: str) -> str:
    r = _run([KAGGLE_CLI, "kernels", "logs", kernel_id], timeout=300)
    return r.stdout if r.returncode == 0 else f"(logs unavailable: {r.stderr[-500:]})"


# --------------------------------------------------------------------------- #
# Checkpoint dataset update
# --------------------------------------------------------------------------- #
def _ckpt_dataset_folder(treatment: str) -> Path:
    return CONTINUATION_DIR / f"ckpt_dataset_{treatment}"


def _ensure_dataset_metadata(folder: Path, treatment: str) -> None:
    meta_path = folder / "dataset-metadata.json"
    if meta_path.is_file():
        return
    meta = {
        "title": f"FlashMini v3 100M Continuation Checkpoint ({treatment})",
        "id": f"{CKPT_DATASET_PREFIX}-{treatment.lower()}",
        "subtitle": f"Latest checkpoint for treatment {treatment} (250M continuation)",
        "description": (
            f"Per-treatment checkpoint dataset for the FlashMini v3 100M -> 250M "
            f"Kaggle continuation. Contains only treatment {treatment}'s latest "
            f"checkpoint so each quota window resumes from the latest state and "
            f"the re-upload stays ~6 GB."
        ),
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["machine-learning", "nlp"],
        "resources": [
            {"path": f"ckpt_{treatment}.pt",
             "description": f"Latest treatment {treatment} checkpoint"}
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def wait_dataset_ready(dataset: str, *, max_wait: int = 900) -> None:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        r = _run([KAGGLE_CLI, "datasets", "status", dataset], timeout=120)
        if r.returncode == 0 and "ready" in r.stdout.lower():
            print(f"Dataset {dataset} ready.")
            return
        time.sleep(30)
    raise SystemExit(f"dataset {dataset} not ready within {max_wait}s")


def _dataset_exists(dataset: str) -> bool:
    r = _run([KAGGLE_CLI, "datasets", "status", dataset], timeout=120)
    if r.returncode != 0:
        return False
    out = r.stdout.lower()
    return "ready" in out or "processing" in out or "error" in out


def sync_checkpoint_dataset(treatment: str, ckpt_local: Path) -> None:
    """Ensure the per-treatment checkpoint dataset contains ``ckpt_local`` as
    ``ckpt_<T>.pt``. Creates the dataset for the first window and pushes a new
    version for later windows. Skips the upload if the folder already holds the
    same checkpoint."""
    dataset = f"{CKPT_DATASET_PREFIX}-{treatment.lower()}"
    folder = _ckpt_dataset_folder(treatment)
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"ckpt_{treatment}.pt"
    current_sha = _sha256_file(dst) if dst.is_file() else None
    new_sha = _sha256_file(ckpt_local)
    if current_sha == new_sha and _dataset_exists(dataset):
        print(f"  checkpoint dataset already has treatment {treatment} "
              f"checkpoint (sha={new_sha[:16]}...)")
        return
    tmp = folder / f"ckpt_{treatment}.pt.tmp"
    tmp.write_bytes(ckpt_local.read_bytes())
    os.replace(tmp, dst)
    _ensure_dataset_metadata(folder, treatment)
    print(f"  uploading treatment {treatment} checkpoint "
          f"({dst.stat().st_size / 1e9:.2f} GB, sha={new_sha[:16]}...)")
    if _dataset_exists(dataset):
        cmd = [KAGGLE_CLI, "datasets", "version",
               "-m", f"checkpoint sync for treatment {treatment}",
               "-p", str(folder)]
    else:
        cmd = [KAGGLE_CLI, "datasets", "create",
               "-p", str(folder)]
    r = _run(cmd, timeout=7200)
    if r.returncode != 0:
        raise SystemExit(f"kaggle dataset upload failed:\n{r.stderr[-2000:]}")
    wait_dataset_ready(dataset)


# --------------------------------------------------------------------------- #
# One window
# --------------------------------------------------------------------------- #
def run_window(
    treatment: str, *, short: bool = False, stop_after_tokens: int | None = None
) -> None:
    state = load_state()
    t = state[treatment]
    if t["status"] == "complete":
        print(f"Treatment {treatment} already complete; nothing to do.")
        return
    if short and stop_after_tokens is not None:
        raise SystemExit("--short and --stop-after-tokens are mutually exclusive")

    tokens_seen = t["tokens_seen"]
    if stop_after_tokens is None:
        stop = compute_stop_after_tokens(treatment, tokens_seen, short=short)
    else:
        stop = int(stop_after_tokens)
        if stop <= tokens_seen:
            raise SystemExit(
                f"explicit stop {stop} must be greater than tokens_seen {tokens_seen}"
            )
        if stop > TARGET_TOKENS:
            raise SystemExit(f"explicit stop {stop} exceeds target {TARGET_TOKENS}")
        if stop != TARGET_TOKENS and not _pause_gate_valid(stop):
            raise SystemExit(f"explicit stop {stop} is not pause-gate valid")
    print(f"\n=== Window for treatment {treatment} ===")
    print(f"  resume tokens_seen: {tokens_seen}")
    print(f"  stop_after_tokens:  {stop}")
    print(f"  window tokens:      {stop - tokens_seen}")

    # 1. Ensure the per-treatment checkpoint dataset is ready (creates it for
    #    the first window; updates it for later windows).
    sync_checkpoint_dataset(treatment, Path(t["latest_checkpoint_local"]))

    # 2. Rewrite the kernel.
    rewrite_kernel(treatment, stop)

    # 3. Push (push also starts the run).
    push_kernel()

    # 4. Poll until COMPLETE or ERROR.
    status = poll_kernel(KERNEL_ID)
    if not status.endswith("COMPLETE"):
        logs = fetch_logs(KERNEL_ID)
        print("KERNEL FAILED. Logs (tail):")
        print(logs[-4000:])
        raise SystemExit(f"kernel {KERNEL_ID} ended with {status}; failing closed")

    # 5. Download output.
    dest = CONTINUATION_DIR / "kernel_output" / f"{treatment}_w{t['windows_completed'] + 1}"
    download_output(KERNEL_ID, dest)

    # 6. Read the result manifest and verify the checkpoint SHA.
    result_path = dest / f"result_{treatment}.json"
    if not result_path.is_file():
        raise SystemExit(f"result manifest not found: {result_path}")
    result = json.loads(result_path.read_text())
    latest_rel = result["latest_checkpoint"]
    # The result manifest records the absolute Kaggle path
    # (/kaggle/working/...). The downloaded output lives under dest/, so
    # resolve the path relative to dest by stripping the /kaggle/working/
    # prefix (an absolute right-hand side in `dest / rel` would discard dest).
    rel = latest_rel
    if rel.startswith("/kaggle/working/"):
        rel = rel[len("/kaggle/working/"):]
    latest_local = dest / rel
    if not latest_local.is_file():
        raise SystemExit(f"latest checkpoint not downloaded: {latest_local}")
    actual_sha = _sha256_file(latest_local)
    expected_sha = result["latest_checkpoint_sha256"]
    if actual_sha != expected_sha:
        raise SystemExit(
            f"checkpoint SHA mismatch: expected {expected_sha[:16]}... "
            f"actual {actual_sha[:16]}...")
    print(f"  latest checkpoint: {latest_local.name} sha={actual_sha[:16]}... (verified)")

    # 7. Update the per-treatment checkpoint dataset for the next window.
    sync_checkpoint_dataset(treatment, latest_local)

    # 8. Record progress.
    t["tokens_seen"] = int(result["stop_after_tokens"])
    t["latest_checkpoint_local"] = str(latest_local)
    t["latest_checkpoint_sha256"] = actual_sha
    t["windows_completed"] += 1
    t["status"] = "complete" if t["tokens_seen"] >= TARGET_TOKENS else "in_progress"
    save_state(state)
    print(f"  treatment {treatment} now at {t['tokens_seen']} tokens "
          f"({t['windows_completed']} window(s) done), status={t['status']}")


def run_all() -> None:
    state = load_state()
    for treatment in ("A", "B", "C"):
        if state[treatment]["status"] == "complete":
            continue
        run_window(treatment)


def show_status() -> None:
    state = load_state()
    for treatment in ("A", "B", "C"):
        t = state[treatment]
        remaining = TARGET_TOKENS - t["tokens_seen"]
        print(f"  {treatment}: tokens_seen={t['tokens_seen']} "
              f"remaining={remaining} windows={t['windows_completed']} "
              f"status={t['status']} sha={t['latest_checkpoint_sha256'][:16]}...")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-controller")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="initialize controller state from origin manifest")
    sub.add_parser("status", help="show per-treatment progress")
    p_run = sub.add_parser("run-window", help="run one quota window for a treatment")
    p_run.add_argument("--treatment", required=True, choices=("A", "B", "C"))
    p_run.add_argument("--short", action="store_true",
                       help="use a short validation window (~5M tokens)")
    p_run.add_argument("--stop-after-tokens", type=int, default=None,
                       help="override the computed stop with an explicit token count")
    sub.add_parser("run-all", help="run windows until all treatments complete")
    args = parser.parse_args(argv)

    if args.cmd == "init":
        init_state()
    elif args.cmd == "status":
        show_status()
    elif args.cmd == "run-window":
        run_window(args.treatment, short=args.short,
                   stop_after_tokens=args.stop_after_tokens)
    elif args.cmd == "run-all":
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
