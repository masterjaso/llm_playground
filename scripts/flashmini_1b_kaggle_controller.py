#!/usr/bin/env python3
"""Operator surface for the FlashMini-1B Kaggle TPU trajectory.

The workstation controller prepares/freeze-validates state, submits a thin
Kaggle worker, and reports durable evidence.  It never constructs a model or
executes training locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(os.environ.get("FLASHMINI_1B_RUN_DIR", str(REPO_ROOT / "runs/flashmini/1b_kaggle")))
WORKER_DIR = REPO_ROOT / "kaggle" / "flashmini_1b_worker"
KERNEL_ID = os.environ.get("FLASHMINI_1B_KERNEL_ID", "masterjaso/flashmini-1b-tpu-v5e-8-production-worker")
KAGGLE_CLI = os.environ.get("KAGGLE_CLI", shutil.which("kaggle") or "kaggle")
OWNER = KERNEL_ID.split("/", 1)[0]
DEFAULT_CHECKPOINT_DATASET = f"{OWNER}/flashmini-1b-checkpoints"
# Notebook kernels only ship the .ipynb; companion code must arrive as a dataset.
# The worker also probes /kaggle/input/flashmini-source explicitly.
DEFAULT_SOURCE_DATASET = os.environ.get("FLASHMINI_SOURCE_DATASET", f"{OWNER}/flashmini-source")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_env() -> dict[str, str]:
    values = dict(os.environ)
    path = REPO_ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values.setdefault(key.strip(), value.strip())
    if values.get("KAGGLE_API_KEY"):
        values.setdefault("KAGGLE_API_TOKEN", values["KAGGLE_API_KEY"])
    return values


def run_kaggle(args: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([KAGGLE_CLI, *args], capture_output=True, text=True,
                          timeout=timeout, env=load_env(), check=False)


def write_json(path: Path, value: dict[str, Any]) -> None:
    from flashmini.observability import atomic_write_json
    atomic_write_json(path, value)


def state_path() -> Path:
    return RUN_DIR / "controller_state.json"


def persist_state(state: dict[str, Any]) -> None:
    write_json(state_path(), state)
    write_json(RUN_DIR / "run_status.json", state)


def load_state() -> dict[str, Any]:
    if not state_path().is_file():
        raise SystemExit(f"run is not prepared: {state_path()} (run `prepare` first)")
    return json.loads(state_path().read_text())


def prepare() -> int:
    from flashmini.production import (
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_AUDIT,
        freeze_manifest,
        sha256_file,
        source_fingerprint,
        validate_freeze_manifest,
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    previous_state = json.loads(state_path().read_text()) if state_path().is_file() else {}
    manifest_path = RUN_DIR / "freeze_manifest.json"
    action = "created"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        stale_reasons = []
        if manifest.get("source_fingerprint") != source_fingerprint():
            stale_reasons.append("source fingerprint changed")
        current_audit_sha = sha256_file(DEFAULT_SOURCE_AUDIT) if DEFAULT_SOURCE_AUDIT.is_file() else None
        if manifest.get("data_view", {}).get("source_audit_sha256") != current_audit_sha:
            stale_reasons.append("source audit changed")
        if stale_reasons:
            progressed = any([
                int(previous_state.get("tokens_seen", 0) or 0) > 0,
                int(previous_state.get("step", 0) or 0) > 0,
                bool(previous_state.get("latest_checkpoint")),
            ])
            if progressed:
                raise SystemExit(
                    "freeze contract changed after progress/checkpoint; refusing to "
                    "replace the resume identity (start a new run_id)"
                )
            manifest = freeze_manifest(manifest_path)
            action = "recreated (" + ", ".join(stale_reasons) + ")"
        else:
            validate_freeze_manifest(manifest)
            action = "validated"
    else:
        manifest = freeze_manifest(manifest_path)
    (RUN_DIR / "status").mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1, "run_id": manifest["run_id"],
        "kernel_id": KERNEL_ID, "status": "prepared",
        "target_tokens": manifest["trajectory"]["total_training_tokens"],
        "preview_pause_tokens": manifest["trajectory"]["preview_pause_tokens"],
        "tokens_seen": 0, "step": 0, "latest_checkpoint": None,
        "latest_checkpoint_sha256": None, "freeze_manifest": str(manifest_path),
        "prepared_at_utc": utc_now(), "config_path": str(DEFAULT_CONFIG),
        "source_fingerprint": manifest.get("source_fingerprint"),
        "data_view_fingerprint": manifest.get("data_view", {}).get("fingerprint"),
        "checkpoint_dataset": load_env().get(
            "FLASHMINI_REMOTE_CHECKPOINT_DATASET", DEFAULT_CHECKPOINT_DATASET
        ),
        "controller": "workstation_only",
    }
    # Re-preparing a run must not erase a queued/quota-paused or already
    # resumed state observed by the controller.  A changed run identity starts
    # from the clean defaults above.
    identity_unchanged = (
        previous_state.get("run_id") == manifest["run_id"]
        and previous_state.get("source_fingerprint") == manifest.get("source_fingerprint")
        and previous_state.get("data_view_fingerprint") == manifest.get("data_view", {}).get("fingerprint")
    )
    if identity_unchanged:
        for key in (
            "status", "session_id", "parent_session_id", "tokens_seen", "step",
            "latest_checkpoint", "latest_checkpoint_sha256", "external_blocker",
            "last_kernel_status", "last_observed_utc",
        ):
            if key in previous_state:
                state[key] = previous_state[key]
    persist_state(state)
    print(f"{action} freeze manifest: {manifest_path}")
    print(f"run_id={manifest['run_id']} params={manifest['parameter_report']['total_learned_parameters']:,}")
    print(f"virtual view={manifest['data_view']['view_id']} fingerprint={manifest['data_view']['fingerprint'][:16]}...")
    return 0


def _copytree_clean(src: Path, dst: Path) -> None:
    """Copy a tree while dropping caches and local package metadata."""
    def _ignore(_directory: str, names: list[str]) -> set[str]:
        drop = {"__pycache__", ".pytest_cache", ".mypy_cache", "flashmini.egg-info", ".venv"}
        return {name for name in names if name in drop or name.endswith(".pyc")}

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore)


def _publish_source_dataset() -> str:
    """Upload worker code as a private Kaggle dataset the notebook can mount.

    Notebook kernels only ship ``code_file``; companion ``.py`` / ``src`` next to
    the notebook never appear under ``/kaggle/working``. The worker already
    probes ``/kaggle/input/flashmini-source``.
    """
    handle = load_env().get("FLASHMINI_SOURCE_DATASET", DEFAULT_SOURCE_DATASET)
    slug = handle.split("/", 1)[-1]
    stage = RUN_DIR / "source_dataset_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "scripts/flashmini_1b_kaggle_worker.py",
                    stage / "flashmini_1b_kaggle_worker.py")
    freeze = RUN_DIR / "freeze_manifest.json"
    if freeze.is_file():
        shutil.copyfile(freeze, stage / "freeze_manifest.json")
    _copytree_clean(REPO_ROOT / "src", stage / "src")
    for relative in ("configs", "training_data/recipes", "training_data/registry",
                     "training_data/tokenizer", "training_data/manifests"):
        src = REPO_ROOT / relative
        if src.is_dir():
            _copytree_clean(src, stage / relative)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "FlashMini Source",
        "id": handle,
        "licenses": [{"name": "other"}],
    }, indent=2))
    status = run_kaggle(["datasets", "status", handle], timeout=120)
    ready = status.returncode == 0 and "ready" in (status.stdout or "").lower()
    if ready:
        published = run_kaggle(
            ["datasets", "version", "-p", str(stage), "-m",
             f"flashmini source {utc_now()}", "--dir-mode", "zip"],
            timeout=1800,
        )
    else:
        published = run_kaggle(
            ["datasets", "create", "-p", str(stage), "--dir-mode", "zip"],
            timeout=1800,
        )
    if published.returncode:
        detail = (published.stderr or published.stdout or "")[-2000:]
        raise SystemExit(f"failed to publish source dataset {handle}: {detail}")
    print(f"published source dataset {handle} (mount /kaggle/input/{slug})")
    return handle


def _ensure_checkpoint_dataset() -> str:
    """Ensure the private checkpoint dataset exists with an empty LATEST pointer."""
    handle = load_env().get("FLASHMINI_REMOTE_CHECKPOINT_DATASET", DEFAULT_CHECKPOINT_DATASET)
    status = run_kaggle(["datasets", "status", handle], timeout=120)
    if status.returncode == 0 and "ready" in (status.stdout or "").lower():
        return handle
    stage = RUN_DIR / "checkpoint_dataset_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "LATEST.json").write_text(json.dumps({
        "checkpoint": None,
        "checkpoint_sha256": None,
        "initialized_at_utc": utc_now(),
    }, indent=2))
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "FlashMini-1B Checkpoints",
        "id": handle,
        "licenses": [{"name": "other"}],
    }, indent=2))
    created = run_kaggle(
        ["datasets", "create", "-p", str(stage), "--dir-mode", "zip"],
        timeout=1800,
    )
    if created.returncode:
        detail = (created.stderr or created.stdout or "")[-2000:]
        raise SystemExit(f"failed to create checkpoint dataset {handle}: {detail}")
    print(f"initialized checkpoint dataset {handle}")
    return handle


def _ensure_worker_surface(*, smoke: bool = False) -> None:
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    source_handle = _publish_source_dataset()
    checkpoint_handle = _ensure_checkpoint_dataset()
    metadata = {
        "id": KERNEL_ID,
        "title": "FlashMini-1B TPU v5e-8 production worker",
        "code_file": "flashmini_1b_kaggle_worker.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "accelerator_type": "TPU VM v5e-8",
        "dataset_sources": [source_handle, checkpoint_handle],
        "competition_sources": [], "enable_internet": True,
    }
    (WORKER_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    # Keep a thin one-cell notebook; the real worker code mounts via dataset.
    smoke_arg = ", '--smoke'" if smoke else ""
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "flashmini-1b-worker-entry",
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, runpy, sys\n",
                "from pathlib import Path\n",
                f"os.environ.setdefault('FLASHMINI_REMOTE_CHECKPOINT_DATASET', {json.dumps(checkpoint_handle)})\n",
                f"os.environ.setdefault('FLASHMINI_SOURCE_DIR', '/kaggle/input/{source_handle.split('/', 1)[-1]}')\n",
                "bundle_roots = [\n",
                "    Path(os.environ['FLASHMINI_SOURCE_DIR']),\n",
                "    Path.cwd(), Path('/kaggle/working'),\n",
                "] + list(Path('/kaggle/input').glob('*'))\n",
                "bundle = next((root for root in bundle_roots if (root / 'flashmini_1b_kaggle_worker.py').is_file()), None)\n",
                "if bundle is None:\n",
                "    raise RuntimeError('FlashMini worker bundle is not mounted under /kaggle/input or /kaggle/working')\n",
                "sys.path.insert(0, str(bundle / 'src'))\n",
                "sys.argv = ['flashmini_1b_kaggle_worker.py', '--run-dir', '/kaggle/working/flashmini_run', '--freeze-manifest', str(bundle / 'freeze_manifest.json')" + smoke_arg + "]\n",
                "runpy.run_path(str(bundle / 'flashmini_1b_kaggle_worker.py'), run_name='__main__')\n",
            ],
        }],
    }
    (WORKER_DIR / "flashmini_1b_kaggle_worker.ipynb").write_text(json.dumps(notebook, indent=2))
    # Keep a local mirror of the worker entry for dry-run inspection only.
    shutil.copyfile(
        REPO_ROOT / "scripts/flashmini_1b_kaggle_worker.py",
        WORKER_DIR / "flashmini_1b_kaggle_worker.py",
    )
    freeze = RUN_DIR / "freeze_manifest.json"
    if freeze.is_file():
        shutil.copyfile(freeze, WORKER_DIR / "freeze_manifest.json")
    bundled_src = WORKER_DIR / "src"
    _copytree_clean(REPO_ROOT / "src", bundled_src)


def run_remote(*, dry_run: bool = False, max_wait: int = 22_200, smoke: bool = False) -> int:
    state = load_state()
    _ensure_worker_surface(smoke=smoke)
    if dry_run:
        print(f"would push {WORKER_DIR} as {KERNEL_ID}")
        return 0
    env = load_env()
    if not env.get("KAGGLE_API_KEY") and not env.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_KEY/KAGGLE_API_TOKEN is required for `run`; no secret was found")
    state.update({"status": "submitting", "submitted_at_utc": utc_now()})
    persist_state(state)
    pushed = run_kaggle(["kernels", "push", "-p", str(WORKER_DIR)], timeout=1800)
    if pushed.returncode:
        state.update({"status": "failed", "failure": "kaggle kernels push", "stderr": pushed.stderr[-1000:]})
        persist_state(state)
        raise SystemExit(pushed.stderr[-2000:] or "kaggle kernels push failed")
    print("Kaggle kernel submitted; workstation remains a controller only.")
    deadline = time.monotonic() + max_wait
    last = ""
    while time.monotonic() < deadline:
        result = run_kaggle(["kernels", "status", KERNEL_ID], timeout=120)
        text = (result.stdout or result.stderr).strip()
        last = text
        print(f"[{utc_now()}] {text[-300:]}", flush=True)
        if "COMPLETE" in text:
            # A clean Kaggle kernel exit can represent a soft quota stop or
            # the 10B preview pause, so consult the worker result before
            # calling the full 100B trajectory complete.
            output_dir = RUN_DIR / "kernel_output" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            downloaded = run_kaggle(["kernels", "output", KERNEL_ID, "-p", str(output_dir)], timeout=3600)
            result_path = output_dir / "result_manifest.json"
            if downloaded.returncode == 0 and result_path.is_file():
                result_manifest = json.loads(result_path.read_text())
                state.update({
                    "status": "complete" if result_manifest.get("status") == "COMPLETE" else "paused",
                    "worker_result": str(result_path),
                    "tokens_seen": result_manifest.get("training", {}).get("exact_tokens", state.get("tokens_seen", 0)),
                    "step": result_manifest.get("training", {}).get("step", state.get("step", 0)),
                })
            else:
                state["status"] = "complete"
            persist_state(state)
            return 0
        if "ERROR" in text or "CANCEL" in text:
            state["status"] = "failed"
            state["failure"] = text[-1000:]
            persist_state(state)
            return 2
        if "permission" in text.lower() or "denied" in text.lower() or "not found" in text.lower():
            state["status"] = "failed"
            state["failure"] = text[-1000:]
            persist_state(state)
            return 2
        time.sleep(30)
    state.update({"status": "paused_quota", "last_kernel_status": last[-1000:]})
    persist_state(state)
    return 3


def show_status(as_json: bool = False) -> int:
    state = load_state()
    status = RUN_DIR / "run_status.json"
    if status.is_file():
        state.update(json.loads(status.read_text()))
    if as_json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    print(f"run_id={state.get('run_id')} status={state.get('status')} session={state.get('session_id')}")
    print(f"tokens={int(state.get('tokens_seen', 0)):,}/{int(state.get('target_tokens', 0)):,} step={state.get('step', 0)}")
    print(f"loss={state.get('recent_loss')} tok/s={state.get('recent_tokens_per_sec')} checkpoint={state.get('latest_checkpoint')}")
    print(f"heartbeat={state.get('last_heartbeat_utc')} progress={state.get('last_progress_utc')}")
    print(f"kernel={state.get('last_kernel_status', 'unknown')}")
    return 0


def refresh_kernel_status(state: dict[str, Any]) -> dict[str, Any]:
    """Refresh the read-only Kaggle state used by ``watch``."""
    result = run_kaggle(["kernels", "status", state.get("kernel_id", KERNEL_ID)], timeout=120)
    text = (result.stdout or result.stderr).strip()
    if text:
        state["last_kernel_status"] = text[-1000:]
        state["last_observed_utc"] = utc_now()
        if "QUEUED" in text:
            if state.get("status") in {"prepared", "submitting", "training"}:
                state["status"] = "paused_quota"
            state["external_blocker"] = "Kaggle worker remains queued; no TPU VM has started"
        elif "RUNNING" in text and state.get("status") in {"prepared", "paused_quota", "submitting"}:
            state["status"] = "training"
    persist_state(state)
    return state


def watch(interval: float = 30.0, *, once: bool = False) -> int:
    while True:
        state = load_state()
        refresh_kernel_status(state)
        show_status(False)
        if once:
            return 0
        time.sleep(max(1.0, interval))


def show_logs(tail: int = 100) -> int:
    candidates = [RUN_DIR / "logs/worker.log", RUN_DIR / "status/events.jsonl"]
    for path in candidates:
        if path.is_file():
            lines = path.read_text().splitlines()
            print(f"--- {path} ---")
            print("\n".join(lines[-tail:]))
            return 0
    print("no worker logs have been downloaded yet")
    return 0


def show_metrics(tail: int = 25) -> int:
    path = RUN_DIR / "metrics.jsonl"
    if not path.is_file():
        print("no metrics have been downloaded yet")
        return 0
    lines = path.read_text().splitlines()
    print("\n".join(lines[-tail:]))
    return 0


def _download_kaggle_checkpoint_dataset() -> tuple[Path, dict[str, Any]]:
    """Download and return the latest private checkpoint dataset snapshot."""
    handle = load_env().get("FLASHMINI_REMOTE_CHECKPOINT_DATASET", DEFAULT_CHECKPOINT_DATASET)
    destination = RUN_DIR / "checkpoint_downloads" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=False)
    downloaded = run_kaggle(
        ["datasets", "download", handle, "-p", str(destination), "--unzip", "--force"],
        timeout=3600,
    )
    if downloaded.returncode:
        raise SystemExit(downloaded.stderr[-2000:] or f"could not download checkpoint dataset {handle}")
    pointer_path = destination / "LATEST.json"
    if not pointer_path.is_file():
        raise SystemExit(f"checkpoint dataset {handle} has no LATEST.json pointer")
    pointer = json.loads(pointer_path.read_text())
    name = pointer.get("checkpoint")
    if name is None:
        raise SystemExit("checkpoint dataset is initialized but has no durable checkpoint yet")
    checkpoint = destination / str(name)
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint dataset pointer references missing directory: {name}")
    return checkpoint, pointer


def verify_latest() -> int:
    from flashmini.production_checkpoint import verify_checkpoint
    state = load_state()
    latest = state.get("latest_checkpoint")
    env = load_env()
    remote_root = env.get("FLASHMINI_REMOTE_CHECKPOINT_DIR", "")
    if remote_root:
        from flashmini.production_checkpoint import FilesystemRemoteBackend
        remote_latest = FilesystemRemoteBackend(remote_root).latest_path()
        if remote_latest is not None:
            latest = str(remote_latest)
    elif env.get("FLASHMINI_REMOTE_CHECKPOINT_DATASET") or DEFAULT_CHECKPOINT_DATASET:
        latest_path, pointer = _download_kaggle_checkpoint_dataset()
        manifest = verify_checkpoint(latest_path)
        if pointer.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise SystemExit("checkpoint dataset pointer checksum does not match its manifest")
        latest = str(latest_path)
    if not latest:
        candidate = sorted(RUN_DIR.glob("checkpoints/checkpoint_step_*"))
        latest = str(candidate[-1]) if candidate else None
    if not latest:
        raise SystemExit("no durable checkpoint is recorded")
    manifest = verify_checkpoint(Path(latest))
    print(json.dumps({"valid": True, "path": latest, "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                      "step": manifest.get("step"), "exact_tokens": manifest.get("exact_tokens")}, indent=2))
    return 0


def download_latest() -> int:
    state = load_state()
    env = load_env()
    remote_root = env.get("FLASHMINI_REMOTE_CHECKPOINT_DIR", "")
    if remote_root:
        from flashmini.production_checkpoint import FilesystemRemoteBackend, verify_checkpoint
        remote_latest = FilesystemRemoteBackend(remote_root).latest_path()
        if remote_latest is None:
            raise SystemExit("durable backend has no LATEST checkpoint")
        verify_checkpoint(remote_latest)
        destination = RUN_DIR / "checkpoints" / remote_latest.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        if temporary.exists():
            raise SystemExit(f"refusing to overwrite an in-progress download: {temporary}")
        shutil.copytree(remote_latest, temporary)
        os.replace(temporary, destination)
        manifest = verify_checkpoint(destination)
        state.update({"latest_checkpoint": str(destination),
                      "latest_checkpoint_sha256": manifest.get("checkpoint_sha256")})
        persist_state(state)
        print(f"downloaded and verified {destination}")
        return 0
    if env.get("FLASHMINI_REMOTE_CHECKPOINT_DATASET") or DEFAULT_CHECKPOINT_DATASET:
        checkpoint, pointer = _download_kaggle_checkpoint_dataset()
        from flashmini.production_checkpoint import verify_checkpoint
        manifest = verify_checkpoint(checkpoint)
        if pointer.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise SystemExit("checkpoint dataset pointer checksum does not match its manifest")
        destination = RUN_DIR / "checkpoints" / checkpoint.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        if temporary.exists():
            raise SystemExit(f"refusing to overwrite an in-progress download: {temporary}")
        shutil.copytree(checkpoint, temporary)
        os.replace(temporary, destination)
        state.update({"latest_checkpoint": str(destination),
                      "latest_checkpoint_sha256": manifest.get("checkpoint_sha256")})
        persist_state(state)
        print(f"downloaded and verified {destination}")
        return 0
    print("no durable checkpoint backend is configured")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flashmini_1b_kaggle_controller")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    p_run = sub.add_parser("run")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--max-wait", type=int, default=22_200)
    p_run.add_argument("--smoke", action="store_true", help="run only the bounded TPU infrastructure smoke gate")
    p_status = sub.add_parser("status")
    p_status.add_argument("--json", action="store_true")
    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--interval", type=float, default=30.0)
    p_watch.add_argument("--once", action="store_true")
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("--tail", type=int, default=100)
    p_metrics = sub.add_parser("metrics")
    p_metrics.add_argument("--tail", type=int, default=25)
    sub.add_parser("verify-latest")
    sub.add_parser("download-latest")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare()
    if args.command == "run":
        return run_remote(dry_run=args.dry_run, max_wait=args.max_wait, smoke=args.smoke)
    if args.command == "status":
        return show_status(args.json)
    if args.command == "watch":
        return watch(args.interval, once=args.once)
    if args.command == "logs":
        return show_logs(args.tail)
    if args.command == "metrics":
        return show_metrics(args.tail)
    if args.command == "verify-latest":
        return verify_latest()
    return download_latest()


if __name__ == "__main__":
    raise SystemExit(main())
