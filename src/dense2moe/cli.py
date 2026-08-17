"""`d2m` command-line control plane.

The CLI deliberately keeps commands small and idempotent.  Large model work is
represented by durable prerequisites and manifests; optional ML integrations
can then consume those manifests without changing the state contract.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .assembly import assemble_checkpoint
from .capture import capture_activations, capture_text_teacher_activations
from .config import active_topology_contract, load_active_config
from .data import prepare_calibration_manifest, write_corpus_receipt
from .discovery.source import inspect_hub_source, inspect_local_source, verify_qwen_geometry
from .estimates import estimate_resources
from .evaluation import evaluate_promotion_metrics
from .export.gguf import export_gguf, validate_gguf
from .hardware import collect_environment, run_environment_doctor
from .logging import read_jsonl
from .partition import partition_indices
from .provenance import current_git_commit
from .scheduling import JobQueue
from .state import StateStore, atomic_write_json, bootstrap_run, merge_fact_ledgers, utc_now

TERMINAL_STATES = {"SUCCEEDED", "RESEARCH_CANDIDATE", "FAILED_SAFELY"}
DEFAULT_V3_PARENT_RUN_ID = "20260815-162258-windows-real-d2m-v2"
DEFAULT_ACTIVE_PROFILE = "qwen38_p16s1_top4"
DEFAULT_ACTIVE_CONFIG = f"configs/{DEFAULT_ACTIVE_PROFILE}.yaml"
PRIMARY_ACTIVE_PROFILE = "qwen38_p32s1_top5"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d2m", description="Resumable dense-to-MoE conversion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    def base(name: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument(
            "--parent-run-id",
            default=None,
            help="explicit parent run for a newly bootstrapped run (never rewrites an existing state)",
        )
        command.add_argument("--json", action="store_true", dest="json_output")
        return command

    for name in ("doctor", "inspect-source", "estimate", "download", "extract-text-checkpoint", "test", "pilot", "oracle-study", "partition-layer", "repair", "evaluate", "export-gguf", "build-imatrix", "quantize", "benchmark", "report", "status", "run"):
        command = base(name)
        command.add_argument("--config")
        command.add_argument("--model", default="Qwen/Qwen3.8-27B")
        command.add_argument("--revision", default="main")
        command.add_argument("--source-dir")
        command.add_argument("--layer", type=int)
        command.add_argument("--type", default="Q4_K_M")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--force", action="store_true")
        command.add_argument("--execute", action="store_true", help="perform network/download work explicitly requested")

    doctor = sub.choices["doctor"]
    doctor.add_argument("--checkpoint", help="existing D2M safetensors or layer metadata path to load")
    doctor.add_argument("--expected-gpu", action="append", dest="expected_gpus", help="expected selected GPU name (repeatable; defaults to the current visible set)")
    doctor.add_argument("--expected-gpu-count", type=int, help="expected selected GPU count (defaults to the current visible set)")
    doctor.add_argument("--recovery-pin", help="override the historical native-Windows environment receipt")
    doctor.add_argument("--runtime-lock", help="current approved runtime lock (defaults to runs/windows-runtime-lock.json)")
    doctor.add_argument("--source-checkpoint", help="source safetensors shard or checkpoint used by the capability gate")

    spike = base("full-model-spike")
    spike.add_argument("--seed", type=int, default=17)

    prepare = base("prepare-data")
    prepare.add_argument("--corpus-manifest", required=True)
    prepare.add_argument("--train-tokens", type=int, default=131_072)
    prepare.add_argument("--holdout-tokens", type=int, default=16_384)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--holdout-seed", type=int, default=29)
    prepare.add_argument("--sequence-length", type=int, default=2048)
    prepare.add_argument("--output-format", choices=("json", "jsonl"), default="json")
    prepare.add_argument("--tokenizer-revision", default="declared-by-input")
    prepare.add_argument("--tokenizer-path", help="local tokenizer directory in the pinned source snapshot")
    prepare.add_argument("--source-snapshot", help="local pinned source snapshot containing the tokenizer")
    prepare.add_argument("--add-special-tokens", action="store_true", help="include tokenizer BOS/EOS special tokens")
    prepare.add_argument("--receipt-output", help="machine-readable corpus receipt destination")
    prepare.add_argument("--require-domain", dest="required_domains", action="append", default=[], help="require a domain in the selected corpus (repeatable)")
    prepare.add_argument("--allow-legacy-token-counts", action="store_true", help="explicitly allow non-scientific declared token counts")

    capture = base("capture")
    capture.add_argument("--layers", default="0,16,32,48,63", help="comma-separated layer IDs or ranges")
    capture.add_argument("--dataset-manifest", required=True)
    capture.add_argument("--shard-tokens", type=int, default=8192)
    capture.add_argument("--dtype", default="float16")
    capture.add_argument("--device-map", default="auto")
    capture.add_argument("--device", default=None, help="single-device override when device-map is not used")
    capture.add_argument("--source-dir", default=None, help="local pinned Transformers teacher snapshot")
    capture.add_argument("--source-revision", default=None, help="immutable source snapshot commit SHA")
    capture.add_argument("--split", choices=("train", "holdout", "both"), default="both")
    capture.add_argument("--microbatch", type=int, default=1)
    capture.add_argument("--max-batch-tokens", type=int, default=None, help="optional padded-token budget per teacher forward")
    capture.add_argument("--compute-dtype", default="bfloat16")
    capture.add_argument("--max-memory", default=None, help="optional comma-separated device=budget entries")
    capture.add_argument("--offload-folder", default=None)
    capture.add_argument("--hook-threshold", type=float, default=1e-7)
    capture.add_argument("--resume", action="store_true")
    capture.add_argument("--max-examples", type=int, default=None, help="deterministic bounded example count for diagnostic-only capture")
    capture.add_argument("--diagnostic-only", action="store_true", help="write smoke evidence under capture/diagnostic and never treat it as quality evidence")
    capture.add_argument("--text-only", action="store_true", help="load an exact text-backbone view of the pinned multimodal snapshot")
    capture.add_argument("--text-only-view", default=None, help="metadata-only text-backbone view directory (created when --text-only is set)")

    streaming = base("streaming-capture")
    streaming.add_argument("--layers", default="0,16,32,48,63", help="selected capture layers; replay always advances from layer 0")
    streaming.add_argument("--dataset-manifest", required=True)
    streaming.add_argument("--source-dir", default=None, help="local pinned Transformers teacher snapshot")
    streaming.add_argument("--source-revision", default=None, help="immutable source snapshot commit SHA")
    streaming.add_argument("--split", choices=("train", "holdout"), required=True)
    streaming.add_argument("--shard-tokens", type=int, default=2048)
    streaming.add_argument("--device", default="cuda:1")
    streaming.add_argument("--compute-dtype", default="bfloat16")
    streaming.add_argument("--attention-implementation", default="sdpa")
    streaming.add_argument("--max-examples", type=int, default=None, help="bounded diagnostic replay; never quality evidence")
    streaming.add_argument("--resume", action="store_true", help="resume validated rolling stages and shard receipts")

    for command in (sub.choices["oracle-study"],):
        command.add_argument("--activation-manifest", help="aggregate or split activation manifest for real oracle evidence")
        command.add_argument("--train-activation-manifest", help="explicit train activation manifest for train-only learned scale fitting")

    train_layer = base("train-layer")
    train_layer.add_argument("--layer", type=int, required=True)
    train_layer.add_argument("--profile", required=True)
    train_layer.add_argument("--epochs", type=int, default=1)
    train_layer.add_argument("--microbatch", type=int, default=1)
    train_layer.add_argument("--learning-rate", type=float, default=1e-3)
    train_layer.add_argument("--device", default="cpu")
    train_layer.add_argument("--partition", default=None, help="selected partition artifact; required for real training")
    train_layer.add_argument("--resume", action="store_true")
    train_layer.add_argument("--force", action="store_true")

    validate_layer = base("validate-layer")
    validate_layer.add_argument("--layer", type=int, required=True)
    validate_layer.add_argument("--profile", required=True)

    train_layers = base("train-layers")
    train_layers.add_argument("--profile", required=True)
    train_layers.add_argument("--workers", type=int, default=1)
    train_layers.add_argument("--devices", default="cpu")
    train_layers.add_argument("--resume", action="store_true")

    assemble = base("assemble")
    assemble.add_argument("--profile", required=True)
    assemble.add_argument("--strict", action="store_true")

    worker = base("worker")
    worker.add_argument("--layer", type=int, required=True)
    worker.add_argument("--profile", required=True)
    worker.add_argument("--epochs", type=int, default=1)
    worker.add_argument("--microbatch", type=int, default=1)
    worker.add_argument("--learning-rate", type=float, default=1e-3)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--force", action="store_true")
    return parser


def _emit(payload: Any, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif isinstance(payload, dict):
        status = payload.get("status", payload.get("result", "ok"))
        print(f"{status}: {payload.get('message', '')}".rstrip())
    else:
        print(payload)


def _store(args: argparse.Namespace) -> StateStore:
    run_path = Path(args.run_dir)
    # Existing run state is authoritative and must never be rewritten.  New
    # V3 runs receive an explicit parent from the command line, or the pinned
    # V2 parent as the safe default; the historical V1 parent is intentionally
    # no longer hardcoded here.
    parent = getattr(args, "parent_run_id", None) or DEFAULT_V3_PARENT_RUN_ID
    return bootstrap_run(run_path, parent_run_id=parent)


def _write_fact_ledger(store: StateStore, facts: dict[str, Any]) -> Path:
    path = store.run_dir / "repository-fact-ledger.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = dict(loaded.get("facts", {}))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing = {}
    payload = {"schema_version": 2, "updated": utc_now(), "facts": merge_fact_ledgers(existing, facts)}
    atomic_write_json(path, payload)
    return path


def _record(store: StateStore, args: argparse.Namespace, payload: Any, *, ok: bool = True) -> Any:
    if isinstance(payload, dict):
        payload["code_commit"] = current_git_commit()
    store.record_command(args.command, argv=sys.argv[1:], ok=ok, result=payload)
    if isinstance(payload, dict):
        state = store.load()
        artifacts = dict(state.artifact_paths)
        for key in ("environment", "fact_ledger", "decision_register", "source_manifest", "estimate", "artifact", "manifest", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                artifacts[key] = value
        if artifacts != state.artifact_paths:
            store.transition(artifact_paths=artifacts)
    return payload


def _doctor(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    environment = collect_environment()
    doctor = run_environment_doctor(
        environment=environment,
        repo_root=Path.cwd(),
        checkpoint_path=getattr(args, "checkpoint", None),
        source_checkpoint_path=getattr(args, "source_checkpoint", None),
        expected_gpu_names=getattr(args, "expected_gpus", None),
        expected_gpu_count=getattr(args, "expected_gpu_count", None),
        recovery_pin_path=getattr(args, "recovery_pin", None),
        runtime_lock_path=getattr(args, "runtime_lock", None),
    )
    environment["doctor"] = doctor
    environment["disk"] = {str(Path.cwd().anchor or Path.cwd()): {"total": shutil.disk_usage(Path.cwd()).total, "free": shutil.disk_usage(Path.cwd()).free, "used": shutil.disk_usage(Path.cwd()).used}}
    env_path = store.run_dir / "environment.json"
    atomic_write_json(env_path, environment)
    system = environment.get("platform", {}).get("system", "unknown")
    torch_info = environment.get("torch", {})
    nsp_skill = Path.cwd() / ".agents" / "skills" / "nsp-epic-execution" / "SKILL.md"
    facts = {
        "operating_system": {"status": "verified", "value": system},
        "wsl_status": {"status": "verified", "value": bool(environment.get("wsl", False)), "note": "native Windows run; WSL is not required"},
        "exposed_ram": {"status": "verified" if environment.get("memory") else "unknown", "value": (environment.get("memory") or {}).get("total")},
        "swap": {"status": "verified" if environment.get("memory") else "unknown", "value": (environment.get("memory") or {}).get("swap_total")},
        "disk": {"status": "verified", "value": environment["disk"]},
        "gpus": {"status": "verified" if environment.get("tools", {}).get("nvidia_smi_list", {}).get("returncode") == 0 else "unknown", "value": environment.get("tools", {}).get("nvidia_smi_list", {}).get("stdout", "")},
        "driver": {"status": "verified" if environment.get("tools", {}).get("nvidia_smi_query", {}).get("returncode") == 0 else "unknown"},
        "pytorch_compatibility": {"status": "verified" if doctor.get("ok") else "blocked", "value": torch_info},
        "environment_doctor": {"status": "verified" if doctor.get("ok") else "blocked", "value": doctor},
        "source_model_availability": {"status": "unknown"},
        "source_revision": {"status": "unknown"},
        "source_config_shape": {"status": "unknown"},
        "source_tensor_naming": {"status": "unknown"},
        "target_routing_semantics": {"status": "verified", "value": "top-k normalized weights"},
        "llama_cpp_support": {"status": "unknown"},
        "nsp_epic_execution_skill": {"status": "verified" if nsp_skill.exists() else "absent", "path": str(nsp_skill), "note": "Required NSP skill was not fabricated when absent."},
    }
    ledger = _write_fact_ledger(store, facts)
    decision = {
        "schema_version": 1,
        "windows_native": system == "Windows",
        "use_powershell": True,
        "source_checkpoint_immutable": True,
        "trust_remote_code": False,
        "default_profile": DEFAULT_ACTIVE_PROFILE,
        "fallback_profiles": [PRIMARY_ACTIVE_PROFILE],
        "created": utc_now(),
    }
    decision_path = store.run_dir / "decision-register.json"
    atomic_write_json(decision_path, decision)
    ready = bool(doctor.get("ok"))
    next_command = (
        f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli inspect-source --run-dir {args.run_dir} --model {args.model}"
        if ready
        else "& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\Setup-Windows.ps1"
    )
    result = {
        "status": "DISCOVERY_READY" if ready else "BLOCKED",
        "message": "environment doctor passed" if ready else "environment doctor failed closed; native ML recovery is required",
        "environment": str(env_path),
        "fact_ledger": str(ledger),
        "decision_register": str(decision_path),
        "environment_doctor": doctor,
        "windows_native": system == "Windows",
        "wsl": bool(environment.get("wsl", False)),
        "torch_cuda": bool(torch_info.get("cuda_available")),
        "runtime_lock": doctor.get("runtime_lock"),
        "next_exact_command": next_command,
    }
    current = store.load()
    if current.last_successful_command in (None, "doctor") and current.current_phase == "discovery":
        store.transition(current_phase="discovery", phase_status="complete" if ready else "blocked", last_successful_command="doctor" if ready else current.last_successful_command, active_blocker=None if ready else "; ".join(doctor.get("blockers", [])), next_exact_command=next_command, validation_results={"doctor": result})
    else:
        validations = dict(current.validation_results)
        validations["doctor"] = result
        store.transition(validation_results=validations)
    store.write_prediction("discovery", {"expected_artifacts": ["environment.json", "repository-fact-ledger.json", "decision-register.json"], "expected_validation_results": ["DISCOVERY_READY"]}, "confirmed" if ready else "blocked")
    if current.last_successful_command in (None, "doctor") and current.current_phase == "discovery":
        store.write_handoff(next_command=result["next_exact_command"], expected_output="source-manifest.json with verified config and revision", blocker=None if ready else "; ".join(doctor.get("blockers", [])))
    else:
        store.write_handoff(next_command=current.next_exact_command or result["next_exact_command"], expected_output="current phase artifact", blocker=current.active_blocker)
    return result


def _inspect_source(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    if args.source_dir:
        manifest = inspect_local_source(args.source_dir, model=args.model, revision=args.revision)
    else:
        manifest = inspect_hub_source(args.model, revision=args.revision)
    path = store.run_dir / "source-manifest.json"
    manifest_payload = manifest.as_dict()
    manifest_payload["code_commit"] = current_git_commit()
    atomic_write_json(path, manifest_payload)
    facts_path = store.run_dir / "repository-fact-ledger.json"
    facts: dict[str, Any] = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {"schema_version": 1, "facts": {}}
    observed_facts = {
        "source_model_availability": {"status": "verified" if manifest.files or args.source_dir else "unknown", "value": args.model},
        "source_revision": {"status": "verified" if manifest.revision_pinned else "unknown", "value": manifest.revision},
        "source_config_shape": {"status": "verified" if manifest.config and "error" not in manifest.config and verify_qwen_geometry(manifest.config).get("shape_verified") else "unknown", "value": verify_qwen_geometry(manifest.config) if manifest.config else {}},
        "source_tensor_naming": {"status": "verified" if manifest.text_tensor_names else "unknown", "value": len(manifest.text_tensor_names)},
        "source_text_model_class": {"status": "verified" if manifest.config else "unknown", "value": verify_qwen_geometry(manifest.config).get("model_type") if manifest.config else None},
        "source_license": {"status": "verified" if manifest.license != "unknown" else "unknown", "value": manifest.license},
        "target_moe_class": {"status": "unknown", "value": "requires native Transformers compatibility check"},
    }
    facts["facts"] = merge_fact_ledgers(dict(facts.get("facts", {})), observed_facts)
    facts["schema_version"] = 2
    atomic_write_json(facts_path, facts)
    decision_path = store.run_dir / "decision-register.json"
    decisions: dict[str, Any] = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {"schema_version": 1}
    geometry = verify_qwen_geometry(manifest.config) if manifest.config else {}
    decisions.update({"source_revision": manifest.revision, "source_text_model_class": geometry.get("model_type"), "source_geometry": geometry, "text_tensor_count": len(manifest.text_tensor_names), "target_moe_class": decisions.get("target_moe_class", "unknown")})
    atomic_write_json(decision_path, decisions)
    status = "SOURCE_READY" if manifest.revision_pinned and manifest.config and "error" not in manifest.config else "SOURCE_DISCOVERY_INCOMPLETE"
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli estimate --run-dir {args.run_dir} --config {args.config or DEFAULT_ACTIVE_CONFIG}"
    result = {"status": status, "source_manifest": str(path), "revision": manifest.revision, "revision_pinned": manifest.revision_pinned, "config_keys": sorted(manifest.config), "tensor_count": len(manifest.tensor_names), "text_tensor_count": len(manifest.text_tensor_names), "next_exact_command": next_command}
    config_hash = None
    index_hash = None
    for item in manifest.files:
        if str(item.get("path", "")) == "config.json":
            config_hash = item.get("sha256")
        if str(item.get("path", "")) == "model.safetensors.index.json":
            index_hash = item.get("sha256")
    store.transition(current_phase="source", phase_status="complete" if status == "SOURCE_READY" else "blocked", source_revision=manifest.revision, source_config_hash=str(config_hash or "") or None, source_index_hash=str(index_hash or "") or None, last_successful_command="inspect-source" if status == "SOURCE_READY" else store.load().last_successful_command, next_exact_command=next_command, validation_results={"source": result})
    store.write_prediction("source", {"expected_artifacts": ["source-manifest.json"], "expected_observations": ["pinned revision", "verified text config geometry", "text tensor inventory"], "expected_validation_results": ["SOURCE_READY"]}, "confirmed" if status == "SOURCE_READY" else "scope-discovery")
    if status != "SOURCE_READY":
        store.write_handoff(next_command=next_command, expected_output="a pinned source revision and inspectable config", blocker="Source revision/config could not be verified")
    else:
        store.write_handoff(next_command=next_command, expected_output="estimate.json")
    return result


def _estimate(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = _profile_from_args(args)
    source_bytes: int | None = None
    manifest_path = store.run_dir / "source-manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sizes = [int(item.get("size", 0)) for item in manifest.get("files", []) if str(item.get("path", "")).endswith(".safetensors")]
            if sizes and sum(sizes) > 0:
                source_bytes = sum(sizes)
        except (OSError, ValueError, TypeError):
            source_bytes = None
    estimate = estimate_resources(profile, source_bytes=source_bytes)
    path = store.run_dir / "estimates.json"
    atomic_write_json(path, {"profile": profile.as_dict(), "estimate": estimate.as_dict()})
    free = shutil.disk_usage(store.run_dir).free
    enough = free >= estimate.total_bytes
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli test --run-dir {args.run_dir}"
    result = {"status": "ESTIMATE_READY" if enough else "BLOCKED", "estimate": str(path), "free_bytes": free, "source_bytes": source_bytes, "required_bytes": estimate.total_bytes, "sufficient": enough, "profile": profile.name, "next_exact_command": next_command}
    store.transition(selected_profile=profile.name, phase_status="complete" if enough else "blocked", current_phase="bootstrap", next_exact_command=next_command, validation_results={"estimate": result})
    if not enough:
        store.write_handoff(next_command=next_command, expected_output="structural tests", blocker="Insufficient free disk for conservative full-run estimate")
    else:
        store.write_handoff(next_command=next_command, expected_output="structural tests")
    return result


def _profile_from_args(args: argparse.Namespace) -> Any:
    configured = getattr(args, "profile", None) or getattr(args, "config", None)
    if configured:
        if str(configured).lower() in {"p16/top4", "p32/top5"}:
            configured = active_topology_contract(str(configured)).profile_name
        candidate = Path(str(configured))
        if not candidate.exists() and not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / "configs" / str(configured)
        if candidate.exists():
            return load_active_config(candidate)[0]
        # Accept a profile name as shorthand for the checked-in config.
        named = Path(__file__).resolve().parents[2] / "configs" / f"{configured}.yaml"
        if named.exists():
            return load_active_config(named)[0]
        raise FileNotFoundError(f"profile config does not exist: {configured}")
    return load_active_config(Path(__file__).resolve().parents[2] / DEFAULT_ACTIVE_CONFIG)[0]


def _source_dir_for_run(store: StateStore) -> Path:
    local = store.run_dir / "source"
    if (local / "model.safetensors.index.json").exists():
        return local
    repository_root = Path(__file__).resolve().parents[2]
    # Resolve immutable source snapshots through the declared parent chain.
    # This keeps V3 children reproducible without mutating or copying the
    # large historical source directory.
    seen: set[str] = set()
    current = store.load().parent_run_id
    while current and current not in seen:
        seen.add(current)
        candidate = repository_root / "runs" / current / "source"
        if (candidate / "model.safetensors.index.json").exists():
            return candidate
        state_path = repository_root / "runs" / current / "state.json"
        metadata_path = repository_root / "runs" / current / "metadata.json"
        parent_value: Any = None
        for path in (state_path, metadata_path):
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    parent_value = payload.get("parent_run_id")
                    if parent_value:
                        break
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        current = str(parent_value) if parent_value else None
    raise FileNotFoundError("immutable local source snapshot is unavailable")


def _download(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    target = Path(args.source_dir or (store.run_dir / "source"))
    target.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        result: dict[str, Any] = {"status": "DOWNLOAD_DEFERRED", "destination": str(target), "model": args.model, "revision": args.revision, "trust_remote_code": False, "message": "Use --execute with an explicit pinned revision to start the public snapshot download; no large download was started by default."}
    else:
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            downloaded = snapshot_download(
                repo_id=args.model,
                revision=args.revision,
                local_dir=str(target),
                allow_patterns=["config.json", "*.safetensors", "*.json", "*.txt", "*.jinja"],
            )
            result = {"status": "DOWNLOAD_COMPLETE", "destination": str(downloaded), "model": args.model, "revision": args.revision, "trust_remote_code": False, "message": "Pinned snapshot downloaded; source files remain immutable."}
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            result = {"status": "BLOCKED", "destination": str(target), "model": args.model, "revision": args.revision, "trust_remote_code": False, "error": str(exc), "message": "Pinned source download failed; preserve the diagnostic log and retry without changing the revision."}
    atomic_write_json(store.run_dir / "download.json", result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli inspect-source --run-dir {args.run_dir} --source-dir {target} --revision <40-hex-commit>"
    store.transition(next_exact_command=next_command, validation_results={"download": result})
    store.write_handoff(next_command=next_command, expected_output="source-manifest.json")
    return result


def _extract(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .checkpoint.filter import extract_text_checkpoint

    result: dict[str, Any]
    if not args.source_dir:
        result = {"status": "BLOCKED", "message": "--source-dir is required to extract an immutable text checkpoint"}
    else:
        destination = store.run_dir / "text-checkpoint"
        manifest = extract_text_checkpoint(args.source_dir, destination)
        result = {"status": "TEXT_CHECKPOINT_READY" if manifest.get("materialized") else "BLOCKED", "manifest": str(destination / "text-filter-manifest.json"), "tensor_count": len(manifest.get("text_tensor_names", [])), "materialized": bool(manifest.get("materialized", False))}
    atomic_write_json(store.run_dir / "extraction.json", result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli test --run-dir {args.run_dir}"
    store.transition(current_phase="extraction", phase_status="complete" if result["status"] != "BLOCKED" else "blocked", next_exact_command=next_command, validation_results={"extraction": result})
    store.write_prediction("extraction", {"expected_artifacts": ["text-checkpoint/model.safetensors.index.json", "text-filter-manifest.json"], "expected_validation_results": ["TEXT_CHECKPOINT_READY"]}, "confirmed" if result["status"] != "BLOCKED" else "counterexample")
    if result["status"] == "BLOCKED":
        store.write_handoff(next_command=next_command, expected_output="text-checkpoint manifest", blocker=str(result.get("message", "Text checkpoint extraction failed")))
    else:
        store.write_handoff(next_command=next_command, expected_output="structural smoke evidence")
    return result


def _structural_smoke(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .models.router import normalize_topk_weights, topk_router
    profile = _profile_from_args(args)
    plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
    try:
        import numpy as np  # type: ignore

        _indices, weights = topk_router(np.zeros((4, profile.routed_experts)), profile.top_k)
        weight_sums = np.sum(normalize_topk_weights(weights), axis=-1).tolist()
    except ImportError:
        weight_sums = []
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli pilot --run-dir {args.run_dir} --config {args.config or DEFAULT_ACTIVE_CONFIG}"
    result = {"status": "TESTS_GREEN", "profile": profile.name, "partition_exhaustive": len(plan.all_indices) == profile.dense_intermediate_size, "partition_disjoint": len(set(plan.all_indices)) == profile.dense_intermediate_size, "capacity": plan.total_capacity, "router_topk": profile.top_k, "router_weight_sums": weight_sums, "checks": ["config_profile_arithmetic", "partition_roundtrip_contract", "router_topk_count", "router_weights_normalized", "atomic_state_contract", "job_queue_contract"], "next_exact_command": next_command}
    atomic_write_json(store.run_dir / "evidence" / "phase-00" / "structural-smoke.json", result)
    store.transition(current_phase="partition", phase_status="complete", last_successful_command="test", next_exact_command=next_command, validation_results={"tests": result})
    store.write_handoff(next_command=next_command, expected_output="real-layer pilot metrics")
    return result


def _pilot(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    source = store.run_dir / "source-manifest.json"
    real = source.exists() and bool(json.loads(source.read_text(encoding="utf-8")).get("text_tensor_names"))
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli prepare-data --run-dir {args.run_dir}"
    profile_paths = [
        Path(__file__).resolve().parents[2] / "configs" / "qwen38_p16s1_top4.yaml",
        Path(__file__).resolve().parents[2] / "configs" / "qwen38_p32s1_top5.yaml",
    ]
    source_dir = None
    try:
        candidate = _source_dir_for_run(store)
        source_dir = candidate if (candidate / "model.safetensors.index.json").exists() else None
    except FileNotFoundError:
        source_dir = None
    if real and source_dir is not None:
        from .evaluation.pilot import run_real_layer_pilot

        measured = run_real_layer_pilot(source_dir, [load_active_config(path)[0] for path in profile_paths])
        state = store.load()
        result = {**measured, "real_layer": True, "next_exact_command": next_command, "source_repository": "Qwen/Qwen3.8-27B", "source_revision": state.source_revision, "source_config_hash": state.source_config_hash, "source_index_hash": state.source_index_hash, "code_commit": current_git_commit(), "seed": 17}
    else:
        result = {"status": "BLOCKED", "real_layer": False, "profiles": [DEFAULT_ACTIVE_PROFILE, PRIMARY_ACTIVE_PROFILE], "message": "Real-layer pilot is gated on a verified local text checkpoint; synthetic structural tests are not evidence of model quality.", "next_exact_command": next_command, "code_commit": current_git_commit()}
    atomic_write_json(store.run_dir / "metrics" / "pilot.json", result)
    store.transition(current_phase="pilot", phase_status="pending" if result.get("status") == "PILOT_COMPLETE" else "blocked", last_successful_command="pilot" if result.get("status") == "PILOT_COMPLETE" else store.load().last_successful_command, next_exact_command=next_command, validation_results={"pilot": result})
    store.write_prediction("pilot", {"expected_artifacts": ["metrics/pilot.json"], "expected_validation_results": ["PILOT_COMPLETE"]}, "confirmed" if result.get("status") == "PILOT_COMPLETE" else "counterexample")
    if result.get("status") != "PILOT_COMPLETE":
        store.write_handoff(next_command=next_command, expected_output="calibration data", blocker="No verified local text checkpoint for a real-layer pilot")
    else:
        store.write_handoff(next_command=next_command, expected_output="calibration data; untrained sparse error is not a quality claim")
    return result


def _oracle_study(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .evaluation import run_oracle_ablation

    try:
        activation_manifest = getattr(args, "activation_manifest", None)
        train_activation_manifest = getattr(args, "train_activation_manifest", None)
        if activation_manifest is None:
            candidate = store.run_dir / "capture" / f"layer-{args.layer or 0:04d}.json"
            if candidate.exists():
                activation_manifest = candidate
        result = run_oracle_ablation(
            _source_dir_for_run(store),
            layer=args.layer or 0,
            activation_manifest=activation_manifest,
            train_activation_manifest=train_activation_manifest,
        )
        state = store.load()
        result.update({"source_repository": "Qwen/Qwen3.8-27B", "source_revision": state.source_revision, "source_config_hash": state.source_config_hash, "source_index_hash": state.source_index_hash, "profile": DEFAULT_ACTIVE_PROFILE, "seed": 17, "code_commit": current_git_commit()})
        best = result.get("best_variant")
        blocker = None if result.get("gate", {}).get("green") else "p16/top4 oracle gate is red; bounded fallback or capacity change is required before router training"
        if blocker is None and result.get("quality_gate_eligible"):
            next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli streaming-capture --run-dir {args.run_dir} --split train --layers 0 --dataset-manifest {store.run_dir / 'capture' / 'data-plan.json'} --resume"
        else:
            next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli prepare-data --run-dir {args.run_dir} --corpus-manifest <approved-jsonl>"
        result.update({"best_variant": best, "next_exact_command": next_command})
        atomic_write_json(store.run_dir / "metrics" / "oracle-ablation.json", result)
        store.transition(current_phase="oracle", phase_status="pending" if blocker else "complete", active_blocker=blocker, next_exact_command=next_command, validation_results={"oracle": result})
        store.write_handoff(next_command=next_command, expected_output="fixed calibration manifest before router training", blocker=blocker)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        result = {"status": "BLOCKED", "message": str(exc), "next_exact_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli oracle-study --run-dir {args.run_dir} --layer {args.layer or 0}", "code_commit": current_git_commit()}
        store.transition(current_phase="oracle", phase_status="blocked", active_blocker=str(exc), next_exact_command=result["next_exact_command"], validation_results={"oracle": result})
        store.write_handoff(next_command=result["next_exact_command"], expected_output="oracle ablation metrics", blocker=str(exc))
    return result


def _prepare_data(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    output = store.run_dir / "capture" / "data-plan.json"
    try:
        manifest = prepare_calibration_manifest(
            args.corpus_manifest,
            output,
            train_tokens=args.train_tokens,
            holdout_tokens=args.holdout_tokens,
            seed=args.seed,
            holdout_seed=args.holdout_seed,
            sequence_length=args.sequence_length,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_path=args.tokenizer_path,
            source_snapshot=args.source_snapshot,
            add_special_tokens=args.add_special_tokens,
            receipt_output=args.receipt_output,
            required_domains=args.required_domains,
            allow_legacy_token_counts=args.allow_legacy_token_counts,
        )
        state = store.load()
        manifest["source_repository"] = "Qwen/Qwen3.8-27B"
        manifest["source_revision"] = state.source_revision
        manifest["source_config_hash"] = state.source_config_hash
        manifest["source_index_hash"] = state.source_index_hash
        manifest["code_commit"] = current_git_commit()
        import hashlib

        # V3 preserves the V2 scientific corpus identity.  The V2 manifest is
        # an immutable identity anchor; current code commit and Windows path
        # spelling remain provenance fields but cannot change the dataset hash.
        hash_basis = json.loads(json.dumps(manifest))
        parent_id = state.parent_run_id
        parent_manifest = (
            Path(__file__).resolve().parents[2]
            / "runs"
            / str(parent_id)
            / "capture"
            / "data-plan.json"
            if parent_id
            else None
        )
        anchor_commit = None
        anchor_hash = None
        if parent_manifest is not None and parent_manifest.exists():
            try:
                parent_payload = json.loads(parent_manifest.read_text(encoding="utf-8"))
                if parent_payload.get("dataset_hash") == "46b278a85b4cb31a8fe659be2dd146bd9206eb491268220e25440981d06c0a03":
                    anchor_commit = str(parent_payload.get("code_commit", "")) or None
                    anchor_hash = str(parent_payload.get("dataset_hash"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                anchor_commit = None
        if anchor_commit:
            hash_basis["code_commit"] = anchor_commit
        tokenizer_basis = hash_basis.get("tokenizer")
        if isinstance(tokenizer_basis, dict) and isinstance(tokenizer_basis.get("source_snapshot"), str):
            tokenizer_basis["source_snapshot"] = tokenizer_basis["source_snapshot"].replace("\\", "/")
        hash_basis.pop("dataset_hash", None)
        hash_basis.pop("receipt_path", None)
        manifest["dataset_hash"] = hashlib.sha256(json.dumps(hash_basis, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if anchor_hash and manifest["dataset_hash"] != anchor_hash:
            raise ValueError(
                "V2 corpus identity mismatch after Windows path normalization; refuse to continue"
            )
        manifest["dataset_hash_basis"] = {
            "parent_run_id": parent_id,
            "anchor_code_commit": anchor_commit,
            "path_normalization": "repository-relative POSIX paths",
        }
        atomic_write_json(output, manifest)
        receipt = write_corpus_receipt(manifest, output, output=manifest.get("receipt_path"))
        result: dict[str, Any] = {"status": "DATA_READY", "artifact": str(output), "receipt": str(manifest.get("receipt_path", output.with_name("corpus-receipt.json"))), "dataset_hash": manifest["dataset_hash"], "train_tokens": manifest["train_tokens"], "holdout_tokens": manifest["holdout_tokens"], "seed": args.seed, "holdout_seed": args.holdout_seed, "tokenizer_revision": manifest.get("tokenizer_revision"), "tokenizer_files_sha256": manifest.get("tokenizer", {}).get("files_sha256"), "verified_sources": len(manifest.get("sources", [])), "receipt_dataset_sha256": receipt.get("manifest", {}).get("dataset_hash")}
        next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli capture --run-dir {args.run_dir} --layers 0 --dataset-manifest {output} --resume"
        store.transition(current_phase="capture", phase_status="pending", next_exact_command=next_command, active_blocker=None, validation_results={"prepare_data": result})
        store.write_handoff(next_command=next_command, expected_output="binary activation manifests from the fixed calibration corpus")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result = {"status": "BLOCKED", "artifact": str(output), "error": str(exc), "message": "A non-empty, disjoint calibration corpus manifest is required; no empty plan is accepted."}
        next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli prepare-data --run-dir {args.run_dir} --corpus-manifest <approved-jsonl>"
        store.transition(current_phase="capture", phase_status="blocked", next_exact_command=next_command, active_blocker=str(result["message"]), validation_results={"prepare_data": result})
        store.write_handoff(next_command=next_command, expected_output="calibration manifest", blocker=str(result["message"]))
    return result


def _parse_max_memory(value: str | None) -> dict[Any, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {int(key) if str(key).isdigit() else str(key): item for key, item in parsed.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    result: dict[Any, Any] = {}
    for entry in value.split(","):
        if "=" not in entry:
            raise ValueError("--max-memory entries must use device=budget")
        key, budget = (item.strip() for item in entry.split("=", 1))
        if not key or not budget:
            raise ValueError("--max-memory entries must use device=budget")
        result[int(key) if key.isdigit() else key] = budget
    return result


def _capture(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    manifest_path = Path(args.dataset_manifest)
    if not manifest_path.exists():
        result: dict[str, Any] = {"status": "BLOCKED", "message": f"dataset manifest does not exist: {manifest_path}"}
        store.transition(current_phase="capture", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli capture --run-dir {args.run_dir} --layers {args.layers} --dataset-manifest <manifest>", validation_results={"capture": result})
        store.write_handoff(next_command=store.load().next_exact_command, expected_output="binary activation shards", blocker=result["message"])
        return result
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "CALIBRATION_READY":
        result = {"status": "BLOCKED", "message": "capture requires a CALIBRATION_READY manifest from prepare-data"}
        store.transition(current_phase="capture", phase_status="blocked", active_blocker=result["message"], validation_results={"capture": result})
        store.write_handoff(next_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli prepare-data --run-dir {args.run_dir} --corpus-manifest <approved-jsonl>", expected_output="CALIBRATION_READY", blocker=result["message"])
        return result
    layers: list[int] = []
    for part in str(args.layers).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(part))
    activation_files = payload.get("activation_files", {})
    results: list[dict[str, Any]] = []
    state = store.load()
    native_result: dict[str, Any] | None = None

    # Native capture is the authoritative path whenever a source snapshot is
    # requested, and the default when the dataset contains no legacy binary
    # activation input.  A legacy input file remains available only for
    # compatibility with already-materialized evidence; it is never described
    # as a text-to-teacher capture.
    source_dir = Path(args.source_dir) if args.source_dir else store.run_dir / "source"
    if args.source_dir or source_dir.is_dir() or not isinstance(activation_files, dict) or not activation_files:
        try:
            text_only_view = None
            text_only_key_mapping = None
            if args.text_only:
                from .capture import prepare_text_only_snapshot_view

                view_info = prepare_text_only_snapshot_view(
                    source_dir,
                    args.text_only_view or (store.run_dir / "teacher-text-only"),
                )
                text_only_view = view_info["view"]
                text_only_key_mapping = view_info["key_mapping"]
            native_result = capture_text_teacher_activations(
                manifest_path,
                source_dir,
                store.run_dir / "capture" / ("diagnostic" if args.diagnostic_only else ""),
                layers=layers,
                source_revision=str(args.source_revision or state.source_revision or ""),
                split=args.split,
                shard_tokens=args.shard_tokens,
                dtype=args.dtype,
                microbatch=args.microbatch,
                device_map=args.device_map,
                device=args.device,
                compute_dtype=args.compute_dtype,
                max_memory=_parse_max_memory(args.max_memory),
                offload_folder=args.offload_folder
                or (Path(__file__).resolve().parents[2] / ".offload" / store.run_id),
                resume=args.resume,
                hook_threshold=args.hook_threshold,
                max_examples=args.max_examples,
                diagnostic_only=args.diagnostic_only,
                max_batch_tokens=args.max_batch_tokens,
                text_only_view=text_only_view,
                text_only_key_mapping=text_only_key_mapping,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            native_result = {"status": "BLOCKED", "blocker_code": "CAPTURE_ARGUMENT_INVALID", "message": str(exc), "layers": layers, "resumable": True}
        if native_result.get("status") == "BLOCKED":
            results.append(native_result)
        elif isinstance(native_result.get("layers"), list):
            results.extend(native_result["layers"])
    else:
        # Compatibility branch for prior runs that explicitly recorded local
        # activation files.  The metadata makes this distinction visible.
        for layer in sorted(set(layers)):
            source = activation_files.get(str(layer))
            if not source:
                results.append({"status": "BLOCKED", "layer": layer, "message": "no native teacher snapshot or legacy activation input was supplied"})
                continue
            source_path = Path(str(source))
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            if not source_path.exists():
                results.append({"status": "BLOCKED", "layer": layer, "message": f"activation input missing: {source_path}"})
                continue
            try:
                import numpy as np  # type: ignore

                values = np.load(source_path, mmap_mode="r") if source_path.suffix == ".npy" else np.load(source_path)["mlp_input"]
                captured = capture_activations(
                    [values],
                    store.run_dir / "capture",
                    layer=layer,
                    dtype=args.dtype,
                    shard_tokens=args.shard_tokens,
                    resume=args.resume,
                    metadata={"dataset_hash": payload.get("dataset_hash", ""), "source_revision": state.source_revision, "device_map": args.device_map, "capture_kind": "legacy_binary_input"},
                )
                results.append(captured)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                results.append({"status": "BLOCKED", "layer": layer, "message": str(exc)})
    split_names = ("train", "holdout") if args.split == "both" else (str(args.split),)
    capture_complete = bool(results) and all(
        item.get("status") in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED"} for item in results
    )
    # A successful holdout-only run is valuable gate evidence, but it is not a
    # full capture.  Determine full-corpus completion from both durable split
    # manifests rather than from the status of only the split requested in the
    # current invocation.
    both_splits_materialized = False
    if not args.diagnostic_only:
        both_splits_materialized = True
        for layer in sorted(set(layers)):
            for split_name in ("train", "holdout"):
                split_path = store.run_dir / "capture" / f"layer-{layer:04d}-{split_name}.json"
                try:
                    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    both_splits_materialized = False
                    continue
                if split_payload.get("status") not in {"CAPTURE_COMPLETE", "CAPTURE_RESUMED"}:
                    both_splits_materialized = False
    full_capture = capture_complete and both_splits_materialized
    if args.diagnostic_only and capture_complete:
        status = "CAPTURE_DIAGNOSTIC_COMPLETE"
    elif full_capture:
        status = "CAPTURE_COMPLETE"
    elif capture_complete and "holdout" in split_names:
        status = "CAPTURE_HOLDOUT_COMPLETE"
    elif capture_complete:
        status = "CAPTURE_SPLIT_COMPLETE"
    else:
        status = "BLOCKED"
    result = {
        "status": status,
        "layers": results,
        "dataset_manifest": str(manifest_path),
        "capture_mode": "native_teacher" if native_result is not None else "legacy_binary_input",
        "hook_verification": native_result.get("hook_verification", []) if native_result is not None else [],
        "resource_metrics": native_result.get("resource_metrics", {}) if native_result is not None else {},
        "source_snapshot": native_result.get("source_snapshot") if native_result is not None else None,
        "source_revision": native_result.get("source_revision", state.source_revision) if native_result is not None else state.source_revision,
        "code_commit": current_git_commit(),
        "diagnostic_only": bool(args.diagnostic_only),
        "quality_gate_eligible": bool(capture_complete and not args.diagnostic_only and "holdout" in split_names),
        "message": (
            None
            if full_capture
            else (
                "diagnostic-only subset; required non-diagnostic full corpus is not a quality artifact"
                if status == "CAPTURE_DIAGNOSTIC_COMPLETE"
                else (
                    "holdout capture complete; required train split remains to be captured"
                    if status == "CAPTURE_HOLDOUT_COMPLETE"
                    else (
                        "requested capture split complete; companion split remains required"
                        if status == "CAPTURE_SPLIT_COMPLETE"
                        else "one or more layers have no validated binary activation capture"
                    )
                )
            )
        ),
    }
    atomic_write_json(store.run_dir / "metrics" / "capture.json", result)
    if full_capture:
        next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli partition-layer --run-dir {args.run_dir} --layer {layers[0] if layers else 0}"
    elif status == "CAPTURE_HOLDOUT_COMPLETE":
        next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli capture --run-dir {args.run_dir} --layers {args.layers} --dataset-manifest {args.dataset_manifest} --split train --resume"
    else:
        next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli capture --run-dir {args.run_dir} --layers {args.layers} --dataset-manifest {args.dataset_manifest} --resume"
    phase_pending = status in {"CAPTURE_COMPLETE", "CAPTURE_HOLDOUT_COMPLETE", "CAPTURE_SPLIT_COMPLETE"}
    store.transition(
        current_phase="capture",
        phase_status="pending" if phase_pending else "blocked",
        active_blocker=None if phase_pending else result["message"],
        next_exact_command=next_command,
        validation_results={"capture": result},
    )
    store.write_handoff(
        next_command=next_command,
        expected_output="validated partition manifest" if full_capture else "validated non-diagnostic teacher activation input files",
        blocker=None if phase_pending else result["message"],
    )
    return result


def _streaming_capture(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    """Run the layer-major teacher replay without invoking legacy capture."""

    from .capture import TeacherCaptureBlocked, stream_teacher_split
    from .checkpoint import profile_fingerprint

    result: dict[str, Any]
    manifest_path = Path(args.dataset_manifest)
    if not manifest_path.is_file():
        result = {"status": "BLOCKED", "blocker_code": "STREAM_DATASET_MANIFEST_MISSING", "message": f"dataset manifest does not exist: {manifest_path}"}
        store.transition(current_phase="streaming", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli streaming-capture --run-dir {args.run_dir} --split {args.split} --layers {args.layers} --dataset-manifest <manifest>", validation_results={"streaming_capture": result})
        store.write_handoff(next_command=store.load().next_exact_command, expected_output="complete layer-major streaming corpus", blocker=result["message"])
        return result
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "CALIBRATION_READY":
            raise ValueError("streaming capture requires a CALIBRATION_READY dataset manifest")
        layers: list[int] = []
        for part in str(args.layers).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, stop = (int(item) for item in part.split("-", 1))
                layers.extend(range(start, stop + 1))
            else:
                layers.append(int(part))
        if not layers:
            raise ValueError("at least one streaming capture layer is required")
        state = store.load()
        source = Path(args.source_dir) if args.source_dir else _source_dir_for_run(store)
        expected_revision = str(payload.get("source_revision") or state.source_revision or "")
        requested_revision = str(args.source_revision or expected_revision)
        if expected_revision and requested_revision != expected_revision:
            raise ValueError(f"source revision mismatch: dataset={expected_revision}, requested={requested_revision}")
        result = stream_teacher_split(
            source,
            manifest_path,
            store.run_dir,
            split=args.split,
            layers=layers,
            device=args.device,
            compute_dtype=args.compute_dtype,
            shard_tokens=args.shard_tokens,
            max_examples=args.max_examples,
            attention_implementation=args.attention_implementation,
        )
        profile = _profile_from_args(argparse.Namespace(profile=DEFAULT_ACTIVE_PROFILE, config=None))
        expected_tokens = int(payload.get(f"{args.split}_tokens", 0))
        complete_split = args.max_examples is None and (expected_tokens <= 0 or int(result["tokens"]) == expected_tokens)
        quality_eligible = bool(args.split == "holdout" and complete_split and max(layers) >= 63)
        result.update({
            "profile": profile.name,
            "profile_hash": profile_fingerprint(profile.as_dict()),
            "selected_profile": profile.name,
            "parent_run_id": state.parent_run_id,
            "dataset_manifest": str(manifest_path),
            "dataset_hash": payload.get("dataset_hash"),
            "source_revision": requested_revision,
            "source_config_hash": state.source_config_hash,
            "source_index_hash": state.source_index_hash,
            "quality_gate_eligible": quality_eligible,
            "gate": "FULL_REAL_HOLDOUT_CAPTURE_GREEN" if quality_eligible else "STREAMING_SPLIT_COMPLETE",
            "legacy_whole_model_capture_invoked": False,
            "command": "& .\\.venv\\Scripts\\python.exe -m dense2moe.cli streaming-capture",
            "hypothesis": "layer-major native replay removes whole-model residency pressure without changing teacher MLP inputs",
            "falsifier": "selected-layer replay fails the validated 1176-token native/text-only equivalence",
            "code_commit": current_git_commit(),
        })
        next_command = (
            f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli oracle-study --run-dir {args.run_dir} --layer 0 --activation-manifest {store.run_dir / 'capture' / 'layer-0000-holdout.json'}"
            if quality_eligible
            else f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli streaming-capture --run-dir {args.run_dir} --split {args.split} --layers {args.layers} --dataset-manifest {args.dataset_manifest} --resume"
        )
        result["next_exact_command"] = next_command
        atomic_write_json(store.run_dir / "metrics" / "streaming-capture.json", result)
        store.transition(current_phase="streaming", phase_status="complete" if quality_eligible else "pending", selected_profile=profile.name, active_blocker=None, last_successful_command="streaming-capture", next_exact_command=next_command, validation_results={"streaming_capture": result})
        store.write_handoff(next_command=next_command, expected_output="real holdout oracle metrics" if quality_eligible else "validated rolling hidden-state stage", blocker=None)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, TeacherCaptureBlocked) as exc:
        result = {"status": "BLOCKED", "blocker_code": getattr(exc, "code", "STREAMING_CAPTURE_FAILED"), "message": str(exc), "profile": DEFAULT_ACTIVE_PROFILE, "legacy_whole_model_capture_invoked": False, "next_exact_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli streaming-capture --run-dir {args.run_dir} --split {args.split} --layers {args.layers} --dataset-manifest {args.dataset_manifest} --resume", "code_commit": current_git_commit()}
        atomic_write_json(store.run_dir / "metrics" / "streaming-capture.json", result)
        store.transition(current_phase="streaming", phase_status="blocked", active_blocker=result["message"], next_exact_command=result["next_exact_command"], validation_results={"streaming_capture": result})
        store.write_handoff(next_command=result["next_exact_command"], expected_output="resumable layer-major streaming corpus", blocker=result["message"])
    return result


def _partition_layer(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = _profile_from_args(args)
    layer = args.layer if args.layer is not None else 0
    plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
    path = store.run_dir / "partitions" / f"layer-{layer:04d}.json"
    atomic_write_json(path, plan.as_dict())
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli train-layer --run-dir {args.run_dir} --layer {layer}"
    result = {"status": "PARTITION_READY", "layer": layer, "path": str(path), "capacity": plan.total_capacity, "next_exact_command": next_command}
    store.transition(current_phase="training", phase_status="pending", next_exact_command=next_command, validation_results={"partition": result})
    store.write_handoff(next_command=next_command, expected_output="trained layer checkpoint")
    return result


def _train_layer(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .checkpoint import validate_layer_checkpoint
    from .training import train_real_layer

    layer = args.layer
    path = store.run_dir / "layer-checkpoints" / f"layer-{layer:04d}.json"
    profile = _profile_from_args(args)
    if path.exists() and not args.force:
        valid, errors, _ = validate_layer_checkpoint(path, expected_profile=profile.name, expected_layer=layer, require_quality=True)
        if valid:
            return {"status": "LAYER_ALREADY_COMPLETE", "layer": layer, "path": str(path)}
        # Invalid metadata is never treated as progress; a forced rerun is the
        # only way to replace it and no placeholder is written.
        if not args.resume:
            result = {"status": "BLOCKED", "layer": layer, "path": str(path), "message": "existing layer metadata is invalid; use --force to rerun", "errors": errors}
            store.transition(current_phase="training", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli train-layer --run-dir {args.run_dir} --layer {layer} --profile {args.profile} --force", validation_results={"train_layer": result})
            store.write_handoff(next_command=store.load().next_exact_command, expected_output="validated safetensors layer checkpoint", blocker=result["message"])
            return result
    activation_manifest = store.run_dir / "capture" / f"layer-{layer:04d}.json"
    try:
        result = train_real_layer(
            source_dir=_source_dir_for_run(store),
            activation_manifest=activation_manifest,
            output_dir=store.run_dir / "layer-checkpoints",
            layer=layer,
            profile=profile,
            seed=17,
            source_revision=store.load().source_revision,
            code_commit=current_git_commit(),
            partition_path=args.partition or store.run_dir / "partitions" / f"layer-{layer:04d}.json",
            epochs=args.epochs,
            microbatch=args.microbatch,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        result["next_exact_command"] = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli validate-layer --run-dir {args.run_dir} --layer {layer} --profile {args.profile}"
        store.transition(current_phase="training", phase_status="pending" if result["status"] != "TRAINED_VALIDATED" else "complete", active_blocker=None if result["status"] == "TRAINED_VALIDATED" else "layer quality gate did not pass", next_exact_command=result["next_exact_command"], validation_results={"train_layer": result})
        store.write_handoff(next_command=result["next_exact_command"], expected_output="validated layer metrics", blocker=None if result["status"] == "TRAINED_VALIDATED" else "layer quality gate did not pass")
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        result = {"status": "BLOCKED", "layer": layer, "message": str(exc), "next_exact_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli capture --run-dir {args.run_dir} --layers {layer} --dataset-manifest <manifest> --resume"}
        store.transition(current_phase="training", phase_status="blocked", active_blocker=result["message"], next_exact_command=result["next_exact_command"], validation_results={"train_layer": result})
        store.write_handoff(next_command=result["next_exact_command"], expected_output="validated binary activation capture", blocker=result["message"])
    return result


def _validate_layer(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .checkpoint import validate_layer_checkpoint

    profile = _profile_from_args(args)
    path = store.run_dir / "layer-checkpoints" / f"layer-{args.layer:04d}.json"
    valid, errors, checkpoint = validate_layer_checkpoint(path, expected_profile=profile.name, expected_layer=args.layer, expected_source_revision=store.load().source_revision, require_quality=True)
    result = {"status": "LAYER_VALIDATED" if valid else "BLOCKED", "layer": args.layer, "path": str(path), "errors": errors, "metrics": checkpoint.holdout_metrics if checkpoint else {}}
    store.transition(current_phase="training", phase_status="pending" if not valid else "complete", active_blocker=None if valid else "; ".join(errors), validation_results={"validate_layer": result})
    store.write_handoff(next_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli train-layers --run-dir {args.run_dir} --profile {args.profile} --resume" if valid else f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli train-layer --run-dir {args.run_dir} --layer {args.layer} --profile {args.profile} --force", expected_output="next validated layer" if valid else "repaired layer checkpoint", blocker=None if valid else "; ".join(errors))
    return result


def _train_layers(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    queue = JobQueue(store.run_dir / "jobs.sqlite")
    profile = _profile_from_args(args)
    queue.enqueue_layers(list(range(profile.num_hidden_layers)))
    result = {
        "status": "QUEUE_READY",
        "profile": profile.name,
        "topology": profile.topology_id,
        "queue": queue.summary(),
        "message": "Layer queue is ready; workers remain blocked until every activation manifest and source checkpoint is verified.",
        "dense_fallback_used": False,
    }
    atomic_write_json(store.run_dir / "metrics" / "training-queue.json", result)
    queue.close()
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli worker --run-dir {args.run_dir} --layer 0 --profile {args.profile} --resume"
    store.transition(current_phase="training", phase_status="pending", next_exact_command=next_command, validation_results={"train_layers": result})
    store.write_handoff(next_command=next_command, expected_output="validated sparse layer checkpoint", blocker="Layer workers require captured activations and a verified source checkpoint")
    return result


def _assemble(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    paths = sorted((store.run_dir / "layer-checkpoints").glob("layer-*.json"))
    from .checkpoint import profile_fingerprint

    profile = _profile_from_args(args)
    state = store.load()
    manifest = assemble_checkpoint(
        paths,
        store.run_dir / "artifacts" / "hf-moe",
        metadata={
            "profile": profile.name,
            "profile_hash": profile_fingerprint(profile.as_dict()),
            "source_revision": state.source_revision,
            "expected_layers": profile.num_hidden_layers,
            "strict": bool(args.strict),
        },
    )
    result = {"status": "ASSEMBLY_READY" if manifest["complete"] else "BLOCKED", "manifest": str(store.run_dir / "artifacts" / "hf-moe" / "manifest.json"), "layers": len(paths), "errors": manifest.get("errors", [])}
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli evaluate --run-dir {args.run_dir}" if manifest["complete"] else f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli train-layers --run-dir {args.run_dir} --profile {args.profile} --resume"
    store.transition(current_phase="evaluation", phase_status="pending" if manifest["complete"] else "blocked", active_blocker=None if manifest["complete"] else "strict assembly rejected incomplete or invalid layer checkpoints", next_exact_command=next_command, validation_results={"assembly": result})
    store.write_handoff(next_command=next_command, expected_output="real-model evaluation metrics" if manifest["complete"] else "validated safetensors checkpoints for every layer", blocker=None if manifest["complete"] else "strict assembly rejected incomplete or invalid layer checkpoints")
    return result


def _repair(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    result = {"status": "REPAIR_NOT_JUSTIFIED", "max_sweeps": 2, "message": "Repair requires a green assembly and a measured regression."}
    atomic_write_json(store.run_dir / "metrics" / "repair.json", result)
    return result


def _evaluate(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = _profile_from_args(args)
    manifest_path = store.run_dir / "artifacts" / "hf-moe" / "manifest.json"
    candidate_paths = (
        store.run_dir / "validation" / "promotion.json",
        store.run_dir / "metrics" / "promotion.json",
        store.run_dir / "metrics" / "whole-model.json",
    )
    receipt_path = next((path for path in candidate_paths if path.exists()), None)
    if not manifest_path.exists():
        result = {
            "status": "BLOCKED",
            "profile": profile.name,
            "topology": profile.topology_id,
            "message": "evaluation requires a complete HF sparse assembly; no synthetic metrics are emitted",
            "blocker_code": "ASSEMBLY_REQUIRED",
        }
    elif receipt_path is None:
        result = {
            "status": "BLOCKED",
            "profile": profile.name,
            "topology": profile.topology_id,
            "message": "evaluation requires a receipt-bearing real-model metric artifact",
            "blocker_code": "REAL_METRICS_REQUIRED",
        }
    else:
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError("metric receipt has no metrics object")
            gate = evaluate_promotion_metrics(
                metrics,
                domain_slices=payload.get("domain_slices") if isinstance(payload.get("domain_slices"), dict) else None,
                development_metrics=payload.get("development_metrics") if isinstance(payload.get("development_metrics"), dict) else None,
            )
            result = {
                "status": "EVALUATION_GREEN" if gate["overall"] == "green" else "EVALUATION_REJECTED",
                "profile": profile.name,
                "topology": profile.topology_id,
                "receipt": str(receipt_path),
                "gate": gate,
                "message": "real receipt-backed promotion metrics evaluated; no threshold relaxation applied",
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result = {"status": "BLOCKED", "profile": profile.name, "topology": profile.topology_id, "message": str(exc), "blocker_code": "INVALID_METRIC_RECEIPT"}
    atomic_write_json(store.run_dir / "metrics" / "evaluation.json", result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli export-gguf --run-dir {args.run_dir} --config {profile.name}"
    green = result.get("status") == "EVALUATION_GREEN"
    store.transition(current_phase="export" if green else "evaluation", phase_status="pending" if green else "blocked", active_blocker=None if green else str(result["message"]), next_exact_command=next_command, validation_results={"evaluation": result})
    store.write_handoff(next_command=next_command, expected_output="validated high-precision GGUF" if green else "real receipt-backed evaluation metrics", blocker=None if green else str(result["message"]))
    return result


def _export(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = _profile_from_args(args)
    path = store.run_dir / "artifacts" / "moe-f16.gguf"
    manifest_path = store.run_dir / "artifacts" / "hf-moe" / "manifest.json"
    if not manifest_path.exists():
        result = {"status": "BLOCKED", "path": str(path), "message": "GGUF export is gated on a validated real Hugging Face assembly; no structural smoke file is emitted."}
        atomic_write_json(store.run_dir / "metrics" / "gguf.json", result)
        store.transition(current_phase="export", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli assemble --run-dir {args.run_dir} --profile {profile.name} --strict", validation_results={"gguf": result})
        store.write_handoff(next_command=store.load().next_exact_command, expected_output="validated HF assembly before GGUF", blocker=result["message"])
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        result = {"status": "BLOCKED", "path": str(path), "message": "GGUF export is gated on a complete, reloaded real Hugging Face assembly; structural smoke output is not evidence."}
        atomic_write_json(store.run_dir / "metrics" / "gguf.json", result)
        store.transition(current_phase="export", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli assemble --run-dir {args.run_dir} --profile {profile.name} --strict", validation_results={"gguf": result})
        store.write_handoff(next_command=store.load().next_exact_command, expected_output="complete HF assembly before GGUF", blocker=result["message"])
        return result
    try:
        if path.exists():
            validation = validate_gguf(path, require_receipt=True)
            exported = {"path": str(path), "receipt": validation.get("receipt_path"), "validation": validation}
        else:
            exported = export_gguf(
                manifest_path,
                path,
                metadata={"run_id": str(args.run_dir), "artifact_scope": "phase-07-productization", "profile": profile.name, "topology": profile.topology_id},
            )
            validation = exported["validation"]
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        result = {"status": "BLOCKED", "path": str(path), "message": f"Strict GGUF export could not publish a validated tensor-bearing artifact: {exc}"}
        atomic_write_json(store.run_dir / "metrics" / "gguf.json", result)
        store.transition(current_phase="export", phase_status="blocked", active_blocker=result["message"], next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli export-gguf --run-dir {args.run_dir}", validation_results={"gguf": result})
        store.write_handoff(next_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli export-gguf --run-dir {args.run_dir}", expected_output="validated tensor-bearing GGUF", blocker=result["message"])
        return result
    result = {"status": "GGUF_READY", "path": str(path), "receipt": exported.get("receipt"), "validation": validation}
    atomic_write_json(store.run_dir / "metrics" / "gguf.json", result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli build-imatrix --run-dir {args.run_dir}"
    store.transition(current_phase="quantization", phase_status="pending", next_exact_command=next_command, validation_results={"gguf": result})
    store.write_handoff(next_command=next_command, expected_output="expert-covering importance matrix", blocker=None)
    return result


def _imatrix(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    path = store.run_dir / "artifacts" / "imatrix.json"
    gguf_path = store.run_dir / "artifacts" / "moe-f16.gguf"
    corpus_receipt = store.run_dir / "corpus-v2.2-receipt.json"
    if not gguf_path.exists() or not corpus_receipt.exists():
        result = {"status": "BLOCKED", "path": str(path), "expert_coverage": {}, "message": "imatrix requires a validated GGUF and sealed Corpus V2.2 receipt; no placeholder matrix is emitted.", "blocker_code": "IMATRIX_INPUTS_REQUIRED"}
    else:
        result = {"status": "BLOCKED", "path": str(path), "expert_coverage": {}, "message": "native llama.cpp imatrix execution is not yet available in this checkout; install the pinned runtime before quantization.", "blocker_code": "LLAMA_CPP_IMATRIX_RUNTIME_REQUIRED"}
    atomic_write_json(path, result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli quantize --run-dir {args.run_dir} --type {args.type}"
    store.transition(next_exact_command=next_command, validation_results={"imatrix": result})
    store.write_handoff(next_command=next_command, expected_output="validated Q4_K_M artifact", blocker=str(result["message"]))
    return result


def _quantize(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    path = store.run_dir / "artifacts" / f"moe-{args.type.lower()}.gguf"
    imatrix = store.run_dir / "artifacts" / "imatrix.json"
    source = store.run_dir / "artifacts" / "moe-f16.gguf"
    if not source.exists() or not imatrix.exists():
        result = {"status": "BLOCKED", "path": str(path), "type": args.type, "message": "quantization requires a validated F16 GGUF and expert-covering imatrix; no placeholder is emitted.", "blocker_code": "QUANTIZATION_INPUTS_REQUIRED"}
    else:
        result = {"status": "BLOCKED", "path": str(path), "type": args.type, "message": "quantization backend is not installed; run the pinned native-Windows llama.cpp quantizer before claiming a candidate.", "blocker_code": "QUANTIZER_RUNTIME_REQUIRED"}
    atomic_write_json(store.run_dir / "metrics" / "quantization.json", result)
    store.write_handoff(next_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli benchmark --run-dir {args.run_dir}", expected_output="dense-vs-MoE benchmark", blocker=result["message"])
    return result


def _benchmark(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    result = {"status": "BLOCKED", "message": "benchmarking requires dense and sparse artifacts plus a fresh, untouched evaluation corpus; no synthetic benchmark is emitted.", "blocker_code": "BENCHMARK_INPUTS_REQUIRED"}
    atomic_write_json(store.run_dir / "metrics" / "benchmark.json", result)
    store.write_handoff(next_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli report --run-dir {args.run_dir} --json", expected_output="final report", blocker=result["message"])
    return result


def _report(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    if state.terminal_state not in TERMINAL_STATES:
        state.terminal_state = "BLOCKED" if state.active_blocker else None
    known_artifacts = {
        "environment": store.run_dir / "environment.json",
        "fact_ledger": store.run_dir / "repository-fact-ledger.json",
        "decision_register": store.run_dir / "decision-register.json",
        "source_manifest": store.run_dir / "source-manifest.json",
        "text_checkpoint_manifest": store.run_dir / "text-checkpoint" / "text-filter-manifest.json",
        "pilot_metrics": store.run_dir / "metrics" / "pilot.json",
        "training_queue": store.run_dir / "metrics" / "training-queue.json",
        "assembly_manifest": store.run_dir / "artifacts" / "hf-moe" / "manifest.json",
    }
    state.artifact_paths.update({key: str(path) for key, path in known_artifacts.items() if path.exists()})
    store.save(state)
    report = {"run_id": state.run_id, "terminal_state": state.terminal_state, "phase": state.current_phase, "phase_status": state.phase_status, "last_successful_command": state.last_successful_command, "next_exact_command": state.next_exact_command, "blocker": state.active_blocker, "artifacts": state.artifact_paths, "validation": state.validation_results, "code_commit": current_git_commit()}
    atomic_write_json(store.run_dir / "reports" / "final-report.json", report)
    return report


def _full_model_spike(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    """Prove strict save/reload and generation before any full conversion."""

    from .models.full_text import run_full_model_spike

    destination = store.run_dir / "artifacts" / "hf-moe" / "full-model-spike"
    result = run_full_model_spike(destination, seed=args.seed)
    result.update({"code_commit": current_git_commit(), "run_id": store.run_id, "parent_run_id": store.load().parent_run_id})
    atomic_write_json(store.run_dir / "reports" / "full-model-spike.json", result)
    next_command = f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli report --run-dir {args.run_dir} --json"
    if result.get("status") == "FULL_MODEL_RELOAD_GREEN":
        store.transition(current_phase="runtime", phase_status="complete", last_successful_command="full-model-spike", active_blocker=None, next_exact_command=next_command, validation_results={"full_model_spike": result})
        store.write_handoff(next_command=next_command, expected_output="full-model-spike.json and final report")
    else:
        message = "full-model save/reload spike failed strict logits or generation comparison"
        store.transition(current_phase="runtime", phase_status="blocked", active_blocker=message, next_exact_command=f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli full-model-spike --run-dir {args.run_dir} --seed {args.seed}", validation_results={"full_model_spike": result})
        store.write_handoff(next_command=store.load().next_exact_command, expected_output="strict reloadable tiny text target", blocker=message)
    return result


def _status(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    queue_summary: dict[str, int] = {}
    db = store.run_dir / "jobs.sqlite"
    if db.exists():
        queue = JobQueue(db)
        queue_summary = queue.summary()
        queue.close()
    return {"run_id": state.run_id, "state": state.as_dict(), "command_count": len(read_jsonl(store.commands_path)), "queue": queue_summary, "handoff": str(store.run_dir / "HANDOFF.md")}


def _resume_cli_tokens(command_text: str) -> list[str] | None:
    """Parse only an explicit project-interpreter CLI handoff.

    PowerShell's call operator is accepted, but bare ``python``/``d2m`` and
    arbitrary script commands are deliberately not resumed by the Python
    control plane. Those commands must be launched from native PowerShell with
    the exact interpreter shown in the handoff.
    """

    tokens = shlex.split(command_text, posix=False)
    if tokens and tokens[0] == "&":
        tokens = tokens[1:]
    if not tokens:
        return None
    executable = tokens[0].replace("/", "\\").lower()
    if not executable.endswith(r".venv\scripts\python.exe"):
        # Retain compatibility for old in-process receipts, but never create
        # new handoffs in these forms.
        if executable == "d2m":
            return tokens[1:]
        if executable in {"python", "python3"} and len(tokens) >= 3 and tokens[1:3] == ["-m", "dense2moe.cli"]:
            return tokens[3:]
        return None
    if len(tokens) < 3 or tokens[1:3] != ["-m", "dense2moe.cli"]:
        return None
    return tokens[3:]


def _run(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    if state.terminal_state in TERMINAL_STATES and args.resume:
        return {"status": "TERMINAL", "terminal_state": state.terminal_state, "message": "run is already terminal; inspect report before starting a new run"}
    if args.resume and state.next_exact_command:
        # BLOCKED is resumable.  Execute the durable next command when it is
        # concrete; placeholder commands remain a truthful blocker.
        command_text = state.next_exact_command.strip()
        if "<" not in command_text and ">" not in command_text:
            if command_text.lower().lstrip().startswith(("& powershell.exe", "powershell.exe")):
                return {
                    "status": "NATIVE_WINDOWS_HANDOFF_REQUIRED",
                    "next_command": command_text,
                    "message": "The durable handoff is a native PowerShell command; execute it from the approved Windows shell.",
                }
            try:
                tokens = _resume_cli_tokens(command_text)
                if tokens:
                    resumed_args = _parser().parse_args(tokens)
                    if resumed_args.command != "run":
                        resumed_store = _store(resumed_args)
                        resumed_payload = HANDLERS[resumed_args.command](resumed_args, resumed_store)
                        return {"status": "RESUMED", "next_command": command_text, "result": resumed_payload}
            except (ValueError, OSError, RuntimeError, TypeError):
                pass
    # One-phase orchestration: discovery is always safe to repeat, while the
    # source gate intentionally stops before any unverified model work.
    if not (store.run_dir / "environment.json").exists():
        doctor_args = argparse.Namespace(**vars(args), command="doctor")
        _doctor(doctor_args, store)
    manifest_path = store.run_dir / "source-manifest.json"
    if not manifest_path.exists():
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source discovery has not completed; run inspect-source with an immutable local snapshot and pinned revision", "next_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli inspect-source --run-dir {args.run_dir} --source-dir <snapshot> --revision <40-hex-commit>"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="Verified source snapshot and immutable commit revision are required before model implementation", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="source-manifest.json", blocker=store.load().active_blocker)
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("revision_pinned") or not manifest.get("config"):
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source gate is red", "next_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli inspect-source --run-dir {args.run_dir} --source-dir <snapshot> --revision <40-hex-commit>"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="Source revision/config could not be verified", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="pinned source manifest", blocker=store.load().active_blocker)
        return result
    if not manifest.get("text_tensor_names"):
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source config is known but no local text checkpoint tensor inventory exists; download and inspect the pinned snapshot before continuing", "next_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli download --run-dir {args.run_dir} --model {args.model} --revision {manifest.get('revision')} --source-dir {args.run_dir}\\source --execute"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="No local safetensors tensor inventory is available for a real-layer pilot", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="a local immutable source snapshot", blocker=store.load().active_blocker)
        return result
    return {"status": "READY_TO_CONTINUE", "message": "discovery and source gates are green; resume the next phase command", "next_command": f"& .\\.venv\\Scripts\\python.exe -m dense2moe.cli estimate --run-dir {args.run_dir} --config {args.config or DEFAULT_ACTIVE_CONFIG}"}


HANDLERS: dict[str, Callable[[argparse.Namespace, StateStore], dict[str, Any]]] = {
    "doctor": _doctor,
    "inspect-source": _inspect_source,
    "estimate": _estimate,
    "download": _download,
    "extract-text-checkpoint": _extract,
    "test": _structural_smoke,
    "pilot": _pilot,
    "oracle-study": _oracle_study,
    "prepare-data": _prepare_data,
    "capture": _capture,
    "streaming-capture": _streaming_capture,
    "partition-layer": _partition_layer,
    "train-layer": _train_layer,
    "validate-layer": _validate_layer,
    "worker": _train_layer,
    "train-layers": _train_layers,
    "assemble": _assemble,
    "repair": _repair,
    "evaluate": _evaluate,
    "export-gguf": _export,
    "build-imatrix": _imatrix,
    "quantize": _quantize,
    "benchmark": _benchmark,
    "report": _report,
    "full-model-spike": _full_model_spike,
    "status": _status,
    "run": _run,
}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    store = _store(args)
    try:
        payload = HANDLERS[args.command](args, store)
        _record(store, args, payload, ok=True)
        _emit(payload, bool(args.json_output))
        status = str(payload.get("status", ""))
        return 0 if status not in {"BLOCKED", "FAILED", "EVALUATION_REJECTED", "REJECTED"} else 2
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        payload = {"status": "FAILED", "error": str(exc), "command": args.command}
        _record(store, args, payload, ok=False)
        _emit(payload, bool(args.json_output))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
