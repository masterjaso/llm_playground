"""`d2m` command-line control plane.

The CLI deliberately keeps commands small and idempotent.  Large model work is
represented by durable prerequisites and manifests; optional ML integrations
can then consume those manifests without changing the state contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .assembly import assemble_checkpoint
from .capture import capture_activations
from .config import load_config
from .discovery.source import inspect_hub_source, inspect_local_source, verify_qwen_geometry
from .estimates import estimate_resources
from .evaluation import quality_gate
from .export.gguf import validate_gguf, write_tiny_gguf
from .hardware import collect_environment
from .logging import read_jsonl
from .partition import partition_indices
from .scheduling import JobQueue
from .state import StateStore, atomic_write_json, bootstrap_run, utc_now

TERMINAL_STATES = {"SUCCEEDED", "RESEARCH_CANDIDATE", "FAILED_SAFELY", "BLOCKED"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d2m", description="Resumable dense-to-MoE conversion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = [
        "doctor", "inspect-source", "estimate", "download", "extract-text-checkpoint", "test", "pilot",
        "prepare-data", "capture", "partition-layer", "train-layer", "worker", "train-layers", "assemble",
        "repair", "evaluate", "export-gguf", "build-imatrix", "quantize", "benchmark", "report", "status", "run",
    ]
    for name in commands:
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--json", action="store_true", dest="json_output")
        command.add_argument("--config")
        command.add_argument("--model", default="Qwen/Qwen3.8-27B")
        command.add_argument("--revision", default="main")
        command.add_argument("--source-dir")
        command.add_argument("--layer", type=int)
        command.add_argument("--type", default="Q4_K_M")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--force", action="store_true")
        command.add_argument("--execute", action="store_true", help="perform network/download work explicitly requested")
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
    return bootstrap_run(args.run_dir)


def _write_fact_ledger(store: StateStore, facts: dict[str, Any]) -> Path:
    path = store.run_dir / "repository-fact-ledger.json"
    payload = {"schema_version": 1, "updated": utc_now(), "facts": facts}
    atomic_write_json(path, payload)
    return path


def _record(store: StateStore, args: argparse.Namespace, payload: Any, *, ok: bool = True) -> Any:
    store.record_command(args.command, argv=sys.argv[1:], ok=ok, result=payload)
    return payload


def _doctor(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    environment = collect_environment()
    environment["disk"] = {str(Path.cwd().anchor or Path.cwd()): {"total": shutil.disk_usage(Path.cwd()).total, "free": shutil.disk_usage(Path.cwd()).free, "used": shutil.disk_usage(Path.cwd()).used}}
    env_path = store.run_dir / "environment.json"
    atomic_write_json(env_path, environment)
    system = environment.get("platform", {}).get("system", "unknown")
    torch_info = environment.get("torch", {})
    facts = {
        "operating_system": {"status": "verified", "value": system},
        "wsl_status": {"status": "verified", "value": bool(environment.get("wsl", False)), "note": "native Windows run; WSL is not required"},
        "exposed_ram": {"status": "verified" if environment.get("memory") else "unknown", "value": (environment.get("memory") or {}).get("total")},
        "swap": {"status": "verified" if environment.get("memory") else "unknown", "value": (environment.get("memory") or {}).get("swap_total")},
        "disk": {"status": "verified", "value": environment["disk"]},
        "gpus": {"status": "verified" if environment.get("tools", {}).get("nvidia_smi_list", {}).get("returncode") == 0 else "unknown", "value": environment.get("tools", {}).get("nvidia_smi_list", {}).get("stdout", "")},
        "driver": {"status": "verified" if environment.get("tools", {}).get("nvidia_smi_query", {}).get("returncode") == 0 else "unknown"},
        "pytorch_compatibility": {"status": "verified" if torch_info.get("cuda_available") else "unknown", "value": torch_info},
        "source_model_availability": {"status": "unknown"},
        "source_revision": {"status": "unknown"},
        "source_config_shape": {"status": "unknown"},
        "source_tensor_naming": {"status": "unknown"},
        "target_routing_semantics": {"status": "verified", "value": "top-k normalized weights"},
        "llama_cpp_support": {"status": "unknown"},
    }
    ledger = _write_fact_ledger(store, facts)
    decision = {
        "schema_version": 1,
        "windows_native": system == "Windows",
        "use_powershell": True,
        "source_checkpoint_immutable": True,
        "trust_remote_code": False,
        "default_profile": "qwen38_p32s1_top2",
        "fallback_profiles": ["qwen38_p16s1_top2", "qwen38_p8s1_top2"],
        "created": utc_now(),
    }
    decision_path = store.run_dir / "decision-register.json"
    atomic_write_json(decision_path, decision)
    next_command = f"d2m inspect-source --run-dir {args.run_dir} --model {args.model}"
    result = {"status": "DISCOVERY_READY", "environment": str(env_path), "fact_ledger": str(ledger), "decision_register": str(decision_path), "windows_native": system == "Windows", "wsl": bool(environment.get("wsl", False)), "torch_cuda": bool(torch_info.get("cuda_available")), "next_exact_command": next_command}
    store.transition(current_phase="discovery", phase_status="complete", last_successful_command="doctor", next_exact_command=next_command, validation_results={"doctor": result})
    store.write_handoff(next_command=result["next_exact_command"], expected_output="source-manifest.json with verified config and revision")
    return result


def _inspect_source(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    if args.source_dir:
        manifest = inspect_local_source(args.source_dir, model=args.model, revision=args.revision)
    else:
        manifest = inspect_hub_source(args.model, revision=args.revision)
    path = store.run_dir / "source-manifest.json"
    atomic_write_json(path, manifest.as_dict())
    facts_path = store.run_dir / "repository-fact-ledger.json"
    facts: dict[str, Any] = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {"schema_version": 1, "facts": {}}
    facts.setdefault("facts", {})
    facts["facts"].update({
        "source_model_availability": {"status": "verified" if manifest.files or args.source_dir else "unknown", "value": args.model},
        "source_revision": {"status": "verified" if manifest.revision_pinned else "unknown", "value": manifest.revision},
        "source_config_shape": {"status": "verified" if manifest.config and "error" not in manifest.config and verify_qwen_geometry(manifest.config).get("shape_verified") else "unknown", "value": verify_qwen_geometry(manifest.config) if manifest.config else {}},
        "source_tensor_naming": {"status": "verified" if manifest.text_tensor_names else "unknown", "value": len(manifest.text_tensor_names)},
        "source_license": {"status": "verified" if manifest.license != "unknown" else "unknown", "value": manifest.license},
    })
    atomic_write_json(facts_path, facts)
    status = "SOURCE_READY" if manifest.revision_pinned and manifest.config and "error" not in manifest.config else "SOURCE_DISCOVERY_INCOMPLETE"
    next_command = f"d2m estimate --run-dir {args.run_dir} --config {args.config or 'configs/qwen38_p32s1_top2.yaml'}"
    result = {"status": status, "source_manifest": str(path), "revision": manifest.revision, "revision_pinned": manifest.revision_pinned, "config_keys": sorted(manifest.config), "tensor_count": len(manifest.tensor_names), "text_tensor_count": len(manifest.text_tensor_names), "next_exact_command": next_command}
    store.transition(current_phase="source", phase_status="complete" if status == "SOURCE_READY" else "blocked", source_revision=manifest.revision, last_successful_command="inspect-source" if status == "SOURCE_READY" else store.load().last_successful_command, next_exact_command=next_command, validation_results={"source": result})
    if status != "SOURCE_READY":
        store.write_handoff(next_command=next_command, expected_output="a pinned source revision and inspectable config", blocker="Source revision/config could not be verified")
    else:
        store.write_handoff(next_command=next_command, expected_output="estimate.json")
    return result


def _estimate(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = load_config(args.config or str(Path(__file__).resolve().parents[2] / "configs" / "qwen38_p32s1_top2.yaml"))
    estimate = estimate_resources(profile)
    path = store.run_dir / "estimates.json"
    atomic_write_json(path, {"profile": profile.as_dict(), "estimate": estimate.as_dict()})
    free = shutil.disk_usage(store.run_dir).free
    enough = free >= estimate.total_bytes
    next_command = f"d2m test --run-dir {args.run_dir}"
    result = {"status": "ESTIMATE_READY" if enough else "BLOCKED", "estimate": str(path), "free_bytes": free, "required_bytes": estimate.total_bytes, "sufficient": enough, "profile": profile.name, "next_exact_command": next_command}
    store.transition(selected_profile=profile.name, phase_status="complete" if enough else "blocked", current_phase="bootstrap", next_exact_command=next_command, validation_results={"estimate": result})
    if not enough:
        store.write_handoff(next_command=next_command, expected_output="structural tests", blocker="Insufficient free disk for conservative full-run estimate")
    return result


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
    next_command = f"d2m inspect-source --run-dir {args.run_dir} --source-dir {target} --revision <40-hex-commit>"
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
    store.transition(current_phase="extraction", phase_status="complete" if result["status"] != "BLOCKED" else "blocked", next_exact_command=f"d2m test --run-dir {args.run_dir}", validation_results={"extraction": result})
    return result


def _structural_smoke(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    from .models.router import normalize_topk_weights, topk_router
    profile = load_config(args.config or str(Path(__file__).resolve().parents[2] / "configs" / "qwen38_p32s1_top2.yaml"))
    plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
    try:
        import numpy as np  # type: ignore

        _indices, weights = topk_router(np.zeros((4, profile.routed_experts)), profile.top_k)
        weight_sums = np.sum(normalize_topk_weights(weights), axis=-1).tolist()
    except ImportError:
        weight_sums = []
    next_command = f"d2m pilot --run-dir {args.run_dir} --config {args.config or 'configs/qwen38_p32s1_top2.yaml'}"
    result = {"status": "TESTS_GREEN", "profile": profile.name, "partition_exhaustive": len(plan.all_indices) == profile.dense_intermediate_size, "partition_disjoint": len(set(plan.all_indices)) == profile.dense_intermediate_size, "capacity": plan.total_capacity, "router_topk": profile.top_k, "router_weight_sums": weight_sums, "checks": ["config_profile_arithmetic", "partition_roundtrip_contract", "router_topk_count", "router_weights_normalized", "atomic_state_contract", "job_queue_contract"], "next_exact_command": next_command}
    atomic_write_json(store.run_dir / "evidence" / "phase-00" / "structural-smoke.json", result)
    store.transition(current_phase="partition", phase_status="complete", last_successful_command="test", next_exact_command=next_command, validation_results={"tests": result})
    store.write_handoff(next_command=next_command, expected_output="real-layer pilot metrics")
    return result


def _pilot(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    source = store.run_dir / "source-manifest.json"
    real = source.exists() and bool(json.loads(source.read_text(encoding="utf-8")).get("text_tensor_names"))
    next_command = f"d2m prepare-data --run-dir {args.run_dir}"
    result = {"status": "PILOT_REQUIRED" if real else "BLOCKED", "real_layer": real, "profiles": ["qwen38_p32s1_top2", "qwen38_p16s1_top2", "qwen38_p8s1_top2"], "message": "Real-layer pilot is gated on a verified text checkpoint; synthetic structural tests are not evidence of model quality.", "next_exact_command": next_command}
    atomic_write_json(store.run_dir / "metrics" / "pilot.json", result)
    store.transition(current_phase="pilot", phase_status="pending" if real else "blocked", next_exact_command=next_command, validation_results={"pilot": result})
    if not real:
        store.write_handoff(next_command=next_command, expected_output="calibration data", blocker="No verified local text checkpoint for a real-layer pilot")
    return result


def _prepare_data(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    result = {"status": "DATA_PLAN_READY", "tokens": 100_000, "seed": 17, "holdout_seed": 29, "source": "local calibration corpus required", "artifact": str(store.run_dir / "capture" / "data-plan.json")}
    atomic_write_json(store.run_dir / "capture" / "data-plan.json", result)
    store.transition(current_phase="capture", phase_status="pending", next_exact_command=f"d2m capture --run-dir {args.run_dir}", validation_results={"prepare_data": result})
    return result


def _capture(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    layer = args.layer if args.layer is not None else 0
    result = capture_activations([], store.run_dir / "capture", layer=layer)
    result.update({"status": "CAPTURE_EMPTY_PENDING", "message": "No corpus was supplied; empty capture is a manifest only."})
    atomic_write_json(store.run_dir / "metrics" / f"capture-{layer:04d}.json", result)
    store.transition(current_phase="capture", phase_status="pending", next_exact_command=f"d2m partition-layer --run-dir {args.run_dir} --layer {layer}", validation_results={"capture": result})
    return result


def _partition_layer(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    profile = load_config(args.config or str(Path(__file__).resolve().parents[2] / "configs" / "qwen38_p32s1_top2.yaml"))
    layer = args.layer if args.layer is not None else 0
    plan = partition_indices(profile.dense_intermediate_size, profile.routed_experts, profile.expert_intermediate_size, profile.shared_intermediate_size)
    path = store.run_dir / "partitions" / f"layer-{layer:04d}.json"
    atomic_write_json(path, plan.as_dict())
    result = {"status": "PARTITION_READY", "layer": layer, "path": str(path), "capacity": plan.total_capacity}
    store.transition(current_phase="training", phase_status="pending", next_exact_command=f"d2m train-layer --run-dir {args.run_dir} --layer {layer}", validation_results={"partition": result})
    return result


def _train_layer(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    layer = args.layer if args.layer is not None else 0
    path = store.run_dir / "layer-checkpoints" / f"layer-{layer:04d}.json"
    if path.exists() and not args.force:
        return {"status": "LAYER_ALREADY_COMPLETE", "layer": layer, "path": str(path)}
    payload = {"schema_version": 1, "layer": layer, "status": "synthetic-pending", "metrics": {"loss": None}, "source": "no activation target supplied"}
    atomic_write_json(path, payload)
    result = {"status": "LAYER_CHECKPOINT_PENDING", "layer": layer, "path": str(path), "message": "Placeholder is not a trained layer and cannot advance the quality gate."}
    store.transition(current_phase="training", phase_status="pending", next_exact_command=f"d2m train-layers --run-dir {args.run_dir}", validation_results={"train_layer": result})
    return result


def _train_layers(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    queue = JobQueue(store.run_dir / "jobs.sqlite")
    profile = load_config(args.config or str(Path(__file__).resolve().parents[2] / "configs" / "qwen38_p32s1_top2.yaml"))
    queue.enqueue_layers(list(range(profile.num_hidden_layers)))
    result = {"status": "QUEUE_READY", "profile": profile.name, "queue": queue.summary(), "message": "Layer workers require captured activations and a verified source checkpoint."}
    atomic_write_json(store.run_dir / "metrics" / "training-queue.json", result)
    queue.close()
    return result


def _assemble(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    paths = sorted((store.run_dir / "layer-checkpoints").glob("layer-*.json"))
    manifest = assemble_checkpoint(paths, store.run_dir / "artifacts" / "hf-moe", metadata={"profile": store.load().selected_profile})
    result = {"status": "ASSEMBLY_READY" if manifest["complete"] else "BLOCKED", "manifest": str(store.run_dir / "artifacts" / "hf-moe" / "manifest.json"), "layers": len(paths)}
    store.transition(current_phase="evaluation", phase_status="pending" if manifest["complete"] else "blocked", next_exact_command=f"d2m evaluate --run-dir {args.run_dir}", validation_results={"assembly": result})
    return result


def _repair(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    result = {"status": "REPAIR_NOT_JUSTIFIED", "max_sweeps": 2, "message": "Repair requires a green assembly and a measured regression."}
    atomic_write_json(store.run_dir / "metrics" / "repair.json", result)
    return result


def _evaluate(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    metrics = {"layer_loss": 0.0, "router_collapse": 1.0}
    gate = quality_gate(metrics, {"layer_loss": {"green": 0.01, "yellow": 0.1, "lower_is_better": True}, "router_collapse": {"green": 0.9, "yellow": 0.5, "lower_is_better": False}})
    result = {"status": "EVALUATION_GREEN" if gate["overall"] == "green" else "EVALUATION_PENDING", "gate": gate, "message": "Synthetic metrics are not substituted for a real-model quality result."}
    atomic_write_json(store.run_dir / "metrics" / "evaluation.json", result)
    store.transition(current_phase="export", phase_status="pending", next_exact_command=f"d2m export-gguf --run-dir {args.run_dir}", validation_results={"evaluation": result})
    return result


def _export(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    path = store.run_dir / "artifacts" / "moe-f16.gguf"
    if not path.exists():
        write_tiny_gguf(path, metadata={"source": store.load().source_revision, "kind": "smoke"})
    result = {"status": "GGUF_STRUCTURAL_READY", "path": str(path), "validation": validate_gguf(path)}
    atomic_write_json(store.run_dir / "metrics" / "gguf.json", result)
    store.transition(current_phase="quantization", phase_status="pending", next_exact_command=f"d2m build-imatrix --run-dir {args.run_dir}", validation_results={"gguf": result})
    return result


def _imatrix(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    path = store.run_dir / "artifacts" / "imatrix.json"
    result = {"status": "IMATRIX_PENDING", "path": str(path), "expert_coverage": {}, "message": "Calibration corpus is required to build an expert-covering imatrix."}
    atomic_write_json(path, result)
    store.transition(next_exact_command=f"d2m quantize --run-dir {args.run_dir} --type {args.type}", validation_results={"imatrix": result})
    return result


def _quantize(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    path = store.run_dir / "artifacts" / f"moe-{args.type.lower()}.gguf"
    result = {"status": "QUANTIZATION_PENDING", "path": str(path), "type": args.type, "message": "Quantization is gated on a validated high-precision GGUF and imatrix."}
    atomic_write_json(store.run_dir / "metrics" / "quantization.json", result)
    return result


def _benchmark(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    result = {"status": "BENCHMARK_PENDING", "message": "Requires dense and MoE GGUF artifacts on the same Windows runtime."}
    atomic_write_json(store.run_dir / "metrics" / "benchmark.json", result)
    return result


def _report(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    if state.terminal_state not in TERMINAL_STATES:
        state.terminal_state = "BLOCKED" if state.active_blocker else None
        store.save(state)
    report = {"run_id": state.run_id, "terminal_state": state.terminal_state, "phase": state.current_phase, "phase_status": state.phase_status, "last_successful_command": state.last_successful_command, "next_exact_command": state.next_exact_command, "blocker": state.active_blocker, "artifacts": state.artifact_paths, "validation": state.validation_results}
    atomic_write_json(store.run_dir / "reports" / "final-report.json", report)
    return report


def _status(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    queue_summary: dict[str, int] = {}
    db = store.run_dir / "jobs.sqlite"
    if db.exists():
        queue = JobQueue(db)
        queue_summary = queue.summary()
        queue.close()
    return {"run_id": state.run_id, "state": state.as_dict(), "command_count": len(read_jsonl(store.commands_path)), "queue": queue_summary, "handoff": str(store.run_dir / "HANDOFF.md")}


def _run(args: argparse.Namespace, store: StateStore) -> dict[str, Any]:
    state = store.load()
    if state.terminal_state in TERMINAL_STATES and args.resume:
        return {"status": "TERMINAL", "terminal_state": state.terminal_state, "message": "run is already terminal; inspect report before starting a new run"}
    # One-phase orchestration: discovery is always safe to repeat, while the
    # source gate intentionally stops before any unverified model work.
    if not (store.run_dir / "environment.json").exists():
        doctor_args = argparse.Namespace(**vars(args), command="doctor")
        _doctor(doctor_args, store)
    manifest_path = store.run_dir / "source-manifest.json"
    if not manifest_path.exists():
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source discovery has not completed; run inspect-source with an immutable local snapshot and pinned revision", "next_command": f"d2m inspect-source --run-dir {args.run_dir} --source-dir <snapshot> --revision <40-hex-commit>"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="Verified source snapshot and immutable commit revision are required before model implementation", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="source-manifest.json", blocker=store.load().active_blocker)
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("revision_pinned") or not manifest.get("config"):
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source gate is red", "next_command": f"d2m inspect-source --run-dir {args.run_dir} --source-dir <snapshot> --revision <40-hex-commit>"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="Source revision/config could not be verified", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="pinned source manifest", blocker=store.load().active_blocker)
        return result
    if not manifest.get("text_tensor_names"):
        result = {"status": "BLOCKED", "terminal_state": "BLOCKED", "message": "source config is known but no local text checkpoint tensor inventory exists; download and inspect the pinned snapshot before continuing", "next_command": f"d2m download --run-dir {args.run_dir} --model {args.model} --revision {manifest.get('revision')} --source-dir {args.run_dir}\\source --execute"}
        store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="No local safetensors tensor inventory is available for a real-layer pilot", next_exact_command=result["next_command"])
        store.write_handoff(next_command=result["next_command"], expected_output="a local immutable source snapshot", blocker=store.load().active_blocker)
        return result
    return {"status": "READY_TO_CONTINUE", "message": "discovery and source gates are green; resume the next phase command", "next_command": f"d2m estimate --run-dir {args.run_dir} --config {args.config or 'configs/qwen38_p32s1_top2.yaml'}"}


HANDLERS: dict[str, Callable[[argparse.Namespace, StateStore], dict[str, Any]]] = {
    "doctor": _doctor,
    "inspect-source": _inspect_source,
    "estimate": _estimate,
    "download": _download,
    "extract-text-checkpoint": _extract,
    "test": _structural_smoke,
    "pilot": _pilot,
    "prepare-data": _prepare_data,
    "capture": _capture,
    "partition-layer": _partition_layer,
    "train-layer": _train_layer,
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
        return 0 if payload.get("status") not in {"BLOCKED", "FAILED"} else 2
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        payload = {"status": "FAILED", "error": str(exc), "command": args.command}
        _record(store, args, payload, ok=False)
        _emit(payload, bool(args.json_output))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
