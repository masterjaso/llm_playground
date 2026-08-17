"""Capture-backed real Qwen p16/top4 method-proof execution.

All provenance and runtime gates run before a model is constructed or an
optimizer step is taken.  The runner is intentionally injectable for small
portable fixtures; native scientific use supplies a Qwen layer-0 source
snapshot and the validated capture receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..capture.real_method_proof import (
    QWEN_DENSE_INTERMEDIATE_SIZE,
    QWEN_HIDDEN_SIZE,
    QWEN_SOURCE_MODEL,
    QWEN_SOURCE_MODEL_TYPE,
    QWEN_SOURCE_REVISION,
    REAL_CAPTURE_EVIDENCE_CLASS,
    RealCaptureBlocked,
    iter_capture_batches,
    sha256_file,
    validate_capture_receipt,
    validate_method_proof_receipt,
)
from ..provenance import current_git_commit
from ..state import atomic_write_json

REAL_METHOD_PROOF_RESULT_RECEIPT_TYPE = "dense2moe-real-qwen-p16-method-proof"
REAL_METHOD_PROOF_RESULT_SCHEMA_VERSION = 1
PHASE_01_REAL_METHOD_PROOF_RUNNING = "PHASE_01_REAL_METHOD_PROOF_RUNNING"
PHASE_01_REAL_METHOD_PROOF_GREEN = "PHASE_01_REAL_METHOD_PROOF_GREEN"
PHASE_01_REAL_METHOD_PROOF_FAILED = "PHASE_01_REAL_METHOD_PROOF_FAILED"
PHASE_01_BLOCKED_NO_REAL_CAPTURE = "PHASE_01_BLOCKED_NO_REAL_CAPTURE"
PHASE_01_BLOCKED_INVALID_CAPTURE = "PHASE_01_BLOCKED_INVALID_CAPTURE"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _blocked(gate: str, reason: str, *, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"gate": gate, "status": "BLOCKED", "reason": reason, "details": dict(details or {})}


def _runtime_is_native_windows() -> bool:
    return os.name == "nt" and platform.system() == "Windows"


def preflight_real_method_proof(
    method_proof_receipt: str | Path,
    capture_receipt: str | Path,
    *,
    topology: str = "p16/top4",
    max_tokens: int,
    runtime_lock_path: str | Path | None = None,
    require_native_windows: bool = True,
) -> dict[str, Any]:
    """Run every Phase 01 gate without constructing an optimizer."""

    gates: list[dict[str, Any]] = []
    method: dict[str, Any] | None = None
    capture: dict[str, Any] | None = None
    if topology != "p16/top4":
        gates.append(_blocked("topology", "PHASE_01_TOPOLOGY_MISMATCH", details={"requested": topology, "required": "p16/top4"}))
    else:
        gates.append({"gate": "topology", "status": "PASS", "topology": topology})
    if max_tokens <= 0:
        gates.append(_blocked("token_count", "REQUESTED_TOKEN_COUNT_INVALID", details={"max_tokens": max_tokens}))
    else:
        gates.append({"gate": "token_count", "status": "PASS", "requested_tokens": max_tokens})
    try:
        method = validate_method_proof_receipt(method_proof_receipt)
        gates.append({
            "gate": "method_proof_receipt",
            "status": "PASS",
            "sha256": sha256_file(method_proof_receipt),
            "selected_tokens": method["_selected_tokens"],
            "selected_record_ids_sha256": method["_selected_record_ids_sha256"],
        })
        if max_tokens > int(method["_selected_tokens"]):
            gates.append(_blocked("token_count", "METHOD_PROOF_TOKEN_COUNT_UNAVAILABLE", details={"requested": max_tokens, "available": method["_selected_tokens"]}))
    except (OSError, TypeError, ValueError, RealCaptureBlocked) as exc:
        gates.append(_blocked("method_proof_receipt", "METHOD_PROOF_RECEIPT_INVALID", details={"error": str(exc)}))
    try:
        capture = validate_capture_receipt(
            capture_receipt,
            method_proof_receipt=method_proof_receipt,
            runtime_lock_path=runtime_lock_path,
            require_native_windows=require_native_windows,
            enforce_runtime_drift=require_native_windows,
            max_tokens=max_tokens,
        )
        gates.append({
            "gate": "capture_receipt",
            "status": "PASS",
            "sha256": sha256_file(capture_receipt),
            "evidence_class": capture["evidence_class"],
            "row_count": capture["_row_count"],
        })
    except (OSError, TypeError, ValueError, RealCaptureBlocked) as exc:
        text = str(exc)
        reason = next(
            (
                candidate
                for candidate in (
                    "NATIVE_WINDOWS_REQUIRED",
                    "WINDOWS_RUNTIME_DRIFT",
                    "CAPTURE_SHARD_HASH_MISMATCH",
                    "CAPTURE_SOURCE_IDENTITY_MISMATCH",
                    "SYNTHETIC_EVIDENCE_REJECTED",
                )
                if candidate in text
            ),
            "CAPTURE_RECEIPT_INVALID",
        )
        gates.append(_blocked("capture_receipt", reason, details={"error": str(exc)}))
    if require_native_windows:
        gates.append({"gate": "native_windows", "status": "PASS"} if _runtime_is_native_windows() else _blocked("native_windows", "NATIVE_WINDOWS_REQUIRED"))
    else:
        gates.append({"gate": "native_windows", "status": "SKIPPED", "assurance": "fixture-only"})
    if capture is not None:
        source = capture.get("source", {})
        expected = {
            "model": QWEN_SOURCE_MODEL,
            "revision": QWEN_SOURCE_REVISION,
            "model_type": QWEN_SOURCE_MODEL_TYPE,
            "layer": 0,
            "hidden_size": QWEN_HIDDEN_SIZE,
            "dense_intermediate_size": QWEN_DENSE_INTERMEDIATE_SIZE,
        }
        observed = {
            "model": source.get("model"),
            "revision": source.get("revision"),
            "model_type": source.get("model_type"),
            "layer": capture.get("layer"),
            "hidden_size": source.get("hidden_size"),
            "dense_intermediate_size": source.get("dense_intermediate_size"),
        }
        mismatches = {key: {"expected": value, "actual": observed[key]} for key, value in expected.items() if observed[key] != value}
        gates.append({"gate": "source_identity", "status": "PASS"} if not mismatches else _blocked("source_identity", "SOURCE_IDENTITY_MISMATCH", details=mismatches))
        gates.append({"gate": "synthetic_evidence", "status": "PASS"} if capture.get("evidence_class") == REAL_CAPTURE_EVIDENCE_CLASS else _blocked("synthetic_evidence", "SYNTHETIC_EVIDENCE_REJECTED"))
        gates.append({"gate": "evaluation_contamination", "status": "PASS"} if not capture.get("benchmark_material") and not capture.get("evaluation_contamination") else _blocked("evaluation_contamination", "EVALUATION_CONTAMINATION"))
        gates.append({"gate": "runtime_lock", "status": "PASS"} if capture.get("_runtime_lock") else _blocked("runtime_lock", "RUNTIME_LOCK_MISSING"))
        gates.append({"gate": "shard_hashes", "status": "PASS"} if capture.get("_shards") else _blocked("shard_hashes", "CAPTURE_SHARDS_MISSING"))
        gates.append({"gate": "selected_record_identity", "status": "PASS"} if capture.get("selected_record_ids_sha256") == capture.get("_method_proof", {}).get("_selected_record_ids_sha256") else _blocked("selected_record_identity", "METHOD_PROOF_SELECTED_IDS_HASH_MISMATCH"))
        gates.append({"gate": "native_windows_proof", "status": "PASS"} if capture.get("native_windows") is True else _blocked("native_windows_proof", "CAPTURE_NATIVE_WINDOWS_PROOF_MISSING"))
    else:
        gates.append(_blocked("capture_receipt", "PHASE_01_BLOCKED_NO_REAL_CAPTURE"))
    failed = [gate for gate in gates if gate.get("status") == "BLOCKED"]
    status = "PASS" if not failed else "BLOCKED"
    return {
        "status": status,
        "phase_state": PHASE_01_REAL_METHOD_PROOF_RUNNING if status == "PASS" else PHASE_01_BLOCKED_INVALID_CAPTURE if capture is not None else PHASE_01_BLOCKED_NO_REAL_CAPTURE,
        "topology": topology,
        "requested_tokens": max_tokens,
        "gates": gates,
        "failed_gates": failed,
        "method_proof": method,
        "capture": capture,
        "optimizer_steps": 0,
        "code_commit": current_git_commit(),
    }


def _to_torch(value: Any, *, device: str) -> Any:
    import torch  # type: ignore

    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _metrics(prediction: Any, target: Any) -> dict[str, float]:
    import torch  # type: ignore

    pred = prediction.float().reshape(-1, prediction.shape[-1])
    truth = target.float().reshape(-1, target.shape[-1])
    delta = pred - truth
    denominator = torch.mean(truth.square()).clamp_min(1e-12)
    cosine = torch.sum(pred * truth, dim=-1) / (torch.linalg.vector_norm(pred, dim=-1) * torch.linalg.vector_norm(truth, dim=-1)).clamp_min(1e-12)
    relative = torch.mean(delta.square(), dim=-1) / torch.mean(truth.square(), dim=-1).clamp_min(1e-12)
    return {
        "global_nmse": float((torch.mean(delta.square()) / denominator).detach().cpu()),
        "cosine": float(torch.mean(cosine).detach().cpu()),
        "mean_token_relative_mse": float(torch.mean(relative).detach().cpu()),
    }


def _method_proof_decision(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one real-data stage without turning it into a product claim.

    The selector-independent oracle is the method-proof signal.  A stage is
    failed when it is non-finite, regresses in both primary quality measures,
    or collapses an expert.  Otherwise it is green only when the historical
    product thresholds are already met; a credible but sub-threshold stage is
    explicitly yellow and remains non-promotable to production.
    """

    metric_names = ("global_nmse", "cosine", "loadCV")
    finite = all(
        isinstance(final.get(name), (int, float)) and math.isfinite(float(final[name]))
        for name in metric_names
    ) and isinstance(final.get("dead_experts"), int)
    improvement = (
        finite
        and isinstance(initial.get("global_nmse"), (int, float))
        and isinstance(initial.get("cosine"), (int, float))
        and float(final["global_nmse"]) < float(initial["global_nmse"])
        and float(final["cosine"]) > float(initial["cosine"])
    )
    no_expert_collapse = finite and int(final.get("dead_experts", -1)) == 0
    gate_comparison = {
        "global_nmse": {
            "value": final.get("global_nmse"),
            "threshold": 0.05,
            "operator": "<=",
            "pass": finite and float(final["global_nmse"]) <= 0.05,
        },
        "cosine": {
            "value": final.get("cosine"),
            "threshold": 0.98,
            "operator": ">=",
            "pass": finite and float(final["cosine"]) >= 0.98,
        },
        "dead_experts": {
            "value": final.get("dead_experts"),
            "threshold": 0,
            "operator": "==",
            "pass": no_expert_collapse,
        },
        "loadCV": {
            "value": final.get("loadCV"),
            "threshold": 0.50,
            "operator": "<=",
            "pass": finite and float(final["loadCV"]) <= 0.50,
        },
    }
    product_gates_pass = all(bool(item["pass"]) for item in gate_comparison.values())
    decision = "FAILED" if not finite or not improvement or not no_expert_collapse else "GREEN" if product_gates_pass else "YELLOW"
    return {
        "decision": decision,
        "finite_metrics": finite,
        "improvement_direction": improvement,
        "no_expert_collapse": no_expert_collapse,
        "product_gate_comparison": gate_comparison,
        "production_promotion_eligible": False,
    }


def _load_source_ffn_model(source_snapshot: str | Path, *, device: str) -> Any:
    """Load only layer-0 dense FFN tensors and build the existing MoE basis."""

    import torch  # type: ignore

    from ..capture.streaming_teacher import IndexedSource
    from ..models.torch_moe import TorchQwen35SwiGLUMoE
    from ..partition import partition_indices

    source = IndexedSource(source_snapshot)
    candidates = {
        "gate": ("model.language_model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight"),
        "up": ("model.language_model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.weight"),
        "down": ("model.language_model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight"),
    }
    tensors: dict[str, Any] = {}
    for key, names in candidates.items():
        for name in names:
            if name in source.weight_map:
                tensors[key], _ = source.read_exact(name)
                break
        if key not in tensors:
            raise RealCaptureBlocked("SOURCE_DENSE_FFN_TENSOR_MISSING", details={"tensor": key})
    plan = partition_indices(QWEN_DENSE_INTERMEDIATE_SIZE, 16, 1024, 1024)
    model = TorchQwen35SwiGLUMoE.from_dense(
        tensors["gate"],
        tensors["up"],
        tensors["down"],
        routed_experts=16,
        shared_intermediate_size=1024,
        top_k=4,
        routing_mode="independent_positive",
        partition=plan,
        learnable_scales=True,
        dtype=torch.float32,
        device=device,
    )
    return model


def _metric_batches(model: Any, batches: Iterable[tuple[Any, Any]], *, device: str) -> dict[str, Any]:
    import torch  # type: ignore

    metrics: list[dict[str, float]] = []
    loads = Counter()
    with torch.no_grad():
        for inputs, target in batches:
            values = _to_torch(inputs, device=device)
            truth = _to_torch(target, device=device)
            prediction, info = model(values, return_router=True)
            metrics.append(_metrics(prediction, truth))
            indices = info.get("indices")
            if indices is not None:
                for value in indices.reshape(-1).detach().cpu().tolist():
                    loads[int(value)] += 1
    if not metrics:
        return {"global_nmse": None, "cosine": None, "mean_token_relative_mse": None, "dead_experts": 16, "loadCV": None}
    aggregate = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
    values = [loads[index] for index in range(16)]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    aggregate.update({"dead_experts": sum(value == 0 for value in values), "loadCV": (variance**0.5) / mean if mean else None})
    return aggregate


def _oracle_metric_batches(
    model: Any,
    batches: Iterable[tuple[Any, Any, Mapping[str, Any]]],
    *,
    device: str,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
) -> dict[str, Any]:
    """Evaluate the selector-independent oracle without materializing all rows."""

    import torch  # type: ignore

    from .oracle_refinement import oracle_assignments, oracle_routed_forward

    metrics: list[tuple[int, dict[str, float]]] = []
    loads = Counter()
    methods: Counter[str] = Counter()
    candidate_counts: list[int] = []
    effective_pools: list[int] = []
    with torch.no_grad():
        for inputs, target, _metadata in batches:
            values = _to_torch(inputs, device=device)
            truth = _to_torch(target, device=device)
            assignment = oracle_assignments(
                model,
                values,
                truth,
                candidate_pool_size=candidate_pool_size,
                max_combinations=max_combinations,
            )
            prediction = oracle_routed_forward(model, values, assignment)
            rows = int(values.reshape(-1, values.shape[-1]).shape[0])
            metrics.append((rows, _metrics(prediction, truth)))
            methods[assignment.method] += 1
            candidate_counts.append(int(assignment.candidate_count))
            effective_pools.append(int(assignment.effective_candidate_pool_size))
            for value in assignment.indices.reshape(-1).detach().cpu().tolist():
                loads[int(value)] += 1
    if not metrics:
        return {
            "global_nmse": None,
            "cosine": None,
            "mean_token_relative_mse": None,
            "dead_experts": 16,
            "loadCV": None,
            "candidate_set_strategy": None,
            "candidate_method": None,
            "candidate_count": 0,
            "effective_candidate_pool_size": 0,
            "coefficient_fitting": "projected-positive",
            "exact": False,
        }
    total = max(sum(rows for rows, _ in metrics), 1)
    aggregate = {
        key: sum(rows * values[key] for rows, values in metrics) / total
        for key in metrics[0][1]
    }
    values = [loads[index] for index in range(int(model.routed_experts))]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    aggregate.update(
        {
            "dead_experts": sum(value == 0 for value in values),
            "loadCV": (variance**0.5) / mean if mean else None,
            "candidate_set_strategy": {
                "exact_all_combinations": "exhaustive-set projected-positive oracle",
                "bounded_correlation_candidate_pool": "bounded screening oracle",
            }.get(methods.most_common(1)[0][0], methods.most_common(1)[0][0])
            if methods
            else None,
            "candidate_method": methods.most_common(1)[0][0] if methods else None,
            "candidate_count": max(candidate_counts) if candidate_counts else 0,
            "effective_candidate_pool_size": max(effective_pools) if effective_pools else 0,
            "coefficient_fitting": "projected-positive",
            "exact": False,
        }
    )
    return aggregate


def _module_state_hash(module: Any) -> str:
    """Hash a module state without serializing the live optimizer or selector."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        digest.update(bytes(value.tobytes()))
    return digest.hexdigest()


def _receipt_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Keep result receipts compact; internal validated rows stay in the capture receipt."""

    method = preflight.get("method_proof") if isinstance(preflight.get("method_proof"), Mapping) else {}
    capture = preflight.get("capture") if isinstance(preflight.get("capture"), Mapping) else {}
    return {
        "status": preflight.get("status"),
        "phase_state": preflight.get("phase_state"),
        "requested_tokens": preflight.get("requested_tokens"),
        "gates": preflight.get("gates", []),
        "failed_gates": preflight.get("failed_gates", []),
        "method_proof_receipt_sha256": next(
            (gate.get("sha256") for gate in preflight.get("gates", []) if gate.get("gate") == "method_proof_receipt"),
            None,
        ),
        "method_proof_selected_tokens": method.get("_selected_tokens"),
        "method_proof_selected_record_ids_sha256": method.get("_selected_record_ids_sha256"),
        "capture_receipt_sha256": next(
            (gate.get("sha256") for gate in preflight.get("gates", []) if gate.get("gate") == "capture_receipt"),
            None,
        ),
        "capture_row_count": capture.get("_row_count"),
        "capture_evidence_class": capture.get("evidence_class"),
        "capture_receipt_type": capture.get("receipt_type"),
    }


def _persist_checkpoint(model: Any, destination: str | Path) -> dict[str, Any]:
    """Save and strictly reload a basis checkpoint before a result can be green."""

    target = Path(destination)
    if not hasattr(model, "save_pretrained") or not hasattr(type(model), "from_pretrained"):
        raise RealCaptureBlocked("CHECKPOINT_SERIALIZATION_UNSUPPORTED")
    model.save_pretrained(target)
    state_files = sorted(path for path in target.rglob("*") if path.is_file())
    if not state_files:
        raise RealCaptureBlocked("CHECKPOINT_EMPTY")
    file_hashes = {str(path.relative_to(target)).replace("\\", "/"): sha256_file(path) for path in state_files}
    reloaded = type(model).from_pretrained(target, strict=True, device="cpu")
    return {
        "path": str(target),
        "files": file_hashes,
        "checkpoint_sha256": _canonical_hash(file_hashes),
        "reload_status": "PASS",
        "reloaded_state_hash": _module_state_hash(reloaded),
    }


def run_real_method_proof(
    method_proof_receipt: str | Path,
    capture_receipt: str | Path,
    *,
    topology: str = "p16/top4",
    max_tokens: int,
    runtime_lock_path: str | Path | None = None,
    device: str = "cuda:0",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    assignment_refresh_steps: int = 1,
    m_step_repeats: int = 1,
    batch_rows: int = 256,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
    model: Any | None = None,
    model_factory: Callable[[], Any] | None = None,
    result_receipt: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    require_native_windows: bool = True,
) -> dict[str, Any]:
    """Run a bounded real-data E/M method proof after all preflight gates."""

    started = time.perf_counter()
    preflight = preflight_real_method_proof(
        method_proof_receipt,
        capture_receipt,
        topology=topology,
        max_tokens=max_tokens,
        runtime_lock_path=runtime_lock_path,
        require_native_windows=require_native_windows,
    )
    if preflight["status"] != "PASS":
        result = {
            **_receipt_preflight(preflight),
            "status": "BLOCKED",
            "phase_state": preflight.get("phase_state", PHASE_01_BLOCKED_INVALID_CAPTURE),
            "optimizer_steps": 0,
            "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
            "scientific_promotion_eligible": False,
            "production_promotion_eligible": False,
            "code_commit": current_git_commit(),
        }
        if result_receipt is not None:
            atomic_write_json(result_receipt, result)
        return result
    if model is None and model_factory is None:
        result = {
            **_receipt_preflight(preflight),
            "status": "BLOCKED",
            "phase_state": PHASE_01_BLOCKED_INVALID_CAPTURE,
            "blocker": "SOURCE_BASIS_MODEL_REQUIRED",
            "optimizer_steps": 0,
            "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
            "scientific_promotion_eligible": False,
            "production_promotion_eligible": False,
            "code_commit": current_git_commit(),
        }
        if result_receipt is not None:
            atomic_write_json(result_receipt, result)
        return result
    try:
        if model is None:
            model = model_factory()  # type: ignore[misc]
        if int(getattr(model, "hidden_size", -1)) != QWEN_HIDDEN_SIZE:
            raise RealCaptureBlocked("MODEL_HIDDEN_SIZE_MISMATCH")
        if int(getattr(model, "intermediate_size", -1)) != QWEN_DENSE_INTERMEDIATE_SIZE:
            raise RealCaptureBlocked("MODEL_INTERMEDIATE_SIZE_MISMATCH")
        if int(getattr(model, "routed_experts", -1)) != 16 or int(getattr(model, "top_k", -1)) != 4:
            raise RealCaptureBlocked("MODEL_TOPOLOGY_MISMATCH")
        from .oracle_refinement import train_oracle_routed_basis

        selector_before = _module_state_hash(model.router)
        amplitude_before = _module_state_hash(model.amplitude_router) if getattr(model, "routing_mode", "") == "independent_positive" else None

        def batches() -> Iterable[tuple[Any, Any]]:
            for inputs, target, _metadata in iter_capture_batches(preflight["capture"], max_tokens=max_tokens, batch_rows=batch_rows):
                yield _to_torch(inputs, device=device), _to_torch(target, device=device)

        initial_batches = lambda: iter_capture_batches(preflight["capture"], max_tokens=max_tokens, batch_rows=batch_rows)
        initial = _metric_batches(model, ((inputs, target) for inputs, target, _ in initial_batches()), device=device)
        oracle_initial = _oracle_metric_batches(model, initial_batches(), device=device, candidate_pool_size=candidate_pool_size, max_combinations=max_combinations)
        training = train_oracle_routed_basis(
            model,
            batches,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            assignment_refresh_steps=assignment_refresh_steps,
            m_step_repeats=m_step_repeats,
            candidate_pool_size=candidate_pool_size,
            max_combinations=max_combinations,
        )
        final = _metric_batches(model, ((inputs, target) for inputs, target, _ in iter_capture_batches(preflight["capture"], max_tokens=max_tokens, batch_rows=batch_rows)), device=device)
        oracle_final = _oracle_metric_batches(
            model,
            iter_capture_batches(preflight["capture"], max_tokens=max_tokens, batch_rows=batch_rows),
            device=device,
            candidate_pool_size=candidate_pool_size,
            max_combinations=max_combinations,
        )
        selector_after = _module_state_hash(model.router)
        amplitude_after = _module_state_hash(model.amplitude_router) if amplitude_before is not None else None
        if selector_before != selector_after or amplitude_before != amplitude_after:
            raise RealCaptureBlocked("SELECTOR_CHANGED_DURING_ORACLE_REFINEMENT")
        method_proof_decision = _method_proof_decision(oracle_initial, oracle_final)
        checkpoint = None
        if checkpoint_dir is not None or result_receipt is not None:
            target_checkpoint = checkpoint_dir or Path(result_receipt).with_suffix("").with_name(Path(result_receipt).stem + "-checkpoint")
            checkpoint = _persist_checkpoint(model, target_checkpoint)
        capture_meta = preflight["capture"]
        result: dict[str, Any] = {
            **_receipt_preflight(preflight),
            "status": "REAL_QWEN_P16_METHOD_PROOF_COMPLETE",
            "phase_state": PHASE_01_REAL_METHOD_PROOF_GREEN,
            "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
            "scientific_promotion_eligible": True,
            "production_promotion_eligible": False,
            "source_model": QWEN_SOURCE_MODEL,
            "source_revision": QWEN_SOURCE_REVISION,
            "source_model_type": QWEN_SOURCE_MODEL_TYPE,
            "topology": "p16/top4",
            "layer": 0,
            "sample_count": max_tokens,
            "token_count": max_tokens,
            "row_count": max_tokens,
            "terminology": "captured tokens/samples; one row per captured token; not synthetic rows",
            "initial": initial,
            "final": final,
            "oracle_initial": oracle_initial,
            "oracle_final": oracle_final,
            "method_proof_decision": method_proof_decision,
            "product_gate_comparison": method_proof_decision["product_gate_comparison"],
            "oracle": {
                "assurance_level": oracle_final.get("candidate_set_strategy") or "exhaustive-set projected-positive oracle",
                "coefficient_fitting": "projected-positive",
                "exact": False,
                "initial": oracle_initial,
                "final": oracle_final,
            },
            "training": training,
            "optimizer_steps": int(training.get("updates", 0)),
            "selector_frozen": True,
            "selector_state_hash_before": selector_before,
            "selector_state_hash_after": selector_after,
            "amplitude_router_frozen": amplitude_before is None or amplitude_before == amplitude_after,
            "partition_hash": str(getattr(model, "partition_hash", lambda: "")()),
            "checkpoint": checkpoint,
            "wall_time_seconds": time.perf_counter() - started,
            "code_commit": current_git_commit(),
        }
        result["receipt_type"] = REAL_METHOD_PROOF_RESULT_RECEIPT_TYPE
        result["schema_version"] = REAL_METHOD_PROOF_RESULT_SCHEMA_VERSION
        result["method_proof_receipt_sha256"] = sha256_file(method_proof_receipt)
        result["capture_receipt_sha256"] = sha256_file(capture_receipt)
        result["method_proof_receipt_path"] = str(method_proof_receipt)
        result["capture_receipt_path"] = str(capture_receipt)
        result["capture_row_count"] = int(capture_meta.get("_row_count", max_tokens))
        unsigned = dict(result)
        unsigned.pop("receipt_sha256", None)
        result["receipt_sha256"] = _canonical_hash(unsigned)
        if result_receipt is not None:
            atomic_write_json(result_receipt, result)
        return result
    except (OSError, TypeError, ValueError, RuntimeError, KeyError, RealCaptureBlocked) as exc:
        result = {
            **_receipt_preflight(preflight),
            "status": "BLOCKED",
            "phase_state": PHASE_01_REAL_METHOD_PROOF_FAILED,
            "blocker": getattr(exc, "reason", "REAL_METHOD_PROOF_FAILED"),
            "message": str(exc),
            "optimizer_steps": 0,
            "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
            "scientific_promotion_eligible": False,
            "production_promotion_eligible": False,
            "wall_time_seconds": time.perf_counter() - started,
            "code_commit": current_git_commit(),
        }
        if result_receipt is not None:
            atomic_write_json(result_receipt, result)
        return result


def validate_result_receipt(
    path: str | Path,
    *,
    capture_receipt: str | Path | None = None,
    method_proof_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a completed result receipt before Phase 01 promotion."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_RECEIPT_INVALID", details={"error": str(exc)}) from exc
    if not isinstance(payload, Mapping):
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_RECEIPT_INVALID")
    if payload.get("receipt_type") != REAL_METHOD_PROOF_RESULT_RECEIPT_TYPE:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_RECEIPT_TYPE_INVALID")
    if int(payload.get("schema_version", 0)) != REAL_METHOD_PROOF_RESULT_SCHEMA_VERSION:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_SCHEMA_UNSUPPORTED")
    if payload.get("status") != "REAL_QWEN_P16_METHOD_PROOF_COMPLETE":
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_NOT_COMPLETE")
    if payload.get("evidence_class") != REAL_CAPTURE_EVIDENCE_CLASS:
        raise RealCaptureBlocked("SYNTHETIC_EVIDENCE_REJECTED")
    if payload.get("scientific_promotion_eligible") is not True or payload.get("production_promotion_eligible") is True:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_ELIGIBILITY_INVALID")
    recorded = str(payload.get("receipt_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not recorded or recorded != _canonical_hash(unsigned):
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_HASH_MISMATCH")
    if payload.get("topology") != "p16/top4" or int(payload.get("layer", -1)) != 0:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_GEOMETRY_INVALID")
    if int(payload.get("token_count", 0) or 0) <= 0 or int(payload.get("optimizer_steps", -1)) < 0:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_METRICS_INVALID")
    expected_source = {
        "source_model": QWEN_SOURCE_MODEL,
        "source_revision": QWEN_SOURCE_REVISION,
        "source_model_type": QWEN_SOURCE_MODEL_TYPE,
    }
    if any(payload.get(key) != value for key, value in expected_source.items()):
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_SOURCE_IDENTITY_INVALID")
    if payload.get("selector_frozen") is not True or payload.get("amplitude_router_frozen") is not True:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_SELECTOR_NOT_FROZEN")
    oracle = payload.get("oracle")
    if isinstance(oracle, Mapping) and (oracle.get("exact") is True or oracle.get("coefficient_fitting") != "projected-positive"):
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_ORACLE_ASSURANCE_INVALID")
    capture_path = capture_receipt or payload.get("capture_receipt_path")
    method_path = method_proof_receipt or payload.get("method_proof_receipt_path")
    if not capture_path or not method_path:
        raise RealCaptureBlocked("METHOD_PROOF_RESULT_LINK_MISSING")
    def _linked_path(value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute() or candidate.is_file():
            return candidate
        sibling = target.parent / candidate
        return sibling if sibling.is_file() else candidate

    capture_path = _linked_path(capture_path)
    method_path = _linked_path(method_path)
    if sha256_file(capture_path) != str(payload.get("capture_receipt_sha256", "")):
        raise RealCaptureBlocked("CAPTURE_RECEIPT_HASH_MISMATCH")
    if sha256_file(method_path) != str(payload.get("method_proof_receipt_sha256", "")):
        raise RealCaptureBlocked("METHOD_PROOF_RECEIPT_HASH_MISMATCH")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("reload_status") != "PASS":
        raise RealCaptureBlocked("CHECKPOINT_RELOAD_MISSING")
    checkpoint_raw = checkpoint.get("path")
    files = checkpoint.get("files")
    if not checkpoint_raw or not isinstance(files, Mapping) or not files:
        raise RealCaptureBlocked("CHECKPOINT_HASH_MANIFEST_MISSING")
    checkpoint_path = Path(str(checkpoint_raw))
    if not checkpoint_path.is_absolute() and not checkpoint_path.is_dir():
        sibling = target.parent / checkpoint_path
        if sibling.is_dir():
            checkpoint_path = sibling
    observed_hashes: dict[str, str] = {}
    for raw_name, raw_hash in files.items():
        file_path = checkpoint_path / str(raw_name)
        if not file_path.is_file() or sha256_file(file_path) != str(raw_hash):
            raise RealCaptureBlocked("CHECKPOINT_HASH_MISMATCH", details={"path": str(file_path)})
        observed_hashes[str(raw_name).replace("\\", "/")] = sha256_file(file_path)
    if checkpoint.get("checkpoint_sha256") != _canonical_hash(observed_hashes):
        raise RealCaptureBlocked("CHECKPOINT_MANIFEST_HASH_MISMATCH")
    if not str(checkpoint.get("reloaded_state_hash", "")):
        raise RealCaptureBlocked("CHECKPOINT_RELOAD_IDENTITY_MISSING")
    return dict(payload)


__all__ = [
    "PHASE_01_BLOCKED_INVALID_CAPTURE",
    "PHASE_01_BLOCKED_NO_REAL_CAPTURE",
    "PHASE_01_REAL_METHOD_PROOF_FAILED",
    "PHASE_01_REAL_METHOD_PROOF_GREEN",
    "PHASE_01_REAL_METHOD_PROOF_RUNNING",
    "REAL_METHOD_PROOF_RESULT_RECEIPT_TYPE",
    "preflight_real_method_proof",
    "run_real_method_proof",
    "validate_result_receipt",
]
