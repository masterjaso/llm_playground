from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dense2moe.capture.real_method_proof import (
    QWEN_DENSE_INTERMEDIATE_SIZE,
    QWEN_HIDDEN_SIZE,
    QWEN_NUM_HIDDEN_LAYERS,
    QWEN_SOURCE_MODEL,
    QWEN_SOURCE_MODEL_TYPE,
    QWEN_SOURCE_REVISION,
    REAL_CAPTURE_EVIDENCE_CLASS,
    REAL_CAPTURE_RECEIPT_TYPE,
    REAL_CAPTURE_STATUS,
    canonical_sha256,
    iter_capture_batches,
    validate_capture_receipt,
    write_real_capture_receipt,
)
from dense2moe.hardware import _canonical_sha256
from dense2moe.phase import (
    PHASE_01_BLOCKED_INVALID_CAPTURE,
    PHASE_01_BLOCKED_NO_REAL_CAPTURE,
    Phase01PromotionBlocked,
    phase_01_promotion_state,
    validate_phase_01_promotion,
)
from dense2moe.training.real_method_proof import _method_proof_decision, preflight_real_method_proof, run_real_method_proof


def _write_method_receipt(root: Path) -> Path:
    source = root / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    rows = []
    for identifier, domain in (("code", "code"), ("technical", "structured")):
        rows.append(
            {
                "id": identifier,
                "split": "FIT-TRAIN",
                "text": identifier,
                "token_count": 16_384,
                "domain": domain,
                "source_record_id": identifier,
                "source_family": "fixture",
                "source_name": "fixture",
                "source_revision": QWEN_SOURCE_REVISION,
            }
        )
    manifest = root / "method-proof.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    body = {
        "schema_version": 1,
        "receipt_type": "dense2moe-method-proof-data",
        "status": "METHOD_PROOF_READY",
        "method_proof_policy": {
            "source_manifest": str(source),
            "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "eligible_split": "FIT-TRAIN",
            "excluded_splits": ["FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C"],
            "minimum_tokens": 32_768,
            "diversity_buckets": ["code", "technical"],
        },
        "manifest": {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "selection": {
            "selected_rows": 2,
            "selected_tokens": 32_768,
            "selected_ids": ["code", "technical"],
        },
        "split_overlap_audit": {
            "status": "PASS",
            "checked_against": ["FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C"],
        },
    }
    body["receipt_sha256"] = canonical_sha256(body)
    path = root / "method-proof-receipt.json"
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return path


def _write_lock(root: Path) -> Path:
    unsigned = {
        "schema_version": 1,
        "status": "APPROVED",
        "runtime_fingerprint": {"platform": "Windows", "python_version": "3.12.6"},
    }
    unsigned["lock_sha256"] = _canonical_sha256(unsigned)
    path = root / "windows-runtime-lock.json"
    path.write_text(json.dumps(unsigned, sort_keys=True), encoding="utf-8")
    return path


def _write_capture(root: Path, method: Path, lock: Path, *, rows: int = 2, **changes: object) -> Path:
    from safetensors.numpy import save_file

    snapshot = root / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    config = {
        "model_type": QWEN_SOURCE_MODEL_TYPE,
        "num_hidden_layers": QWEN_NUM_HIDDEN_LAYERS,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "intermediate_size": QWEN_DENSE_INTERMEDIATE_SIZE,
    }
    config_path = snapshot / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    index_path = snapshot / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {}}, sort_keys=True), encoding="utf-8")
    tokenizer_path = snapshot / "tokenizer.json"
    tokenizer_path.write_text("fixture-tokenizer\n", encoding="utf-8")
    shard = root / "capture.safetensors"
    values = np.arange(rows * QWEN_HIDDEN_SIZE, dtype=np.float32).reshape(rows, QWEN_HIDDEN_SIZE) / 1000
    save_file({"ffn_input": values, "dense_ffn_target": values.copy()}, str(shard))
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    method_ids = hashlib.sha256(b"code\ntechnical").hexdigest()
    body: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": REAL_CAPTURE_RECEIPT_TYPE,
        "status": REAL_CAPTURE_STATUS,
        "evidence_class": REAL_CAPTURE_EVIDENCE_CLASS,
        "scientific_promotion_eligible": True,
        "production_promotion_eligible": False,
        "source": {
            "model": QWEN_SOURCE_MODEL,
            "revision": QWEN_SOURCE_REVISION,
            "model_type": QWEN_SOURCE_MODEL_TYPE,
            "num_hidden_layers": QWEN_NUM_HIDDEN_LAYERS,
            "hidden_size": QWEN_HIDDEN_SIZE,
            "dense_intermediate_size": QWEN_DENSE_INTERMEDIATE_SIZE,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_index_path": str(index_path),
            "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        },
        "tokenizer": {
            "revision": QWEN_SOURCE_REVISION,
            "hash": canonical_sha256({str(tokenizer_path): hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()}),
            "files": {str(tokenizer_path): hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()},
        },
        "layer": 0,
        "split": "FIT-TRAIN",
        "input_activation": {"name": "X", "tensor": "ffn_input", "shape": [rows, QWEN_HIDDEN_SIZE]},
        "dense_ffn_target": {"name": "Y", "tensor": "dense_ffn_target", "shape": [rows, QWEN_HIDDEN_SIZE]},
        "row_count": rows,
        "token_count": rows,
        "capture_dtype": "float32",
        "capture_shard_format": "safetensors",
        "shards": [{
            "path": str(shard),
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "count": rows,
            "shape": [rows, QWEN_HIDDEN_SIZE],
            "dtype": "float32",
            "format": "safetensors",
            "input_tensor": "ffn_input",
            "target_tensor": "dense_ffn_target",
        }],
        "method_proof_receipt": {
            "path": str(method),
            "sha256": hashlib.sha256(method.read_bytes()).hexdigest(),
            "selected_record_ids_sha256": method_ids,
            "selected_tokens": 32_768,
        },
        "selected_record_ids_sha256": method_ids,
        "selected_tokens": 32_768,
        "excluded_evaluation_identities": {"status": "PASS", "splits": ["FIT-DEV", "GATE-A"]},
        "benchmark_material": False,
        "evaluation_contamination": False,
        "native_windows": True,
        "runtime_lock": {
            "path": str(lock),
            "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "lock_sha256": lock_payload["lock_sha256"],
        },
    }
    body.update(changes)
    path = root / "capture-receipt.json"
    write_real_capture_receipt(body, path)
    return path


@pytest.fixture
def fixture_receipts(tmp_path: Path) -> tuple[Path, Path, Path]:
    method = _write_method_receipt(tmp_path)
    lock = _write_lock(tmp_path)
    capture = _write_capture(tmp_path, method, lock)
    return method, capture, lock


def test_real_capture_receipt_accepts_small_deterministic_fixture(fixture_receipts) -> None:
    method, capture, lock = fixture_receipts
    result = validate_capture_receipt(capture, method_proof_receipt=method, runtime_lock_path=lock, max_tokens=2)
    batches = list(iter_capture_batches(result, max_tokens=2, batch_rows=1))
    assert len(batches) == 2
    assert batches[0][0].shape == (1, QWEN_HIDDEN_SIZE)
    assert batches[0][1].shape == (1, QWEN_HIDDEN_SIZE)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("layer", 1, "CAPTURE_LAYER_NOT_ZERO"),
        ("native_windows", False, "CAPTURE_NATIVE_WINDOWS_PROOF_MISSING"),
        ("benchmark_material", True, "CAPTURE_EVALUATION_CONTAMINATION"),
    ],
)
def test_capture_receipt_rejects_required_negative_fields(fixture_receipts, field, value, message) -> None:
    method, capture, lock = fixture_receipts
    payload = json.loads(capture.read_text(encoding="utf-8"))
    payload[field] = value
    write_real_capture_receipt(payload, capture)
    with pytest.raises(ValueError, match=message):
        validate_capture_receipt(capture, method_proof_receipt=method, runtime_lock_path=lock, max_tokens=2)


def test_phase_01_rejects_synthetic_receipt_even_when_rows_are_large(fixture_receipts) -> None:
    _method, capture, _lock = fixture_receipts
    synthetic = {
        "status": "ORACLE_ROUTED_BASIS_SYNTHETIC_SMOKE_GREEN",
        "evidence_class": "synthetic-smoke",
        "scientific_promotion_eligible": False,
        "production_promotion_eligible": False,
        "topology": "p16/top4",
        "layer": 0,
        "sample_count": 32_768,
    }
    with pytest.raises(Phase01PromotionBlocked):
        validate_phase_01_promotion(synthetic, capture_receipt=capture)
    assert phase_01_promotion_state(None, capture_receipt=capture) == PHASE_01_BLOCKED_NO_REAL_CAPTURE
    assert phase_01_promotion_state(synthetic, capture_receipt=capture) == PHASE_01_BLOCKED_INVALID_CAPTURE


def test_preflight_and_runner_do_zero_steps_on_linux_or_missing_model(fixture_receipts) -> None:
    method, capture, lock = fixture_receipts
    preflight = preflight_real_method_proof(method, capture, max_tokens=2, runtime_lock_path=lock, require_native_windows=False)
    assert preflight["status"] == "PASS"
    result = run_real_method_proof(method, capture, max_tokens=2, runtime_lock_path=lock, device="cpu", require_native_windows=False)
    assert result["status"] == "BLOCKED"
    assert result["optimizer_steps"] == 0


def test_wrong_topology_blocks_before_any_optimizer_step(fixture_receipts) -> None:
    method, capture, lock = fixture_receipts
    result = run_real_method_proof(method, capture, topology="p32/top5", max_tokens=2, runtime_lock_path=lock, require_native_windows=False)
    assert result["status"] == "BLOCKED"
    assert result["optimizer_steps"] == 0


def test_method_proof_decision_distinguishes_yellow_and_failed() -> None:
    initial = {"global_nmse": 0.12, "cosine": 0.93}
    improving = {"global_nmse": 0.04, "cosine": 0.979, "loadCV": 0.7, "dead_experts": 0}
    assert _method_proof_decision(initial, improving)["decision"] == "YELLOW"
    green = {"global_nmse": 0.04, "cosine": 0.985, "loadCV": 0.4, "dead_experts": 0}
    assert _method_proof_decision(initial, green)["decision"] == "GREEN"
    collapsed = {"global_nmse": 0.04, "cosine": 0.985, "loadCV": 0.4, "dead_experts": 1}
    assert _method_proof_decision(initial, collapsed)["decision"] == "FAILED"
