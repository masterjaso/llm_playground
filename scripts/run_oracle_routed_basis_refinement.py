"""Compatibility wrapper for the legacy synthetic oracle smoke.

The implementation in this file creates a tiny local SwiGLU fixture.  It is
useful for unit/smoke testing only and is never scientific or production
evidence.  Real Phase 01 work must use
``scripts/run_real_oracle_routed_basis_refinement.py`` with both a validated
METHOD_PROOF_ONLY receipt and a real Qwen layer-0 capture receipt.

The gate is deliberately fail-closed: old corpus receipts that only describe
tokenization are not accepted as a frozen corpus-v2 receipt.  A receipt must
declare a v2 marker, a frozen marker, successful provenance/overlap and
benchmark-denylist checks, and byte-verified manifest/split artifacts.
For the bounded Phase 01 method proof, ``--method-proof-receipt`` accepts the
separate receipt emitted by ``prepare_method_proof_data.py``.  It verifies the
clean FIT-TRAIN manifest and its token/diversity policy without treating that
subset as production-corpus evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _normalise_marker(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _find_named_value(payload: Any, names: set[str]) -> Any | None:
    """Find a named value recursively without accepting arbitrary booleans."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalise_marker(key) in names:
                return value
        for value in payload.values():
            found = _find_named_value(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_named_value(value, names)
            if found is not None:
                return found
    return None


def _check_passed(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("status", "result", "verified", "passed", "ok"):
            if key in value:
                return _check_passed(value[key])
        booleans = [item for item in value.values() if isinstance(item, bool)]
        if booleans:
            return all(booleans)
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == 1:
        return True
    return _normalise_marker(value) in {
        "pass",
        "passed",
        "ok",
        "green",
        "true",
        "verified",
        "success",
        "successful",
    }


def _resolve_receipt_artifact(receipt_path: Path, value: Any) -> Path:
    """Resolve a receipt locator without allowing a missing artifact to pass."""

    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    options = [Path.cwd() / candidate, receipt_path.parent / candidate, receipt_path.parent.parent / candidate]
    for option in options:
        if option.is_file():
            return option
    raise FileNotFoundError(f"receipt artifact is missing: {value}")


def _verify_frozen_artifacts(receipt_path: Path, payload: dict[str, Any]) -> None:
    """Verify the bytes and split IDs named by the receipt before training."""

    manifest = payload.get("manifest")
    if not isinstance(manifest, dict) or not manifest.get("path") or not manifest.get("sha256"):
        raise ValueError("receipt must include a hashed corpus manifest")
    manifest_path = _resolve_receipt_artifact(receipt_path, manifest["path"])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != str(manifest["sha256"]):
        raise ValueError(f"corpus manifest hash mismatch: expected {manifest['sha256']}, got {manifest_hash}")
    manifest_rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("corpus manifest rows must be objects")
                manifest_rows.append(row)
    expected_records_value = manifest.get("records", payload.get("records"))
    if expected_records_value is not None and len(manifest_rows) != int(expected_records_value):
        raise ValueError(f"corpus manifest record count mismatch: expected {expected_records_value}, got {len(manifest_rows)}")
    manifest_ids = [str(row.get("id", "")) for row in manifest_rows]
    if not all(manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("corpus manifest IDs must be present and unique")

    split_identity = payload.get("split_identity")
    if split_identity is None and isinstance(payload.get("artifacts"), dict):
        split_identity = payload["artifacts"].get("corpus-v2.1-splits.json")
    if not isinstance(split_identity, dict) or not split_identity.get("path") or not split_identity.get("sha256"):
        raise ValueError("receipt must include a hashed frozen split manifest")
    split_path = _resolve_receipt_artifact(receipt_path, split_identity["path"])
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != str(split_identity["sha256"]):
        raise ValueError(f"split manifest hash mismatch: expected {split_identity['sha256']}, got {split_hash}")
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("frozen split manifest is not valid JSON") from exc
    split_records = split_payload.get("records") if isinstance(split_payload, dict) else None
    if not isinstance(split_records, dict):
        raise ValueError("frozen split manifest must contain records by role")  # noqa: TRY004 - malformed receipt is a semantic validation error
    role_ids: list[str] = []
    seen: set[str] = set()
    for role, values in split_records.items():
        if not isinstance(values, list):
            raise ValueError(f"split role {role!r} is not a list")  # noqa: TRY004 - malformed receipt is a semantic validation error
        for value in values:
            identifier = str(value)
            if identifier in seen:
                raise ValueError(f"split ID appears in multiple roles: {identifier}")
            seen.add(identifier)
            role_ids.append(identifier)
    if set(role_ids) != set(manifest_ids) or len(role_ids) != len(manifest_ids):
        raise ValueError("frozen split IDs do not exactly cover the corpus manifest")
    expected_id_hash = split_identity.get("record_ids_sha256")
    if expected_id_hash:
        actual_id_hash = hashlib.sha256("\n".join(manifest_ids).encode("utf-8")).hexdigest()
        if actual_id_hash != str(expected_id_hash):
            raise ValueError("corpus record ID hash mismatch")


def require_frozen_corpus_v2(receipt_path: str | Path) -> dict[str, Any]:
    """Load and validate the mandatory frozen corpus-v2 receipt.

    The helper accepts equivalent field spellings so receipts can evolve, but
    it never treats an absent marker as success.  It is kept separate from the
    smoke implementation so callers can test the policy without importing
    PyTorch.
    """

    path = Path(receipt_path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen corpus-v2 receipt is required: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"corpus-v2 receipt is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("corpus-v2 receipt must be a JSON object")

    version = _find_named_value(payload, {"corpus-version", "corpus-v2-version"})
    version_marker = _normalise_marker(version) if version is not None else ""
    if version_marker not in {"v2", "2", "corpus-v2", "corpus-v2.0", "2.0", "v2.1", "corpus-v2.1", "2.1"}:
        raise ValueError("receipt must explicitly declare corpus_version=corpus-v2 or corpus-v2.1")

    frozen = _find_named_value(payload, {"frozen", "corpus-frozen", "splits-frozen"})
    status = _find_named_value(payload, {"status", "corpus-status"})
    frozen_ok = _check_passed(frozen) or _normalise_marker(status) in {
        "corpus-v2-frozen",
        "corpus-v21-frozen",
        "corpus-v2.1-frozen",
        "corpus-frozen",
        "frozen",
        "corpus-verified-frozen",
    }
    if not frozen_ok:
        raise ValueError("receipt must explicitly prove that corpus-v2/v2.1 splits are frozen")

    if version_marker in {"v2.1", "corpus-v2.1", "2.1"}:
        phase_gate = payload.get("phase_gate")
        if not isinstance(phase_gate, dict) or not _check_passed(phase_gate.get("status")):
            raise ValueError("Corpus V2.1 phase gate is not green; acquire required tasks and capture capacity first")
        parent = payload.get("parent_v2")
        overlap_audit = payload.get("overlap_audit")
        benchmark_audit = payload.get("benchmark_exclusion")
        provenance = {"status": "pass" if isinstance(parent, dict) and parent.get("receipt_verified") else "blocked"}
        overlap = overlap_audit.get("status") if isinstance(overlap_audit, dict) else None
        denylist = benchmark_audit.get("status") if isinstance(benchmark_audit, dict) else None

    if version_marker not in {"v2.1", "corpus-v2.1", "2.1"}:
        provenance = _find_named_value(
            payload,
            {
                "provenance-check",
                "provenance-checks",
                "provenance-verified",
                "source-provenance",
                "provenance",
            },
        )
        overlap = _find_named_value(
            payload,
            {
                "overlap-check",
                "overlap-checks",
                "repo-document-overlap-check",
                "repository-document-overlap",
                "repo-document-overlap",
                "repository-document-disjoint",
                "overlap-verified",
            },
        )
        denylist = _find_named_value(
            payload,
            {
                "benchmark-denylist-check",
                "benchmark-denylist-checks",
                "benchmark-denylist-verified",
                "denylist-check",
                "denylist-checks",
                "benchmark-denylist",
            },
        )
    missing = [
        name
        for name, value in (
            ("provenance", provenance),
            ("repo/document overlap", overlap),
            ("benchmark denylist", denylist),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"receipt is missing required successful checks: {', '.join(missing)}")
    failed = [
        name
        for name, value in (
            ("provenance", provenance),
            ("repo/document overlap", overlap),
            ("benchmark denylist", denylist),
        )
        if not _check_passed(value)
    ]
    if failed:
        raise ValueError(f"corpus-v2 receipt checks are not successful: {', '.join(failed)}")
    _verify_frozen_artifacts(path, payload)
    return payload


def _has_method_provenance(row: dict[str, Any]) -> bool:
    return all(
        str(row.get(key, "")).strip().casefold() not in {"", "unknown", "none", "null"}
        for key in ("source_record_id", "source_family", "source_name", "source_revision")
    )


def require_method_proof_receipt(receipt_path: str | Path) -> dict[str, Any]:
    """Load and verify the clean, non-production method-proof receipt."""

    path = Path(receipt_path)
    if not path.is_file():
        raise FileNotFoundError(f"method-proof receipt is required: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"method-proof receipt is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("receipt_type") != "dense2moe-method-proof-data":
        raise ValueError("receipt_type must be dense2moe-method-proof-data")
    recorded_receipt_hash = str(payload.get("receipt_sha256", ""))
    unsigned_receipt = dict(payload)
    unsigned_receipt.pop("receipt_sha256", None)
    if not recorded_receipt_hash or recorded_receipt_hash != hashlib.sha256(
        json.dumps(unsigned_receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise ValueError("method-proof receipt hash mismatch")
    if payload.get("status") != "METHOD_PROOF_READY":
        raise ValueError("method-proof data receipt is not ready")
    policy = payload.get("method_proof_policy")
    if not isinstance(policy, dict) or policy.get("eligible_split") != "FIT-TRAIN":
        raise ValueError("method-proof receipt must be restricted to FIT-TRAIN")
    source_manifest_value = policy.get("source_manifest")
    source_manifest_hash = str(policy.get("source_manifest_sha256", ""))
    if not source_manifest_value or not source_manifest_hash:
        raise ValueError("method-proof receipt must include a hashed source manifest")
    source_manifest_path = _resolve_receipt_artifact(path, source_manifest_value)
    if hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() != source_manifest_hash:
        raise ValueError("method-proof source manifest hash mismatch")
    diversity_buckets = policy.get("diversity_buckets")
    if not isinstance(diversity_buckets, list) or not {str(value) for value in diversity_buckets} >= {"code", "technical"}:
        raise ValueError("method-proof receipt must require code/technical diversity")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict) or not manifest.get("path") or not manifest.get("sha256"):
        raise ValueError("method-proof receipt must include a hashed manifest")
    manifest_path = _resolve_receipt_artifact(path, manifest["path"])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != str(manifest["sha256"]):
        raise ValueError("method-proof manifest hash mismatch")
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("method-proof manifest rows must be objects")
                rows.append(row)
    if not rows:
        raise ValueError("method-proof manifest is empty")
    if any(str(row.get("split", "")) != "FIT-TRAIN" for row in rows):
        raise ValueError("method-proof manifest contains a non-FIT-TRAIN row")
    if any(
        bool(row.get("benchmark_quarantine"))
        or row.get("benchmark_quarantine_reason")
        or row.get("benchmark_membership")
        or row.get("benchmark_denylist")
        or row.get("benchmark_context")
        or row.get("benchmark_original_split")
        for row in rows
    ):
        raise ValueError("method-proof manifest contains benchmark-derived material")
    if any(not _has_method_provenance(row) for row in rows):
        raise ValueError("method-proof manifest contains a row without retained provenance")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("method-proof manifest IDs must be present and unique")
    selected_tokens = sum(int(row.get("token_count", 0) or 0) for row in rows)
    minimum_tokens = int(policy.get("minimum_tokens", 0) or 0)
    if selected_tokens < minimum_tokens or minimum_tokens < 32_768:
        raise ValueError("method-proof token threshold is below the required 32768-token gate")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or int(selection.get("selected_tokens", -1)) != selected_tokens or int(selection.get("selected_rows", -1)) != len(rows):
        raise ValueError("method-proof receipt token count does not match its manifest")
    bucket_tokens = {"code": 0, "technical": 0}
    for row in rows:
        domain = str(row.get("domain", "")).casefold().replace("_", "-")
        bucket = "code" if domain == "code" or domain.startswith("code/") or domain.endswith("/code") else "technical" if "agentic" in domain or "software-engineering" in domain or domain == "structured" else "other"
        if bucket in bucket_tokens:
            bucket_tokens[bucket] += int(row.get("token_count", 0) or 0)
    bucket_target = max(1, int(minimum_tokens * 0.25))
    if any(bucket_tokens[bucket] < bucket_target for bucket in ("code", "technical")):
        raise ValueError("method-proof manifest does not satisfy code/technical diversity")
    return payload


def run_synthetic_smoke(
    receipt_path: str | Path | None = None,
    *,
    topology: str = "p16/top4",
    seed: int = 20260816,
    rows: int = 2,
    epochs: int = 1,
    assignment_refresh_steps: int = 1,
    m_step_repeats: int = 1,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
    receipt_kind: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the tiny random SwiGLU smoke and label it as non-promotable.

    ``receipt_path`` remains an optional compatibility hook for older callers
    that wanted to exercise the corpus/method-proof receipt validators.  The
    receipt, when supplied, does not change the evidence class of this run.
    """

    if receipt_path is not None and receipt_kind == "method-proof":
        receipt = require_method_proof_receipt(receipt_path)
    elif receipt_path is not None and receipt_kind == "corpus":
        receipt = require_frozen_corpus_v2(receipt_path)
    elif receipt_path is not None:
        raise ValueError(f"unknown receipt kind: {receipt_kind}")
    else:
        receipt = {}
        receipt_kind = None
    if topology not in {"p16/top4", "p32/top5"}:
        raise ValueError("topology must be p16/top4 or p32/top5")
    if rows <= 0 or epochs < 0:
        raise ValueError("rows must be positive and epochs non-negative")

    import numpy as np
    import torch

    from dense2moe.models.torch_moe import TorchQwen35SwiGLUMoE
    from dense2moe.partition import partition_indices
    from dense2moe.training.oracle_refinement import train_oracle_routed_basis

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    experts, top_k = (16, 4) if topology == "p16/top4" else (32, 5)
    hidden = 8
    shared_width = 2
    expert_width = 2
    intermediate = shared_width + experts * expert_width
    gate = rng.normal(size=(intermediate, hidden)).astype("float32")
    up = rng.normal(size=(intermediate, hidden)).astype("float32")
    down = rng.normal(size=(hidden, intermediate)).astype("float32")
    gate_tensor = torch.as_tensor(gate, device=device)
    up_tensor = torch.as_tensor(up, device=device)
    down_tensor = torch.as_tensor(down, device=device)
    model = TorchQwen35SwiGLUMoE.from_dense(
        gate,
        up,
        down,
        routed_experts=experts,
        shared_intermediate_size=shared_width,
        top_k=top_k,
        routing_mode="independent_positive",
        partition=partition_indices(intermediate, experts, expert_width, shared_width),
        learnable_scales=True,
    ).to(device)
    inputs = torch.as_tensor(rng.normal(size=(rows, hidden)).astype("float32"), device=device)
    dense_target = torch.nn.functional.silu(inputs @ gate_tensor.T)
    dense_target = (dense_target * (inputs @ up_tensor.T)) @ down_tensor.T
    router_before = {
        name: value.detach().clone() for name, value in model.router.state_dict().items()
    }
    amplitude_before = {
        name: value.detach().clone() for name, value in model.amplitude_router.state_dict().items()
    }
    result = train_oracle_routed_basis(
        model,
        [(inputs, dense_target)],
        epochs=epochs,
        learning_rate=1e-3,
        device=device,
        assignment_refresh_steps=assignment_refresh_steps,
        m_step_repeats=m_step_repeats,
        candidate_pool_size=candidate_pool_size,
        max_combinations=max_combinations,
    )
    router_unchanged = all(
        torch.equal(router_before[name], value) for name, value in model.router.state_dict().items()
    )
    amplitude_unchanged = all(
        torch.equal(amplitude_before[name], value)
        for name, value in model.amplitude_router.state_dict().items()
    )
    if not router_unchanged or not amplitude_unchanged:
        raise AssertionError("oracle-routed smoke changed selector parameters")
    return {
        "status": "ORACLE_ROUTED_BASIS_SYNTHETIC_SMOKE_GREEN",
        "evidence_class": "synthetic-smoke",
        "scientific_promotion_eligible": False,
        "production_promotion_eligible": False,
        "topology": topology,
        "device": device,
        "receipt_kind": receipt_kind,
        "corpus_receipt": str(receipt_path) if receipt_kind == "corpus" else None,
        "method_proof_receipt": str(receipt_path) if receipt_kind == "method-proof" else None,
        "corpus_version": _find_named_value(receipt, {"corpus-version", "corpus-v2-version"}) if receipt_kind == "corpus" else None,
        "method_proof_only": receipt_kind == "method-proof",
        "sample_count": int(rows),
        "row_count": int(rows),
        "token_count": None,
        "terminology": "rows/samples; no token claim",
        "selector_frozen": True,
        "router_unchanged": router_unchanged,
        "amplitude_router_unchanged": amplitude_unchanged,
        "training": result,
    }


# Keep the import-level API used by historical unit tests, but make the
# semantic class impossible to mistake for a real method-proof result.
run_smoke = run_synthetic_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        required=True,
        help="required compatibility flag; this command can never emit scientific evidence",
    )
    receipts = parser.add_mutually_exclusive_group(required=False)
    receipts.add_argument("--corpus-receipt", type=Path)
    receipts.add_argument("--method-proof-receipt", type=Path)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), default="p16/top4")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--assignment-refresh-steps", type=int, default=1)
    parser.add_argument("--m-step-repeats", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.method_proof_receipt or args.corpus_receipt,
                topology=args.topology,
                seed=args.seed,
                rows=args.rows,
                epochs=args.epochs,
                assignment_refresh_steps=args.assignment_refresh_steps,
                m_step_repeats=args.m_step_repeats,
                candidate_pool_size=args.candidate_pool_size,
                max_combinations=args.max_combinations,
                receipt_kind="method-proof" if args.method_proof_receipt else "corpus" if args.corpus_receipt else None,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
