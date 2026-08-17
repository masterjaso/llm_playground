"""Run a corpus-gated, bounded oracle-routed basis refinement pilot.

This driver is intentionally smoke-test friendly.  It creates a tiny local
SwiGLU fixture and exercises the same E/M implementation used by a real basis
pilot; no model download, activation capture, or GPU is required.  A real
pilot must pass an explicit frozen corpus-v2 receipt with ``--corpus-receipt``
before any optimizer step is allowed.

The gate is deliberately fail-closed: old corpus receipts that only describe
tokenization are not accepted as a frozen corpus-v2 receipt.  A receipt must
declare a v2 marker, a frozen marker, successful provenance/overlap and
benchmark-denylist checks, and byte-verified manifest/split artifacts.
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


def run_smoke(
    receipt_path: str | Path,
    *,
    topology: str = "p16/top4",
    seed: int = 20260816,
    rows: int = 2,
    epochs: int = 1,
    assignment_refresh_steps: int = 1,
    m_step_repeats: int = 1,
    candidate_pool_size: int | None = None,
    max_combinations: int = 4096,
) -> dict[str, Any]:
    """Run a tiny CPU-only E/M smoke after the corpus gate passes."""

    receipt = require_frozen_corpus_v2(receipt_path)
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
    ).to("cpu")
    inputs = torch.as_tensor(rng.normal(size=(rows, hidden)).astype("float32"))
    dense_target = torch.nn.functional.silu(inputs @ torch.as_tensor(gate).T)
    dense_target = (dense_target * (inputs @ torch.as_tensor(up).T)) @ torch.as_tensor(down).T
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
        device="cpu",
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
        "status": "ORACLE_ROUTED_BASIS_SMOKE_GREEN",
        "topology": topology,
        "device": "cpu",
        "corpus_receipt": str(receipt_path),
        "corpus_version": _find_named_value(receipt, {"corpus-version", "corpus-v2-version"}),
        "selector_frozen": True,
        "router_unchanged": router_unchanged,
        "amplitude_router_unchanged": amplitude_unchanged,
        "training": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--topology", choices=("p16/top4", "p32/top5"), default="p16/top4")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--assignment-refresh-steps", type=int, default=1)
    parser.add_argument("--m-step-repeats", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.corpus_receipt,
                topology=args.topology,
                seed=args.seed,
                rows=args.rows,
                epochs=args.epochs,
                assignment_refresh_steps=args.assignment_refresh_steps,
                m_step_repeats=args.m_step_repeats,
                candidate_pool_size=args.candidate_pool_size,
                max_combinations=args.max_combinations,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
