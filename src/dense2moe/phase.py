"""Explicit phase contracts and resumable gate receipts.

The control-plane CLI records command outcomes, while this module describes
the semantic contract those outcomes must satisfy.  It is intentionally
side-effect-light: no model, CUDA, corpus, or network work is performed here.
Contracts are immutable and receipts are atomically persisted so a later
invocation can resume from validated gates without trusting chat history.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ACTIVE_TOPOLOGY_IDS, FORBIDDEN_TOPOLOGY_IDS

PHASE_SCHEMA_VERSION = 1
PHASE_00A_ID = "phase-00a"
PHASE_00B_ID = "phase-00b"
PHASE_00_ID = "phase-00"
PHASE_01_ID = "phase-01"
PHASE_02_ID = "phase-02"
PHASE_03_ID = "phase-03"
PHASE_04_ID = "phase-04"
PHASE_05_ID = "phase-05"
PHASE_06_ID = "phase-06"
PHASE_07_ID = "phase-07"
PHASE_08_ID = "phase-08"
PHASE_09_ID = "phase-09"
PHASE_10_ID = "phase-10"
LEGACY_PHASE_IDS = tuple(f"phase-{index:02d}" for index in range(11))
PHASE_IDS = (PHASE_00A_ID, PHASE_00B_ID, *tuple(f"phase-{index:02d}" for index in range(1, 10)))
PHASE_STATUSES = frozenset({"pending", "running", "blocked", "complete"})
GATE_STATUSES = frozenset({"pending", "running", "passed", "failed", "blocked", "skipped"})
PREDICTION_DEPTHS = frozenset({"none", "compact", "expanded"})
PHASE_01_BLOCKED_NO_REAL_CAPTURE = "PHASE_01_BLOCKED_NO_REAL_CAPTURE"
PHASE_01_BLOCKED_INVALID_CAPTURE = "PHASE_01_BLOCKED_INVALID_CAPTURE"
PHASE_01_REAL_METHOD_PROOF_RUNNING = "PHASE_01_REAL_METHOD_PROOF_RUNNING"
PHASE_01_REAL_METHOD_PROOF_GREEN = "PHASE_01_REAL_METHOD_PROOF_GREEN"
PHASE_01_REAL_METHOD_PROOF_FAILED = "PHASE_01_REAL_METHOD_PROOF_FAILED"
PHASE_01_REQUIRED_EVIDENCE_CLASS = "real-qwen-layer-capture"
PHASE_01_MIN_TOKENS = 32_768
PHASE_01_SCIENTIFIC_STATES = frozenset(
    {
        PHASE_01_BLOCKED_NO_REAL_CAPTURE,
        PHASE_01_BLOCKED_INVALID_CAPTURE,
        PHASE_01_REAL_METHOD_PROOF_RUNNING,
        PHASE_01_REAL_METHOD_PROOF_GREEN,
        PHASE_01_REAL_METHOD_PROOF_FAILED,
    }
)

# Every executable phase handoff is a native PowerShell command. Keeping the
# prefixes centralized prevents a later blueprint from silently reintroducing
# a bare interpreter, console alias, or POSIX environment assignment.
WINDOWS_PYTHON = r"& .\.venv\Scripts\python.exe "
WINDOWS_CLI = WINDOWS_PYTHON + "-m dense2moe.cli"
WINDOWS_POWERSHELL_CUDA = r"& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Test-Windows-Cuda.ps1"
WINDOWS_NSP = "& _nsp"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Phase01PromotionBlocked(ValueError):
    """Raised when Phase 01 evidence is not a real capture-backed result."""


def _load_phase_artifact(value: Mapping[str, Any] | str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Phase01PromotionBlocked(f"{label} receipt is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Phase01PromotionBlocked(f"{label} receipt must be a JSON object")
    return dict(payload)


def validate_phase_01_promotion(
    result_receipt: Mapping[str, Any] | str | os.PathLike[str],
    *,
    capture_receipt: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate the evidence class and receipt linkage required for Phase 01.

    A synthetic result, an old ``ORACLE_ROUTED_BASIS_SMOKE_GREEN`` status, or
    a real-looking result without a validated capture receipt is rejected.
    """

    result_path = Path(result_receipt) if isinstance(result_receipt, (str, os.PathLike)) else None
    result = _load_phase_artifact(result_receipt, label="Phase 01 result")
    evidence_class = str(result.get("evidence_class", ""))
    status = str(result.get("status", ""))
    if evidence_class != PHASE_01_REQUIRED_EVIDENCE_CLASS:
        raise Phase01PromotionBlocked(PHASE_01_BLOCKED_NO_REAL_CAPTURE)
    if status in {"ORACLE_ROUTED_BASIS_SMOKE_GREEN", "ORACLE_ROUTED_BASIS_SYNTHETIC_SMOKE_GREEN"} or "synthetic" in status.casefold():
        raise Phase01PromotionBlocked("SYNTHETIC_EVIDENCE_REJECTED")

    capture_value = capture_receipt or result.get("capture_receipt") or result.get("capture_receipt_path")
    method_value = result.get("method_proof_receipt") or result.get("method_proof_receipt_path")
    if capture_value is None or method_value is None:
        raise Phase01PromotionBlocked(PHASE_01_BLOCKED_NO_REAL_CAPTURE)
    if not isinstance(capture_value, (str, os.PathLike)) or not isinstance(method_value, (str, os.PathLike)):
        raise Phase01PromotionBlocked("PHASE_01_RECEIPT_PATHS_REQUIRED")

    # A result receipt is the final training/checkpoint contract, not merely a
    # JSON status string.  Its canonical hash, linked receipt hashes, and
    # reloadable checkpoint are verified before any phase state can turn green.
    if result_path is None:
        raise Phase01PromotionBlocked("PHASE_01_RESULT_RECEIPT_PATH_REQUIRED")
    try:
        from .training.real_method_proof import validate_result_receipt

        result = validate_result_receipt(
            result_path,
            capture_receipt=Path(capture_value),
            method_proof_receipt=Path(method_value),
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise Phase01PromotionBlocked(str(exc)) from exc

    if result.get("scientific_promotion_eligible") is not True or result.get("production_promotion_eligible") is True:
        raise Phase01PromotionBlocked("PHASE_01_ELIGIBILITY_FLAGS_INVALID")
    if result.get("topology") != "p16/top4" or int(result.get("layer", -1)) != 0:
        raise Phase01PromotionBlocked("PHASE_01_GEOMETRY_OR_TOPOLOGY_INVALID")
    expected_source = {
        "source_model": "Qwen/Qwen3.8-27B",
        "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "source_model_type": "qwen3_5_text",
    }
    for key, expected in expected_source.items():
        if result.get(key) != expected:
            raise Phase01PromotionBlocked("PHASE_01_SOURCE_IDENTITY_INVALID")
    token_count = int(result.get("token_count", result.get("sample_count", 0)) or 0)
    if token_count < PHASE_01_MIN_TOKENS:
        raise Phase01PromotionBlocked("PHASE_01_TOKEN_COUNT_BELOW_MINIMUM")

    capture = _load_phase_artifact(capture_value, label="capture")
    if capture.get("evidence_class") != PHASE_01_REQUIRED_EVIDENCE_CLASS:
        raise Phase01PromotionBlocked("SYNTHETIC_EVIDENCE_REJECTED")
    if capture.get("receipt_type") != "dense2moe-real-qwen-layer-capture":
        raise Phase01PromotionBlocked(PHASE_01_BLOCKED_INVALID_CAPTURE)
    try:
        from .capture.real_method_proof import validate_capture_receipt

        capture = validate_capture_receipt(
            Path(capture_value),
            method_proof_receipt=Path(method_value),
            require_native_windows=True,
            enforce_runtime_drift=True,
            max_tokens=token_count,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise Phase01PromotionBlocked(str(exc)) from exc
    if result.get("capture_receipt_sha256") != hashlib.sha256(Path(capture_value).read_bytes()).hexdigest():
        raise Phase01PromotionBlocked("CAPTURE_RECEIPT_HASH_MISMATCH")
    if result.get("method_proof_receipt_sha256") != hashlib.sha256(Path(method_value).read_bytes()).hexdigest():
        raise Phase01PromotionBlocked("METHOD_PROOF_RECEIPT_HASH_MISMATCH")
    return {
        "status": PHASE_01_REAL_METHOD_PROOF_GREEN,
        "evidence_class": PHASE_01_REQUIRED_EVIDENCE_CLASS,
        "scientific_promotion_eligible": True,
        "production_promotion_eligible": False,
        "result": result,
        "capture": capture,
    }


def phase_01_promotion_state(
    result_receipt: Mapping[str, Any] | str | os.PathLike[str] | None,
    *,
    capture_receipt: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> str:
    """Return an explicit fail-closed Phase 01 state for orchestration."""

    if result_receipt is None or capture_receipt is None:
        return PHASE_01_BLOCKED_NO_REAL_CAPTURE
    try:
        validate_phase_01_promotion(result_receipt, capture_receipt=capture_receipt)
    except Phase01PromotionBlocked:
        return PHASE_01_BLOCKED_INVALID_CAPTURE
    return PHASE_01_REAL_METHOD_PROOF_GREEN


@dataclass(frozen=True)
class GateContract:
    """One observable acceptance gate inside a phase."""

    gate_id: str
    description: str
    validation_commands: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        if not self.gate_id.strip():
            raise ValueError("gate_id must not be empty")
        if not self.description.strip():
            raise ValueError(f"gate {self.gate_id!r} requires a description")
        if any(not command.strip() for command in self.validation_commands):
            raise ValueError(f"gate {self.gate_id!r} contains an empty validation command")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "validation_commands": list(self.validation_commands),
            "expected_artifacts": list(self.expected_artifacts),
        }


@dataclass(frozen=True)
class PhaseContract:
    """Stable semantic destination for a resumable epic phase."""

    phase_id: str
    objective: str
    gates: tuple[GateContract, ...]
    validation_commands: tuple[str, ...]
    prediction_depth: str
    expected_artifacts: tuple[str, ...]
    next_phase: str
    handoff_commands: tuple[str, ...] = ()
    active_topologies: tuple[str, ...] = ACTIVE_TOPOLOGY_IDS
    forbidden_topologies: tuple[str, ...] = tuple(sorted(FORBIDDEN_TOPOLOGY_IDS))
    predecessor_phase: str | None = None

    def __post_init__(self) -> None:
        if not self.phase_id.strip():
            raise ValueError("phase_id must not be empty")
        if not self.objective.strip():
            raise ValueError("phase objective must not be empty")
        if self.prediction_depth not in PREDICTION_DEPTHS:
            raise ValueError(f"prediction_depth must be one of {sorted(PREDICTION_DEPTHS)}")
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("phase gate IDs must be unique")
        if tuple(self.active_topologies) != ACTIVE_TOPOLOGY_IDS:
            raise ValueError("phase contract must retain the two active product topologies")
        if set(self.active_topologies) & set(self.forbidden_topologies):
            raise ValueError("active and forbidden topology sets must be disjoint")
        if self.predecessor_phase is not None and self.predecessor_phase == self.phase_id:
            raise ValueError("phase cannot depend on itself")

    @property
    def gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.gate_id for gate in self.gates)

    @property
    def contract_fingerprint(self) -> str:
        return _canonical_digest(self.as_dict())

    def gate(self, gate_id: str) -> GateContract:
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        raise KeyError(f"unknown gate {gate_id!r} for {self.phase_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_SCHEMA_VERSION,
            "phase_id": self.phase_id,
            "objective": self.objective,
            "gates": [gate.as_dict() for gate in self.gates],
            "validation_commands": list(self.validation_commands),
            "prediction_depth": self.prediction_depth,
            "expected_artifacts": list(self.expected_artifacts),
            "next_phase": self.next_phase,
            "handoff_commands": list(self.handoff_commands),
            "active_topologies": list(self.active_topologies),
            "forbidden_topologies": list(self.forbidden_topologies),
            "predecessor_phase": self.predecessor_phase,
        }


@dataclass
class GateReceipt:
    """Evidence for one gate, safe to update as work resumes."""

    gate_id: str
    status: str = "pending"
    checked_at: str | None = None
    command_results: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    blocker: str | None = None

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUSES:
            raise ValueError(f"unknown gate status: {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateReceipt:
        return cls(
            gate_id=str(payload.get("gate_id", "")),
            status=str(payload.get("status", "pending")),
            checked_at=payload.get("checked_at"),
            command_results=[dict(item) for item in payload.get("command_results", [])],
            evidence_refs=[str(item) for item in payload.get("evidence_refs", [])],
            observations=dict(payload.get("observations", {})),
            artifact_hashes={str(key): str(value) for key, value in dict(payload.get("artifact_hashes", {})).items()},
            blocker=payload.get("blocker"),
        )


@dataclass
class PhaseReceipt:
    """Persisted progress for a phase contract.

    The contract fingerprint is mandatory on every receipt.  A changed phase
    contract therefore cannot silently reuse old gate evidence; the caller
    must create a fresh receipt after revising the plan.
    """

    phase_id: str
    contract_fingerprint: str
    run_id: str | None = None
    status: str = "pending"
    gates: dict[str, GateReceipt] = field(default_factory=dict)
    actual_observations: dict[str, Any] = field(default_factory=dict)
    prediction_result: str | None = None
    residual_risks: list[str] = field(default_factory=list)
    next_phase_eligibility: str = "blocked"
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    code_commit: str | None = None

    @classmethod
    def new(cls, contract: PhaseContract, *, run_id: str | None = None) -> PhaseReceipt:
        return cls(
            phase_id=contract.phase_id,
            contract_fingerprint=contract.contract_fingerprint,
            run_id=run_id,
            gates={gate.gate_id: GateReceipt(gate.gate_id) for gate in contract.gates},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseReceipt:
        raw_gates = payload.get("gates", {})
        if isinstance(raw_gates, list):
            raw_gates = {str(item.get("gate_id", "")): item for item in raw_gates}
        return cls(
            phase_id=str(payload.get("phase_id", "")),
            contract_fingerprint=str(payload.get("contract_fingerprint", "")),
            run_id=payload.get("run_id"),
            status=str(payload.get("status", "pending")),
            gates={str(key): GateReceipt.from_dict(value) for key, value in dict(raw_gates).items()},
            actual_observations=dict(payload.get("actual_observations", {})),
            prediction_result=payload.get("prediction_result"),
            residual_risks=[str(item) for item in payload.get("residual_risks", [])],
            next_phase_eligibility=str(payload.get("next_phase_eligibility", "blocked")),
            artifact_hashes={str(key): str(value) for key, value in dict(payload.get("artifact_hashes", {})).items()},
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
            code_commit=payload.get("code_commit"),
        )

    def validate_for(self, contract: PhaseContract) -> None:
        if self.phase_id != contract.phase_id:
            raise ValueError(f"receipt phase {self.phase_id!r} does not match {contract.phase_id!r}")
        if self.contract_fingerprint != contract.contract_fingerprint:
            raise ValueError("phase contract fingerprint mismatch; stale receipt cannot be resumed")
        if self.status not in PHASE_STATUSES:
            raise ValueError(f"unknown phase status: {self.status!r}")
        unknown = set(self.gates) - set(contract.gate_ids)
        if unknown:
            raise ValueError(f"receipt contains unknown gates: {sorted(unknown)}")
        missing = set(contract.gate_ids) - set(self.gates)
        if missing:
            raise ValueError(f"receipt is missing gates: {sorted(missing)}")

    @property
    def all_required_gates_passed(self) -> bool:
        return all(receipt.status == "passed" for receipt in self.gates.values())

    @property
    def is_complete(self) -> bool:
        # ``record_gate`` computes ``status`` from the contract's required
        # flags. Requiring every optional gate here would make a receipt that
        # correctly completed a contract appear incomplete on reload.
        return self.status == "complete"

    def can_resume(self, contract: PhaseContract, *, artifact_hashes: Mapping[str, str] | None = None) -> bool:
        try:
            self.validate_for(contract)
        except ValueError:
            return False
        if self.status == "complete":
            return False
        if artifact_hashes is not None:
            expected = {str(key): str(value) for key, value in artifact_hashes.items()}
            if any(self.artifact_hashes.get(key) != value for key, value in expected.items()):
                return False
        return True

    def record_gate(
        self,
        contract: PhaseContract,
        gate_id: str,
        status: str,
        *,
        command_results: list[Mapping[str, Any]] | None = None,
        evidence_refs: list[str] | None = None,
        observations: Mapping[str, Any] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        blocker: str | None = None,
    ) -> GateReceipt:
        self.validate_for(contract)
        contract.gate(gate_id)
        receipt = self.gates[gate_id]
        updated = GateReceipt(
            gate_id=gate_id,
            status=status,
            checked_at=_utc_now(),
            command_results=[dict(item) for item in command_results or []],
            evidence_refs=[str(item) for item in evidence_refs or []],
            observations=dict(observations or {}),
            artifact_hashes={str(key): str(value) for key, value in (artifact_hashes or {}).items()},
            blocker=blocker,
        )
        self.gates[gate_id] = updated
        if artifact_hashes:
            self.artifact_hashes.update({str(key): str(value) for key, value in artifact_hashes.items()})
        self.updated_at = _utc_now()
        required_ids = {gate.gate_id for gate in contract.gates if gate.required}
        if all(self.gates[required_id].status == "passed" for required_id in required_ids):
            self.status = "complete"
            self.next_phase_eligibility = "eligible"
        elif any(item.status in {"failed", "blocked"} for item in self.gates.values()):
            self.status = "blocked"
            self.next_phase_eligibility = "blocked"
        elif any(item.status in {"running", "passed", "skipped"} for item in self.gates.values()):
            self.status = "running"
            self.next_phase_eligibility = "blocked"
        else:
            self.status = "pending"
            self.next_phase_eligibility = "blocked"
        return receipt

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_SCHEMA_VERSION,
            "phase_id": self.phase_id,
            "contract_fingerprint": self.contract_fingerprint,
            "run_id": self.run_id,
            "status": self.status,
            "gates": {key: value.as_dict() for key, value in sorted(self.gates.items())},
            "actual_observations": self.actual_observations,
            "prediction_result": self.prediction_result,
            "residual_risks": list(self.residual_risks),
            "next_phase_eligibility": self.next_phase_eligibility,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "code_commit": self.code_commit,
        }


class PhaseReceiptStore:
    """Atomic JSON persistence for one phase's resumable receipt."""

    def __init__(self, path: str | os.PathLike[str], contract: PhaseContract, *, run_id: str | None = None):
        self.path = Path(path)
        self.contract = contract
        self.run_id = run_id

    def load(self) -> PhaseReceipt:
        if not self.path.exists():
            return PhaseReceipt.new(self.contract, run_id=self.run_id)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid phase receipt: {self.path}") from exc
        if not isinstance(payload, Mapping):
            raise TypeError(f"phase receipt must be a JSON object: {self.path}")
        receipt = PhaseReceipt.from_dict(payload)
        receipt.validate_for(self.contract)
        return receipt

    def save(self, receipt: PhaseReceipt) -> Path:
        receipt.validate_for(self.contract)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(receipt.as_dict(), indent=2, sort_keys=True, default=str) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f"{self.path.stem}-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.path

    def record_gate(self, gate_id: str, status: str, **kwargs: Any) -> PhaseReceipt:
        receipt = self.load()
        receipt.record_gate(self.contract, gate_id, status, **kwargs)
        self.save(receipt)
        return receipt


PHASE_00_VALIDATION_COMMANDS = (
    WINDOWS_PYTHON + "-m pytest -q tests/test_data_v2.py tests/test_fetch_public_corpus_v2.py tests/test_real_pipeline.py",
    WINDOWS_PYTHON + "-m pytest -q",
    WINDOWS_PYTHON + "-m ruff check src tests scripts",
    WINDOWS_PYTHON + r"scripts\freeze_corpus_v21.py --source data\public_v2\corpus-v2-source.jsonl --v2-manifest data\public_v2\corpus.jsonl --v2-splits data\public_v2\corpus-v2-splits.json --output data\public_v21 --require-agent-tasks 96 --json",
    WINDOWS_PYTHON + "-m dense2moe.cli doctor --run-dir <phase-00-run-dir> --json",
    WINDOWS_POWERSHELL_CUDA,
    WINDOWS_NSP + " plan-substrate discovery validate --target . --path .nsp/artifacts/runs/<run-id>/planning/repository-fact-ledger.json",
)

PHASE_01_COMMANDS = (
    WINDOWS_PYTHON + r"scripts\capture_real_qwen_layer0.py --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json --source-snapshot <pinned-qwen-source> --run-dir <phase-01-run-dir> --runtime-lock runs\windows-runtime-lock.json --shard-tokens 2048 --resume --json",
    WINDOWS_PYTHON + r"scripts\run_real_oracle_routed_basis_refinement.py --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json --runtime-lock runs\windows-runtime-lock.json --source-snapshot <pinned-qwen-source> --topology p16/top4 --max-tokens 2048 --batch-rows 256 --learning-rate 0.0001 --epochs 1 --device cuda:0 --result-receipt <phase-01-run-dir>\metrics\p16-2k.json --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-2k --json",
    WINDOWS_PYTHON + r"scripts\run_real_oracle_routed_basis_refinement.py --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json --runtime-lock runs\windows-runtime-lock.json --source-snapshot <pinned-qwen-source> --topology p16/top4 --max-tokens 4096 --batch-rows 512 --learning-rate 0.0001 --epochs 1 --device cuda:0 --result-receipt <phase-01-run-dir>\metrics\p16-4k.json --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-4k --json",
    WINDOWS_PYTHON + r"scripts\run_real_oracle_routed_basis_refinement.py --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json --runtime-lock runs\windows-runtime-lock.json --source-snapshot <pinned-qwen-source> --topology p16/top4 --max-tokens 32768 --batch-rows 512 --learning-rate 0.0001 --epochs 1 --device cuda:0 --result-receipt <phase-01-run-dir>\metrics\p16-32k.json --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-32k --json",
)

PHASE_00A_COMMANDS = (
    WINDOWS_PYTHON + "-m pytest -q tests/test_environment_doctor.py tests/test_phase_contract.py",
    WINDOWS_PYTHON + "-m ruff check src tests scripts",
    r"& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Setup-Windows.ps1",
    r"& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Test-Windows-Cuda.ps1",
    WINDOWS_PYTHON + "-m dense2moe.cli doctor --run-dir <phase-00a-run-dir> --runtime-lock runs\\windows-runtime-lock.json --json",
)

PHASE_00B_COMMANDS = (
    WINDOWS_PYTHON + r"scripts\prepare_method_proof_data.py --corpus-manifest data\public_v21\corpus-v2.1.jsonl --output <phase-00b-run-dir>\method-proof --min-tokens 32768 --json",
    WINDOWS_PYTHON + "-m pytest -q tests/test_method_proof_data.py",
)

METHOD_PROOF_COMMANDS = (
    PHASE_01_COMMANDS[1],
    PHASE_01_COMMANDS[2],
    PHASE_01_COMMANDS[3],
)


def phase_00_contract() -> PhaseContract:
    """Return the active Phase 0 contract from the epic ATDD gates."""

    freeze = PHASE_00_VALIDATION_COMMANDS[3]
    doctor = PHASE_00_VALIDATION_COMMANDS[4]
    tests = PHASE_00_VALIDATION_COMMANDS[1:3]
    discovery = PHASE_00_VALIDATION_COMMANDS[6]
    gates = (
        GateContract("corpus-v21-freeze", "Corpus V2.1 is immutable and all required receipts exist.", (freeze,), ("data/public_v21/corpus-v2.1.jsonl", "data/public_v21/corpus-v2.1-receipt.json")),
        GateContract("benchmark-quarantine", "Benchmark-derived records are excluded from optimization and promotion splits.", (freeze,)),
        GateContract("agent-task-diversity", "At least 96 independent non-benchmark agent tasks are present or blocked with evidence.", (freeze,)),
        GateContract("split-overlap-audit", "Repository, task/issue, and document forbidden-overlap audits are zero.", (freeze,)),
        GateContract("tokenizer-audit", "Tokenizer, recount, special-token, and chat-template behavior are audited.", (freeze,)),
        GateContract("balanced-capture-plan", "The balanced activation plan and concentration bounds are recorded.", (freeze,)),
        GateContract("environment-doctor", "All required native-Windows ML probes pass or fail closed with a named blocker.", (doctor,), ("environment.json",)),
        GateContract("relevant-ml-tests", "The complete relevant test collection is green on the recovered environment.", tests),
        GateContract("provenance-reconciliation", "HEAD, source, corpus, split, and checkpoint provenance are reconciled.", (discovery,)),
        GateContract("phase1-handoff", "Exact Phase 1 capture/oracle commands and residual risks are emitted.", (), ("handoff.md",)),
    )
    return PhaseContract(
        phase_id=PHASE_00_ID,
        objective="Create immutable Corpus V2.1 and recover a reproducible native-Windows ML environment.",
        gates=gates,
        validation_commands=PHASE_00_VALIDATION_COMMANDS,
        prediction_depth="expanded",
        expected_artifacts=(
            "repository-fact-ledger.json",
            "data/public_v21/corpus-v2.1.jsonl",
            "data/public_v21/corpus-v2.1-splits.json",
            "data/public_v21/corpus-v2.1-receipt.json",
            "data/public_v21/corpus-v2.1-benchmark-exclusion.json",
            "data/public_v21/corpus-v2.1-tokenizer-audit.json",
            "data/public_v21/corpus-v2.1-activation-plan.json",
            "environment.json",
            "handoff.md",
        ),
        next_phase="phase-01",
        handoff_commands=PHASE_01_COMMANDS,
    )


_LATER_PHASE_BLUEPRINTS: dict[str, dict[str, Any]] = {
    PHASE_01_ID: {
        "objective": "Prove selector-independent oracle-routed basis refinement with a real p16/top4 pilot.",
        "prediction_depth": "expanded",
        "next_phase": PHASE_02_ID,
        "gates": (
            ("phase-00-green", "Phase 0 corpus, environment, and provenance gates are green.", (WINDOWS_CLI + " status --run-dir <phase-01-run-dir> --json",), ("phase-00-receipt.json",)),
            ("balanced-teacher-capture", "Real Qwen layer-0 X/Y capture is hash- and split-closed.", (PHASE_01_COMMANDS[0],), ("capture/real-qwen-layer0-receipt.json",)),
            ("oracle-independence", "E-step assignments are selector-independent and checkpoint-reloadable on real captured Qwen data.", (PHASE_01_COMMANDS[1],), ("metrics/p16-2k.json",)),
            ("method-falsifier", "The real 2k, 4k, and 32k stages satisfy the method-proof falsifier without synthetic evidence.", (PHASE_01_COMMANDS[1], PHASE_01_COMMANDS[2], PHASE_01_COMMANDS[3]), ("metrics/p16-32k.json",)),
            ("phase-01-handoff", "The selected p16 recipe and unresolved risks are immutable and resumable.", (), ("handoff.md",)),
        ),
        "expected_artifacts": ("capture/data-plan-receipt.json", "oracle/assignment-receipt.json", "metrics/method-proof.json", "handoff.md"),
    },
    PHASE_02_ID: {
        "objective": "Lock a robust-green p16/top4 layer-0 recipe and selector.",
        "prediction_depth": "compact",
        "next_phase": PHASE_03_ID,
        "gates": (
            ("phase-01-green", "The p16 method-proof falsifier is green.", (WINDOWS_CLI + " status --run-dir <phase-02-run-dir> --json",), ("phase-01-receipt.json",)),
            ("oracle-quality", "The real capture-backed p16 method-proof receipt reports the historical oracle quality gates before any production work.", (WINDOWS_PYTHON + r"scripts\run_real_oracle_routed_basis_refinement.py --method-proof-receipt <phase-01-run-dir>\method-proof\receipt.json --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json --runtime-lock runs\windows-runtime-lock.json --source-snapshot <pinned-qwen-source> --topology p16/top4 --max-tokens 32768 --result-receipt <phase-01-run-dir>\metrics\p16-32k.json --json",), ("metrics/oracle-quality.json",)),
            ("load-quality", "p16 load CV is at most 0.50 with zero dead experts on meaningful counts.", (WINDOWS_CLI + " validate-layer --run-dir <phase-02-run-dir> --layer 0 --profile qwen38_p16s1_top4 --json",), ("metrics/load-quality.json",)),
            ("selector-quality", "The selector is trained only after basis freeze and remains change-sensitive on held-out labels.", (WINDOWS_CLI + " evaluate --run-dir <phase-02-run-dir> --json",), ("metrics/selector-quality.json",)),
            ("p16-robust-green", "Independent A/B/C evidence and preservation canary support promotion.", (WINDOWS_CLI + " report --run-dir <phase-02-run-dir> --json",), ("metrics/robust-green.json",)),
        ),
        "expected_artifacts": ("metrics/oracle-quality.json", "metrics/load-quality.json", "metrics/selector-quality.json", "metrics/robust-green.json"),
    },
    PHASE_03_ID: {
        "objective": "Transfer the locked p16 method to the primary p32/top5 product and assess candidate-pool adequacy.",
        "prediction_depth": "compact",
        "next_phase": PHASE_04_ID,
        "gates": (
            ("p16-lock-input", "A robust-green p16 recipe is frozen as the transfer input.", (WINDOWS_CLI + " status --run-dir <phase-03-run-dir> --json",), ("p16-method-lock.json",)),
            ("structured-initialization", "p16 routed experts are split into p32 512-wide experts with contribution evidence.", (WINDOWS_PYTHON + r"scripts\materialize_p32_product_targets.py --help",), ("p32/initialization-receipt.json",)),
            ("candidate-pool-adequacy", "The bounded p32 oracle candidate pool is measured and expanded when coverage is insufficient.", (WINDOWS_PYTHON + r"scripts\run_oracle_routed_basis_refinement.py --corpus-receipt data\public_v21\corpus-v2.1-receipt.json --topology p32/top5 --rows 2048 --epochs 1 --device cuda:0",), ("p32/candidate-pool-receipt.json",)),
            ("p32-quality", "p32/top5 quality, load, and selector evidence are reported against the same gates.", (WINDOWS_CLI + " validate-layer --run-dir <phase-03-run-dir> --layer 0 --profile qwen38_p32s1_top5 --json",), ("metrics/p32-quality.json",)),
            ("p32-decision", "The p32 product decision is recorded without blocking the p16 completion path.", (), ("p32/decision.json",)),
        ),
        "expected_artifacts": ("p32/initialization-receipt.json", "p32/candidate-pool-receipt.json", "metrics/p32-quality.json", "p32/decision.json"),
    },
    PHASE_04_ID: {
        "objective": "Publish an immutable training-method lock for the proven p16 recipe and understood p32 transfer.",
        "prediction_depth": "compact",
        "next_phase": PHASE_05_ID,
        "gates": (
            ("promotion-inputs", "p16 robust-green and p32 decision receipts are present.", (WINDOWS_CLI + " status --run-dir <phase-04-run-dir> --json",), ("promotion-inputs.json",)),
            ("method-specification", "Corpus, split, sampler, partition, solver, optimizer, schedule, selector, and validation cadence are specified.", (), ("TRAINING_METHOD_LOCK.json",)),
            ("method-hash", "The method lock is content-addressed and reloadable.", (WINDOWS_PYTHON + "-m json.tool <phase-04-run-dir>\\TRAINING_METHOD_LOCK.json",), ("TRAINING_METHOD_LOCK.sha256",)),
            ("transfer-boundary", "Representative phases consume the lock without arbitrary per-layer architecture search.", (), ("transfer-boundary.json",)),
        ),
        "expected_artifacts": ("TRAINING_METHOD_LOCK.json", "TRAINING_METHOD_LOCK.sha256", "transfer-boundary.json"),
    },
    PHASE_05_ID: {
        "objective": "Validate transfer across the 12-layer representative matrix and all attention-cycle classes.",
        "prediction_depth": "expanded",
        "next_phase": PHASE_06_ID,
        "gates": (
            ("method-lock-input", "The immutable training method lock is green.", (WINDOWS_CLI + " status --run-dir <phase-05-run-dir> --json",), ("TRAINING_METHOD_LOCK.json",)),
            ("representative-layer-set", "Exactly layers 0–3, 28–31, and 60–63 are captured and evaluated.", (), ("representative/layers.json",)),
            ("cycle-coverage", "LINEAR_A/B/C and FULL_ATTENTION cycle classes are represented.", (), ("representative/cycle-coverage.json",)),
            ("transfer-quality", "The locked method meets layer quality/load gates across early, middle, and late layers.", (WINDOWS_CLI + " report --run-dir <phase-05-run-dir> --json",), ("representative/quality.json",)),
            ("representative-decision", "The p16 method transfers or a bounded corrective PIV is recorded.", (), ("representative/decision.json",)),
        ),
        "expected_artifacts": ("representative/layers.json", "representative/cycle-coverage.json", "representative/quality.json", "representative/decision.json"),
    },
    PHASE_06_ID: {
        "objective": "Convert all 64 layers with guarded, resumable p16 execution and selective escalation.",
        "prediction_depth": "expanded",
        "next_phase": PHASE_07_ID,
        "gates": (
            ("representative-green", "The representative matrix authorizes full64 p16 conversion.", (WINDOWS_CLI + " status --run-dir <phase-06-run-dir> --json",), ("representative/decision.json",)),
            ("layer-queue", "All 64 layers have deterministic queue receipts and bounded budgets.", (), ("full64/layer-queue.json",)),
            ("layer-checkpoints", "Every layer has a validated checkpoint, hash, source pin, and dataset fingerprint.", (WINDOWS_CLI + " assemble --run-dir <phase-06-run-dir> --profile qwen38_p16s1_top4 --strict --json",), ("full64/checkpoints-manifest.json",)),
            ("full64-quality", "All layer-level quality gates pass or have an explicit bounded blocker; no silent substitution occurs.", (), ("full64/quality-report.json",)),
        ),
        "expected_artifacts": ("full64/layer-queue.json", "full64/checkpoints-manifest.json", "full64/quality-report.json"),
    },
    PHASE_07_ID: {
        "objective": "Assemble and reload a canonical BF16 sparse master while preserving the Qwen backbone.",
        "prediction_depth": "expanded",
        "next_phase": PHASE_08_ID,
        "gates": (
            ("full64-input", "All 64 p16 layer checkpoints are complete and hash-consistent.", (WINDOWS_CLI + " status --run-dir <phase-07-run-dir> --json",), ("full64/checkpoints-manifest.json",)),
            ("tensor-inventory", "All intended FFNs are replaced and non-FFN tensor inventory is preserved.", (WINDOWS_CLI + " assemble --run-dir <phase-07-run-dir> --profile qwen38_p16s1_top4 --strict --json",), ("BF16_SPARSE_MASTER/manifest.json",)),
            ("backbone-preservation", "Attention, norms, residual, embeddings, LM head, tokenizer, chat template, and generation metadata are verified.", (), ("BF16_SPARSE_MASTER/preservation-receipt.json",)),
            ("bf16-reload", "The assembled BF16 master reloads and passes representative forward checks.", (), ("BF16_SPARSE_MASTER/reload-receipt.json",)),
        ),
        "expected_artifacts": ("BF16_SPARSE_MASTER/manifest.json", "BF16_SPARSE_MASTER/preservation-receipt.json", "BF16_SPARSE_MASTER/reload-receipt.json"),
    },
    PHASE_08_ID: {
        "objective": "Validate the BF16 sparse master against dense distribution, coding-agent behavior, and preservation canaries.",
        "prediction_depth": "compact",
        "next_phase": PHASE_09_ID,
        "gates": (
            ("bf16-master-input", "The canonical BF16 sparse master is frozen.", (WINDOWS_CLI + " status --run-dir <phase-08-run-dir> --json",), ("BF16_SPARSE_MASTER/manifest.json",)),
            ("distribution-quality", "PPL delta, token KL, and top-1 agreement meet the accepted envelope.", (WINDOWS_CLI + " evaluate --run-dir <phase-08-run-dir> --json",), ("validation/distribution.json",)),
            ("coding-agent-quality", "Unseen coding-agent tasks cover generation, debugging, navigation, tools, retries, and long context.", (), ("validation/coding-agent.json",)),
            ("preservation-quality", "General/STEM/OOD canaries show no unexplained catastrophic collapse.", (), ("validation/preservation.json",)),
            ("finalist-holdout-policy", "Official holdout remains closed unless this artifact is the selected finalist.", (), ("validation/holdout-policy.json",)),
        ),
        "expected_artifacts": ("validation/distribution.json", "validation/coding-agent.json", "validation/preservation.json", "validation/holdout-policy.json"),
    },
    PHASE_09_ID: {
        "objective": "Prove sparse serialization, runtime, and quantization-backend compatibility without quantizing research candidates.",
        "prediction_depth": "compact",
        "next_phase": PHASE_10_ID,
        "gates": (
            ("bf16-validation-input", "BF16 whole-model validation is green before quantization work.", (WINDOWS_CLI + " status --run-dir <phase-09-run-dir> --json",), ("validation/distribution.json",)),
            ("runtime-discovery", "A runtime/backend capable of the active sparse topology is identified and exercised structurally.", (WINDOWS_CLI + " export-gguf --run-dir <phase-09-run-dir> --json",), ("quant/runtime-discovery.json",)),
            ("serialization-contract", "Expert layout, router precision, shared branch precision, and metadata survive round-trip.", (), ("quant/serialization-contract.json",)),
            ("imatrix-contract", "Calibration/imatrix generation is receipt-bearing and source/BF16 fingerprints are closed.", (WINDOWS_CLI + " build-imatrix --run-dir <phase-09-run-dir> --json",), ("quant/imatrix-receipt.json",)),
        ),
        "expected_artifacts": ("quant/runtime-discovery.json", "quant/serialization-contract.json", "quant/imatrix-receipt.json"),
    },
    PHASE_10_ID: {
        "objective": "Quantize the frozen BF16 sparse master and validate incremental degradation.",
        "prediction_depth": "expanded",
        "next_phase": "complete",
        "gates": (
            ("bf16-freeze", "Quantization starts only from the exact validated BF16 sparse master.", (WINDOWS_CLI + " status --run-dir <phase-10-run-dir> --json",), ("BF16_SPARSE_MASTER/manifest.json",)),
            ("conservative-quantization", "A conservative Q8-like or equivalent baseline reloads successfully.", (WINDOWS_CLI + " quantize --run-dir <phase-10-run-dir> --json",), ("quant/candidate-q8/receipt.json",)),
            ("incremental-quality", "BF16-to-quantized degradation is measured separately from dense-to-BF16 degradation.", (WINDOWS_CLI + " evaluate --run-dir <phase-10-run-dir> --json",), ("quant/incremental-quality.json",)),
            ("practical-candidate", "At least one quantized candidate remains within the accepted whole-model quality envelope.", (), ("quant/final-candidate.json",)),
            ("closeout", "The full source-to-quantized pipeline is reproducible from a clean checkout with hashes and receipts.", (), ("FINAL_CLOSEOUT.md",)),
        ),
        "expected_artifacts": ("quant/candidate-q8/receipt.json", "quant/incremental-quality.json", "quant/final-candidate.json", "FINAL_CLOSEOUT.md"),
    },
}


def _later_phase_contract(phase_id: str, blueprint: Mapping[str, Any]) -> PhaseContract:
    gates = tuple(
        GateContract(
            gate_id=str(gate_id),
            description=str(description),
            validation_commands=tuple(str(command) for command in commands),
            expected_artifacts=tuple(str(path) for path in artifacts),
        )
        for gate_id, description, commands, artifacts in blueprint["gates"]
    )
    return PhaseContract(
        phase_id=phase_id,
        objective=str(blueprint["objective"]),
        gates=gates,
        validation_commands=tuple(command for gate in gates for command in gate.validation_commands),
        prediction_depth=str(blueprint["prediction_depth"]),
        expected_artifacts=tuple(str(path) for path in blueprint["expected_artifacts"]),
        next_phase=str(blueprint["next_phase"]),
        handoff_commands=(f"{WINDOWS_CLI} status --run-dir <{phase_id}-run-dir> --json",),
        predecessor_phase=f"phase-{int(phase_id[-2:]) - 1:02d}",
    )


LEGACY_PHASE_CONTRACTS: dict[str, PhaseContract] = {
    PHASE_00_ID: phase_00_contract(),
    **{phase_id: _later_phase_contract(phase_id, blueprint) for phase_id, blueprint in _LATER_PHASE_BLUEPRINTS.items()},
}


def phase_00a_contract() -> PhaseContract:
    return PhaseContract(
        phase_id=PHASE_00A_ID,
        objective="Establish and approve the reusable native-Windows runtime capability lock.",
        gates=(
            GateContract("platform-policy", "WSL/Linux is rejected and no fallback execution path is offered.", (PHASE_00A_COMMANDS[0],)),
            GateContract("capability-gate", "The project Windows interpreter proves Torch, CUDA, BF16, GEMM, checkpoint, p16, teacher, oracle, and source probes.", (PHASE_00A_COMMANDS[4],)),
            GateContract("runtime-lock", "A green capability result creates runs\\windows-runtime-lock.json with a content hash and receipt hashes.", (PHASE_00A_COMMANDS[3],), ("runs/windows-runtime-lock.json",)),
            GateContract("runtime-drift", "Changes from the current lock classify as WINDOWS_RUNTIME_DRIFT.", (PHASE_00A_COMMANDS[4],)),
            GateContract("handoff", "The next phase receives the lock path and exact native commands.", (), ("runs/windows-runtime-lock.json",)),
        ),
        validation_commands=PHASE_00A_COMMANDS,
        prediction_depth="compact",
        expected_artifacts=("runs/windows-environment-receipt.json", "runs/windows-runtime-lock.json"),
        next_phase=PHASE_00B_ID,
        handoff_commands=(PHASE_00B_COMMANDS[0],),
        predecessor_phase=None,
    )


def phase_00b_contract() -> PhaseContract:
    return PhaseContract(
        phase_id=PHASE_00B_ID,
        objective="Derive a clean non-benchmark METHOD_PROOF_ONLY dataset from immutable Corpus V2.1.",
        gates=(
            GateContract("runtime-lock-green", "The current Windows runtime lock is approved and reusable.", (PHASE_00A_COMMANDS[4],), ("runs/windows-runtime-lock.json",)),
            GateContract("method-proof-data", "At least 32k usable states/tokens are selected with known provenance and no benchmark rows.", (PHASE_00B_COMMANDS[0],), ("method-proof/receipt.json",)),
            GateContract("method-proof-diversity", "The subset includes code/technical diversity and remains disjoint from evaluation cohorts.", (PHASE_00B_COMMANDS[1],)),
        ),
        validation_commands=PHASE_00B_COMMANDS,
        prediction_depth="compact",
        expected_artifacts=("method-proof/manifest.jsonl", "method-proof/receipt.json"),
        next_phase=PHASE_01_ID,
        handoff_commands=METHOD_PROOF_COMMANDS,
        predecessor_phase=PHASE_00A_ID,
    )


def _variant(
    base: PhaseContract,
    *,
    phase_id: str,
    predecessor_phase: str | None,
    next_phase: str,
    objective: str | None = None,
    gates: tuple[GateContract, ...] | None = None,
    validation_commands: tuple[str, ...] | None = None,
    expected_artifacts: tuple[str, ...] | None = None,
    prediction_depth: str | None = None,
    handoff_commands: tuple[str, ...] | None = None,
) -> PhaseContract:
    selected_gates = gates if gates is not None else base.gates
    selected_commands = validation_commands if validation_commands is not None else tuple(command for gate in selected_gates for command in gate.validation_commands)
    return PhaseContract(
        phase_id=phase_id,
        objective=objective or base.objective,
        gates=selected_gates,
        validation_commands=selected_commands,
        prediction_depth=prediction_depth or base.prediction_depth,
        expected_artifacts=expected_artifacts or base.expected_artifacts,
        next_phase=next_phase,
        handoff_commands=handoff_commands or (WINDOWS_CLI + f" status --run-dir <{phase_id}-run-dir> --json",),
        active_topologies=base.active_topologies,
        forbidden_topologies=base.forbidden_topologies,
        predecessor_phase=predecessor_phase,
    )


def _canonical_phase_contracts() -> dict[str, PhaseContract]:
    legacy = LEGACY_PHASE_CONTRACTS
    phase_01 = _variant(
        legacy[PHASE_01_ID],
        phase_id=PHASE_01_ID,
        predecessor_phase=PHASE_00B_ID,
        next_phase=PHASE_02_ID,
        objective="Prove selector-independent oracle-routed p16/top4 basis refinement at 2k, 4k, and 32k.",
        gates=(
            GateContract("phase-00-green", "The runtime-lock and capability gates are green before method proof begins.", (PHASE_00A_COMMANDS[4],), ("runs/windows-runtime-lock.json",)),
            GateContract("runtime-lock-green", "The approved current Windows runtime lock is present.", (PHASE_00A_COMMANDS[4],), ("runs/windows-runtime-lock.json",)),
            GateContract("method-proof-data-green", "The clean METHOD_PROOF_ONLY receipt is green.", (PHASE_00B_COMMANDS[0],), ("method-proof/receipt.json",)),
            GateContract("real-capture-receipt", "A validated real Qwen layer-0 X/Y capture receipt is present; synthetic smoke is informational only.", (PHASE_01_COMMANDS[0],), ("capture/real-qwen-layer0-receipt.json",)),
            GateContract("oracle-independence", "The exhaustive p16 E-step remains selector-independent and checkpoint-reloadable on real Qwen captures.", (METHOD_PROOF_COMMANDS[0],), ("metrics/p16-2k.json",)),
            GateContract("method-falsifier", "The real 2k, 4k, and 32k stages show a repeatable reconstruction signal or classify the real method proof as failed.", METHOD_PROOF_COMMANDS, ("metrics/p16-32k.json",)),
            GateContract("synthetic-evidence-rejection", "Synthetic-smoke receipts and legacy synthetic status names cannot satisfy Phase 01.", (), ("metrics/synthetic-rejection.json",)),
            GateContract("phase-01-handoff", "The real method-proof verdict and next production command are immutable and resumable.", (), ("handoff.md",)),
        ),
        validation_commands=(*PHASE_00A_COMMANDS[4:5], *PHASE_00B_COMMANDS, *PHASE_01_COMMANDS),
        expected_artifacts=("method-proof/receipt.json", "capture/real-qwen-layer0-receipt.json", "metrics/p16-2k.json", "metrics/p16-4k.json", "metrics/p16-32k.json", "handoff.md"),
        handoff_commands=(WINDOWS_CLI + " status --run-dir <phase-01-run-dir> --json",),
    )
    phase_02 = PhaseContract(
        phase_id=PHASE_02_ID,
        objective="Acquire/freeze production Corpus V2.2 independently and begin production-balanced p16 training.",
        gates=(
            GateContract("runtime-lock-green", "The current Windows runtime lock remains green.", (PHASE_00A_COMMANDS[4],), ("runs/windows-runtime-lock.json",)),
            GateContract("method-proof-green", "Phase 01 reports a real capture-backed METHOD_PROOF_GREEN result; synthetic smoke is never sufficient.", (WINDOWS_CLI + " status --run-dir <phase-02-run-dir> --json",), ("phase-01-real-method-proof-receipt.json",)),
            GateContract("production-corpus-green", "Corpus V2.2 contains the required independent non-benchmark agent-task distribution.", (WINDOWS_PYTHON + r"scripts\freeze_corpus_v21.py --help",), ("corpus-v2.2-receipt.json",)),
            GateContract("production-p16-training", "The bounded 128k production p16 continuation completes with receipt-backed metrics.", (WINDOWS_CLI + " train-layer --run-dir <phase-02-run-dir> --layer 0 --profile qwen38_p16s1_top4 --json",), ("metrics/p16-production.json",)),
        ),
        validation_commands=(PHASE_00A_COMMANDS[4], WINDOWS_CLI + " status --run-dir <phase-02-run-dir> --json", WINDOWS_PYTHON + r"scripts\freeze_corpus_v21.py --help", WINDOWS_CLI + " train-layer --run-dir <phase-02-run-dir> --layer 0 --profile qwen38_p16s1_top4 --json"),
        prediction_depth="compact",
        expected_artifacts=("corpus-v2.2-receipt.json", "metrics/p16-production.json"),
        next_phase=PHASE_03_ID,
        handoff_commands=(WINDOWS_CLI + " status --run-dir <phase-02-run-dir> --json",),
        predecessor_phase=PHASE_01_ID,
    )
    phase_03 = _variant(legacy[PHASE_02_ID], phase_id=PHASE_03_ID, predecessor_phase=PHASE_02_ID, next_phase=PHASE_04_ID, objective="Train the p16 selector and validate joint load/generalization across FIT-DEV, GATE-A, SHADOW-B, and SHADOW-C.")
    phase_04 = _variant(legacy[PHASE_03_ID], phase_id=PHASE_04_ID, predecessor_phase=PHASE_03_ID, next_phase=PHASE_05_ID, objective="Transfer the proven method to the primary p32/top5 product without blocking the p16 fallback.")
    phase_05 = _variant(legacy[PHASE_04_ID], phase_id=PHASE_05_ID, predecessor_phase=PHASE_04_ID, next_phase=PHASE_06_ID)
    phase_06 = _variant(legacy[PHASE_05_ID], phase_id=PHASE_06_ID, predecessor_phase=PHASE_05_ID, next_phase=PHASE_07_ID)
    phase_07 = _variant(
        legacy[PHASE_06_ID],
        phase_id=PHASE_07_ID,
        predecessor_phase=PHASE_06_ID,
        next_phase=PHASE_08_ID,
        gates=legacy[PHASE_06_ID].gates
        + (
            GateContract(
                "full64-input",
                "The guarded full64 p16 queue and layer checkpoints are the sole input to BF16 assembly.",
                (WINDOWS_CLI + " status --run-dir <phase-07-run-dir> --json",),
                ("full64/checkpoints-manifest.json",),
            ),
        ),
    )
    # The canonical graph collapses the historical assembly/validation pair
    # into Phase 08 and the historical runtime/quantization pair into Phase
    # 09.  Preserve every required gate when translating those legacy
    # contracts; otherwise a canonical phase could report green while silently
    # dropping whole-model, serialization, or imatrix evidence.
    phase_08_assembly_validation_gates = legacy[PHASE_07_ID].gates + legacy[PHASE_08_ID].gates
    phase_08 = _variant(
        legacy[PHASE_07_ID],
        phase_id=PHASE_08_ID,
        predecessor_phase=PHASE_07_ID,
        next_phase=PHASE_09_ID,
        objective="Assemble the BF16 sparse model and validate whole-model preservation and quality.",
        gates=phase_08_assembly_validation_gates,
        validation_commands=tuple(command for gate in phase_08_assembly_validation_gates for command in gate.validation_commands),
        expected_artifacts=legacy[PHASE_07_ID].expected_artifacts + legacy[PHASE_08_ID].expected_artifacts,
    )
    phase_09_runtime_quantization_gates = legacy[PHASE_09_ID].gates + legacy[PHASE_10_ID].gates
    phase_09 = _variant(
        legacy[PHASE_09_ID],
        phase_id=PHASE_09_ID,
        predecessor_phase=PHASE_08_ID,
        next_phase="complete",
        objective="Prove sparse runtime/serialization compatibility, then quantize the frozen BF16 sparse master and validate incremental degradation.",
        gates=phase_09_runtime_quantization_gates,
        validation_commands=tuple(command for gate in phase_09_runtime_quantization_gates for command in gate.validation_commands),
        expected_artifacts=legacy[PHASE_09_ID].expected_artifacts + legacy[PHASE_10_ID].expected_artifacts,
    )
    return {PHASE_00A_ID: phase_00a_contract(), PHASE_00B_ID: phase_00b_contract(), PHASE_01_ID: phase_01, PHASE_02_ID: phase_02, PHASE_03_ID: phase_03, PHASE_04_ID: phase_04, PHASE_05_ID: phase_05, PHASE_06_ID: phase_06, PHASE_07_ID: phase_07, PHASE_08_ID: phase_08, PHASE_09_ID: phase_09}


PHASE_CONTRACTS: dict[str, PhaseContract] = _canonical_phase_contracts()


def get_phase_contract(phase_id: str) -> PhaseContract:
    key = str(phase_id)
    if key in PHASE_CONTRACTS:
        return PHASE_CONTRACTS[key]
    try:
        return LEGACY_PHASE_CONTRACTS[key]
    except KeyError as exc:
        raise KeyError(f"unknown phase contract: {phase_id!r}") from exc


def get_execution_phase_contract(phase_id: str) -> PhaseContract:
    """Return only a canonical 00A/00B/01–09 contract."""

    try:
        return PHASE_CONTRACTS[str(phase_id)]
    except KeyError as exc:
        raise KeyError(f"unknown canonical execution phase: {phase_id!r}") from exc


__all__ = [
    "GATE_STATUSES",
    "LEGACY_PHASE_IDS",
    "METHOD_PROOF_COMMANDS",
    "PHASE_00A_COMMANDS",
    "PHASE_00A_ID",
    "PHASE_00B_COMMANDS",
    "PHASE_00B_ID",
    "PHASE_00_ID",
    "PHASE_00_VALIDATION_COMMANDS",
    "PHASE_01_BLOCKED_INVALID_CAPTURE",
    "PHASE_01_BLOCKED_NO_REAL_CAPTURE",
    "PHASE_01_COMMANDS",
    "PHASE_01_ID",
    "PHASE_01_MIN_TOKENS",
    "PHASE_01_REAL_METHOD_PROOF_FAILED",
    "PHASE_01_REAL_METHOD_PROOF_GREEN",
    "PHASE_01_REAL_METHOD_PROOF_RUNNING",
    "PHASE_01_REQUIRED_EVIDENCE_CLASS",
    "PHASE_01_SCIENTIFIC_STATES",
    "PHASE_02_ID",
    "PHASE_03_ID",
    "PHASE_04_ID",
    "PHASE_05_ID",
    "PHASE_06_ID",
    "PHASE_07_ID",
    "PHASE_08_ID",
    "PHASE_09_ID",
    "PHASE_10_ID",
    "PHASE_CONTRACTS",
    "PHASE_IDS",
    "PHASE_SCHEMA_VERSION",
    "GateContract",
    "GateReceipt",
    "Phase01PromotionBlocked",
    "PhaseContract",
    "PhaseReceipt",
    "PhaseReceiptStore",
    "get_execution_phase_contract",
    "get_phase_contract",
    "phase_00_contract",
    "phase_00a_contract",
    "phase_00b_contract",
    "phase_01_promotion_state",
    "validate_phase_01_promotion",
]
