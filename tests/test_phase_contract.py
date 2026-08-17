from __future__ import annotations

import json

import pytest

from dense2moe.cli import _resume_cli_tokens
from dense2moe.config import (
    ACTIVE_TOPOLOGY_IDS,
    FORBIDDEN_TOPOLOGY_IDS,
    active_topology_contract,
    load_active_config,
    load_config,
    validate_active_profile,
)
from dense2moe.phase import (
    LEGACY_PHASE_CONTRACTS,
    PHASE_CONTRACTS,
    PHASE_IDS,
    PhaseReceiptStore,
    get_phase_contract,
    phase_00_contract,
)


def test_executable_phase_contracts_are_explicit_native_windows_commands() -> None:
    contracts = [*PHASE_CONTRACTS.values(), *LEGACY_PHASE_CONTRACTS.values()]
    commands = [command for contract in contracts for command in (*contract.validation_commands, *contract.handoff_commands) if command.strip()]
    assert commands
    for command in commands:
        assert "PYTHONPATH=" not in command
        assert not command.lstrip().startswith(("python ", "python3 ", "pytest ", "d2m "))
        if "_nsp" not in command and "powershell.exe" not in command:
            assert ".venv\\Scripts\\python.exe" in command


def test_cli_resume_accepts_only_explicit_windows_project_interpreter() -> None:
    command = r"& .\.venv\Scripts\python.exe -m dense2moe.cli status --run-dir C:\workplace\run --json"
    assert _resume_cli_tokens(command) == ["status", "--run-dir", r"C:\workplace\run", "--json"]
    assert _resume_cli_tokens("d2m status --run-dir run") == ["status", "--run-dir", "run"]
    assert _resume_cli_tokens(r"& powershell.exe -File .\scripts\Setup-Windows.ps1") is None


def test_only_product_topologies_are_active() -> None:
    assert ACTIVE_TOPOLOGY_IDS == ("p16/top4", "p32/top5")
    assert FORBIDDEN_TOPOLOGY_IDS == {"p32/top4"}
    assert active_topology_contract("p16/top4").active_intermediate_size == 5120
    assert active_topology_contract("p32/top5").active_intermediate_size == 3584


def test_topology_selector_handles_ids_and_config_paths() -> None:
    assert active_topology_contract("p16/top4").profile_name == "qwen38_p16s1_top4"
    assert active_topology_contract("configs/qwen38_p32s1_top5.yaml").topology_id == "p32/top5"
    assert active_topology_contract(r"configs\qwen38_p16s1_top4.yaml").topology_id == "p16/top4"


def test_forbidden_and_legacy_topologies_fail_closed() -> None:
    with pytest.raises(ValueError, match="explicitly forbidden"):
        active_topology_contract("p32/top4")
    with pytest.raises(ValueError, match="inactive topology"):
        active_topology_contract("qwen38_p16s1_top3")


def test_active_config_checks_geometry_not_just_profile_name() -> None:
    profile, contract = load_active_config("configs/qwen38_p16s1_top4.yaml")
    assert validate_active_profile(profile, topology="p16/top4") == contract

    mismatched = load_config("configs/qwen38_p16s1152_top4.yaml")
    with pytest.raises(ValueError, match="does not satisfy active topology"):
        validate_active_profile(mismatched, topology="p16/top4")


def test_phase_zero_contract_is_expanded_and_contains_active_topology_invariants() -> None:
    contract = phase_00_contract()
    assert get_phase_contract("phase-00") == contract
    assert contract.prediction_depth == "expanded"
    assert len(contract.gates) == 10
    assert contract.active_topologies == ("p16/top4", "p32/top5")
    assert "p32/top4" in contract.forbidden_topologies
    assert "freeze_corpus_v21.py" in contract.validation_commands[3]
    assert contract.contract_fingerprint == phase_00_contract().contract_fingerprint


def test_all_later_phases_have_coarse_receipt_bearing_contracts() -> None:
    assert tuple(PHASE_CONTRACTS) == PHASE_IDS
    for index, phase_id in enumerate(PHASE_IDS[1:], start=1):
        contract = get_phase_contract(phase_id)
        assert contract.predecessor_phase == PHASE_IDS[index - 1]
        assert contract.next_phase == (PHASE_IDS[index + 1] if index < len(PHASE_IDS) - 1 else "complete")
        assert contract.gates
        assert contract.expected_artifacts
        assert contract.validation_commands
        assert contract.active_topologies == ACTIVE_TOPOLOGY_IDS
        assert "p32/top4" in contract.forbidden_topologies
        assert all(gate.gate_id for gate in contract.gates)


def test_later_phase_contracts_keep_scientific_predecessor_gates() -> None:
    phase1 = get_phase_contract("phase-01")
    phase7 = get_phase_contract("phase-07")
    phase8 = get_phase_contract("phase-08")
    phase9 = get_phase_contract("phase-09")
    phase10 = get_phase_contract("phase-10")
    assert "phase-00-green" in phase1.gate_ids
    assert "full64-input" in phase7.gate_ids
    assert {"bf16-reload", "distribution-quality", "preservation-quality"} <= set(phase8.gate_ids)
    assert {"runtime-discovery", "serialization-contract", "imatrix-contract", "conservative-quantization"} <= set(phase9.gate_ids)
    assert "bf16-freeze" in phase10.gate_ids
    assert phase10.next_phase == "complete"


def test_phase_receipt_resumes_only_with_matching_contract_and_artifacts(tmp_path) -> None:
    contract = phase_00_contract()
    store = PhaseReceiptStore(tmp_path / "phase-00.json", contract, run_id="run-1")
    initial = store.load()
    assert initial.can_resume(contract)
    assert not initial.is_complete

    receipt = store.record_gate(
        "corpus-v21-freeze",
        "passed",
        evidence_refs=["corpus-v21-receipt.json"],
        artifact_hashes={"corpus-v21-receipt.json": "abc"},
    )
    assert receipt.status == "running"
    restored = store.load()
    assert restored.gates["corpus-v21-freeze"].status == "passed"
    assert restored.can_resume(contract, artifact_hashes={"corpus-v21-receipt.json": "abc"})
    assert not restored.can_resume(contract, artifact_hashes={"corpus-v21-receipt.json": "changed"})

    for gate_id in contract.gate_ids[1:]:
        store.record_gate(gate_id, "passed")
    complete = store.load()
    assert complete.status == "complete"
    assert complete.next_phase_eligibility == "eligible"
    assert complete.is_complete
    assert not complete.can_resume(contract)


def test_phase_receipt_rejects_stale_contract(tmp_path) -> None:
    contract = phase_00_contract()
    path = tmp_path / "phase-00.json"
    store = PhaseReceiptStore(path, contract)
    store.save(store.load())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_fingerprint"] = "stale"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        store.load()
