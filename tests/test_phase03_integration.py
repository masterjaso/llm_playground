from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase03_integration import (
    _canonical_hash,
    _partition_from_payload,
    run_integration,
)


def _lock(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "status": "METHOD_LOCKED",
        "method_version": "moe-v22-m01",
        "external_tuning_forbidden": True,
        "opened_evaluation_tiers": [],
        "finalists": {},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase03_contract_mode_is_profile_scoped_and_full64_blocked(tmp_path: Path) -> None:
    result = run_integration(
        run_dir=tmp_path / "run",
        source_dir=tmp_path / "source",
        finalist_lock=_lock(tmp_path / "lock.json"),
        activation_manifest=tmp_path / "FIT-DEV.json",
        execute=False,
    )
    assert result["status"] == "PROFILE_INTEGRATION_READY"
    assert result["profiles"] == ["qwen38_p16s1_top4", "qwen38_p32s1_top5"]
    assert result["full64_assembly"] == "BLOCKED_UNTIL_64_VALIDATED_LAYERS"


def test_phase03_rejects_open_evaluation_tier_before_integration(tmp_path: Path) -> None:
    result = run_integration(
        run_dir=tmp_path / "run",
        source_dir=tmp_path / "source",
        finalist_lock=_lock(tmp_path / "lock.json", opened_evaluation_tiers=["GATE-A"]),
        activation_manifest=tmp_path / "FIT-DEV.json",
        execute=False,
    )
    assert result["status"] == "BLOCKED"
    assert result["blocker_code"] == "EVALUATION_TIERS_ALREADY_OPEN"


def test_phase03_partition_payload_is_strictly_reconstructed() -> None:
    payload = {
        "plan": {
            "dense_intermediate_size": 16,
            "routed_experts": 4,
            "expert_intermediate_size": 2,
            "shared_intermediate_size": 8,
            "shared_indices": list(range(8)),
            "expert_indices": [list(range(8 + 2 * index, 10 + 2 * index)) for index in range(4)],
        }
    }
    plan = _partition_from_payload(payload)
    assert plan.total_capacity == 16
    assert _canonical_hash(plan.as_dict())
