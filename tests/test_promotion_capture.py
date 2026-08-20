from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dense2moe.capture import iter_activation_shards
from dense2moe.capture import promotion as promotion_module
from dense2moe.capture.real_method_proof import QWEN_SOURCE_REVISION, RealCaptureBlocked
from dense2moe.data import write_immutable_json
from dense2moe.hardware import _canonical_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return _sha256(path)


def _fixture_corpus(root: Path) -> tuple[Path, dict[str, list[str]]]:
    root.mkdir(parents=True, exist_ok=True)
    ids_by_tier = {
        "FIT-TRAIN": ["train-0"],
        "FIT-DEV": ["dev-0"],
        "GATE-A": ["gate-0", "gate-1"],
        "SHADOW-B": ["shadow-b-0"],
        "SHADOW-C": ["shadow-c-0"],
    }
    rows: list[dict[str, object]] = []
    for tier, ids in ids_by_tier.items():
        for row_id in ids:
            text = f"frozen {tier} example {row_id}"
            rows.append(
                {
                    "id": row_id,
                    "tier": tier,
                    "split": tier,
                    "text": text,
                    "token_count": 1,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "normalized_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "source_record_id": f"source:{row_id}",
                    "source_revision": QWEN_SOURCE_REVISION,
                    "source_family": f"family:{row_id}",
                }
            )
    manifest = root / "corpus-v2.2.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest_hash = _sha256(manifest)
    splits = {"schema_version": 1, "records": ids_by_tier}
    splits_hash = _write_json(root / "corpus-v2.2-splits.json", splits)
    overlap_hash = _write_json(
        root / "corpus-v2.2-overlap-audit.json",
        {
            "schema_version": 2,
            "status": "PASS",
            "checked_tiers": list(ids_by_tier),
            "zero_forbidden_overlap": True,
            "records_checked": len(rows),
            "group_conflicts": [],
            "exact_conflicts": [],
            "near_duplicate_conflicts": [],
        },
    )
    dataset_hashes = {
        tier: hashlib.sha256("".join(ids).encode()).hexdigest() for tier, ids in ids_by_tier.items()
    }
    tier_entries = {
        tier: {"status": "FROZEN", "opened": False, "optimizer_updates": False, "dataset_sha256": dataset_hashes[tier]}
        for tier in promotion_module.PROMOTION_EVALUATION_TIERS
    }
    tier_ledger_hash = _write_json(
        root / "corpus-v2.2-tier-ledger.json",
        {
            "schema_version": 1,
            "status": "SEALED",
            "one_way_opening": True,
            "manifest_sha256": manifest_hash,
            "method_version": "moe-v22-m01",
            "tiers": tier_entries,
        },
    )
    activation_plan_hash = _write_json(
        root / "corpus-v2.2-activation-plan.json",
        {
            "schema_version": 1,
            "status": "READY_FOR_BALANCED_CAPTURE",
            "eligible_splits": ["FIT-TRAIN", "FIT-DEV"],
            "records": {"FIT-TRAIN": ids_by_tier["FIT-TRAIN"], "FIT-DEV": ids_by_tier["FIT-DEV"]},
        },
    )
    artifacts = {
        "manifest": {"path": manifest.name, "sha256": manifest_hash},
        "splits": {"path": "corpus-v2.2-splits.json", "sha256": splits_hash},
        "overlap": {"path": "corpus-v2.2-overlap-audit.json", "sha256": overlap_hash},
        "tier_ledger": {"path": "corpus-v2.2-tier-ledger.json", "sha256": tier_ledger_hash},
        "activation_plan": {"path": "corpus-v2.2-activation-plan.json", "sha256": activation_plan_hash},
    }
    receipt = {
        "schema_version": 1,
        "status": "CORPUS_V22_FROZEN",
        "immutable": True,
        "component": "development-internal",
        "required_tiers": list(ids_by_tier),
        "method_version": "moe-v22-m01",
        "threshold_fingerprint": "sealed-qwen38-promotion-v1",
        "manifest": {"path": manifest.name, "sha256": manifest_hash},
        "artifacts": artifacts,
    }
    _write_json(root / "corpus-v2.2-receipt.json", receipt)
    return root, ids_by_tier


def _fixture_runtime_and_source(root: Path) -> tuple[Path, Path, Path]:
    runtime_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "APPROVED",
        "runtime_fingerprint": {"platform": "Windows", "python_version": "3.12.6"},
    }
    runtime_payload["lock_sha256"] = _canonical_sha256(runtime_payload)
    runtime_lock = root / "windows-runtime-lock.json"
    _write_json(runtime_lock, runtime_payload)

    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    _write_json(source / "config.json", {"model_type": "qwen3_5"})
    _write_json(source / "model.safetensors.index.json", {"weight_map": {}})

    ledger = root / "promotion" / "contamination-ledger.json"
    _write_json(
        ledger,
        {
            "schema_version": 1,
            "ledger_type": "dense2moe-evaluation-contamination",
            "status": "SEALED",
            "method_version": "moe-v22-m01",
            "code_commit": "fixture-commit",
            "thresholds_fingerprint": "sealed-qwen38-promotion-v1",
            "runtime_lock_sha256": "",
            "corpus_hashes": {},
            "tiers": {},
            "history": [],
        },
    )
    return runtime_lock, source, ledger


@pytest.fixture
def frozen_fixture(tmp_path: Path) -> dict[str, object]:
    corpus, ids_by_tier = _fixture_corpus(tmp_path / "corpus")
    runtime_lock, source, ledger = _fixture_runtime_and_source(tmp_path)
    return {"corpus": corpus, "ids": ids_by_tier, "runtime_lock": runtime_lock, "source": source, "ledger": ledger, "root": tmp_path}


def test_frozen_gate_a_plan_contains_only_frozen_rows(frozen_fixture: dict[str, object]) -> None:
    result = promotion_module.build_frozen_evaluation_plan(frozen_fixture["corpus"], "GATE-A")
    plan = result["plan"]
    assert plan["GATE-A"]
    assert plan["row_ids"] == frozen_fixture["ids"]["GATE-A"]
    assert {row["split"] for row in plan["GATE-A"]} == {"GATE-A"}
    assert set(plan["row_ids"]).isdisjoint(frozen_fixture["ids"]["FIT-TRAIN"] + frozen_fixture["ids"]["FIT-DEV"])


@pytest.mark.parametrize("tier,reason", [("FIT-TRAIN", "PROMOTION_TIER_DEVELOPMENT_UNAUTHORIZED"), ("NOPE", "PROMOTION_TIER_UNKNOWN")])
def test_development_and_unknown_tiers_are_rejected(frozen_fixture: dict[str, object], tier: str, reason: str) -> None:
    with pytest.raises(RealCaptureBlocked, match=reason):
        promotion_module.build_frozen_evaluation_plan(frozen_fixture["corpus"], tier)


def test_frozen_validation_rejects_development_overlap(frozen_fixture: dict[str, object]) -> None:
    root = frozen_fixture["corpus"]
    splits_path = root / "corpus-v2.2-splits.json"
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    payload["records"]["FIT-TRAIN"].append(payload["records"]["GATE-A"][0])
    splits_hash = _write_json(splits_path, payload)
    receipt_path = root / "corpus-v2.2-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["splits"]["sha256"] = splits_hash
    _write_json(receipt_path, receipt)
    with pytest.raises(RealCaptureBlocked, match="FROZEN_EVALUATION_DEVELOPMENT_OVERLAP"):
        promotion_module.build_frozen_evaluation_plan(root, "GATE-A")


def test_frozen_validation_rejects_dataset_and_receipt_hash_mismatch(frozen_fixture: dict[str, object]) -> None:
    root = frozen_fixture["corpus"]
    tier_ledger_path = root / "corpus-v2.2-tier-ledger.json"
    tier_ledger = json.loads(tier_ledger_path.read_text(encoding="utf-8"))
    tier_ledger["tiers"]["GATE-A"]["dataset_sha256"] = "0" * 64
    tier_hash = _write_json(tier_ledger_path, tier_ledger)
    receipt_path = root / "corpus-v2.2-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["tier_ledger"]["sha256"] = tier_hash
    _write_json(receipt_path, receipt)
    with pytest.raises(RealCaptureBlocked, match="FROZEN_TIER_DATASET_HASH_MISMATCH"):
        promotion_module.build_frozen_evaluation_plan(root, "GATE-A")

    manifest = root / "corpus-v2.2.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RealCaptureBlocked, match="FROZEN_CORPUS_ARTIFACT_HASH_MISMATCH"):
        promotion_module.build_frozen_evaluation_plan(root, "GATE-A")


def test_capture_publishes_gate_a_only_and_is_evaluator_resolvable(frozen_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.numpy import save_file

    calls = {"count": 0}

    def fake_stream(source_snapshot: Path, plan_path: Path, run_dir: Path, *, split: str, layers: tuple[int, ...], **_: object) -> dict[str, object]:
        calls["count"] += 1
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rows = plan[split]
        capture_root = Path(run_dir) / "capture"
        shard_root = capture_root / "layer-0000"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_path = shard_root / "shard-00000.safetensors"
        values = np.arange(len(rows) * 2, dtype=np.float32).reshape(len(rows), 2)
        save_file({"ffn_input": values, "dense_ffn_target": values.copy()}, str(shard_path))
        shard_record = {
            "path": "layer-0000/shard-00000.safetensors",
            "sha256": _sha256(shard_path),
            "count": len(rows),
            "records": [{"example_id": row["id"], "split": split, "length": int(row["token_count"]), "chunk_index": 0} for row in rows],
            "input_tensor": "ffn_input",
            "target_tensor": "dense_ffn_target",
        }
        manifest = {
            "status": "CAPTURE_COMPLETE",
            "layer": 0,
            "split": split,
            "count": len(rows),
            "dataset_hash": plan["dataset_hash"],
            "shards": [shard_record],
            "input_tensor": "ffn_input",
            "target_tensor": "dense_ffn_target",
        }
        manifest_path = capture_root / "layer-0000-GATE-A.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "STREAMING_CAPTURE_COMPLETE", "split": split, "layers": list(layers)}

    monkeypatch.setattr(promotion_module, "stream_teacher_split", fake_stream)
    output = frozen_fixture["root"] / "candidate-search" / "activations" / "promotion" / "GATE-A"
    staging = frozen_fixture["root"] / "capture-work" / "GATE-A"
    ledger = frozen_fixture["ledger"]
    before_ledger = _sha256(ledger)
    result = promotion_module.capture_frozen_evaluation_tier(
        frozen_fixture["corpus"],
        frozen_fixture["source"],
        output,
        frozen_fixture["runtime_lock"],
        tier="GATE-A",
        staging_root=staging,
        contamination_ledger=ledger,
    )
    assert result["status"] == "PROMOTION_CAPTURE_COMPLETE"
    manifest = output / "layer-0000.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["tier"] == "GATE-A"
    assert payload["promotion_provenance"]["row_ids"] == frozen_fixture["ids"]["GATE-A"]
    assert _sha256(ledger) == before_ledger
    assert sorted(path.name for path in output.iterdir()) == ["layer-0000", "layer-0000.json"]

    from scripts.evaluate_promotion import _manifest_for_tier

    assert _manifest_for_tier(output.parent, "GATE-A") == manifest
    shards = list(iter_activation_shards(manifest))
    assert len(shards) == 1
    assert shards[0].shape == (len(frozen_fixture["ids"]["GATE-A"]), 2)
    resumed = promotion_module.capture_frozen_evaluation_tier(
        frozen_fixture["corpus"],
        frozen_fixture["source"],
        output,
        frozen_fixture["runtime_lock"],
        tier="GATE-A",
        staging_root=staging,
        contamination_ledger=ledger,
    )
    assert resumed["status"] == "PROMOTION_CAPTURE_RESUMED"
    assert calls["count"] == 1


def test_capture_rejects_existing_manifest_with_wrong_receipt(frozen_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.numpy import save_file

    def fake_stream(_source_snapshot: Path, plan_path: Path, run_dir: Path, *, split: str, **_: object) -> dict[str, object]:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        capture_root = Path(run_dir) / "capture"
        shard_root = capture_root / "layer-0000"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_path = shard_root / "shard.safetensors"
        values = np.zeros((len(plan[split]), 2), dtype=np.float32)
        save_file({"ffn_input": values, "dense_ffn_target": values}, str(shard_path))
        payload = {
            "status": "CAPTURE_COMPLETE",
            "layer": 0,
            "split": split,
            "count": len(plan[split]),
            "dataset_hash": plan["dataset_hash"],
            "shards": [{"path": "layer-0000/shard.safetensors", "sha256": _sha256(shard_path), "count": len(plan[split]), "records": [{"example_id": row["id"], "split": split, "length": int(row["token_count"]), "chunk_index": 0} for row in plan[split]], "input_tensor": "ffn_input", "target_tensor": "dense_ffn_target"}],
        }
        (capture_root).mkdir(parents=True, exist_ok=True)
        (capture_root / "layer-0000-GATE-A.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "STREAMING_CAPTURE_COMPLETE"}

    monkeypatch.setattr(promotion_module, "stream_teacher_split", fake_stream)
    output = frozen_fixture["root"] / "out"
    common = {
        "corpus_root": frozen_fixture["corpus"],
        "source_snapshot": frozen_fixture["source"],
        "output_root": output,
        "runtime_lock": frozen_fixture["runtime_lock"],
        "tier": "GATE-A",
        "staging_root": frozen_fixture["root"] / "capture-work" / "tamper-test",
        "contamination_ledger": frozen_fixture["ledger"],
    }
    promotion_module.capture_frozen_evaluation_tier(**common)
    manifest_path = output / "layer-0000.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["promotion_provenance"]["corpus_receipt_sha256"] = "0" * 64
    manifest_path.unlink()
    write_immutable_json(manifest_path, payload)
    with pytest.raises(RealCaptureBlocked, match="PROMOTION_CAPTURE_PROVENANCE_MISMATCH"):
        promotion_module.capture_frozen_evaluation_tier(**common)
