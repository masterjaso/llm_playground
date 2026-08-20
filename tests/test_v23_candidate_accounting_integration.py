from __future__ import annotations

import json

from scripts.run_candidate_search import _write_v23_accounting_receipt, run_candidate_search


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v23_readiness_receipt_contains_complete_accounting(tmp_path) -> None:
    train = tmp_path / "FIT-TRAIN.json"
    dev = tmp_path / "FIT-DEV.json"
    _write(train, {"split": "train", "count": 32, "dataset_hash": "fit"})
    _write(dev, {"split": "train", "count": 16, "dataset_hash": "dev"})

    result = run_candidate_search(
        run_dir=tmp_path / "run",
        activation_manifest=train,
        dev_manifest=dev,
        topology="p16/top4",
        exhaustive=True,
        expected_combinations=1820,
        method_version="moe-v23-m01",
    )

    accounting = result["accounting"]
    assert result["schema_version"] == 4
    assert result["method_version"] == "moe-v23-m01"
    assert accounting["version"] == "dense2moe-v2.3"
    assert accounting["topology"]["id"] == "p16/top4"
    assert accounting["parameters"]["shared_parameters"] > 0
    assert accounting["parameters"]["active_routed_parameters"] > 0
    assert accounting["parameters"]["router_parameters"] > 0
    assert accounting["parameters"]["scale_parameters"] > 0
    assert accounting["flops"]["active_flops"] > 0
    assert accounting["active_ffn_flop_reduction"] >= 0.50


def test_v23_accounting_receipt_is_immutable_at_execute_boundary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_v23_accounting_receipt(
        run_dir=run_dir,
        topologies=("p16/top4", "p32/top5"),
        tokens=4096,
    )
    receipt_path = run_dir / "development" / "v23-accounting-receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert sorted(payload["topologies"]) == ["p16/top4", "p32/top5"]
    # Re-emitting identical content is allowed, while a changed token window
    # would fail closed through write_immutable_json.
    _write_v23_accounting_receipt(
        run_dir=run_dir,
        topologies=("p16/top4", "p32/top5"),
        tokens=4096,
    )
