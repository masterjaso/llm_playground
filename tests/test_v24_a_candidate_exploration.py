from __future__ import annotations

import json
from pathlib import Path

import pytest

from dense2moe.partition import partition_indices
from scripts import run_v24_a_candidate_exploration as runner


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    _write_json(source / "config.json", {"model_type": "qwen3_5"})
    _write_json(source / "model.safetensors.index.json", {"weight_map": {}})
    fit = _write_json(tmp_path / "FIT-TRAIN.json", {"split": "FIT-TRAIN", "count": 8, "dataset_hash": "fit-hash", "source_revision": "rev-1"})
    dev = _write_json(tmp_path / "FIT-DEV.json", {"split": "FIT-DEV", "count": 4, "dataset_hash": "dev-hash", "source_revision": "rev-1"})
    plan = partition_indices(17_408, 16, 960, 2_048, strategy="interleave")
    partition = _write_json(tmp_path / "partition.json", plan.as_dict())
    return {"source": source, "fit": fit, "dev": dev, "partition": partition}


def _schedule(config_id: str = "router-warm-start") -> dict[str, object]:
    return {
        "config_id": config_id,
        "stages": [
            {
                "name": "oracle_router_warm_start",
                "epochs": 1,
                "train_selection_router": True,
                "train_amplitude_router": True,
                "train_scales": True,
                "train_experts": False,
                "train_shared": False,
                "use_oracle_targets": True,
                "oracle_target_mode": "residual_correlation",
                "oracle_loss_mode": "multilabel_bce",
                "oracle_amplitude_mode": "mixed",
                "teacher_forcing_ratio": 0.5,
                "learning_rates": {"selection_router": 5e-4, "amplitude_router": 5e-4, "expert_scales": 1e-4},
                "loss_coefficients": {"mse": 1.0, "cosine": 0.1, "load_balance": 0.05, "hard_load_balance": 0.05, "router_z_loss": 0.001, "oracle": 0.2, "oracle_amplitude": 0.1},
                "oracle_regret_weight": 0.5,
            }
        ],
    }


def _run_kwargs(paths: dict[str, Path], run_root: Path, schedule: object) -> dict[str, object]:
    return {
        "source_dir": paths["source"],
        "fit_train_manifest": paths["fit"],
        "fit_dev_manifest": paths["dev"],
        "partition_path": paths["partition"],
        "run_root": run_root,
        "seed": 17,
        "device": "cpu",
        "microbatch": 2,
        "schedule": schedule,
        "source_revision": "rev-1",
    }


def test_dry_run_validates_partition_binding_and_immutable_cache(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    kwargs = _run_kwargs(paths, tmp_path / "run", _schedule())
    first = runner.run_exploration(**kwargs, execute=False)
    second = runner.run_exploration(**kwargs, execute=False)

    assert first == second
    assert first["status"] == "DRY_RUN"
    assert first["partition"]["geometry"] == {
        "dense_intermediate_size": 17_408,
        "routed_experts": 16,
        "expert_intermediate_size": 960,
        "shared_intermediate_size": 2_048,
    }
    assert first["opened_evaluation_tiers"] == []
    assert first["promotion_eligible"] is False
    assert first["fit_dev_gradient_contract"]["fit_dev_gradients"] is False

    changed = _schedule()
    changed["stages"][0]["teacher_forcing_ratio"] = 0.25  # type: ignore[index]
    with pytest.raises(RuntimeError, match="lineage/config mismatch"):
        runner.run_exploration(**_run_kwargs(paths, tmp_path / "run", changed), execute=False)


def test_split_validation_rejects_a_single_manifest_for_both_sides(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="split FIT-DEV"):
        runner.validate_split_manifests(paths["fit"], paths["fit"])


def test_execute_binds_fit_train_for_gradients_and_fit_dev_for_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    observed: dict[str, object] = {}

    def fake_train(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "metadata": str(tmp_path / "checkpoint" / "layer-0000.json"),
            "tensor_file": str(tmp_path / "checkpoint" / "layer-0000.safetensors"),
            "initial_fit": {"normalized_mse": 0.2, "cosine": 0.8, "load_cv": 0.2, "dead_experts": 0, "selected_counts": [1]},
            "final_fit": {"normalized_mse": 0.1, "cosine": 0.9, "load_cv": 0.2, "dead_experts": 0, "selected_counts": [1]},
            "initial_selection": {"normalized_mse": 0.2, "cosine": 0.8, "load_cv": 0.2, "dead_experts": 0, "selected_counts": [1]},
            "final_selection": {"normalized_mse": 0.08, "cosine": 0.95, "load_cv": 0.2, "dead_experts": 0, "selected_counts": [1]},
            "training_config": {"stage_selection_metrics": [], "stages": []},
        }

    monkeypatch.setattr("dense2moe.training.torch_distill.train_torch_layer", fake_train)
    monkeypatch.setattr(runner, "strict_reload_checkpoint", lambda *args, **kwargs: {"passed": True, "tensor_path": "tensor.safetensors", "tensor_sha256": "abc"})
    result = runner.run_exploration(**_run_kwargs(paths, tmp_path / "run", _schedule()), execute=True)

    assert result["status"] == "TRAINED"
    assert Path(str(observed["activation_manifest"])).resolve() == paths["fit"].resolve()
    assert Path(str(observed["selection_manifest"])).resolve() == paths["dev"].resolve()
    assert observed["selection_split"] == "FIT-DEV"
    assert observed["evaluate_holdout"] is False
    assert result["fit_dev_gradient_contract"]["fit_dev_gradients"] is False
    assert result["candidate_gate_met"] is False


def test_strict_reload_failure_is_recorded_and_not_promoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def fake_train(**_: object) -> dict[str, object]:
        return {"metadata": "missing.json", "tensor_file": "missing.safetensors"}

    monkeypatch.setattr("dense2moe.training.torch_distill.train_torch_layer", fake_train)
    result = runner.run_exploration(**_run_kwargs(paths, tmp_path / "run", _schedule("reload-failure")), execute=True)

    assert result["status"] == "TRAINING_FAILED"
    assert result["failure"]["type"] == "FileNotFoundError"
    assert result["promotion_eligible"] is False
    assert result["candidate_gate_met"] is False
