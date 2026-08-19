import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from dense2moe.evaluation.replay import (
    ReplayInputError,
    iter_paired_activation_batches,
    iter_paired_activation_shards,
)


def _write_capture(tmp_path, *, count=5):
    shard = tmp_path / "shard.safetensors"
    inputs = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
    targets = inputs + 1
    save_file({"ffn_input": inputs, "dense_ffn_target": targets}, str(shard))
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = tmp_path / "layer-0000-FIT-TRAIN.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "CAPTURE_COMPLETE",
                "quality_gate_eligible": True,
                "split": "FIT-TRAIN",
                "count": count,
                "input_tensor": "ffn_input",
                "target_tensor": "dense_ffn_target",
                "shards": [
                    {
                        "path": shard.name,
                        "sha256": digest,
                        "shape": [count, 4],
                        "input_tensor": "ffn_input",
                        "target_tensor": "dense_ffn_target",
                        "records": [
                            {"offset": 0, "length": 2, "example_id": "example-a", "split": "FIT-TRAIN"},
                            {"offset": 2, "length": count - 2, "example_id": "example-b", "split": "FIT-TRAIN"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, inputs, targets


def test_paired_reader_loads_input_target_and_expands_groups(tmp_path):
    manifest, inputs, targets = _write_capture(tmp_path)

    batches = list(iter_paired_activation_shards(manifest, expected_split="FIT-TRAIN", repo_root=tmp_path))

    assert len(batches) == 1
    assert torch.equal(batches[0].inputs, inputs)
    assert torch.equal(batches[0].targets, targets)
    assert [item["independent_group"] for item in batches[0].metadata] == ["example-a", "example-a", "example-b", "example-b", "example-b"]


def test_paired_reader_batches_are_bounded_and_preserve_metadata(tmp_path):
    manifest, inputs, targets = _write_capture(tmp_path, count=7)

    batches = list(iter_paired_activation_batches(manifest, expected_split="train", repo_root=tmp_path, batch_tokens=3))

    assert [int(batch.inputs.shape[0]) for batch in batches] == [3, 3, 1]
    assert torch.equal(torch.cat([batch.inputs for batch in batches]), inputs)
    assert torch.equal(torch.cat([batch.targets for batch in batches]), targets)
    assert sum(len(batch.metadata) for batch in batches) == 7


def test_paired_reader_fails_closed_on_hash_mismatch(tmp_path):
    manifest, _inputs, _targets = _write_capture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shards"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayInputError, match="SHA256 mismatch"):
        list(iter_paired_activation_shards(manifest, expected_split="FIT-TRAIN", repo_root=tmp_path))


def test_paired_reader_rejects_wrong_split(tmp_path):
    manifest, _inputs, _targets = _write_capture(tmp_path)

    with pytest.raises(ReplayInputError, match="split mismatch"):
        list(iter_paired_activation_shards(manifest, expected_split="FIT-DEV", repo_root=tmp_path))
