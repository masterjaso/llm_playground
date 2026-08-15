from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense2moe.assembly import assemble_checkpoint
from dense2moe.capture import capture_activations, iter_activation_shards
from dense2moe.checkpoint import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
    validate_layer_checkpoint,
)
from dense2moe.models import DenseSwiGLU, Qwen35SwiGLUMoE
from dense2moe.state import StateStore, merge_fact_ledgers


class RealContractTests(unittest.TestCase):
    def test_blocked_state_resumes_and_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "run")
            store.transition(phase_status="blocked", terminal_state="BLOCKED", active_blocker="needs corpus", next_exact_command="d2m prepare-data --run-dir run --corpus-manifest corpus.json")
            store.write_handoff(next_command="d2m test --run-dir run", blocker=None)
            state = store.load()
            self.assertIsNone(state.active_blocker)
            self.assertIsNone(state.terminal_state)

    def test_fact_merge_is_monotonic(self) -> None:
        old = {"source_revision": {"status": "verified", "value": "immutable"}}
        merged = merge_fact_ledgers(old, {"source_revision": {"status": "unknown", "value": None}})
        self.assertEqual(merged["source_revision"]["value"], "immutable")
        self.assertEqual(merged["source_revision"]["status"], "verified")

    def test_capture_is_binary_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            values = np.arange(30, dtype=np.float32).reshape(10, 3)
            first = capture_activations([values], tmp, layer=0, shard_tokens=4, metadata={"dataset_hash": "d"})
            self.assertEqual(first["status"], "CAPTURE_COMPLETE")
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in first["shards"]))
            self.assertNotIn("values", json.loads((Path(tmp) / "layer-0000.json").read_text()))
            second = capture_activations([values], tmp, layer=0, shard_tokens=4, resume=True, metadata={"dataset_hash": "d"})
            self.assertEqual(second["status"], "CAPTURE_RESUMED")
            np.testing.assert_array_equal(np.concatenate(list(iter_activation_shards(Path(tmp) / "layer-0000.json"))), values)

    def test_target_swiglu_strict_reload(self) -> None:
        rng = np.random.default_rng(4)
        dense = DenseSwiGLU(rng.normal(size=(20, 8)).astype("float32"), rng.normal(size=(20, 8)).astype("float32"), rng.normal(size=(8, 20)).astype("float32"))
        model = Qwen35SwiGLUMoE.from_dense(dense, routed_experts=4, shared_intermediate_size=4, top_k=2)
        x = rng.normal(size=(9, 8)).astype("float32")
        np.testing.assert_allclose(dense(x), model(x, all_experts=True), atol=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = Qwen35SwiGLUMoE.from_pretrained(tmp, strict=True)
            np.testing.assert_array_equal(model(x), reloaded(x))

    def test_placeholder_assembly_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "layers"
            checkpoints.mkdir()
            for layer in range(4):
                save_layer_checkpoint(LayerCheckpoint(layer, "p8", status="synthetic-pending"), checkpoints / f"layer-{layer:04d}.json")
            manifest = assemble_checkpoint(sorted(checkpoints.glob("*.json")), root / "assembled", metadata={"profile": "p8", "expected_layers": 4})
            self.assertFalse(manifest["complete"])
            self.assertTrue(any("status" in error for error in manifest["errors"]))

    def test_real_layer_schema_requires_hash_and_finite_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors_path, inventory, digest = publish_tensor_artifact({"x": np.ones((2, 3), dtype=np.float32)}, root / "x.safetensors")
            metadata = LayerCheckpoint(0, "p8", status="TRAINED_VALIDATED", profile_hash=profile_fingerprint("p8"), source_revision="rev", source_config_hash="config", source_index_hash="index", dataset_hash="dataset", partition_hash="partition", tensor_file=tensors_path.name, tensor_sha256=digest, tensor_inventory=inventory, quality_gate={"overall": "green"}, code_commit="test")
            path = save_layer_checkpoint(metadata, root / "layer-0000.json")
            valid, errors, _ = validate_layer_checkpoint(path, expected_profile="p8", expected_source_revision="rev")
            self.assertTrue(valid, errors)
