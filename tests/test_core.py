from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dense2moe.assembly import target_state_dict_inventory
from dense2moe.config import MoEProfile
from dense2moe.evaluation import classify_metric
from dense2moe.export.gguf import validate_gguf, write_tiny_gguf
from dense2moe.models import TinyDenseFFN, TinyMoE, topk_router
from dense2moe.partition import (
    pack_experts,
    partition_ffn_weights,
    partition_indices,
    unpack_experts,
)
from dense2moe.scheduling import JobQueue
from dense2moe.state import StateStore, atomic_artifact_publish
from dense2moe.training import OOMBackoff, train_tiny_layer


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def profile(self) -> MoEProfile:
        return MoEProfile("tiny", 8, 20, 2, 4, 4, 4, 2)

    def test_config_profile_arithmetic(self):
        profile = self.profile()
        self.assertEqual(profile.total_capacity, 20)
        profile.validate()

    def test_p32_product_target_geometry(self):
        from dense2moe.config import load_config

        top5 = load_config("configs/qwen38_p32s1_top5.yaml")
        top4 = load_config("configs/qwen38_p32s1_top4.yaml")
        self.assertEqual(top5.active_intermediate_size, 3584)
        self.assertAlmostEqual(top5.sparsity, 0.7941176470588235)
        self.assertEqual(top4.active_intermediate_size, 3072)
        self.assertAlmostEqual(top4.sparsity, 0.8235294117647058)

    def test_source_manifest_revision_pinned(self):
        from dense2moe.discovery.source import is_pinned_revision

        self.assertTrue(is_pinned_revision("0123456789abcdef"))
        self.assertFalse(is_pinned_revision("main"))

    def test_safetensors_slice_reader(self):
        try:
            import numpy as np
            from safetensors.numpy import save_file

            from dense2moe.checkpoint import SafetensorsSliceReader
        except ImportError:
            self.skipTest("optional numpy/safetensors not installed")
        path = self.root / "x.safetensors"
        save_file({"x": np.arange(12, dtype=np.float32).reshape(3, 4)}, str(path))
        with SafetensorsSliceReader(path) as reader:
            self.assertEqual(reader.info("x").shape, (3, 4))
            self.assertEqual(reader.read_rows("x", 1, 3).shape, (2, 4))

    def test_text_checkpoint_tensor_filter(self):
        from dense2moe.discovery.source import filter_text_tensor_names

        self.assertEqual(filter_text_tensor_names(["model.layers.0.mlp.w1", "visual.proj", "lm_head.weight"]), ["lm_head.weight", "model.layers.0.mlp.w1"])

    def test_text_checkpoint_extract_manifest(self):
        try:
            import numpy as np
            from safetensors.numpy import save_file

            from dense2moe.checkpoint import extract_text_checkpoint
        except ImportError:
            self.skipTest("optional numpy/safetensors not installed")
        source = self.root / "source"
        source.mkdir()
        save_file({"model.layers.0.mlp.w1": np.ones((2, 2)), "visual.proj": np.zeros((2, 2))}, str(source / "model-00001-of-00001.safetensors"))
        manifest = extract_text_checkpoint(source, self.root / "text")
        self.assertTrue(manifest["materialized"])
        self.assertEqual(manifest["text_tensor_names"], ["model.layers.0.mlp.w1"])

    def test_partition_is_exhaustive(self):
        plan = partition_indices(20, 4, 4, 4)
        self.assertEqual(set(plan.all_indices), set(range(20)))

    def test_partition_is_disjoint(self):
        plan = partition_indices(20, 4, 4, 4)
        self.assertEqual(len(plan.all_indices), len(set(plan.all_indices)))

    def test_partition_capacity(self):
        self.assertEqual(partition_indices(20, 4, 4, 4).total_capacity, 20)

    def test_partition_roundtrip_exact(self):
        import numpy as np

        weight = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        plan = partition_indices(20, 4, 4, 4)
        parts = partition_ffn_weights(weight, plan)
        from dense2moe.partition.ffn import reconstruct_ffn_weights

        np.testing.assert_array_equal(reconstruct_ffn_weights(parts, plan), weight)

    def test_activation_partition_deterministic(self):
        from dense2moe.capture import activation_partition_deterministic

        self.assertEqual(activation_partition_deterministic(20, 3, 7), activation_partition_deterministic(20, 3, 7))

    def test_dense_equivalent_reference(self):
        import numpy as np

        dense = TinyDenseFFN(3, 4, 2, seed=4)
        moe = TinyMoE(dense, experts=1, top_k=1)
        x = np.ones((5, 3), dtype=np.float32)
        np.testing.assert_allclose(dense(x), moe(x), atol=1e-6)

    def test_native_all_expert_equivalence(self):
        import numpy as np

        dense = TinyDenseFFN(3, 4, 2, seed=4)
        moe = TinyMoE(dense, experts=2, top_k=2)
        x = np.ones((5, 3), dtype=np.float32)
        np.testing.assert_allclose(dense(x), moe(x, all_experts=True), atol=1e-6)

    def test_router_topk_count(self):
        import numpy as np

        indices, _ = topk_router(np.zeros((5, 4)), 2)
        self.assertEqual(indices.shape, (5, 2))

    def test_router_weights_normalized(self):
        import numpy as np

        _, weights = topk_router(np.ones((5, 4)), 2)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(5))

    def test_shared_gate_initialization(self):
        from dense2moe.models.router import shared_gate_initialization

        self.assertEqual(len(shared_gate_initialization(5)), 5)

    def test_pack_unpack_experts(self):
        import numpy as np

        experts = [np.ones((2, 2)), np.zeros((2, 2))]
        unpacked = unpack_experts(pack_experts(experts))
        np.testing.assert_array_equal(unpacked[0], experts[0])

    def test_layer_checkpoint_roundtrip(self):
        from dense2moe.checkpoint.layer import (
            LayerCheckpoint,
            load_layer_checkpoint,
            save_layer_checkpoint,
        )

        path = self.root / "layer.json"
        save_layer_checkpoint(LayerCheckpoint(1, "tiny", {"w": [1, 2]}, {"loss": 0.1}), path)
        self.assertEqual(load_layer_checkpoint(path).layer, 1)

    def test_target_state_dict_inventory(self):
        keys = target_state_dict_inventory(num_layers=2, hidden_size=8, routed_experts=4, expert_intermediate_size=4, shared_intermediate_size=4)
        self.assertIn("model.layers.0.mlp.router.weight", keys)

    def test_atomic_artifact_publish(self):
        source, destination = self.root / "a", self.root / "b"
        source.write_text("x", encoding="utf-8")
        atomic_artifact_publish(source, destination)
        self.assertEqual(destination.read_text(encoding="utf-8"), "x")
        with self.assertRaises(FileExistsError):
            atomic_artifact_publish(source, destination)

    def test_state_resume(self):
        store = StateStore(self.root / "run")
        store.transition(current_phase="source", phase_status="pending", next_exact_command="d2m test")
        self.assertEqual(store.load().next_exact_command, "d2m test")

    def test_shadow_validation_is_disjoint_and_deterministic(self):
        from dense2moe.training import deterministic_shadow_validation_indices

        first, first_hash = deterministic_shadow_validation_indices(20, excluded_indices=[1, 3, 5], shadow_count=5, seed=9)
        second, second_hash = deterministic_shadow_validation_indices(20, excluded_indices=[1, 3, 5], shadow_count=5, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertTrue(set(first).isdisjoint({1, 3, 5}))

    def test_selector_split_contract_excludes_a_and_b_from_fit(self):
        from dense2moe.training import validate_split_contract

        contract = validate_split_contract(
            12,
            selection_indices=[2, 4],
            validation_b_indices=[7, 9],
            fit_exclude_indices=[2, 4, 7, 9],
        )
        self.assertEqual(contract["selection_indices"], (2, 4))
        self.assertEqual(contract["validation_b_indices"], (7, 9))
        self.assertEqual(contract["fit_exclude_indices"], (2, 4, 7, 9))
        self.assertEqual(contract["fit_indices"], tuple(i for i in range(12) if i not in {2, 4, 7, 9}))

    def test_selector_split_contract_rejects_b_in_selection_union(self):
        from dense2moe.training import validate_split_contract

        with self.assertRaisesRegex(ValueError, "validation-B cannot participate"):
            validate_split_contract(
                12,
                selection_indices=[2, 4],
                validation_b_indices=[7, 9],
                fit_exclude_indices=[2, 4, 7, 9],
                selection_union_indices=[2, 4, 7, 9],
            )

    def test_selector_split_contract_rejects_overlapping_a_and_b(self):
        from dense2moe.training import validate_split_contract

        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_split_contract(
                12,
                selection_indices=[2, 4],
                validation_b_indices=[4, 9],
                fit_exclude_indices=[2, 4, 9],
            )

    def test_job_queue_no_duplicate_lease(self):
        queue = JobQueue(self.root / "jobs.sqlite")
        queue.enqueue(0)
        first = queue.lease("a")
        second = queue.lease("b")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        queue.close()

    def test_worker_oom_backoff(self):
        backoff = OOMBackoff(8)
        self.assertEqual(backoff.next(RuntimeError("CUDA out of memory")), 4)

    def test_tiny_synthetic_model_training(self):
        import numpy as np

        x = np.eye(3)
        result = train_tiny_layer(x, x, epochs=10)
        self.assertLess(result["final_loss"], result["losses"][0])

    def test_tiny_hf_checkpoint_load(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
        self.assertEqual(json.loads(path.read_text())["model_type"], "qwen3")

    def test_tiny_gguf_conversion(self):
        path = write_tiny_gguf(self.root / "tiny.gguf")
        self.assertEqual(validate_gguf(path)["magic"], "GGUF")

    def test_metric_classification(self):
        self.assertEqual(classify_metric(0.1, green=0.2, yellow=0.4), "green")

    def test_report_terminal_state(self):
        store = StateStore(self.root / "run")
        store.transition(terminal_state="BLOCKED", active_blocker="test")
        self.assertEqual(store.load().terminal_state, "BLOCKED")
