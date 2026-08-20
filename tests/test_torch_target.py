from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense2moe.models import TorchQwen35SwiGLUMoE, run_full_model_spike


class TorchTargetTests(unittest.TestCase):
    def test_trainable_moe_routes_and_reloads_strictly(self) -> None:
        import torch

        rng = np.random.default_rng(23)
        gate = rng.normal(size=(20, 8)).astype("float32")
        up = rng.normal(size=(20, 8)).astype("float32")
        down = rng.normal(size=(8, 20)).astype("float32")
        model = TorchQwen35SwiGLUMoE.from_dense(
            gate,
            up,
            down,
            routed_experts=4,
            shared_intermediate_size=4,
            top_k=2,
            learnable_scales=True,
        )
        inputs = torch.randn(5, 8)
        output, routing = model(inputs, return_router=True)
        self.assertEqual(tuple(output.shape), (5, 8))
        self.assertTrue(torch.allclose(routing["weights"].sum(dim=-1), torch.ones(5)))
        self.assertEqual(tuple(routing["indices"].shape), (5, 2))
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = TorchQwen35SwiGLUMoE.from_pretrained(tmp, strict=True)
            torch.testing.assert_close(model(inputs), restored(inputs))

    def test_shared_output_feature_router_is_opt_in_and_reloadable(self) -> None:
        import torch

        rng = np.random.default_rng(31)
        gate = rng.normal(size=(12, 4)).astype("float32")
        up = rng.normal(size=(12, 4)).astype("float32")
        down = rng.normal(size=(4, 12)).astype("float32")
        model = TorchQwen35SwiGLUMoE.from_dense(
            gate,
            up,
            down,
            routed_experts=2,
            shared_intermediate_size=4,
            top_k=1,
            router_hidden_size=6,
            router_feature_mode="shared_output",
        )
        inputs = torch.randn(3, 4)
        output, info = model(inputs, return_router=True)
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertEqual(info["router_feature_mode"], "shared_output")
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = TorchQwen35SwiGLUMoE.from_pretrained(tmp, strict=True)
            torch.testing.assert_close(model(inputs), restored(inputs))

    def test_full_model_spike_is_strict_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_full_model_spike(tmp)
        self.assertEqual(result["status"], "FULL_MODEL_RELOAD_GREEN")
        self.assertTrue(result["strict_tensor_inventory"])
        self.assertTrue(result["generation_equal"])
        self.assertLessEqual(result["max_logit_delta"], 1e-6)

    def test_actual_torch_distillation_uses_explicit_fixed_splits(self) -> None:
        """Exercise the AdamW router/scale/expert path on a tiny checkpoint."""

        from safetensors.numpy import save_file

        from dense2moe.capture import capture_activation_shards
        from dense2moe.config import MoEProfile
        from dense2moe.training import train_torch_layer

        rng = np.random.default_rng(7)
        gate = rng.normal(size=(8, 4)).astype("float32")
        up = rng.normal(size=(8, 4)).astype("float32")
        down = rng.normal(size=(4, 8)).astype("float32")
        train_x = rng.normal(size=(12, 4)).astype("float32")
        holdout_x = rng.normal(size=(6, 4)).astype("float32")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            shard_name = "model-00001-of-00001.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.mlp.gate_proj.weight": gate,
                    "model.language_model.layers.0.mlp.up_proj.weight": up,
                    "model.language_model.layers.0.mlp.down_proj.weight": down,
                },
                str(source / shard_name),
            )
            index = {
                "weight_map": {
                    "model.language_model.layers.0.mlp.gate_proj.weight": shard_name,
                    "model.language_model.layers.0.mlp.up_proj.weight": shard_name,
                    "model.language_model.layers.0.mlp.down_proj.weight": shard_name,
                }
            }
            (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            (source / "config.json").write_text(json.dumps({"model_type": "fixture"}), encoding="utf-8")
            captures = root / "capture"
            capture_activation_shards(
                train_x,
                captures,
                layer=0,
                split="train",
                manifest_name="layer-0000-train.json",
                shard_tokens=8,
                metadata={"dataset_hash": "fixture-dataset"},
            )
            capture_activation_shards(
                holdout_x,
                captures,
                layer=0,
                split="holdout",
                manifest_name="layer-0000-holdout.json",
                shard_tokens=8,
                metadata={"dataset_hash": "fixture-dataset"},
            )
            wrapper = captures / "layer-0000.json"
            wrapper.write_text(
                json.dumps(
                    {
                        "train_manifest": "layer-0000-train.json",
                        "holdout_manifest": "layer-0000-holdout.json",
                        "dataset_hash": "fixture-dataset",
                    }
                ),
                encoding="utf-8",
            )
            partition = root / "partition.json"
            partition.write_text(
                json.dumps(
                    {
                        "dense_intermediate_size": 8,
                        "routed_experts": 2,
                        "expert_intermediate_size": 3,
                        "shared_intermediate_size": 2,
                        "shared_indices": [0, 1],
                        "expert_indices": [[2, 3, 4], [5, 6, 7]],
                    }
                ),
                encoding="utf-8",
            )
            profile = MoEProfile("fixture", 4, 8, 1, 2, 3, 2, 2, revision="a" * 40)
            result = train_torch_layer(
                source_dir=source,
                activation_manifest=wrapper,
                output_dir=root / "trained",
                layer=0,
                profile=profile,
                partition_path=partition,
                epochs=1,
                microbatch=4,
                learning_rate=1e-2,
                device="cpu",
                source_revision="a" * 40,
                selection_indices=[0, 1],
                fit_exclude_indices=[0, 1, 2],
                validation_b_indices=[2],
                selection_identity_hash="validation-a-id",
                validation_b_identity_hash="validation-b-id",
            )
            self.assertIn(result["status"], {"TRAINED_VALIDATED", "VALIDATION_FAILED"})
            self.assertTrue(Path(result["metadata"]).exists())
            self.assertTrue(Path(result["tensor_file"]).exists())
            config = result["training_config"]
            self.assertEqual(config["selection_count"], 2)
            self.assertEqual(config["validation_a_count"], 2)
            self.assertEqual(config["validation_b_count"], 1)
            self.assertEqual(config["fit_count"], 9)
            self.assertEqual(config["fit_excluded_count"], 3)
            self.assertEqual(config["validation_a_identity_hash"], "validation-a-id")
            self.assertEqual(config["validation_b_identity_hash"], "validation-b-id")
            self.assertEqual(result["validation_b_metrics"]["split"], "validation-b")

            # FIT-DEV and a confirmation split may be independent frozen
            # manifests; neither identity is allowed to become a positional
            # row from FIT-TRAIN or an optimizer input.
            dev_x = rng.normal(size=(5, 4)).astype("float32")
            shadow_x = rng.normal(size=(4, 4)).astype("float32")
            capture_activation_shards(
                dev_x,
                captures,
                layer=0,
                split="train",
                manifest_name="layer-0000-fit-dev.json",
                shard_tokens=8,
                metadata={"dataset_hash": "fixture-fit-dev"},
            )
            capture_activation_shards(
                shadow_x,
                captures,
                layer=0,
                split="train",
                manifest_name="layer-0000-shadow-b.json",
                shard_tokens=8,
                metadata={"dataset_hash": "fixture-shadow-b"},
            )
            independent = train_torch_layer(
                source_dir=source,
                activation_manifest=wrapper,
                selection_manifest=captures / "layer-0000-fit-dev.json",
                validation_b_manifest=captures / "layer-0000-shadow-b.json",
                output_dir=root / "trained-independent",
                layer=0,
                profile=profile,
                partition_path=partition,
                epochs=0,
                microbatch=4,
                learning_rate=1e-2,
                device="cpu",
                source_revision="a" * 40,
                evaluate_holdout=False,
            )
            independent_config = independent["training_config"]
            self.assertEqual(independent_config["selection_split"], "independent-manifest")
            self.assertEqual(independent_config["selection_count"], 5)
            self.assertEqual(independent_config["validation_b_count"], 4)
            self.assertEqual(independent_config["selection_identity_hash"], "fixture-fit-dev")
            self.assertEqual(independent_config["validation_b_identity_hash"], "fixture-shadow-b")
            self.assertFalse(independent_config["split_opened_for"]["validation_a"]["gradient_updates"])


if __name__ == "__main__":
    unittest.main()
