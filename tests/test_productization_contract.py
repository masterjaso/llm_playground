from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense2moe.assembly.checkpoint import assemble_checkpoint, validate_assembly_receipt
from dense2moe.checkpoint import (
    LayerCheckpoint,
    profile_fingerprint,
    publish_tensor_artifact,
    save_layer_checkpoint,
)
from dense2moe.export.gguf import export_gguf, validate_gguf, write_tiny_gguf
from dense2moe.provenance import current_git_commit


class ProductizationContractTests(unittest.TestCase):
    def _write_layer(self, root: Path, layer: int) -> Path:
        tensor_path, inventory, tensor_hash = publish_tensor_artifact(
            {f"model.layers.{layer}.mlp.experts.0.gate_proj.weight": np.ones((2, 3), dtype=np.float32)},
            root / f"layer-{layer:04d}.safetensors",
        )
        checkpoint = LayerCheckpoint(
            layer=layer,
            profile="fixture",
            status="TRAINED_VALIDATED",
            profile_hash=profile_fingerprint("fixture"),
            source_revision="source-revision",
            source_config_hash="source-config",
            source_index_hash="source-index",
            dataset_hash="dataset",
            partition_hash="partition",
            tensor_file=tensor_path.name,
            tensor_sha256=tensor_hash,
            tensor_inventory=inventory,
            quality_gate={"overall": "green"},
            code_commit=current_git_commit(),
        )
        return save_layer_checkpoint(checkpoint, root / f"layer-{layer:04d}.json")

    def _assemble(self, root: Path, count: int = 2) -> tuple[Path, dict[str, object]]:
        paths = [self._write_layer(root / "layers", layer) for layer in range(count)]
        output = root / "assembled"
        manifest = assemble_checkpoint(
            paths,
            output,
            metadata={
                "profile": "fixture",
                "profile_hash": profile_fingerprint("fixture"),
                "source_revision": "source-revision",
                "expected_layers": count,
            },
        )
        return output / "manifest.json", manifest

    def test_assembly_receipt_preserves_complete_layer_and_tensor_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, manifest = self._assemble(Path(tmp))
            self.assertTrue(manifest["complete"], manifest["errors"])
            self.assertEqual(manifest["format"], "dense2moe-manifest-v3")
            self.assertEqual(manifest["layer_inventory"]["observed"], [0, 1])
            self.assertEqual(len(manifest["tensor_inventory"]), 2)
            self.assertTrue(manifest["tensor_inventory_sha256"])
            self.assertTrue(all(item["tensor_sha256"] for item in manifest["layers"]))
            self.assertTrue((manifest_path.parent / "assembly-receipt.json").is_file())
            receipt = validate_assembly_receipt(manifest_path)
            self.assertTrue(receipt["valid"], receipt["errors"])
            self.assertEqual(receipt["receipt"]["status"], "ASSEMBLY_COMPLETE")
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["tensor_inventory"], manifest["tensor_inventory"])

    def test_assembly_fails_closed_on_incomplete_layer_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_layer(root / "layers", 0)
            manifest = assemble_checkpoint(
                [path],
                root / "assembled",
                metadata={"profile": "fixture", "expected_layers": 2},
            )
            self.assertFalse(manifest["complete"])
            self.assertTrue(any("missing layers" in error for error in manifest["errors"]))
            receipt = validate_assembly_receipt(root / "assembled")
            self.assertTrue(receipt["valid"], receipt["errors"])
            self.assertEqual(receipt["receipt"]["status"], "ASSEMBLY_BLOCKED")

    def test_gguf_export_is_tensor_bearing_and_provenance_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _ = self._assemble(root)
            result = export_gguf(manifest_path, root / "model.gguf", metadata={"purpose": "contract"})
            validation = validate_gguf(root / "model.gguf", require_receipt=True)
            self.assertEqual(result["validation"]["tensor_count"], 2)
            self.assertEqual(validation["receipt"]["status"], "GGUF_EXPORT_COMPLETE")
            self.assertGreater(validation["size"], 24)
            self.assertEqual(validation["receipt"]["tensor_count"], validation["tensor_count"])
            self.assertTrue(Path(result["receipt"]).is_file())

    def test_gguf_export_rejects_changed_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _ = self._assemble(root)
            layer_tensor = root / "layers" / "layer-0000.safetensors"
            layer_tensor.write_bytes(layer_tensor.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "tensor hash mismatch"):
                export_gguf(manifest_path, root / "model.gguf")
            self.assertFalse((root / "model.gguf").exists())

    def test_tiny_gguf_is_explicitly_structural_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tiny_gguf(Path(tmp) / "tiny.gguf")
            self.assertEqual(validate_gguf(path)["magic"], "GGUF")
            with self.assertRaisesRegex(ValueError, "provenance receipt"):
                validate_gguf(path, require_receipt=True)


if __name__ == "__main__":
    unittest.main()
