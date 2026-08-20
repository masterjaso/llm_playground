from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dense2moe.data import (
    prepare_calibration_manifest,
    resolve_corpus_record,
    verify_corpus_manifest,
)


class _CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    add_bos_token = False
    add_eos_token = False
    chat_template = None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = [ord(character) for character in text]
        if add_special_tokens:
            values = [self.bos_token_id, *values, self.eos_token_id]
        return values


def _record(text: str, index: int, *, domain: str = "general", license_name: str = "MIT") -> dict[str, object]:
    return {
        "text": text,
        "source_name": "fixture/public",
        "source_revision": "fixture-v1",
        "source_license": license_name,
        "source_record_id": f"fixture-{index}",
        "domain": domain,
        "rationale": "Small local fixture used to test deterministic corpus preparation.",
    }


class CalibrationDataV2Tests(unittest.TestCase):
    def test_exact_counts_locators_dedup_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            rows = [
                _record("alpha", 0),
                _record("bravo", 1),
                _record("charlie", 2),
                _record("delta", 3),
                _record("echo", 4),
                _record("foxtrot", 5),
                _record("  alpha  ", 6),  # normalized-content duplicate
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "run" / "data-plan.json"
            tokenizer = _CharacterTokenizer()
            payload = prepare_calibration_manifest(
                source,
                output,
                train_tokens=20,
                holdout_tokens=10,
                tokenizer=tokenizer,
                tokenizer_revision="fixture-v1",
                tokenizer_metadata={
                    "source_snapshot": "fixture-tokenizer",
                    "files": [{"path": "tokenizer.json", "sha256": hashlib.sha256(b"fixture").hexdigest()}],
                },
            )

            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(payload["tokenization_method"], "source_tokenizer_exact")
            self.assertEqual(payload["tokenizer"]["files_sha256"], payload["tokenizer"]["files_sha256"])
            self.assertTrue(Path(payload["receipt_path"]).exists())
            self.assertGreaterEqual(payload["train_tokens"], 20)
            self.assertGreaterEqual(payload["holdout_tokens"], 10)
            train_ids = {item["id"] for item in payload["train"]}
            holdout_ids = {item["id"] for item in payload["holdout"]}
            self.assertTrue(train_ids.isdisjoint(holdout_ids))
            self.assertEqual(len(train_ids | holdout_ids), 6)
            selected = next(iter(payload["train"] + payload["holdout"]))
            self.assertIn("source_file", selected)
            self.assertIn("source_record_index", selected)
            self.assertIn("content_sha256", selected)
            self.assertEqual(len(resolve_corpus_record(selected, base_dir=root, tokenizer=tokenizer)), selected["token_count"])
            evidence = verify_corpus_manifest(output, tokenizer=tokenizer)
            self.assertEqual(evidence["status"], "CORPUS_VERIFIED")
            receipt = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["receipt_type"], "dense2moe-corpus-receipt")
            self.assertEqual(receipt["manifest"]["dataset_hash"], payload["dataset_hash"])
            self.assertFalse(receipt["resolvability"]["text_embedded"])

    def test_missing_tokenizer_is_not_whitespace_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps(_record("alpha", 0)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact source-tokenizer"):
                prepare_calibration_manifest(source, root / "manifest.json", train_tokens=1, holdout_tokens=1)

    def test_unclear_license_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps(_record("alpha", 0, license_name="unknown")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unclear licenses"):
                prepare_calibration_manifest(
                    source,
                    root / "manifest.json",
                    train_tokens=1,
                    holdout_tokens=1,
                    tokenizer=_CharacterTokenizer(),
                    tokenizer_revision="fixture-v1",
                )

    def test_locator_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps(_record(text, index)) for index, text in enumerate(("alpha", "bravo"))) + "\n",
                encoding="utf-8",
            )
            payload = prepare_calibration_manifest(
                source,
                root / "manifest.json",
                train_tokens=1,
                holdout_tokens=1,
                tokenizer=_CharacterTokenizer(),
                tokenizer_revision="fixture-v1",
            )
            source.write_text(json.dumps(_record("changed", 0)) + "\n", encoding="utf-8")
            selected = payload["train"][0]
            with self.assertRaisesRegex(ValueError, "source file hash mismatch"):
                resolve_corpus_record(selected, base_dir=root, tokenizer=_CharacterTokenizer())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
