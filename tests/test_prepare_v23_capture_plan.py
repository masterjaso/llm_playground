from __future__ import annotations

import hashlib
import json

from scripts.prepare_v23_capture_plan import prepare_v23_capture_plan


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_v23_capture_plan_joins_frozen_rows_and_caps_both_splits(tmp_path) -> None:
    corpus = tmp_path / "v23-corpus.jsonl"
    rows = []
    for split, prefix in (("FIT-TRAIN", "train"), ("FIT-DEV", "dev")):
        for index in range(3):
            text = f"{prefix} row {index} with enough immutable text"
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "source_record_id": f"source:{prefix}:{index}",
                    "source_family": f"family:{prefix}",
                    "source_name": "fixture",
                    "source_revision": "rev",
                    "source_license": "MIT",
                    "source_id": "fixture",
                    "source_lineage": "lineage",
                    "content_sha256": content_hash,
                    "normalized_content_sha256": content_hash,
                    "token_count": 5,
                    "domain": "general",
                    "split": split,
                    "task_id": f"task:{prefix}:{index}",
                    "tree_id": f"tree:{prefix}:{index}",
                    "trajectory_id": f"trajectory:{prefix}:{index}",
                    "document_id": f"document:{prefix}:{index}",
                    "group_identity": f"group:{prefix}:{index}",
                    "text": text,
                }
            )
    corpus.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    freeze = tmp_path / "v23-freeze-receipt.json"
    freeze.write_text(
        json.dumps(
            {
                "artifacts": {
                    "v23-corpus.jsonl": {"path": corpus.name, "sha256": _sha(corpus)}
                }
            }
        ),
        encoding="utf-8",
    )
    activation = tmp_path / "v23-activation-plan.json"
    activation.write_text(
        json.dumps(
            {
                "status": "READY_FOR_BALANCED_CAPTURE",
                "selected_rows": [
                    {"id": "FIT-TRAIN-0", "split": "FIT-TRAIN"},
                    {"id": "FIT-TRAIN-1", "split": "FIT-TRAIN"},
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "capture" / "v23-data-plan.json"
    payload = prepare_v23_capture_plan(
        freeze_receipt=freeze,
        activation_plan=activation,
        output=output,
        train_tokens=8,
        dev_tokens=10,
    )

    assert payload["status"] == "V23_CAPTURE_PLAN_READY"
    assert payload["selected_tokens"] == {"FIT-TRAIN": 8, "FIT-DEV": 10}
    assert len(payload["FIT-TRAIN"]) == 2
    assert len(payload["FIT-DEV"]) == 2
    assert payload["source"]["sha256"] == _sha(corpus)
    assert payload["FIT-TRAIN"][0]["source_record_index"] == 0
