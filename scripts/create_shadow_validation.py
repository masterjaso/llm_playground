"""Create an untouched validation-B identity from FIT activation rows.

The command writes only a deterministic index receipt.  It does not copy
activations, train a model, or read the holdout manifest.  Evaluators can pass
``selected_global_indices`` to ``ActivationShardDataset.iter_selected_batches``
after the router is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep this receipt-only utility runnable from a fresh checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense2moe.provenance import current_git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-a", type=Path, required=True, help="architecture-dev/validation-A identity JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-count", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    # Delay the training package import until after argument parsing so this
    # receipt-only utility can still show help in a minimal checkout without
    # optional PyTorch installed.
    from dense2moe.training import deterministic_shadow_validation_indices

    train_payload = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    validation_payload = json.loads(args.validation_a.read_text(encoding="utf-8"))
    validation_indices = validation_payload.get("selected_global_indices")
    if not isinstance(validation_indices, list):
        raise TypeError("validation-A receipt must contain selected_global_indices")
    count = int(train_payload.get("count", 0))
    selected, identity_hash = deterministic_shadow_validation_indices(
        count,
        excluded_indices=[int(value) for value in validation_indices],
        shadow_count=args.shadow_count,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "status": "SHADOW_VALIDATION_READY",
        "classification": "FIT_ONLY_UNTOUCHED_VALIDATION_B",
        "train_manifest": str(args.train_manifest),
        "train_dataset_hash": train_payload.get("dataset_hash"),
        "validation_a_identity_hash": validation_payload.get("selected_row_key_hash"),
        "validation_a_count": len(validation_indices),
        "selected_global_indices": list(selected),
        "selected_count": len(selected),
        "selected_identity_hash": identity_hash,
        "selection_method": "sha256_ranked_train_rows_excluding_validation_a",
        "selection_seed": int(args.seed),
        "holdout_opened": False,
        "code_commit": current_git_commit(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_count": payload["selected_count"], "identity_hash": identity_hash, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
