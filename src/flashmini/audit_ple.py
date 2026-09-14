"""Measure head collisions on distinct n-grams, separately from repeated lookups."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import FlashMiniConfig
from .data import MemmapDataset, sha256_file
from .models.ple import PLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", type=int, default=128)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cfg = FlashMiniConfig.from_dict(yaml.safe_load(Path(args.config).read_text()))
    memory = PLE(cfg.ple)
    data = MemmapDataset(Path(args.data_dir), "val")
    tokens, _ = data.get_batch(np.arange(min(args.sequences, len(data))))
    keys = memory._ngram_keys(torch.tensor(tokens)).numpy()
    distinct = [{} for _ in range(cfg.ple.num_heads)]
    for batch, row in enumerate(tokens):
        context = []
        for position, token in enumerate(row):
            if token == cfg.ple.eos_id:
                context = []
            context.append(int(token) + 1)
            context = context[-cfg.ple.ngram:]
            for head, order in enumerate(memory.orders.tolist()):
                suffix = context[-order:]
                exact = tuple([0] * (order - len(suffix)) + suffix)
                key = int(keys[batch, position, head])
                previous = distinct[head].setdefault(exact, key)
                if previous != key:
                    raise AssertionError("Same n-gram produced different keys")
    heads = []
    for head, mapping in enumerate(distinct):
        contexts, rows = len(mapping), len(set(mapping.values()))
        heads.append({"head": head, "ngram": int(memory.orders[head]),
                      "capacity": memory.sizes[head], "distinct_contexts": contexts,
                      "distinct_rows": rows,
                      "context_collisions": contexts - rows,
                      "context_collision_fraction": (contexts - rows) / max(1, contexts)})
    signatures = []
    for order in range(2, cfg.ple.ngram + 1):
        selected = [h for h, n in enumerate(memory.orders.tolist()) if n == order]
        contexts = distinct[selected[0]]
        codes = {tuple(distinct[h][context] for h in selected) for context in contexts}
        signatures.append({"ngram": order, "heads": selected,
                           "distinct_contexts": len(contexts), "distinct_signatures": len(codes),
                           "full_signature_collisions": len(contexts) - len(codes)})
    result = {"purpose": "finite_validation_slice_hash_audit_not_universal_collision_proof",
              "combined_signatures": signatures,
              "sequences": len(tokens), "positions": int(tokens.size), "heads": heads,
              "config_sha256": sha256_file(Path(args.config)),
              "data_manifest_sha256": sha256_file(Path(args.data_dir) / "data_manifest.json")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
