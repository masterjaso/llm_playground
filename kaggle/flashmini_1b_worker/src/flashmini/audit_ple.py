"""Measure head collisions on distinct n-grams, separately from repeated lookups."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import FlashMiniConfig
from .data import MemmapDataset, sha256_file
from .models.ple import PLE, PLEV3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", type=int, default=2048)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cfg = FlashMiniConfig.from_dict(yaml.safe_load(Path(args.config).read_text()))
    if args.sequences <= 0:
        raise ValueError("sequences must be positive")
    hash_config = copy.deepcopy(cfg.ple)
    hash_config.d_model = 4
    hash_config.head_dim = 1
    if cfg.architecture_version >= 3:
        hash_config.embed_dim = (hash_config.ngram - 1) * hash_config.heads_per_ngram
    memory = PLEV3(hash_config) if cfg.architecture_version >= 3 else PLE(hash_config)
    data = MemmapDataset(Path(args.data_dir), "val")
    selected_rows = np.linspace(0, len(data) - 1, min(args.sequences, len(data)), dtype=np.int64)
    tokens, _ = data.get_batch(selected_rows)
    keys = memory._ngram_keys(torch.tensor(tokens)).numpy()
    orders = ([n for n in range(2, cfg.ple.ngram + 1) for _ in range(cfg.ple.heads_per_ngram)]
              if cfg.architecture_version >= 3 else memory.orders.tolist())
    distinct = [{} for _ in orders]
    for batch, row in enumerate(tokens):
        context = []
        previous = None
        for position, token in enumerate(row):
            if (previous == cfg.ple.eos_id if cfg.architecture_version >= 3 else token == cfg.ple.eos_id):
                context = []
            context.append(int(token) + (0 if cfg.architecture_version >= 3 else 1))
            context = context[-cfg.ple.ngram:]
            previous = token
            for head, order in enumerate(orders):
                suffix = context[-order:]
                sentinel = (cfg.ple.eos_id or 0) if cfg.architecture_version >= 3 else 0
                exact = tuple([sentinel] * (order - len(suffix)) + suffix)
                key = int(keys[batch, position, head])
                previous_key = distinct[head].setdefault(exact, key)
                if previous_key != key:
                    raise AssertionError("Same n-gram produced different keys")
    heads = []
    for head, mapping in enumerate(distinct):
        contexts, rows = len(mapping), len(set(mapping.values()))
        heads.append({"head": head, "ngram": orders[head],
                      "capacity": memory.sizes[head], "distinct_contexts": contexts,
                      "distinct_rows": rows,
                      "context_collisions": contexts - rows,
                      "context_collision_fraction": (contexts - rows) / max(1, contexts)})
    signatures = []
    for order in range(2, cfg.ple.ngram + 1):
        selected = [h for h, n in enumerate(orders) if n == order]
        contexts = distinct[selected[0]]
        codes = {tuple(distinct[h][context] for h in selected) for context in contexts}
        signatures.append({"ngram": order, "heads": selected,
                           "distinct_contexts": len(contexts), "distinct_signatures": len(codes),
                           "full_signature_collisions": len(contexts) - len(codes)})
    result = {"purpose": "finite_validation_slice_hash_audit_not_universal_collision_proof",
              "combined_signatures": signatures,
              "sequences": len(tokens), "positions": int(tokens.size), "heads": heads,
              "sampling": "evenly_spaced_validation_sequences",
              "observed_eos_markers": int((tokens == cfg.ple.eos_id).sum()),
              "document_count_note": "EOS markers observed; sampled rows may contain partial documents",
              "ple_config": cfg.to_dict()["ple"],
              "architecture_version": cfg.architecture_version,
              "config_sha256": sha256_file(Path(args.config)),
              "data_manifest_sha256": sha256_file(Path(args.data_dir) / "data_manifest.json")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
