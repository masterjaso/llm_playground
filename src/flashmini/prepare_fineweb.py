"""Prepare FineWeb-Edu frozen dataset for FlashMini.

Streams a slice of FineWeb-Edu, tokenizes with a frozen tokenizer, packs
deterministically, and stores memory-mapped shards on local NVMe. Records
manifest hashes for reproducibility.

Usage:
    python -m flashmini.prepare_fineweb --out-dir data/fineweb --max-tokens 30000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import prepare_dataset


def stream_fineweb_tokens(
    tokenizer,
    max_tokens: int,
    sample: str = "CC-MAIN-2013-20",
    max_docs: int = 2000,
    seed: int = 0,
):
    """Stream FineWeb-Edu documents, tokenize, and yield token IDs.

    Uses a deterministic document ordering (fixed sample + seed) so reruns
    produce the same token sequence.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample,
        split="train",
        streaming=True,
    )
    total = 0
    count = 0
    for doc in ds:
        if count >= max_docs:
            break
        text = doc.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Add EOS to mark document boundary
        for t in ids:
            yield t
            total += 1
            if total >= max_tokens:
                return
        yield tokenizer.eos_token_id
        total += 1
        count += 1
        if total >= max_tokens:
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=30_000_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--sample", default="CC-MAIN-2013-20")
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer {args.tokenizer}@{args.tokenizer_revision}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    eos_id = tokenizer.eos_token_id

    print(f"Streaming FineWeb-Edu sample={args.sample} up to {args.max_tokens} tokens...")
    t0 = time.time()
    token_stream = stream_fineweb_tokens(
        tokenizer,
        args.max_tokens,
        sample=args.sample,
        max_docs=args.max_docs,
        seed=args.seed,
    )
    manifest = prepare_dataset(
        token_stream,
        Path(args.out_dir),
        seq_len=args.seq_len,
        eos_id=eos_id,
        seed=args.seed,
        val_fraction=args.val_fraction,
        document_split=True,
    )
    manifest["tokenizer"] = args.tokenizer
    manifest["tokenizer_revision"] = args.tokenizer_revision
    manifest["dataset"] = "HuggingFaceFW/fineweb-edu"
    manifest["dataset_sample"] = args.sample
    manifest["requested_max_tokens"] = args.max_tokens
    manifest["max_docs"] = args.max_docs
    manifest["elapsed_seconds"] = time.time() - t0
    manifest_path = Path(args.out_dir) / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
