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
import platform
import time
from importlib import metadata
from pathlib import Path

from .data import prepare_dataset, prepare_streaming_documents

# These are immutable Hub commits resolved during the v3 hardening pass.  The
# CLI still permits an explicit override, but a bare v3 command is pinned and
# cannot silently follow a moving ``main`` branch.
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def stream_fineweb_tokens(
    tokenizer,
    max_tokens: int,
    sample: str = "CC-MAIN-2013-20",
    max_docs: int = 2000,
    seed: int = 0,
):
    """Stream FineWeb-Edu documents, tokenize, and yield token IDs.

    Legacy v2 compatibility: source order is retained and seed has no effect
    here. New experiments must use the pinned v3 document partition pipeline,
    where the recorded seed/salt controls deterministic split assignment.
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


def stream_fineweb_documents(
    tokenizer,
    *,
    sample: str,
    dataset_revision: str,
    max_docs: int | None = None,
    max_source_tokens: int | None = None,
):
    """Yield one complete FineWeb-Edu document at a time for v3 preparation.

    The dataset revision is mandatory here so an interrupted/repeated
    preparation cannot accidentally mix moving upstream data.  No EOS is
    yielded; ``prepare_streaming_documents`` adds exactly one marker per
    accepted document after deduplication.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )
    docs = 0
    body_tokens = 0
    for record in ds:
        if max_docs is not None and docs >= max_docs:
            break
        text = record.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        if max_source_tokens is not None and body_tokens + len(ids) > max_source_tokens:
            break
        body_tokens += len(ids)
        docs += 1
        yield ids


def prepare_fineweb_v3(
    out_dir: Path,
    *,
    target_train_tokens: int,
    seq_len: int,
    sample: str = "CC-MAIN-2013-20",
    max_docs: int | None = None,
    max_source_tokens: int | None = None,
    tokenizer_name: str = "gpt2",
    tokenizer_revision: str = GPT2_REVISION,
    dataset_revision: str = FINEWEB_EDU_REVISION,
    seed: int = 0,
    split_salt: str | None = None,
    val_fraction: float = 0.02,
) -> dict:
    """Freeze a fresh-token v3 FineWeb corpus with pinned provenance."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("v3 preparation requires a tokenizer with an EOS token")
    documents = stream_fineweb_documents(
        tokenizer,
        sample=sample,
        dataset_revision=dataset_revision,
        max_docs=max_docs,
        max_source_tokens=max_source_tokens,
    )
    package_versions = {}
    for package in ("datasets", "transformers", "tokenizers", "huggingface-hub", "numpy"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    manifest = prepare_streaming_documents(
        documents,
        Path(out_dir),
        seq_len=seq_len,
        eos_id=int(tokenizer.eos_token_id),
        target_train_tokens=target_train_tokens,
        max_source_tokens=max_source_tokens,
        max_docs=max_docs,
        seed=seed,
        split_salt=split_salt,
        val_fraction=val_fraction,
        provenance={
            "dataset_id": "HuggingFaceFW/fineweb-edu",
            "dataset_config": sample,
            "dataset_split": "train",
            "dataset_revision": dataset_revision,
            "tokenizer_id": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
            "streaming_order": "datasets_streaming_fixed_shard_order_v1",
            "package_versions": package_versions,
            "python_version": platform.python_version(),
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--format-version", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--target-train-tokens", type=int)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--sample", default="CC-MAIN-2013-20")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--tokenizer", default="gpt2")
    # ``None`` keeps the historical v2 default (moving ``main``) while v3
    # selects immutable defaults below.  Existing v2 semantics must not be
    # silently rewritten by this hardening pass.
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-salt")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-source-tokens", type=int)
    args = parser.parse_args()

    if args.format_version == 3:
        target = args.target_train_tokens
        if target is None:
            raise ValueError("--target-train-tokens is required for v3 preparation")
        max_docs = None if args.max_docs in (None, 0) else args.max_docs
        print(
            f"Preparing v3 FineWeb-Edu {args.dataset_revision or FINEWEB_EDU_REVISION} "
            f"with tokenizer {args.tokenizer}@{args.tokenizer_revision or GPT2_REVISION}; "
            f"target train tokens={target}..."
        )
        manifest = prepare_fineweb_v3(
            Path(args.out_dir),
            target_train_tokens=target,
            seq_len=args.seq_len,
            sample=args.sample,
            max_docs=max_docs,
            max_source_tokens=args.max_source_tokens,
            tokenizer_name=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision or GPT2_REVISION,
            dataset_revision=args.dataset_revision or FINEWEB_EDU_REVISION,
            seed=args.seed,
            split_salt=args.split_salt,
            val_fraction=args.val_fraction,
        )
        manifest_path = Path(args.out_dir) / "data_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return 0

    from transformers import AutoTokenizer

    max_tokens = args.max_tokens if args.max_tokens is not None else 30_000_000
    max_docs = args.max_docs if args.max_docs is not None else 2000
    print(f"Loading tokenizer {args.tokenizer}@{args.tokenizer_revision or 'main'}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision or "main")
    eos_id = tokenizer.eos_token_id

    print(f"Streaming FineWeb-Edu sample={args.sample} up to {max_tokens} tokens...")
    t0 = time.time()
    token_stream = stream_fineweb_tokens(
        tokenizer,
        max_tokens,
        sample=args.sample,
        max_docs=max_docs,
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
    manifest["tokenizer_revision"] = args.tokenizer_revision or "main"
    manifest["dataset"] = "HuggingFaceFW/fineweb-edu"
    manifest["dataset_sample"] = args.sample
    manifest["requested_max_tokens"] = max_tokens
    manifest["max_docs"] = max_docs
    manifest["elapsed_seconds"] = time.time() - t0
    manifest_path = Path(args.out_dir) / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
