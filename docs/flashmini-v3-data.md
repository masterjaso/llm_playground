<!-- nsp:meta
id: docs.flashmini-v3-data
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-v3-data.md
graphTags: flashmini,v3,training
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-14
replaces:
replacedBy:
-->

# FlashMini v3 frozen data

Version 3 uses a fresh, frozen FineWeb-Edu stream.  The source revision and
tokenizer revision are immutable Hub commits recorded in
`data_manifest.json`; do not use `main` for an official run.

The current pinned inputs are:

| Input | Identity |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu`, sample `CC-MAIN-2013-20` |
| Dataset revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Tokenizer | `gpt2` |
| Tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |

The prepared corpus contains **2,000,000,289 scored training tokens** and
**39,577,129 scored validation tokens**, occupying about 15.3 GiB including the
deduplication database. The preparer observed 1,930,744 documents and removed
15 exact tokenized-document duplicates before splitting. The resulting
1,892,346 training and 38,383 validation documents have disjoint deduplication
keys. This does not guarantee absence of near duplicates or shared passages.

Training budgets of 100M, 250M and 1B imply approximately 0.05, 0.125 and 0.5
passes respectively. The exact manifest, shard hashes and verification scope
are in [the data validation record](validation/flashmini-v3-data.json).
Manifest SHA-256:
`b06f559f65e6969a9dae36d392873610330b8a2425fe27f47518c4d200d42dab`.

The preparation completed its files and manifest, but the upstream streaming
library stalled during HTTP teardown. The preparer was terminated only after
an independent full integrity check passed. This was not a clean process exit;
the completed corpus does not need to be regenerated.

Prepare the 2B scored-token corpus with:

```bash
.venv/bin/python -m flashmini.prepare_fineweb \
  --format-version 3 \
  --out-dir data/fineweb_v3_2b \
  --target-train-tokens 2000000000 \
  --seq-len 256 \
  --sample CC-MAIN-2013-20 \
  --max-docs 0 \
  --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --tokenizer gpt2 \
  --tokenizer-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --seed 17 \
  --split-salt flashmini-v3-frozen-seed17 \
  --val-fraction 0.02
```

For a capacity-constrained 250M PoC, 500M scored tokens are sufficient, but
the later 1B-token stage needs the larger corpus. A preparation
stops only at a complete document boundary, so the scored train count may be a
little above the requested target.  `--max-source-tokens` and `--max-docs` are
optional bounded-stream controls; `--max-docs 0` means no document-count cap.

The preparer first writes each tokenized document to one of two raw staging
streams.  A SQLite `WITHOUT ROWID` table keyed by the SHA-256 of the complete
tokenized document removes exact duplicates before a salted SHA-256 threshold
partition assigns the document to train or validation.  Therefore a document
cannot occur in both splits, and the split does not depend on the order in
which a document happens to arrive.  The SQLite table is retained as compact
audit evidence in `document_dedupe.sqlite3`.

Without an explicit salt, `seed` determines the salt. An explicit `--split-salt`
overrides this derivation and is the authoritative partition input; changing
only the seed while supplying the same salt does not change the split. Source
order is pinned, not shuffled. Deduplication removes exact tokenized documents
only; it does not detect near duplicates, shared passages or semantic overlap.

The final `*_input.npy` and `*_labels.npy` shards are compact int32 memmaps.
`MemmapDataset.get_batch()` casts only the requested batch to int64 for the
existing model and loss interfaces.  The manifest records raw and scored token
counts, document counts, duplicate count, split salt and algorithm versions,
provenance, the dedupe database hash, and each shard hash.  The staging raw
streams are removed after successful packing.

The resulting v3 data directory is a generated artifact and must remain
gitignored.  Keep its manifest, shard hashes and pinned command with the run
metadata; do not commit the dataset itself.

Verify all shard/database hashes, shapes, scored-token counts and nonempty-label
rows before an official run:

```bash
.venv/bin/python -c 'from pathlib import Path; from flashmini.data import verify_dataset_integrity; print(verify_dataset_integrity(Path("data/fineweb_v3_2b")))'
```

Decisive training and comparison run this integrity check themselves. Dataset
preparation omits any final sequence with no scored labels and never truncates
an admitted document to satisfy a source-token cap.
