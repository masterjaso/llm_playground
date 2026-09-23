# FlashMini production pretraining data

This branch owns the tokenizer-independent canonical lake and the shared
training-view machinery for the 1B and future 50B base models.  It does not
contain SFT, DPO, GRPO, RLHF, or assistant/chat data.

## Release identity

`mjaso/flashmini-data-v1` is the frozen pilot and is never overwritten by the
production release.  Production artifacts use a versioned prefix/repository,
an immutable Hub revision, a source-lock hash, and separate canonical and
tokenized-view fingerprints.

The live pilot audit at revision
`e83398462169164d9e4127627ad4f72d95b05a41` found 85 `shards/*.parquet` files,
15 `clean/*.parquet` files, 95 manifest rows, 83 rows marked published,
91,173 manifest-published documents, and 1,714,354,474 estimated published
tokens.  The manifest has no exact production-tokenizer counts.  Six remote
shards disagree with the manifest size/hash and one unpublished shard plus one
unmanifested shard are present remotely.  These facts are recorded in the
run-scoped audit evidence; they are not silently repaired in corpus-v1.

## Pipeline

1. Source registry and `source_snapshot.lock.json` pin dataset, config, split,
   revision, license, and redistribution class.
2. Canonicalization and conservative filters produce stable document IDs.
3. Exact SHA dedupe uses a durable SQLite index; MinHash/LSH near-dedupe uses a
   second SQLite index, never performs an all-pairs comparison, and retains at
   most 4,096 deterministic word-5 shingles per document.
4. A deficit-aware scheduler consumes exact tokenizer tokens by domain. Source
   cursors include config/split/revision and native file/row-group/row-index
   fields; generic HF streaming retains a bounded offset fallback. Source
   streams pass the repository Hub token and use the managed dataset cache.
5. Canonical shards are Parquet+ZSTD and roll over by physical-size/token
   limits.  Manifests record checksums, distributions, provenance, and exact
   token totals when a frozen tokenizer is supplied.
6. Canonical production ingestion is a bounded stream-to-Hub loop: a completed
   shard is uploaded, checked for remote size and SHA/LFS identity, committed to
   resumable state, and evicted locally. `BUILD_STATE.json` is published in the
   release prefix after verified progress. The managed source cache is bounded
   and the builder stops at its free-space watermark.
7. The production tokenizer is pinned in `tokenizer/production.yaml` to the
   existing GPT-2 commit used by the validated FlashMini data path.  The
   tokenizer-specific representation is a contiguous `uint16`/`uint32` token
   store with document offsets.  Packing combines short documents with EOS
   boundaries and consumes every token from long documents.
8. Distributed sampling is a deterministic shard/document permutation with a
   fail-closed resume contract.  The bounded cache verifies checksums before
   atomic install, protects active mmap files, evicts by LRU, and exposes
   prefetch/download telemetry.

## Curriculum contracts

The full 1B recipe is the exact aggregate of its 80B/15B/5B stage contracts:
100B train tokens with 100M additional validation tokens.  The full 50B recipe
is the exact aggregate of its 6.5T/1T/0.5T stages: 8T train tokens with a
bounded 200M validation contract.  `recipes.load_recipe` rejects aggregate
drift; document limits are operational chunking only.

Validation membership is globally excluded from train and bounded by an
explicit token budget.  Benchmark decontamination uses exact hashes and
MinHash/LSH candidates from `eval/contamination_sources.yaml`; benchmark names
alone do not remove ordinary prose. The committed rules file may be partial
during canonical ingestion, while final freeze still requires every immutable
benchmark/private-eval export under `corpus_files`.

## Readiness

`FLASHMINI_1B_DATA_READY` remains false until source locks, exact counts,
dedupe/decontamination, tokenized artifacts, remote checksums, deterministic
sampling/resume, cache/prefetch smoke, the exact 100B view, and private-eval
decontamination all pass. Canonical ingestion may run before that gate; the
current bounded continuation has published 1,039,496 exact train tokens across
12 verified shards and retained no completed shard locally. The live release
prefix is `releases/pretrain-production-v1/1b/canonical`, with progress at
Hub revision `31a09d868966466a90459678b5eba9532267d02c`. Approved windows
cover FineWeb-Edu, FineWeb, public-domain books, FineMath, Open-Web-Math, and
Cosmopedia. The pinned `stack_edu` code source is retired because its Python
configuration exposes metadata without a configured `content` field. The
production 1B view uses the pinned, content-bearing `stackv2_edu` revision
upstream-only; review-required and gated sources remain held from mirroring.
`FLASHMINI_50B_DATA_PIPELINE_READY` is a code-and-small-scale proof state;
`FLASHMINI_50B_DATA_MATERIALIZED_READY` requires the actual 8T artifacts. The
50B canonical stream is paused at the user's request at
`releases/pretrain-production-v1/50b/canonical`; its latest durable checkpoint
is 3,941 verified shards, 958,979 documents, and 1,390,902,314 exact train tokens.
The sixteen shards that hit a transient DNS failure were preserved locally,
retried, and verified together in one atomic batch. Four later local staging
shards remain preserved but unverified for a future resumable continuation.
Production now uses 16-shard Hub commits and bounded 4,096-record windows.
This is durable progress, not completion: code/reasoning inputs remain held or
schema-blocked, Nemotron inputs are gated, and the final tokenized 8T view is
not materialized. The workstation has neither the remote CPU/storage
allocation nor the exact-token corpus needed to claim a finished release. A
bounded exact-token slice reached 397 documents and 149,487 publishable tokens
in 108 seconds with 5.56 GiB peak RSS; this is path evidence, not production
capacity.

The blocked target descriptors are recorded in
`manifests/releases/flashmini-1b-100b.preflight.json` and
`manifests/releases/flashmini-50b-8t.preflight.json`; they are not substitutes
for the final manifests produced after materialization.
