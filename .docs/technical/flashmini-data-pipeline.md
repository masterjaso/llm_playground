<!-- nsp:meta
id: technical.flashmini-data-pipeline
kind: technical-doc
audience: developer
status: active
description: Specify the resumable v4 source, deduplication, sharding, release, and token-store pipeline.
resource: ezra://technical/flashmini-data-pipeline
scope: technical
persona: release-operations
source: model
confidence: high
reviewStatus: unreviewed
graphNode: technical:flashmini-data-pipeline
graphTags: flashmini,data,provenance,release
validation: context-header-audit,manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# FlashMini v4 data pipeline

## Purpose

Describe the tokenizer-independent canonical lake and deterministic training
views used by the production data operator. The v4 path is additive to the
frozen v3 preparation path and is responsible for provenance, bounded storage,
resumable ingestion, release verification, and tokenizer-specific materialization.

## Feature links

- [Production training-data releases](../features/flashmini-production-data.md)
- [Validation and release gates](../features/flashmini-validation-release.md)

## Architecture / design

- The registry and immutable source lock identify dataset, configuration, split,
  revision, license, and redistribution class. Recipes declare exact domain
  targets and stage aggregates.
- Source windows pass through canonicalization, exact SQLite deduplication,
  bounded MinHash/LSH near-deduplication, and benchmark exclusion before
  deficit-aware scheduling and Parquet+ZSTD shard rollover.
- Build state records cursors, hashes, source status, release prefix, and verified
  progress. Production mode uploads bounded shard batches, verifies remote size
  and identity, then evicts completed local shards under cache and disk-watermark
  policy.
- Materialization uses the pinned tokenizer to produce contiguous token stores
  with document offsets. Remote-shard sampling applies deterministic
  shard/document permutations and persists exact resume state.
- Overlay creates a separate decontaminated view from published canonical shards;
  it never rewrites the canonical release.

## Invariants

- Decisive sources require immutable revisions; mutable aliases cannot satisfy a
  frozen release identity.
- Recipe stage aggregates must equal the declared 1B or 50B total; operational
  record or document limits only bound a resumable chunk.
- Validation membership is excluded from training and bounded by an explicit
  token budget. Exact and near-deduplication state survives restart.
- Local cleanup is limited to managed cache/staging, protects active shards, and
  verifies checksums before atomic install or eviction.
- Redistribution rules fail closed: publishable bytes and provenance-only sources
  are kept distinct, and licenses remain attached to the source evidence.

## Code anchors

- repo://src/flashmini/data_v4/registry.py
- repo://src/flashmini/data_v4/source.py
- repo://src/flashmini/data_v4/recipes.py
- repo://src/flashmini/data_v4/filters.py
- repo://src/flashmini/data_v4/dedupe.py
- repo://src/flashmini/data_v4/build.py
- repo://src/flashmini/data_v4/manifests.py
- repo://src/flashmini/data_v4/hf_store.py
- repo://src/flashmini/data_v4/cache.py
- repo://src/flashmini/data_v4/remote_dataset.py
- repo://src/flashmini/data_v4/materialize.py
- repo://src/flashmini/data_v4/overlay.py
- repo://src/flashmini/data_v4/tokenizer.py
- repo://src/flashmini/data_v4/cli.py
- repo://training_data/registry/source_snapshot.lock.json
- repo://training_data/production_job_spec.yaml

## Validation

- `python -m pytest tests/flashmini/test_data_v4.py tests/flashmini/test_data_v4_production.py`
- `python -m pytest tests/flashmini/test_v3_data.py`
- `flashmini-data source-lock-validate`
- `flashmini-data verify --recipe <recipe>`
- `repo://training_data/PRODUCTION_READINESS.md` records the release gate
  contract and current blockers.

## Risks

- The preserved pilot has remote manifest/hash mismatches and estimated rather
  than exact production-tokenizer counts; it must not be silently reused.
- The 1B view remains blocked on exact final counts and private-evaluation
  decontamination. The 50B stream is paused and not materialized.
- Held, gated, or schema-incompatible sources remain visible in state and are
  not replaced with unpinned data.
