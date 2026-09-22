# Manifests (v4)

`build_state.json` (resumable, atomic) and `corpus_manifest.json` (frozen
shard hashes, recipe hash, and corpus fingerprint) are runtime indexes. They
are not bulk data and are gitignored.

## Current release: corpus-v1 (pilot, immutable)

The live Hub tree at revision `e83398462169164d9e4127627ad4f72d95b05a41`
contains 85 `shards/*.parquet` files and 15 `clean/*.parquet` files. The
remote manifest has 95 rows (83 marked published), 91,173 published-manifest
documents, 2,619,337,547 published bytes, and 1,714,354,474 estimated tokens.
It does not carry exact production-tokenizer counts. Six remote shard
metadata mismatches, one unpublished remote shard, and one unmanifested shard
are retained as pilot audit findings. Do not use old hand-entered counts as
production evidence.

The pilot release is preserved. Production views must use a new versioned
prefix or repository, an immutable Hub revision, exact tokenizer counts, and a
separate frozen training-view manifest.

## Production scale-up

Production builds use a new release identity, SQLite exact and near-dedupe
indexes, the deficit-aware exact-token scheduler, explicit validation budgets,
and a frozen tokenizer/token-store contract. `corpus-v1` state is never
overwritten. See `training_data/PRODUCTION_READINESS.md` for the full gate
contract and recovery rules.

The `releases/*.preflight.json` files are immutable target descriptors for the
blocked, not-yet-materialized views. They intentionally leave Hub revisions,
corpus/view fingerprints, shard lists, and source exact totals empty until the
remote build and verification gates succeed.
