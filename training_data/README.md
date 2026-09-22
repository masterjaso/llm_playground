# FlashMini training data (v4)

One logical master data lake, multiple deterministic training views. The
workstation never holds the whole corpus: bounded local cache (default 20-40
GiB via `FLASHMINI_DATA_CACHE_GB` / `FLASHMINI_DATA_CACHE_DIR`), stream one
window, validate, shard to Parquet+ZSTD, upload to Hugging Face, verify
remote, evict local, continue. Disk watermark stops ingestion below ~10 GiB
free. Never delete outside the cache/work dir.

v3 (`src/flashmini/data.py`) is frozen for PoC reproduction; v4 is additive.

## Quick start

```bash
flashmini-data auth-check
flashmini-data source-list
flashmini-data source-probe --all --limit 3
flashmini-data plan --recipe training_data/recipes/flashmini_1b_full_v1.yaml
flashmini-data build --recipe training_data/recipes/flashmini_1b_foundation_v1.yaml --max-docs 5000 --no-publish
flashmini-data status
flashmini-data freeze --recipe training_data/recipes/flashmini_1b_foundation_v1.yaml
flashmini-data verify --recipe training_data/recipes/flashmini_1b_foundation_v1.yaml
flashmini-data train-smoke
flashmini-data cache-status
```

## Production stream operator loop

The 1B production command writes only bounded staging and publishes verified
progress under `releases/pretrain-production-v1/1b/canonical`:

```bash
flashmini-data build --production \
  --recipe training_data/recipes/flashmini_1b_full_v1.yaml \
  --source-lock training_data/registry/source_snapshot.lock.json \
  --tokenizer-spec training_data/tokenizer/production.yaml \
  --contamination-config training_data/eval/contamination_sources.yaml \
  --hf-repo mjaso/flashmini-data-v1 \
  --hf-prefix releases/pretrain-production-v1/1b/canonical \
  --upload-workers 1 --max-pending-shards 16 --window 4096 --shard-docs 256 \
  --free-space-watermark-gib 10
flashmini-data resume --production --recipe training_data/recipes/flashmini_1b_full_v1.yaml \
  --state <state.json> --out-dir <staging-dir> --cache-dir <managed-cache>
flashmini-data progress --state <state.json> --cache-dir <managed-cache>
flashmini-data cache-status --cache-dir <managed-cache>
flashmini-data hf-audit --repo mjaso/flashmini-data-v1
```

When immutable benchmark/private-eval exports arrive, run `flashmini-data
overlay` against the published canonical state. It downloads and filters one
remote shard at a time, publishes a separate decontaminated view prefix, and
never rewrites the canonical release.

For a controlled operational slice, add `--max-records`, `--max-docs`, and an
optional `--source-ids` allowlist. These limits pause after a bounded chunk;
they never redefine the exact recipe target. A verified shard is evicted
immediately, while cursors, SQLite dedupe state, and release progress remain.

`--max-pending-shards` controls the number of shards in one atomic Hub commit.
Production runs use 16 to avoid the repository commit rate limit; a partial
batch remains local until every member is remotely verified.

## HF auth

Set `HF_TOKEN` (preferred; `HUGGINGFACE_HUB_TOKEN` also honored, plus `.env`
fallback without printing). Never commit the token. `auth-check` calls
`HfApi().whoami()` and derives `<HF_USER>` for repo slugs
`<HF_USER>/flashmini-data-v1` (public) and `<HF_USER>/flashmini-eval-v1`
(private, small, isolated).

## Adding a source

1. Add entry to `training_data/registry/sources.yaml` with `dataset_id`,
   `domain`, `redistribution_class` (mirror_allowed | recipe_only |
   gated_recipe_only | review_required | generated_owned).
2. `flashmini-data source-probe --sources <id>` then `source-lock`.
3. Decisive corpora require immutable 40-hex `revision` (no `main`).

## Building a recipe / resuming

Recipes in `training_data/recipes/` declare domain weights summing to 1.0.
`build` checkpoints cursors + hashes to `training_data/manifests/build_state.json`
(atomically) after every source window; rerun the same command or
`flashmini-data resume --recipe ...` to continue. Gated/missing sources record
`SOURCE_BLOCKED` and continue.

## Training from HF

Canonical shards (`Parquet+ZSTD`) + `corpus_manifest.json` (shard hashes,
recipe hash, `corpus_fingerprint_sha256`) feed `RemoteShardDataset`:
deterministic shard permutation -> within-shard permutation -> batches, with
`sampler_state()` / `restore_sampler_state()` for exact resume. Tokenizer stays
independent: record `tokenizer_id@revision` per run; see `tokenizer/`.

## Cache

`cache-status` / `cache-prune`; LRU eviction, atomic download/rename,
checksum-before-use, never evict in-use shard, watermark stop.

## Provenance / licensing

Every source carries `redistribution_class`. Only `mirror_allowed` and
`generated_owned` content bytes are published; everything else stores
pinned recipe/provenance (fail-closed). Dataset cards retain source licenses;
no relicensing under the code license. See `SOURCES.md`, `ARCHITECTURE.md`.

## Current corpus-v1 status

The pilot remains immutable.  Re-audit the live Hub tree before using it:

```bash
flashmini-data hf-audit --repo mjaso/flashmini-data-v1 \
  --revision e83398462169164d9e4127627ad4f72d95b05a41
```

At that pinned revision the live inventory contains 85 `shards/*.parquet`
files and 15 `clean/*.parquet` files.  The remote manifest contains 95 rows,
83 marked published, 91,173 published-manifest documents, and
1,714,354,474 estimated published tokens.  It has no exact production-tokenizer
counts.  Six shard size/hash mismatches, one unpublished remote shard, and
one unmanifested remote shard are recorded in the run audit.  These are pilot
release findings; production code never mutates or silently reuses the pilot.

The production release must use a new versioned prefix or dataset repository,
an immutable Hub revision, exact tokenizer counts, and a separate frozen
training-view manifest.  Machine-readable readiness facts live in
`production_release_status.json`; the blocked target descriptors are under
`manifests/releases/`. See `PRODUCTION_READINESS.md` for the gate contract.
