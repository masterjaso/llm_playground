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
