# Manifests (v4)

`build_state.json` (resumable, atomic) and `corpus_manifest.json` (frozen:
shard hashes, recipe hash, `corpus_fingerprint_sha256`) live here. Small
indexes only; never bulk data in Git.

## Current release: corpus-v1 (live build)

| Field | Value |
|---|---|
| Recipe | `flashmini_1b_full_v1` (hash `6a38be9e15cc…`) |
| HF repo | `mjaso/flashmini-data-v1` @ `e83398462169…` |
| Shards | 91 published + verified (indices 0–96, dedup applied) |
| Documents | 99,728 (train 99,220 / val 508, 0.51%) |
| Published bytes | ~2.88 GB |
| Estimated tokens | ~2.01 B |
| Fingerprint | `ae3a4dd74c761d4ce2c63811f40a9b3b0f78a99f0334cea033f17311eaf662fe` |
| Verify | recipe_match=True, 91/91 shard hashes valid |
| Train-smoke | 99,220 sequences, integrity valid, 1 download from HF |
| Resume-check | identical=True (resume == uninterrupted continuation) |

### Dedup applied

6 duplicate shard entries (000084, 000086, 000088, 000089, 000093, 000095)
were removed — the build overwrote their HF content with newer shards, so
only the newest entry (matching HF) is kept.

### Known gaps

- 12 shards from the original build (000002, 000011, 000019, …, 000091) were
  lost during interrupted builds and cannot be exactly re-derived.
- 6 held shards (000085-held, 000087-held, 000090-held, 000092-held,
  000094-held, 000096-held) are recipe-only (stackv2_edu / openthoughts) and
  not published to HF.

### Per-document split

The exact train/val split is stored per document in each parquet file's
`split` column (derived from `assign_split(did, salt="flashmini-v4-split-v1")`
with `val_fraction=0.005`). Exact counts were read from every published
shard on HF and recorded in `corpus_manifest.json.split_distribution` and
`split_totals`. Validation membership is stable regardless of source
iteration order.

### Scale-up

The build loop continues from `build_state.json` cursors toward the
100B-token recipe target. Source cursors were jumped to fresh regions
(+50K) to avoid duplicate-heavy overlap between fineweb_edu and fineweb.



