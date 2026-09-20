# Manifests (v4)

`build_state.json` (resumable, atomic) and `corpus_manifest.json` (frozen:
shard hashes, recipe hash, `corpus_fingerprint_sha256`) live here. Small
indexes only; never bulk data in Git.

## Current release: corpus-v1

| Field | Value |
|---|---|
| Recipe | `flashmini_1b_full_v1` (hash `6a38be9e15cc…`) |
| HF repo | `mjaso/flashmini-data-v1` @ `f13e9cee67eb…` |
| Shards | 84 published + verified (indices 0–95 minus 12 lost) |
| Documents | 92,373 (train 91,907 / val 466, 0.50%) |
| Published bytes | ~2.70 GB |
| Estimated tokens | ~1.76 B |
| Fingerprint | `af4439de1e1a6d25c97759cba67947397f39740cb7d99b61df22453c9bd74aa5` |
| Verify | recipe_match=True, 84/84 shard hashes valid |
| Train-smoke | 91,907 sequences, integrity valid, consumed across shard boundaries |
| Resume-check | identical=True (resume == uninterrupted continuation) |

### Known gap

12 shards (indices 000002, 000011, 000019, 000027, 000035, 000043, 000051,
000059, 000067, 000075, 000083, 000091) were built but never published —
their local parquet files were evicted and no remote copy exists. They
represent ~13,900 docs (~1.5% of corpus) of finemath + stackv2_edu content.
Source cursors have advanced past the point where these shards were
created, so they cannot be exactly re-derived. The gap is documented here
and accepted for this release.

### Per-document split

The exact train/val split is stored per document in each parquet file's
`split` column (derived from `assign_split(did, salt="flashmini-v4-split-v1")`
with `val_fraction=0.005`). Exact counts were read from every published
shard on HF and recorded in `corpus_manifest.json.split_distribution` and
`split_totals`. Validation membership is stable regardless of source
iteration order.


