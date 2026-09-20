# Manifests (v4)

`build_state.json` (resumable, atomic) and `corpus_manifest.json` (frozen:
shard hashes, recipe hash, `corpus_fingerprint_sha256`) live here. Small
indexes only; never bulk data in Git.

## Releases

| Release | Namespace | State | Gating | Status |
|---|---|---|---|---|
| pilot-1 | `shards/` | `build_state.json` | shard-level (first-document class) | frozen, 95 shards / 91k docs / ~1.71B tokens published |
| clean-1 | `clean/` | `build_state_clean.json` | **document-level** (per-class shards) | building (resumable) |

The pilot-1 release was produced before per-class partitioning existed, so a
published pilot shard may contain `review_required` documents mixed with
`mirror_allowed` ones. Treat pilot-1 as an infrastructure/mixture pilot;
use clean-1 (or later) for decisive training. `build.py` now partitions each
buffer by `redistribution_class` and writes separate `*-held` shards for
non-publishable classes, so only document-level-allowed content is uploaded.


- Recipe `flashmini_1b_full_v1` (hash `6a38be9e15cc…`), 2800 docs selected,
  4 canonical Parquet+ZSTD shards, ~36.1M estimated tokens, 55.0 MiB.
- Domains covered: books_reference, general_web, code, math_stem,
  synthetic_edu, tech_docs, multilingual. Rejections: too_short 414,
  too_long 56, boilerplate 1, too_few_words 3; exact duplicates 0.
- `freeze` fingerprint `b20baa94e449af32…107a`; `verify` recipe_match=True,
  4/4 shard hashes valid; `train-smoke` consumed sequences across shards,
  integrity valid.
- Source locks: all 12 sources pinned in
  `registry/source_snapshot.lock.json` (immutable revisions; no `main`).

## Known blocker (publishing)

The provided HF token authenticates (`whoami` OK, user `mjaso`) but has
`role=read`; repo create/upload returns 403. Unblocking options: issue a
fine-grained token with dataset write permission (or classic write token)
for the account, or pre-create `mjaso/flashmini-data-v1` (public) and
`mjaso/flashmini-eval-v1` (private) and grant write. The 3 NVIDIA gated
sources are correctly `SOURCE_BLOCKED` (terms not accepted) and remain
recipe-only. Scale-up beyond the pilot is prepared but not started:
`flashmini-data build` resumes from `build_state.json` cursors.

