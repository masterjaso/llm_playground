---
license: other
license_name: flashmini-data-v1-terms
task_categories:
- text-generation
language: [en]
---

# FlashMini data v4 (card)

Deterministic FlashMini training corpus. Canonical documents live in
Parquet+ZSTD shards under `shards/`; each shard carries a manifest with
sha256, counts, and distributions; the frozen corpus identity is
`corpus_fingerprint_sha256`.

Sources and redistribution: each source carries one of mirror_allowed,
recipe_only, gated_recipe_only, review_required, generated_owned
(fail-closed; see registry/sources.yaml + source_snapshot.lock.json).
Content shards are published only for mirror_allowed/generated_owned
sources; recipe-only sources contribute provenance only and stream from
upstream at training time. Gated NVIDIA sources are not mirrored. Licenses
are recorded per source; no blanket license is claimed over mixed content.

Provenance: registry hash + immutable source revisions, recipe hash,
filter/dedupe versions, split salt, per-shard sha256, exact dedupe,
MinHash/LSH near-dedupe coverage (documented per release).

# FlashMini data v4 dataset card template

Use one card per frozen corpus release. Retain source-specific licenses;
do not declare a single permissive license over mixed content.
