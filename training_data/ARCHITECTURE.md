# FlashMini corpus architecture (v4)

One logical master lake; deterministic views per recipe. Flow: HF/upstream
-> registry (`sources.yaml` + immutable `source_snapshot.lock.json`) ->
recipe (exact stage aggregate and token deficits) -> bounded cache window ->
canonicalize/filter -> SQLite exact + MinHash-LSH near-dedupe -> benchmark
exclusion -> explicit bounded validation selection -> Parquet+ZSTD shards
(256MiB-1GiB physical target) -> verified immutable release -> tokenizer
specific contiguous `uint16`/`uint32` token store with document offsets ->
distributed deterministic sampling, prefetch, and exact resume. Canonical text
is never regenerated for a different sequence length. Corpus identity is the
SHA256 of registry lock, recipe, filter/dedupe versions, split/validation
contracts, shard hashes, tokenizer identity, and token format.
