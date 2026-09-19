# FlashMini corpus architecture (v4)

One logical master lake; deterministic views per recipe. Flow: HF/upstream
-> registry (`sources.yaml` + immutable `source_snapshot.lock.json`) ->
recipe (domain weights) -> bounded cache window -> canonicalize/filter ->
exact + MinHash-LSH near-dedupe -> hash-salted train/val split ->
Parquet+ZSTD shards (256MiB-1GiB) -> HF publish + remote verify -> evict
local -> `RemoteShardDataset` training with hierarchical deterministic
sampling and exact resume. Corpus identity: `corpus_fingerprint_sha256`
(registry hash, revisions, recipe hash, filter/dedupe versions, split salt,
shard hashes, tokenizer identity).
