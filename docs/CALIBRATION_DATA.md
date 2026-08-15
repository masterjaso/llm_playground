# Calibration data contract

The calibration manifest is a small, reproducible index rather than a corpus
copy.  Each example has a stable ID, source/revision, content SHA-256, token
count, and split.  Train and holdout IDs must be disjoint.  The manifest records
tokenizer revision, sequence length, deduplication method, and a domain label
(`code`, `reasoning/math`, `instruction/dialogue`, `general`, or
`long-context`).

Activation capture stores only the input to the selected MLP in binary
`safetensors` shards.  Dense targets are recomputed from immutable source
weights during training.  A shard is published atomically only after its
tensor shapes, finite-value check, metadata, and SHA-256 digest pass.  A
resumed capture skips only a shard whose digest and metadata still validate.

The pilot is intentionally local-data driven.  If no approved corpus manifest
is supplied, `prepare-data` returns `BLOCKED` with an executable next command;
it does not create an empty activation iterable or call synthetic metrics a
calibration result.
