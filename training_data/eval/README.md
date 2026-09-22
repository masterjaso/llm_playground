# Evaluation isolation (v4)

Final held-out eval lives in `<HF_USER>/flashmini-eval-v1` (private, small),
never in the public training corpus. Canonical ingestion may proceed with the
currently available exports and records a partial-exclusion status. Track
benchmark revisions and exclusion method; do not claim perfect
decontamination or freeze a final training view until immutable exports,
including the private FlashMini evaluation set, are added to
`contamination_sources.yaml:corpus_files`.
