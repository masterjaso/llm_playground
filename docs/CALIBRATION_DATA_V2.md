<!-- nsp:meta
id: docs.calibration.data.v2
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/CALIBRATION_DATA_V2.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Calibration corpus v2

Phase 2 calibration data is a reproducible, source-backed manifest.  The
manifest is an index, not a copied corpus: every selected record contains a
relative or absolute `source_file`, `source_record_index`, stable
`source_record_id`, source-file SHA-256, raw and normalized content SHA-256,
the source revision and license, and the exact source-tokenizer count.  The
`corpus-receipt.json` sibling is a bounded machine-readable record of the
same provenance and selected IDs.

## Approved public mixture

The initial vertical slice uses only public, ungated sources whose dataset
cards state a permissive or public-domain license.  A source is not accepted
merely because it appears in this table: the local source record must repeat
its source name, revision/version, license, download hash (when available),
and selection rationale.  Unclear or missing licenses are rejected by
`prepare-data`.

| Domain | Source | Revision/version | License | Why it is included |
| --- | --- | --- | --- | --- |
| code | `openai/openai_humaneval` | `6d43fb980f9fee3c892a914eda09951f772ad10d` | MIT | Small Python code-generation problems. |
| reasoning/math | `openai/gsm8k` | `3101c7d5072418e28b9008a6636bde82a006892c` | MIT | Grade-school math problems with worked reasoning. |
| instruction/dialogue | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | Public instruction and dialogue turns. |
| general | Project Gutenberg public-domain texts | `ebook-1342` | Public Domain | Public-domain book 1342 selected by stable Gutenberg ebook ID. |
| long-context | Project Gutenberg public-domain texts | `ebook-1342` | Public Domain | Long chunks of the same stable public-domain ebook. |

These are source choices, not a request to download the full datasets.  A
small local, hashed subset is sufficient for the vertical slice.  For Project
Gutenberg, verify the selected ebook's public-domain status in the applicable
jurisdiction and retain its ebook ID and download hash.  Do not use private,
gated, leaked, proprietary, or mixed-license corpora.  If a source card
changes its license or access status, remove it and record the decision in
the run receipt.

## Source file format

JSONL is preferred because its physical line index is a stable locator.  Each
record should contain `text` (or `content`/`messages`) and the provenance
fields below.  A JSON object containing `records`/`examples` is also accepted;
top-level source metadata is inherited by each record.

```json
{
  "text": "A short source example.",
  "source_name": "example/public-dataset",
  "source_revision": "2025-01-01",
  "source_license": "MIT",
  "download_sha256": "<sha256 of the downloaded source artifact>",
  "source_record_id": "stable-provider-id-001",
  "domain": "general",
  "rationale": "Representative public prose for the vertical slice."
}
```

When records are transformed from a downloaded dataset, retain the original
record ID and write the transformed JSONL hash into the run evidence.  A
record may instead point at another JSON/JSONL/TSV file with `source_file` and
`source_record_index`; the preparation and verification code always reopens
that locator before accepting the text.

## Exact tokenizer contract

`prepare-data` requires a tokenizer loaded from the pinned local source
snapshot (or an explicitly supplied tokenizer in tests).  It calls the
tokenizer with an explicit `add_special_tokens` setting and records:

- tokenizer snapshot/revision and every recognized tokenizer file hash;
- aggregate tokenizer-file hash;
- BOS/EOS and other special-token IDs and whether they were added;
- whether a chat template was available and whether it was applied;
- sequence length and truncation policy;
- exact `input_ids` count for every selected record.

Whitespace or word counts are never used by the scientific path.  The
`--allow-legacy-token-counts` option is retained only as an explicit migration
escape hatch and marks the resulting manifest
`declared_count_legacy_non_scientific`; such a manifest is not calibration
evidence.

Example:

```text
d2m prepare-data \
  --run-dir runs/<run> \
  --corpus-manifest data/public-subset.jsonl \
  --source-snapshot /snapshots/qwen-pinned \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --train-tokens 131072 \
  --holdout-tokens 16384 \
  --sequence-length 2048 \
  --require-domain code \
  --require-domain reasoning/math \
  --require-domain instruction/dialogue \
  --require-domain general \
  --require-domain long-context
```

`capture` or another consumer can call `resolve_corpus_record` before using a
record.  It verifies the source file hash, raw content hash, normalized
deduplication hash, and (when given the tokenizer) the exact token count.
Train and holdout are selected by seeded hash order after normalized-content
deduplication, and stable IDs are never shared between splits.

## Receipt schema

The receipt has `receipt_type: "dense2moe-corpus-receipt"` and includes:

1. the manifest path, manifest SHA-256, schema version, and dataset hash;
2. a source table with dataset name, revision, license, URL, rationale, and
   download hash;
3. tokenizer metadata and tokenizer-file hashes;
4. selected train/holdout record IDs, source locators, content hashes, domains,
   and exact token counts;
5. a resolvability statement requiring source reopen, hash checks, and a
   tokenizer recount.

The dataset hash excludes output paths, so moving a manifest or receipt does
not change the scientific identity of the selected corpus.  Changing source
content, source revision/license, tokenizer files, tokenizer settings, split
seeds, or selected IDs does change it.
