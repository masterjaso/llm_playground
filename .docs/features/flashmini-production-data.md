<!-- nsp:meta
id: feature.flashmini-production-data
kind: feature-doc
audience: end-user
status: active
description: Build and resume bounded, provenance-preserving FlashMini training-data releases.
resource: ezra://feature/flashmini-production-data
scope: features
persona: release-operations
source: model
confidence: high
reviewStatus: unreviewed
graphNode: feature:flashmini-production-data
graphTags: flashmini,data,production
validation: context-header-audit,manifest-check,secret-scan
owner: features
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# Production training-data releases

## What this feature does

The data operator can turn pinned public sources into deterministic training
views while keeping the workstation bounded. The workflow streams a window,
filters and deduplicates it, publishes verified shards, and resumes from saved
progress without rebuilding the whole corpus.

## Who uses it

Use it for corpus operators, release managers, and training owners preparing
the 1B view or the future 50B view. A release owner also uses it to inspect
source licensing, contamination exclusions, remote integrity, and readiness
before handing data to training.

## How to use it

Start with authentication and source inspection, plan the selected recipe, then
run a bounded build or resume. Keep the immutable source lock, tokenizer
revision, contamination configuration, and release prefix fixed. Verify the
published manifest before freezing a release; use the status and cache views
to monitor progress and disk safety.

## Expected behavior

Every progress checkpoint is resumable and records source identity, cursors,
hashes, deduplication state, and release progress. Completed shards are checked
remotely before local eviction. Cache limits and the free-space watermark stop
the run before it can consume unrelated disk space. Canonical data is not
rewritten when a separate decontaminated view is produced.

## Limits / known constraints

The pilot release remains immutable and has recorded remote integrity and exact
token-count blockers. The 1B release still needs private-evaluation
decontamination and exact final counts. The 50B stream is paused with durable
progress but is not a materialized 8T training view; blocked or gated sources
remain explicit rather than silently substituted.

## Technical implementation

See [the data-pipeline technical reference](../technical/flashmini-data-pipeline.md).
