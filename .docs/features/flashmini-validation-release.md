<!-- nsp:meta
id: feature.flashmini-validation-release
kind: feature-doc
audience: end-user
status: active
description: Apply reproducible validation and release gates before treating FlashMini results or data as ready.
resource: ezra://feature/flashmini-validation-release
scope: features
persona: qa-validation
source: model
confidence: high
reviewStatus: unreviewed
graphNode: feature:flashmini-validation-release
graphTags: flashmini,validation,release
validation: context-header-audit,manifest-check,secret-scan
owner: features
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# Validation and release gates

## What this feature does

This capability turns training and corpus checks into explicit evidence before
an operator calls a run or release ready. It separates implementation health,
fresh-data validity, matched comparisons, scaling evidence, and final release
readiness so an early pass cannot be mistaken for completion.

## Who uses it

Researchers use the gates to qualify an experiment. Data and release operators
use them to qualify a corpus snapshot. Reviewers use the recorded reports to
see which evidence is complete, which gate is blocked, and which claims must
remain provisional.

## How to use it

Run the project test suite and the documented structural, upstream, data, and
runtime checks for the selected track. For data, audit the pinned remote
revision, verify the manifest, and freeze only after the release contract is
complete. Record failed or unavailable evidence as a blocker rather than
raising a readiness flag manually.

## Expected behavior

Gate results are tied to immutable configuration, source, tokenizer, and
candidate identities. The policy reports finite metrics, clipping and routing
health, data membership, remote checksums, and evaluation coverage. A missing
control, stale manifest, or unresolved private-evaluation export keeps the
release blocked.

## Limits / known constraints

Short structural probes do not prove long-context capability, and a green test
suite does not establish a model winner. The current production data
status intentionally retains pilot integrity mismatches, held sources, and
paused materialization as visible blockers.

## Technical implementation

See [the validation and operations technical reference](../technical/flashmini-validation-operations.md).
