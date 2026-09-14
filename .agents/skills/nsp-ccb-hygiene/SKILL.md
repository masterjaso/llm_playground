---
name: nsp-ccb-hygiene
description: Maintains the Context-to-Code Bridge by reviewing deterministic coverage candidates, resolving anchors, and promoting only evidence-backed links into the reviewed ledger. Use when CCB coverage reports, stale or missing doc-to-code links, repair candidates, or bridge validation need review and promotion. Do not use for general docs hygiene, source-code repair, or treating CLI-generated candidates as already reviewed.
user-invocable: false
---

# CCB hygiene

## Outcome
Promote only evidence-backed doc-to-code links for assigned candidates.

## Workflow
Read candidate, feature/technical docs, actual code anchor, and relevant tests. Confirm relationship type, source, target, and current meaning. Repair supported anchors; leave guesses unreviewed.

Write supported reviewer, time, status, and evidence fields to `.docs/graph/ccb-reviewed-links.json`. Build/validate affected graph surfaces before claiming trusted results. Structural, linked, anchored, reviewed, and trusted are distinct evidence tiers; CLI candidates are not reviewed links. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for continuation.

## Evidence
Return dispositions, inspected evidence, promoted IDs, unresolved anchors, and bridge validation via `ccb-promotion-report` when requested. The deterministic CLI generates candidates/validates records; promotion is agent-owned.

## Boundaries
No broad repair, fabricated metadata, filename-only promotion, or trusted tier before required checks. Preserve target-owned reviewed links unless current evidence warrants change.
