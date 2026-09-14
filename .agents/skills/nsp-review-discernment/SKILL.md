---
name: nsp-review-discernment
description: Performs exhaustive evidence-backed review of a pull request, merge request, patch, commit range, or changed-file set across correctness, security, tests, hygiene, knowledge, and release readiness. Use when a user asks for PR review, code review, merge review, release-readiness assessment, or severity-ranked findings. Do not use for implementing fixes, planning a feature, or replacing Maintain's diff-alignment pass.
---

# Review Discernment

## Outcome
Give an evidence-backed verdict over the entire change set across correctness, security, tests, hygiene, knowledge, and release readiness.

## Workflow
1. Resolve the integration base with `review-manifest resolve-base` before PR/range review. Prior-commit fallback is only for an explicitly requested single commit. Inventory every changed file, including tests, with exhaustive review-manifest accounting.
2. Run [Maintain](../nsp-maintain-steward/SKILL.md), finish its enabled direct-path queue, and inspect proof-of-read. Review all files and affected interfaces against acceptance behavior, trust boundaries, regressions, and durability. Generated candidates are not semantic proof.
3. Check required commands and evidence freshness. Required independence/fresh-context assurance blocks approval when absent; preferred reduction is disclosed. Do not claim independence without bounded evidence. If prediction is active, read [prediction.md](../nsp-prompt-router/references/prediction.md) and assess classifier/action.
4. Large diffs increase the queue, not reduce coverage. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for resumption and [review-output.md](../nsp-prompt-router/references/review-output.md) for a requested full human report.

## Evidence
Approval requires `maintain_ready`, `reviewComplete`, `unreviewedFileCount: 0`, required validation, and supported file-level conclusions. Missing proof-of-read or blocking findings prevent approval. Cite precise paths/lines and practical impact. The deterministic CLI validates records; the agent supplies the verdict.

## Boundaries
No invented findings to fill categories, incomplete-queue approval, narrative substituted for checks, or treating review as merge permission. Label interim coverage. Bot state uses records/references; human reports explain findings and limits.
