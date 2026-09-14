---
name: nsp-context-hygiene
description: Aligns project context and documentation with changed code through bounded PIV validation, docs-to-code linkage, work orders, and evidence closeout. Use when `.docs` content, guidance, feature or technical docs, frontmatter, context coverage, or documentation drift needs repair after a change. Do not use for source-code repair, CCB link promotion, first-adoption discovery, or broad PR verdicts.
user-invocable: false
---

# Context hygiene

## Outcome
Repair scoped documentation/context drift from current source evidence.

## Workflow
Read affected docs, source/tests, and [context-model.md](../nsp-prompt-router/references/context-model.md). Apply the feature/technical/persona/guidance/CCB delta matrix where the change matters. Preserve target ownership, select the right document type, and remove unsupported claims.

Plan one alignment result, edit its necessary docs, and validate before the next slice. Run affected frontmatter, manifest, context, and graph checks. Hand code repairs to Code Hygiene and link promotion to CCB Hygiene. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for coordinated continuation.

## Evidence
Record proof-of-read, changed paths, source anchors, skips, and validation. The deterministic CLI checks shape/coverage candidates; accuracy remains agent-owned. Human docs use prose passes; bot queue state stays compact.

## Boundaries
No full discovery sweep for a diff, unrelated source repair, guessed Domain Language, or automatic CCB promotion. Do not replace customization with generic NSP text or claim alignment with failed checks.
