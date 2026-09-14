---
name: nsp-maintain-steward
description: Keeps changed code, context, Domain Language, and Context-to-Code Bridge artifacts aligned through diff-scoped hygiene before handoff. Use when a task is complete and the current diff needs alignment, before or during PR preparation, after docs or code changes, or when a user asks to reconcile drift or run a maintain pass. Do not use for full-repository adoption, isolated code repair, bridge-only promotion, or merge verdicts.
---

# Maintain Steward

## Outcome
Align changed code, context, Domain Language, and CCB before handoff. Reuse accepted scope and verified integration base; resume the current queue.

## Workflow
1. Run `_nsp hygiene maintain --target <repo> --base <verified-base>`. Add `--working-tree` when the requested review includes current uncommitted edits; the default committed comparison omits them. Inspect enabled phases, direct changed paths, related context, and gaps. Deterministic candidates guide inspection, not semantic proof.
2. Account for each direct changed path once per enabled phase. Related paths are context, not extra items. Inventory-only exclusions `tests/**`, `.nsp/skills/**`, and `README.md` do not create worker items. Record justified not-applicable decisions without hiding missing coverage.
3. Execute enabled code, context, then CCB work through their workers. Read [context-model.md](../nsp-prompt-router/references/context-model.md) when coverage changes; check the feature, technical, persona, guidance, and CCB delta matrix. Preserve target-owned material and evidence for durable claims.
4. Continue until every item is validated or explicitly blocked. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for real queue continuation/delegation and [prediction.md](../nsp-prompt-router/references/prediction.md) only for material activated hypotheses.

## Evidence
Record proof-of-read, queue status, skips, deviations, and fresh required validation. Re-run maintain for status 0 and `maintain_ready`. Required acceptance/assurance failures block readiness; preferred reduction is disclosed. Run-scoped evidence is canonical; latest paths are compatibility pointers. The deterministic CLI checks reports, while the agent owns alignment.

## Boundaries
A partial queue is interim work, never completion or review approval. Do not expand maintenance into adoption, promote guessed links, borrow another run's proof, or close a run for one worker. No human report is needed for each bot transition.
