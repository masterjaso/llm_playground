---
name: nsp-context-genesis
description: Extracts a deep evidence-backed model of an existing repository including features, technical architecture, personas, guidance, and relationships for downstream NSP workflows. Use when a repository needs exhaustive discovery, context generation, re-baselining input, architecture inventory, or evidence-backed project understanding. Do not use for first-adoption orchestration, routine diff alignment, or code repair execution.
user-invocable: false
---

# Context Genesis

## Outcome
Extract an evidence-backed repository model for adoption or explicit re-baselining from the supplied scope, coverage goals, and discovery state.

## Workflow
Read [context-model.md](../nsp-prompt-router/references/context-model.md). Inspect scoped entrypoints, source, configuration, dependencies, tests, CI/delivery, and existing docs. Connect features, mechanisms, actors/personas, guidance, Domain Language, and doc-to-code relationships. Support every durable claim and retain gaps for unavailable evidence.

Preserve useful context and target guidance. Produce inventories appropriate to the actual project, then hand bounded repairs to workers. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for parent-coordinated continuation; a child never starts another epic or unassigned sweep.

## Evidence
Return coverage, anchors, gaps, and validated artifacts through the `genesis-sweep-summary` contract when requested. The deterministic CLI indexes and validates records, not project meaning. Inventory completion is distinct from evidence quality.

## Boundaries
No speculation promoted to facts, unrelated repair, adoption ownership in a child, or implicit CCB promotion. Working inventories use compact ephemeral state; durable prose serves human understanding.
