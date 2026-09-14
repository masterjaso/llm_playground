---
name: nsp-epic-execution
description: Runs large multi-phase initiatives with progressive elaboration, milestone gates, Ralph continuation state, PIV and ATDD validation, and resumable handoffs. Use when work is an EPIC, migration, roadmap, release-hardening effort, or several dependent workstreams that must continue across sessions. Do not use for a single bounded implementation, prompt-only planning, or ordinary PR review.
user-invocable: false
---

# Epic execution

## Outcome
Carry an accepted EPIC through dependent phases and integration gates. The original primary owns it; delegated workers execute only their envelope and never initiate another epic.

## Workflow
Read [run-state.md](../nsp-prompt-router/references/run-state.md) and [execution.md](../nsp-prompt-router/references/execution.md). Reuse run, immutable accepted contract, claims, and phase state. Ralph lives at `.nsp/artifacts/tmp/ralph/<id>/`; `.docs/roadmap/**` contains durable intent, never in-flight state.

Elaborate only the active phase. Retain coarse future objectives/dependencies without repeated expansion. Each phase uses observable ATDD gates and vertical PIV feedback. Read [prediction.md](../nsp-prompt-router/references/prediction.md) for active-phase uncertainty; child records are optional unless they add a distinct hypothesis. Revise only dependencies invalidated by evidence.

## Evidence
Advance only after required acceptance and assurance have fresh proof. Required independence/fresh-context assurance blocks advancement when unmet; preferred reduction needs evidence/disclosure. Resume current phase, IDs, validations, unresolved decisions, and next command. The deterministic CLI validates state; the agent owns judgment.

## Boundaries
No completion with failed gates, partial queues, or unresolved material decisions. Keep one compact bot state and evidence references, not parallel mandatory prose logs. Do not repeat Genesis for unchanged contracts or classify small work as EPIC by time/file count.
