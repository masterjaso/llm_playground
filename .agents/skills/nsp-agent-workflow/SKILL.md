---
name: nsp-agent-workflow
description: Coordinates bounded and workstream execution with resumable Work Capsules or Packages, PIV loops, ATDD gates, Ralph state, and handoff evidence. Use when work spans multiple prompts, needs explicit acceptance gates, fresh-context handoffs, resumability, or workstream orchestration. Do not use for simple DIRECT tasks, persona routing, domain-specific hygiene, or deterministic validation.
user-invocable: false
---

# Agent workflow

## Outcome
Coordinate an accepted objective when dispatch, handoff, resumption, or multiple outcomes require it. DIRECT needs no workflow record; BOUNDED adds a Work Capsule when useful; WORKSTREAM adds a Work Package and integration gates; EPIC uses its owner skill.

## Workflow
Read [execution.md](../nsp-prompt-router/references/execution.md) for vertical Plan, Implement, Validate and [run-state.md](../nsp-prompt-router/references/run-state.md) for claims, delegation, and continuation. Reuse the active run and immutable accepted contract. Ralph, when needed, lives at `.nsp/artifacts/tmp/ralph/<id>/`; human reports are optional projections of compact state.

Set observable ATDD acceptance before risky behavior changes, implement a slice, and run focused feedback before the next. Realize roles from available harness capabilities. Preserve required independence/fresh-context assurance; disclose reductions only when preferred. Read [prediction.md](../nsp-prompt-router/references/prediction.md) for uncertain mechanisms, without empty records for known work.

## Evidence
Return result receipts, acceptance commands/results, unresolved decisions, next action, and ownership references. Verify relevant inputs before reusing evidence. Capsule completion is not whole-run closure. The deterministic CLI validates records/lifecycle; semantic judgment remains agent-owned.

## Boundaries
Never close another run, borrow foreign evidence, mutate accepted contracts, or claim independence from role labels. Original `EPIC_OWNER` remains primary. Valid delegated envelopes skip broad routing and Genesis. No mandatory `nsp-agent` runtime, provider, model, or redundant narrative logs.
