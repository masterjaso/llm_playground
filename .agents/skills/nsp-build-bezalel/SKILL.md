---
name: nsp-build-bezalel
description: Orchestrates implementation of an accepted bounded objective or Work Package with right-sized PIV, ATDD, verification-before-claim, and adaptive test-first discipline. Use when a user asks to implement, build, modify, add, integrate, or safely change code or docs from an accepted scope. Do not use for vague planning, defect diagnosis without reproduction, PR review, or routine diff alignment.
---

# Build Bezalel

## Outcome
Implement the accepted bounded objective and prove its acceptance behavior. Start with current scope, contract, context receipt, and path-local safety. Resume them without rerouting or repeating Genesis.

## Workflow
Read [execution.md](../nsp-prompt-router/references/execution.md) for behavioral implementation: Plan one observable behavior, Implement its vertical slice, then Validate before the next. Capture regression/characterization first for risky existing behavior. Mechanical or documentation edits need proportionate checks, not implementation-shaped tests. Apply the Minimum Code Gate before adding code.

DIRECT uses one action and proof. Use [agent workflow](../nsp-agent-workflow/SKILL.md) only for dispatch, handoff, resumption, or workstream integration; [epic execution](../nsp-epic-execution/SKILL.md) only for EPIC. An isolated [prototype](../nsp-prototype/SKILL.md) may resolve material technical uncertainty before production changes.

## Evidence
Record changed behavior, fresh commands/results, evidence paths, and residual risks. Keep machine coordination compact and human summaries outcome-first. The deterministic CLI validates records; the agent owns design and semantic correctness.

## Boundaries
Stay within the contract. Preserve safety, accessibility, public behavior, recoverability, and required validation. No provider, model, capsule, Ralph, or prediction placeholder is mandatory for a clear local change. Failed acceptance prevents completion.
