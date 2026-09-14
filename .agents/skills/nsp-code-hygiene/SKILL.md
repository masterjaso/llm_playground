---
name: nsp-code-hygiene
description: Repairs code structure, seams, module boundaries, tests, and code-graph coverage through bounded PIV work orders and validation evidence. Use when a hygiene report, repair packet, seam issue, structural drift, oversized module, missing test, or code-graph gap needs a scoped repair. Do not use for documentation alignment, CCB promotion, new feature implementation without a repair packet, or final review.
user-invocable: false
---

# Code hygiene

## Outcome
Repair the specified structure, seam, test, or graph gap while preserving observable behavior.

## Workflow
1. Read the packet, owning module, callers, tests, and local guidance. Apply the Minimum Code Gate: reuse repository behavior, then platform/runtime or installed dependencies, before adding code.
2. Read [execution.md](../nsp-prompt-router/references/execution.md). Characterize risky behavior, repair one slice, migrate a representative caller, verify, then migrate the others and remove proven duplication. Keep acceptance feedback vertical.
3. Prefer deep modules with understandable interfaces and coherent ownership. A layer must hide complexity or reduce caller knowledge. Do not split by line count or add forwarding wrappers. Preserve locality when it clarifies invariants. Comments explain non-obvious intent/constraints, not adjacent code.
4. Run behavior checks, affected code validation, and code-graph checks when inputs changed. Read [run-state.md](../nsp-prompt-router/references/run-state.md) for continuation only; uncertain mechanisms may use an isolated [prototype](../nsp-prototype/SKILL.md).

## Evidence
Complete the work order with source/test proof, seam ownership, commands/results, and residual gaps. The deterministic CLI finds candidates and validates substrate; the agent judges understandability. Read [prediction.md](../nsp-prompt-router/references/prediction.md) only when uncertainty warrants it.

## Boundaries
A plan is not a repair. No arbitrary cleanup, unassigned features, generated files as source of truth, or lost safety/recoverability for fewer bytes. Completion depends on acceptance, not cosmetic file-size targets.
