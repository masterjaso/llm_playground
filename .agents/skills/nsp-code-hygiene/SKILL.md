---
name: nsp-code-hygiene
description: Repairs code structure, seams, module boundaries, tests, and code-graph coverage through bounded PIV work orders and validation evidence. Use when a hygiene report, repair packet, seam issue, structural drift, oversized module, missing test, or code-graph gap needs a scoped repair. Do not use for documentation alignment, CCB promotion, new feature implementation without a repair packet, or final review.
user-invocable: false
---

# nsp-code-hygiene

Use this skill for code hygiene cleanup, safe refactoring, module repair, seam extraction, adapters, test or harness closure, queue reconciliation, full multi-target PIV execution, and code hygiene closeout.

## Trigger Conditions

Code structure, seams, module boundaries, tests, code graph coverage, or implementation hygiene.

## Operating Model

The CLI provides deterministic findings, reports, repair packets, safe target-owned setup repairs, and managed work-order state under `.nsp/artifacts/reports/`. This skill performs semantic source repair by inspecting bounded source context, selecting one narrow target, adding or confirming tests, editing code, validating, updating queues and coverage debt, and continuing until the requested scope is complete or blocked. Deterministic CLI output is bounded evidence, not final alignment authority.

Do not stop at a plan. Execute the PIV loop, validate, update state, and record residual risk. Code hygiene PIVs are task-level work and may be nested inside a Ralph phase when a sweep needs resumable outer coordination.

## Minimum Code Gate

Before semantic repair, refactoring, or source generation, apply this gate in order:

1. Can the request be satisfied without new source code?
2. Does the target already have a helper, module, scene, adapter, command, resource, or pattern that should be reused?
3. Does the language runtime, standard library, framework, game engine, or platform already provide the behavior?
4. Does an already-installed dependency cover the behavior without new dependency ownership?
5. Can the change be one focused function, one narrow seam, or one small local edit?
6. Preserve the safety floor: validation at trust boundaries, data-loss prevention, security, accessibility, public behavior, golden-path user behavior, migration safety, and required tests/checks.
7. State the validation evidence that will prove the smaller implementation is correct.

Use minimum viable code, reuse-before-new-code, native/platform-first implementation, abstraction restraint, dependency restraint, and the smallest safe behavior-preserving change. Do not simplify away the safety floor.

## Design Pressure Execution Rules

Detailed rationale lives in `.docs/guidance/nsp/code-hygiene.md#design-pressure-checks`; keep execution decisions aligned with it:

1. For durable or high-risk design changes, complete NSP's existing draft-plan, adversarial-review, and hardened-plan workflow before implementation.
2. Do not begin implementation while must-fix adversarial findings remain unresolved.
3. Validate idempotency and recovery behavior when an operation may be repeated, retried, interrupted, or partially completed.
4. Do not choose the smallest textual diff when it increases change amplification, duplicated knowledge, hidden coupling, or caller coordination.
5. Require new layers to provide measurable simplification, isolation, normalization, enforcement, translation, or ownership value.

## Required Start

```bash
_nsp hygiene full-code-hygiene --target <repo>
_nsp hygiene code repair-packet --target <repo>
_nsp hygiene code validate --target <repo>
_nsp hygiene code minimize-review --target <repo> --base <ref>
_nsp code-graph validate --target <repo>
```

For safe target-owned setup repairs:

```bash
_nsp hygiene full-code-hygiene --target <repo> --apply-safe
```

## Required Inputs

- The work order / repair packet naming the code targets (`.nsp/artifacts/reports/**`).
- Bounded source context for each target: the file, callers, nearby tests, and graph impact.

## Owns

- reviewing code structure, seams, service/module boundaries, tests, graph coverage
- code repair PIV, seam isolation, test-backed source repair
- proposing changes through ATDD/PIV
- bounded repair execution with validation evidence

## Does Not Own

- deterministic validation authority (CLI)
- context/doc alignment (use `nsp-context-hygiene`)
- bridge links (use `nsp-ccb-hygiene`)

## PIV Loop

PIV is the inner task loop for one bounded code hygiene target. Use Ralph through `nsp-agent-workflow` only when the work spans multiple sessions, phases, or handoffs. For each target:

1. Plan: read the work order, repair packet, source, callers, nearby tests, docs references, generated-output rules, and local guidance.
2. Plan: identify behavior ownership, public interface, hidden implementation details, validation commands, coverage debt, and forbidden changes.
3. Implement: add characterization, seam, contract, golden-output, or golden-path coverage before moving risky behavior.
4. Implement: extract one cohesive seam or repair one cohesive module boundary while keeping behavior stable.
5. Validate: run focused tests first, then `_nsp hygiene code validate --target <repo>` and batch-boundary checks.
6. Manage state: update the work order, runbook, queue, coverage debt, completed seams, new seams, validation results, and residual risks.

Continue through the inferred phase targets unless the user asked for one target only, validation fails, ownership is ambiguous, or human review is required. Suggested phase sizing is guidance for resumable execution, not a universal hard cap.

## Safety Rules

- Use line count as a review signal, not as permission to split behavior.
- Avoid broad rewrites and mixed-concern cleanup.
- Preserve public behavior unless tests and migration protect the change.
- Avoid generated, vendored, archive, or forbidden paths unless selected by the work order.
- Record validation evidence before claiming completion.
- Remove completed seams from active queue items.

## Validation Gates

- Focused tests for each repaired target green before batch checks.
- `_nsp hygiene code validate --target <repo>` green (or blocked-with-evidence recorded).
- Queue/coverage-debt state updated for every completed target.

## Handoff Artifacts

- `.nsp/artifacts/reports/code-hygiene-latest.json` (and repair packet/report artifacts).
- Updated work-order queue with completed seams removed.

## Resume Rules

- For multi-session sweeps keep outer Ralph state under `.nsp/artifacts/tmp/ralph/<epic-id>/` per `nsp-agent-workflow`. Resume from the work order, queue state, and last PIV validation evidence, never chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- pretend the CLI did holistic code review
- claim hygiene without test/validator output

## Expected Outputs

Report completed targets, files changed, tests added or updated, validation results, queue and coverage-debt updates, residual risks, and next targets.
