---
name: nsp-acceptance-prover
description: Qualifies completed controlled work against the accepted Work Capsule or Work Package with candidate-specific deterministic evidence before handoff or status closure. Use when a user asks to prove acceptance, run UAT, or verify a completed slice against its accepted gates. Do not use for implementation, repair, active defect diagnosis, planning, PR review, deployment, or production operation.
---

# Acceptance Prover

## Outcome

Prove that the delivered candidate satisfies the accepted Work Capsule or Work Package before it can advance through the controlled lifecycle. Acceptance is independent qualification, not construction-time PIV, repair, review, deployment, or production operation. Start from the accepted root contract, the active controlled run, the current candidate, and path-local safety. Resume from existing acceptance artifacts without rerouting or repeating Genesis.

## Workflow

Read [execution.md](../nsp-prompt-router/references/execution.md) for bounded verification. In order:

1. Confirm the accepted root contract, active top-level controlled run, `acceptancePolicy: "required"`, and delivered candidate. Do not invent acceptance gates or qualify work that was never delivered to disk.
2. Derive and validate the immutable, revision-scoped Acceptance Plan from the accepted Work Capsule or Work Package. Record it before sweeping; the Plan is the authority for criterion identity, source gates, dependencies, contract revision, and acceptance cycle.
3. Execute a complete dependency-aware Acceptance Sweep. Evaluate every independently runnable criterion in stable Plan order; one failure must not stop unrelated criteria. Record the exact evidence and candidate identity, then validate and record the Sweep.
4. If the Sweep requires repair, use its generated Repair Set and routing record to hand failures to Build or Debug. Acceptance does not repair product code. After mutation, Maintain and a new complete Sweep are required.

Use the highest-fidelity local, user-observable surface available. Use `npm run surfaces:check`, `npm run test`, or the relevant deterministic runner for applicable gates, and inspect code or graph artifacts only for structural gates. An isolated [prototype](../nsp-prototype/SKILL.md) may resolve a material uncertainty about what constitutes proof before the verifier runs.

DIRECT uses one bounded qualification action: enumerate the accepted criteria, gather evidence, validate the records, and report the verdict per criterion. Use [agent workflow](../nsp-agent-workflow/SKILL.md) only for dispatch, handoff, resumption, or workstream integration; [epic execution](../nsp-epic-execution/SKILL.md) only for EPIC-level milestone gates.

## Acceptance Plan

The accepted Work Capsule or Work Package is authoritative. Derive a revision-scoped Plan body at `.nsp/artifacts/runs/<runId>/acceptance/plans/PLAN-<contractRevision>.json` and record the validated current pointer at `.nsp/artifacts/runs/<runId>/acceptance/PLAN.json` from its declared acceptance gates without adding unrelated criteria. The Plan must:

- cover every accepted gate, with each criterion bound to its exact `sourceGate`;
- use unique criterion IDs and only declared `dependsOn` criterion IDs, with no dependency cycles; and
- preserve the controlled run’s `runId`, `contractRef`, SHA-256 `contractDigest`, and `contractRevision`.

Use `_nsp work acceptance plan validate` before `_nsp work acceptance plan record`. Revision Plan bodies are serialized and immutable; a different rewrite requires a new accepted contract revision. `work revise` removes the current pointer while preserving historical Plan bodies.

## Acceptance Sweep

Record each complete Sweep at `.nsp/artifacts/runs/<runId>/acceptance/sweeps/<sweepId>.json` and its current `LATEST.json` pointer. It must be bound to the Plan and include the current `acceptanceCycle`, one current `candidateIdentity`, and exactly one result for every criterion. Run a criterion only when all dependencies are `PASS`; a terminal non-`PASS` dependency yields `BLOCKED_DEPENDENCY`. Continue with independent criteria after failures and do not stop at the first failure.

Use only the defined result statuses: `PASS`, `FAIL`, `BLOCKED_DEPENDENCY`, `BLOCKED_ENVIRONMENT`, and `BLOCKED_AMBIGUITY`. `PASS` and `FAIL` require evidence. `FAIL` also requires expected behavior, observed behavior, and a failure class. Environment and ambiguity blocks require a concrete reason. The recorded outcome is derived: only all-`PASS` criteria produce `PASS`; product failures or dependency blocks produce `REPAIR_REQUIRED`; environment or ambiguity blockers produce `BLOCKED`.

Validate and record with `_nsp work acceptance sweep validate` and `_nsp work acceptance sweep record`. Recording is allowed only for the active controlled run in phase `accepting`, and the submitted candidate digest must still match the current candidate.

## Repair Set

For `REPAIR_REQUIRED`, use the generated `.nsp/artifacts/runs/<runId>/acceptance/REPAIR-<sweepId>.json` and its routing record. Every Sweep `FAIL` appears exactly once; `PASS` and `BLOCKED_DEPENDENCY` criteria are not product failures. Preserve each failure’s criterion, `sourceGate`, failure class, evidence references, and Sweep candidate digest. `incomplete-implementation` routes to Build; behavioral defect, regression, or flaky behavior routes to Debug. Do not silently collapse independent failures or replace a dependency block with an invented defect.

For `BLOCKED`, report the concrete environment or ambiguity evidence and resolve that blocker before retrying. Do not claim acceptance or manufacture a repair classification for an unqualified result.

## Evidence

Acceptance evidence is target-contained under `.nsp/artifacts/runs/<runId>/acceptance/`. For every criterion, record the verdict, exact command/result or file path, and bounded evidence references. Never record an evidence-free `PASS` or `FAIL`; do not substitute prose for command or artifact evidence. The deterministic CLI validates record structure, bindings, and greedy completeness; the prover owns semantic user-observable judgment. Report residual risks and whether the candidate is approved to advance.

## Current Candidate

Acceptance is candidate-specific. Capture the Sweep’s full `candidateIdentity`; its `finalCandidateDigest` must match the current Git candidate when the Sweep is recorded and when the controlled run advances. Any code, context, or other candidate mutation makes earlier Acceptance stale. After repair or any later mutation, run Maintain and a new complete Sweep; never reuse an older PASS or candidate digest.

## Controlled Lifecycle

Acceptance runs only in the active controlled run’s `accepting` phase. The Plan and Sweep must remain bound to the controlled contract reference, digest, revision, `acceptanceCycle`, and `runId`. A `PASS` Sweep advances `accepting` to `accepted`; only then may required-acceptance work enter Review. A failed Sweep routes to `acceptance-repair`, then back through Maintain and requalification. Do not mark the controlled run complete while Acceptance is missing, stale, blocked, or not `PASS`; GIT-READY completion also requires current Maintain, Review, and candidate-identity evidence. Work whose policy is `not-required` does not enter this skill’s Acceptance phase.

## Boundaries

Stay within the accepted contract. Do not extend criteria, alter the Work Capsule or Package to obtain a pass, repair product code, or treat acceptance as Review approval. Use local target-contained evidence only; do not use cloud or remote CI/CD, deployment, production operation, external trackers, or pull-request side effects as acceptance steps. Failed or blocked acceptance prevents completion. NSP stops at GIT-READY; it does not create or push a remote PR, synchronize an external tracker, trigger CI/CD, deploy, or publish.
