---
name: nsp-agent-workflow
description: Coordinates bounded and workstream execution with resumable Work Capsules or Packages, PIV loops, ATDD gates, Ralph state, and handoff evidence. Use when work spans multiple prompts, needs explicit acceptance gates, fresh-context handoffs, resumability, or workstream orchestration. Do not use for simple DIRECT tasks, persona routing, domain-specific hygiene, or deterministic validation.
user-invocable: false
---

# nsp-agent-workflow

## Trigger Conditions

Use for multi-step work, resumable epics, handoffs between agents, or when ATDD gates must be defined before implementation.

## Required Start

Confirm the deterministic substrate before defining gates (the CLI validates. This skill owns the loop discipline):

```bash
_nsp project status --target <repo>
_nsp project validate --target <repo>
# for material uncertainty or WORKSTREAM work:
_nsp plan-substrate discovery validate --target <repo> --run-id <run-id>
```

## Required Inputs

- The work item or epic goal with acceptance expectations.
- Existing Ralph state when resuming (`.nsp/artifacts/tmp/ralph/<epic-id>/STATE.json`).
- A green Repository Discovery Gate for WORKSTREAM work or explicitly uncertain BOUNDED work; DIRECT and clear low-risk BOUNDED work remain exempt unless explicitly gated.

## Owns

- creating/updating Ralph state files
- defining ATDD gates
- running PIV loops
- preparing handoffs
- making work resumable
- bounded-mode orchestration
- workstream-mode orchestration

## Does Not Own

- persona routing (use `nsp-prompt-router`)
- domain-specific hygiene (use hygiene skills)
- CLI validation authority

## Bounded Mode

Use bounded mode for one coherent capability or defect outcome with one acceptance contract.

- Define one Work Capsule, one acceptance contract, and one smallest-valid PIV slice.
- Keep required reads, write scopes, and validation commands tightly bounded to the outcome.
- When the compact gate applies, bind exact paths/commands only to verified fact IDs; rejected facts are excluded and unknown/conflicted facts block Plan.
- Route execution through `_nspx` capability requirements, not named-harness paths.
- Prefer `_nspx work --requires <capabilities> --target <repo>` when orchestration is needed.
- Do not branch on harness brand, provider name, or model identity.

## Workstream Mode

Use workstream mode when one delivery objective contains multiple independently verifiable bounded outcomes.

- Define one Work Package with child outcomes, child dependencies, and package-level integration gates.
- Treat each child outcome as a bounded capsule with its own acceptance contract and validation commands.
- Use `_nspx` to select execution realization from capabilities. Do not assign named-harness execution paths in the contract.
- Prefer `_nspx executor --requires <capabilities> --target <repo>` or `_nspx work --requires <capabilities> --target <repo>` for capability-based realization.
- Keep integration gates, shared invariants, and package handoff evidence at the workstream level.
- Complete Repository Fact Extraction and `DISCOVERY_READY` before selecting child implementation paths; register architecture decisions separately from facts.
- Record package invariants plus selected child or integration predictions where uncertainty, risk, or cross-child behavior warrants them, not a Prediction Contract for every child.

## Assurance in PIV

Select assurance requirements during Plan: independence, fresh-context, evidence, and unmet policy. Keep routine PIVs without extra assurance ceremony. During Validate, record assurance on the Result Receipt. Required unmet assurance blocks acceptance; preferred unmet assurance follows disclosure or reduced-assurance policy. Do not treat role labels as executor assignments.

## Prediction Contracts inside PIV

Acceptance criteria define the destination; a Prediction Contract describes the path: the proposed mechanism, expected blast radius and observations, non-effects/invariants, falsifiers, and mismatch response. A prediction must not merely restate "the feature works" or "tests pass."

The agent selects prediction depth during Plan as `none | compact | expanded`; the deterministic CLI does not make this semantic choice.

- DIRECT mechanical work uses `none` and emits no placeholder artifact.
- Low-risk mechanical BOUNDED work may use `none` when the mechanism, blast radius, and validation are already clear.
- Uncertain, risky, high-impact, or costly-to-reverse BOUNDED work requires `compact`.
- Use `expanded` only when alternatives or diagnostic probes materially improve a risky decision.
- WORKSTREAM uses its package invariants plus selected child or integration predictions; keep routine children light.
- Reassess after contradictory evidence. A prior counterexample prevents silent reuse of the prior hypothesis or depth decision.

Experiments are optional and decision-triggered. Use at most the bounded discriminating probes allowed by the prediction contract when competing hypotheses, conflicting evidence, unlocalized causes, expensive changes, or a prior contradiction make a cheap safe probe decision-relevant. Stop when the implementation decision is resolved.

## PIV Loop

PIV is the inner task loop. Each Ralph phase may contain one or more PIV loops, and each PIV loop must be small enough to validate.

1. **Plan** — define ATDD gates (observable behaviors, validation commands, fixtures, negative cases), select prediction depth, author the activated contract, and optionally run a bounded discriminating experiment
2. **Implement** — smallest coherent change satisfying gates
3. **Validate** — run the exact gates, compare actual observations with the activated Prediction Contract, and record exactly one classifier

During Validate, record exactly one semantic classifier: `expected-match`, `benign-deviation`, `scope-discovery`, `counterexample`, or `invalid-validation`.

- `benign-deviation` cannot excuse a material scope or invariant change. Record evidence that the difference is understood, material invariants hold, scope did not materially expand, and the hypothesis remains usable; otherwise use `scope-discovery` or `counterexample`.
- For `scope-discovery`, update the plan, impact contract, context receipt/deviations, claims, and validation before continuing.
- For `counterexample`, stop the affected PIV, abandon or materially revise the hypothesis, and re-plan.
- For `invalid-validation`, repair the validation method and rerun it before claiming success.
- `scope-discovery`, `counterexample`, and `invalid-validation` reopen/block the Repository Discovery Gate until the ledger, plan, claims, and validation are refreshed.
- Missing or unclassified results and unresolved material contradictions block PIV completion.

Each PIV loop must record the plan, prediction mode and activated artifact reference when any, implementation summary, exact validation commands and result, Prediction Result classifier, actual observations, files changed, bounded evidence references, contradictions, residual risk, and next action. `none` records the mode without creating a placeholder Prediction Contract. A PIV loop is incomplete until validation evidence and any required Prediction Result are recorded.

The CLI validates form, budgets, provenance, and lifecycle only. The agent/skill owns hypotheses, confidence, experiment selection, the semantic classifier, and revise/abandon decisions.

When a PIV loop needs execution realization, delegate via `_nspx` capability requirements instead of naming a harness or provider.

## Clean-context sub-agent preference

Keep NSP's PIV and Ralph loops. When the harness can spawn sub-agents or nested sessions, **prefer them** for independently verifiable Implement/Validate/research slices instead of stuffing the full parent transcript into one worker.

This is Context Marshalling applied to live agents — not a third-party loop brand.

1. **Parent orchestrates** — primary agent owns routing, epic ownership, acceptance judgment, integration, and Ralph state. Children do not become `EPIC_OWNER`.
2. **Child gets a clean purpose** — spawn with a valid `NSP_ENTRY_MODE: delegated` envelope (`ROUTING_STATUS: complete`, objective, scope, exclusions, expected output, optional `REQUIRED_CONTEXT`). Pass only what the child needs; **do not** forward the parent chat, full epic narrative, or unrelated tool dumps.
3. **Envelope-only context policy** — delegated agents load path-local safety plus supplied `REQUIRED_CONTEXT` / scoped paths. They skip top-level genesis, broad routing, persona discovery, and epic classification. Malformed envelopes fail closed.
4. **Separate build from critique when useful** — for Validate or adversarial review, prefer a **fresh** delegated agent that sees the artifact + ATDD/acceptance bar (and exact commands), not the implementer's rationalization history. Parent integrates the compact critique into `validation.md`.
5. **Return compact evidence** — child returns only `EXPECTED_OUTPUT` (commands, pass/fail, paths touched, residual risk, blockers). Parent marshals results into Ralph/PIV records and decides next action.
6. **Lowest sufficient effort** — when the harness can choose effort or model class, assign mechanical implement/verify slices to the cheapest capable realization; keep expensive reasoning for Plan, routing, integration, and hard trade-offs. Record optional `EFFORT_CLASS` / `SEPARATION` on the envelope as harness-owned hints — never as provider/model identity in the contract.
7. **When sub-agents are unavailable** — same-session PIV with Context Marshalling to Ralph/run artifacts remains correct. Do not invent fake multi-agent machinery or claim isolation you do not have.

Prefer sub-agent fan-out when: multiple independent PIV slices, Implement vs Validate role separation, bounded research that would bloat the parent, or parallel non-overlapping claims. Keep work in-session when: tiny DIRECT edits, missing harness support, or the slice cannot be described without continuous parent judgment.

## Ralph Loop

Ralph is the outer agent continuation loop. It controls how work is resumed, sequenced, handed off, and continued across prompts, agents, context windows, or sessions.

**Rehydrate → Assess → Lay the plan → Produce → Handoff** is NSP's flow-control shorthand, not a required RALPH acronym expansion.

NSP does not require a RALPH backronym. When a receipt-style format is useful, operators may record read/reproduce, analysis, localization, patch/proof, and handoff details, but the canonical Ralph contract is the outer continuation loop and the canonical task execution contract is PIV.

Write ephemeral state to `.nsp/artifacts/tmp/ralph/<epic-id>/`:

```text
STATE.md / STATE.json
phase-XX/context-receipt.md
phase-XX/atdd.md
phase-XX/plan.md
phase-XX/validation.md
phase-XX/handoff.md
```

Each Ralph phase should name the PIV loop(s) it contains. A phase can contain zero PIV loops only when it is assessment, routing, or handoff-only and records that fact explicitly.

## Expected Outputs

- Ralph state tree under `.nsp/artifacts/tmp/ralph/<epic-id>/` (STATE.md, STATE.json, per-phase atdd/plan/validation/handoff).
- ATDD gate definitions before any implementation edit.
- Per-PIV records with plan, implementation summary, validation commands, validation result, files changed, and residual risk.

## Validation Gates

- Every PIV slice runs its exact gate commands. Failures are recorded honestly in `phase-XX/validation.md`.
- Do not claim Ralph phase completion until all required PIV loops are complete or honestly blocked with evidence.
- `STATE.json` stays valid JSON and reflects real phase statuses.

## Handoff Artifacts

- `phase-XX/handoff.md` ending with the "Resume here" footer (current phase/work item, completed and blocked PIV loops, active and released claims, last validations, files changed, decisions, blockers, next exact command, next file to inspect).

Required handoff footer:

```md
## Resume here

- run_id:
- ralph_epic_id:
- current_phase:
- completed_piv_loops:
- blocked_piv_loops:
- active_claims:
- released_claims:
- last_validation:
- files_changed:
- residual_risk:
- next_piv_target:
- next_exact_command:
- next_file_to_inspect:
```

## Resume Rules

- Rehydrate from `STATE.json` + latest `handoff.md` at session start. Verify prior validation actually passed before building on it. Never rely on chat history.
- Revalidate the same-run fact ledger before resuming gated work; do not resume from stale or cross-run discovery evidence.
- Resume prediction work from only the accepted hypothesis, mode, latest mismatch status, unresolved material contradictions, next action, and bounded evidence references. Load that compact state by reference; do not copy full contracts, logs, or reasoning transcripts into Ralph state.

## Required References

- `.docs/technical/agent-workflow-terms.md`

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- skip validation and mark work complete
- treat Ralph artifacts as durable truth
- claim CLI performed semantic judgment
- dump the parent transcript into a sub-agent when a delegated envelope would suffice
- ask a child to re-run top-level routing, genesis, or epic ownership
- bind PIV/Ralph contracts to a named model, provider, or harness brand
