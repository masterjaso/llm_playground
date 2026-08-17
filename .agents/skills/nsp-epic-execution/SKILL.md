---
name: nsp-epic-execution
description: Runs large multi-phase initiatives with progressive elaboration, milestone gates, Ralph continuation state, PIV and ATDD validation, and resumable handoffs. Use when work is an EPIC, migration, roadmap, release-hardening effort, or several dependent workstreams that must continue across sessions. Do not use for a single bounded implementation, prompt-only planning, or ordinary PR review.
user-invocable: false
---

# nsp-epic-execution

## Trigger Conditions

Large multi-phase initiatives, architectural epics, release hardening, **PR review maintain queues with many direct changed paths**, or work requiring resumable milestone tracking.

**Primary entry only.** Delegated sub-agents must not invoke this skill to own or initiate an epic. If `NSP_ENTRY_MODE: delegated` is present, return control to the parent `EPIC_OWNER`.

## Required Start

Deterministic substrate at the start of every epic session (the CLI validates. This skill owns milestone judgment):

```bash
_nsp status --target <repo>
_nsp validate --target <repo>
_nsp plan-substrate discovery validate --target <repo> --run-id <run-id>
```

## Required Inputs

- The epic goal, ephemeral roadmap/plan under `.nsp/artifacts/tmp/ralph/<epic-id>/` (e.g. `roadmap.md` / `STATE.md` / `STATE.json`), and existing Ralph state when resuming.
- ATDD gates for the current phase (defined before editing).
- A green run-scoped Repository Discovery Gate at `.nsp/artifacts/runs/<runId>/planning/repository-fact-ledger.json` plus its Markdown companion. If it is absent, stale, structurally invalid, or `DISCOVERY_BLOCKED`, stop planning and run discovery before selecting implementation files.
- Do **not** use `.docs/roadmap/**` as in-flight epic execution state, resume source of truth, or required input.

## Progressive Elaboration

Epic planning keeps durable phase objectives and gates, but fully elaborates only the active phase.

- Record the full phase list, stable objectives, ATDD gates, and next-phase links durably.
- Expand only the active phase into detailed work packages, capsules, and PIV slices.
- Compose work as Work Packages → Work Capsules → PIV loops.
- Prefer clean-context delegated sub-agents for Implement/Validate slices when the harness supports them (see `nsp-agent-workflow` clean-context preference). Parent keeps orchestration and Ralph state; children receive only a delegated envelope + required context.
- Keep future phases intentionally coarse until they become active.
- Do not assign harnesses, providers, or model identities in the epic contract. Realization happens later through capability-based orchestration.

### Assurance progression

Keep future-phase assurance coarse and elaborate only the active phase or capsule. Record independence and fresh-context requirements in run-scoped artifacts. Never treat a role label as a named executor assignment. The original primary remains `EPIC_OWNER`; children cannot become owner.
- Do not synthesize implementation plans from memory before discovery. Stage A extracts facts, Stage B gates readiness, and Stage C registers architecture decisions separately before phase plans bind to verified fact IDs.

## Phase Prediction

During Plan, record one phase prediction that distinguishes the phase destination (ATDD acceptance gates) from the proposed path (mechanism, expected observations, invariants, falsifiers, and mismatch response). Do not force a Prediction Contract for every PIV: activate PIV-level `compact` or `expanded` contracts only for risky, uncertain, cross-cutting, high-impact, or previously contradicted slices. Routine later PIVs may use `none`.

Reassess prediction depth at every phase boundary using current evidence, impact, reversibility, incorrect-change cost, and unresolved contradictions. A prior counterexample forces reassessment of the hypothesis and prediction depth before the affected phase or PIV continues; do not silently reuse it.

## PR review maintain queue (mandatory resumability)

When `nsp-review-discernment` or `nsp-maintain-steward` builds a maintain PIV queue, treat it as an epic:

- **Queue items** = one PIV per **direct changed path** (not blast-radius related paths)
- **Blast radius** = `phaseRelatedContext` loaded during each direct PIV
- **No file-count limit** — continue until `maintain_ready=true`

Ralph state path:

```text
.nsp/artifacts/tmp/ralph/maintain-pr-<pr-or-branch>/
  STATE.md
  STATE.json
```

Loop (resume across sessions):

```bash
_nsp hygiene maintain next --target <repo>
# PIV direct path + blastRadiusContext
_nsp hygiene maintain record --target <repo> --phase <code|context|ccb> --path <direct-file> --status complete
_nsp hygiene maintain status --target <repo>
```

Update Ralph STATE after each recorded item. **Never** stop or approve because the queue is large — that is exactly when epic-execution applies.

## Owns

- milestone phase definitions
- ATDD gates before implementation
- PIV loops within each phase
- Ralph state artifacts for resumability
- validation execution and recording
- structured handoffs between phases

## Does Not Own

- routine single-file fixes
- persona routing (use `nsp-prompt-router`)

## Phase Structure

Each phase must have:

| Field | Content |
|-------|---------|
| phase id | e.g. `phase-04` |
| objective | one-sentence goal |
| ATDD gates | observable acceptance criteria |
| changed files | expected touch points |
| validation commands | exact commands to run |
| status | not-started / in-progress / blocked / done |
| handoff notes | what the next agent needs |
| next phase | phase id or `complete` |

Every active phase also records the discovery ledger path, the verified fact
IDs it binds, excluded rejected IDs, ledger freshness, and the architecture
decision/register IDs it depends on.

## Status Rules

Phase status transitions are one-way unless a blocker reopens work: `not-started → planned → in-progress → implemented → validating → done`, with `blocked`, `deferred`, and `superseded` as explicit exits. Never mark a phase `done` if its gates are not green.

A missing or unclassified Prediction Result, `invalid-validation` before repair/rerun, `counterexample` pending hypothesis revision and replanning, `scope-discovery` pending plan/context/claim/validation updates, or any unresolved material contradiction blocks PIV and phase completion. An agent-authored `benign-deviation` is non-blocking only with evidence that material invariants hold, scope did not materially expand, the difference is understood, and the hypothesis remains usable.

The same classifiers govern discovery freshness: `scope-discovery`,
`counterexample`, and `invalid-validation` reopen/block the Repository
Discovery Gate until facts, decisions, claims, and validation are refreshed.

## Ralph Artifacts

```text
.nsp/artifacts/tmp/ralph/<epic-id>/
  STATE.md
  STATE.json
  decision-log.md
  validation-ledger.md
  risk-register.md
  repository-fact-ledger.md
  phase-XX/
    atdd.md
    plan.md
    validation.md
    handoff.md
```

Update the ephemeral Ralph roadmap/plan under `.nsp/artifacts/tmp/ralph/<epic-id>/` (e.g. `roadmap.md`, `STATE.md`, `STATE.json`). Never write in-flight phase plans, status, ATDD, validation ledgers, or handoffs to `.docs/roadmap/**`. Optional promotion of lasting product/architecture decisions into committed `.docs/**` is a separate human-owned step — not default epic storage or resume path.

## Validation Ledger

Record every gate run in `validation-ledger.md` (date, phase, exact command, honest PASS/FAIL, notes). A phase's `done` claim must point at ledger rows, not prose.

Record the discovery gate result and ledger path in the same ledger. A green
phase must cite only verified fact IDs; rejected facts are traceability-only.

## Handoff Template

Every session handoff (`phase-XX/handoff.md`) must end with:

```markdown
## Resume here

- Current phase:
- Current work item:
- Last successful validation:
- Last failed validation:
- Files changed:
- Decisions made:
- Discovery result and ledger:
- Verified fact IDs / excluded rejected IDs:
- Architecture decision/register IDs:
- Risks/blockers:
- Next exact command:
- Next exact file to inspect/edit:
```

## Interruption / Resume Protocol

1. **Rehydrate** — read STATE.json, the latest handoff, and the validation ledger.
2. **Assess** — verify branch, pending work, blockers, and whether the last validation actually passed.
3. **Discover and gate** — validate the same-run Repository Fact Ledger; if it is not green, extract missing facts before planning.
4. **Lay the plan** — pick the next smallest PIV slice. Keep decisions separate from facts and define ATDD gates before editing.
5. **Produce** — implement one bounded slice.
6. **Handoff** — update state, ledger, decisions, and the Resume-here footer before ending.

If interrupted mid-slice, record the partial state as `in-progress` with the exact next command. Never leave the state claiming more than was validated.

## Max-Scope Guardrails

- One bounded PIV slice per iteration. No opportunistic refactors outside the slice.
- Never skip or shrink large queues to finish faster — large queues are exactly why this skill exists.
- Scope expansion requires a recorded decision in `decision-log.md` with rationale.

## Expected Outputs

- Updated Ralph state tree, validation ledger, and ephemeral roadmap/plan status per phase under `.nsp/artifacts/tmp/ralph/<epic-id>/`.

## Validation Gates

- Phase ATDD gates green and recorded in the ledger before `done`.
- After every phase: repository build/test/governance gates per the Ralph roadmap's validation ladder.

## Handoff Artifacts

- `phase-XX/handoff.md` with the Resume-here footer. STATE.json accurate for a cold-start agent.

## Resume Rules

- Resume only from Ralph state + ephemeral roadmap/plan + ledger under `.nsp/artifacts/tmp/ralph/<epic-id>/`. Never treat `.docs/roadmap/` as execution or resume source of truth. Distrust any claim in prose that lacks a ledger row. Never rely on chat history.
- Resume only when the run-scoped Repository Fact Ledger is fresh and `DISCOVERY_READY`; never bind a phase to unknown, conflicted, or rejected facts.
- Resume prediction work by reference from the compact state only: accepted hypothesis, mode, latest mismatch status, unresolved material contradictions, next action, and bounded evidence references. Reassess rather than reuse when the latest result is a counterexample.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.
- Parallel phases are allowed only when their path/scope/artifact claims are disjoint.
- Closeout must wait for every phase claim to be released and `_nsp run closeout --target <repo> --run-id <id>` to succeed.

## Never Do

- approve PRs while `maintain_ready=false`
- skip maintain queue items because of PR size
- complete phases without validation evidence
- expand scope without a recorded decision
- rely on chat history for resumption
- write or resume in-flight epic state from `.docs/roadmap/**`
- dual-write execution state to both Ralph and `.docs/roadmap/`
