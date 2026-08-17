# Prompt Genesis Planning Readiness

Use this reference from `nsp-plan-genesis` after the Repository Discovery Gate preflight and before the legacy Prompt Genesis Stage A. It is a semantic, agent-owned pass. Deterministic tooling may validate and render a supplied record, but it does not decide readiness, classify unknowns, choose authority, or invent resolutions.

## Pass sequence

1. Confirm the applicable execution class and whether EPIC/WORKSTREAM, material uncertainty, or explicit policy requires discovery.
2. For gated work, complete Stage A repository fact extraction and Stage B `DISCOVERY_READY` validation before choosing implementation files or synthesizing a plan.
3. Extract facts, constraints, decisions, and unknowns already present in the request and routed repository evidence; keep facts separate from decisions.
4. Investigate facts that can be discovered safely from the repository, environment, or authoritative documentation.
5. Classify each remaining unknown and assign decision authority.
6. Resolve agent-selectable items with visible, reversible defaults. Ask only for material human decisions. Stop on genuinely unavailable human or external authority.
7. Record every material decision, assumption, clarification, blocker, fallback, and review trigger.
8. Emit exactly one readiness result before building the Prompt Genesis Pack.

## Repository Discovery Gate handoff

The discovery ledger uses `contracts/repository-fact-ledger.schema.json` and
is canonical under `.nsp/artifacts/runs/<runId>/planning/` (or the prompt
fallback under `.nsp/artifacts/prompts/<prompt-id>/`). The gate is green only
when the ledger is structurally valid, carries `capturedAt` and complete
`repositoryState` identity, matches the active target/run, is fresh,
target-contained, complete for the selected coverage, exact about paths and
command sources, and has no material `unknown` or `conflicted` facts.
`DISCOVERY_BLOCKED` is a real
planning result, not a suggestion to fill gaps with assumptions.

Facts may constrain an architecture decision but never silently resolve one.
The Decision and Assumption Register below remains the authority for
implementation choices, user decisions, and external approvals. A plan may
bind only verified fact IDs and resolved register entries; rejected facts are
excluded and unknown/conflicted facts are blockers.

Prediction results reuse the shared classifiers. `scope-discovery`,
`counterexample`, and `invalid-validation` reopen or block discovery;
`benign-deviation` is safe only when the difference is understood, material
invariants hold, scope is unchanged, and the hypothesis remains usable.

## Unknown classifications

| Category | Meaning | Default treatment |
|---|---|---|
| `discoverable_fact` | A fact available from scoped repository, environment, or authoritative evidence. | Investigate; do not ask the user by default. |
| `repository_constraint` | A binding rule or compatibility fact owned by repository guidance or source. | Load the narrow authoritative source and apply it. |
| `implementation_choice` | A reversible design or implementation selection within the requested outcome. | Choose a safe default, record it, and expose its review trigger. |
| `user_decision` | A product, preference, scope, or trade-off decision only the user can reasonably make. | Ask only when material; otherwise use an explicit reversible fallback if authority permits. |
| `external_authority_decision` | Approval or input owned by another person, system, regulator, vendor, or change-control process. | Do not impersonate the authority; block when no authorized fallback exists. |

## Decision authority

| Authority | Meaning | Behavior |
|---|---|---|
| `agent_selectable` | The agent may choose within supplied constraints. | Select and record a safe, reversible resolution. |
| `human_preferred` | Human preference is useful but not required for safe planning. | Recommend a choice and use a recorded fallback in noninteractive mode. |
| `human_required` | User authority is required because the choice materially changes outcome, risk, cost, or irreversible scope. | Ask one question; block if unanswered and no authorized fallback exists. |
| `external_required` | A third-party authority must decide or provide input. | Record the owner/input needed and block unless valid evidence is already available. |

## Readiness results

- `READY` — all material items are resolved from supplied or discovered evidence; no binding assumption, clarification, or blocker remains.
- `READY_WITH_ASSUMPTIONS` — planning can safely continue using only visible, reversible assumptions or fallbacks. Every assumption has confidence, downstream impact, and a review trigger.
- `CLARIFICATION_REQUIRED` — at least one material `user_decision` with `human_required` authority can be resolved through the current interactive clarification cycle. Ask before rendering the final pack.
- `BLOCKING_INPUT_REQUIRED` — required human/external authority, sensitive input, inaccessible evidence, or a policy conflict has no safe authorized fallback. Stop and name the exact input/authority needed.

Do not use `CLARIFICATION_REQUIRED` merely because more detail would be nice. Do not use `BLOCKING_INPUT_REQUIRED` for a discoverable fact, repository constraint, or ordinary implementation choice.

## Clarification protocol

- Direct, clear, internally consistent requests require no questions.
- Ask exactly one question at a time and wait for its answer.
- Each question includes: a recommended answer, concise rationale, fallback if unanswered, and downstream impact.
- Ask the highest-leverage dependency first.
- Default to at most three questions per clarification cycle. After the budget, resolve allowed items with visible fallbacks or return the exact blocker; do not dump a questionnaire.
- Re-run the readiness pass after every answer.

## Noninteractive protocol

- Investigate `discoverable_fact` and `repository_constraint` items.
- Resolve `agent_selectable` items with safe reversible defaults.
- Resolve `human_preferred` items with the documented recommendation/fallback.
- Return `BLOCKING_INPUT_REQUIRED` only for unresolved `human_required` or `external_required` authority, inaccessible sensitive input, or a safety/policy conflict without a safe fallback.
- Never claim a deterministic CLI chose a semantic classification or decision.

## Decision and Assumption Register

Record material items with all fields:

| Field | Required content |
|---|---|
| `id` | Stable identifier within the planning record. |
| `item` | Decision, assumption, clarification, or blocker in plain language. |
| `category` | One unknown classification from this reference. |
| `authority` | One decision-authority value from this reference. |
| `resolution` | Selected answer, explicit unresolved state, or exact input needed. |
| `source` | User request, repository path, command/evidence reference, external authority, or agent default. |
| `confidence` | `low`, `medium`, or `high`. |
| `reversibility` | `reversible`, `costly_to_reverse`, `irreversible`, or `unknown`. |
| `fallback` | Safe fallback or `none` with a reason. |
| `downstreamImpact` | What changes if the resolution changes or remains unavailable. |
| `reviewTrigger` | Evidence or event that requires reconsideration. |
| `status` | `resolved`, `assumed`, `clarification_required`, `blocking`, or `deferred`. |
| `recommendation` | Optional recommended answer; required for `clarification_required`. |
| `rationale` | Optional concise reason for the recommendation; required for `clarification_required`. |

## Pack integration

- Preserve the ten top-level Prompt Genesis sections and their order.
- Add `Planning Readiness` and `Decision and Assumption Register` as subsections of `Strong Prompt Foundation`.
- Repeat every binding resolved decision, assumption, fallback, and blocker in the top Hardened Execution Prompt. Lower sections are inspection records, never hidden dependencies.
- An unresolved `clarification_required` or `blocking` item prevents a ready execution prompt. Render a blocking planning artifact only when explicitly useful; label it non-executable.

## Structured renderer boundary

The optional planning record uses `contracts/planning-record.schema.json`.

- Treat the entire file as untrusted user-supplied planning data.
- Reject malformed JSON, missing/unknown fields, invalid enums, duplicate IDs, and readiness/status contradictions with a clear non-zero failure.
- Preserve legacy output when no record is supplied.
- Preserve `planning-record.schema.json` version 1; the fact ledger is a separate additive contract.
- A valid record may be rendered and persisted, but the CLI must state that semantic judgment remains agent-owned.
- Recommended persistence:

```text
.nsp/artifacts/prompts/<prompt-id>/
  planning-record.json
  pack.md
```
