# Planning readiness

The agent owns readiness and decision authority. Deterministic tooling validates and renders supplied records. Use the existing version 1 planning record; optional `dependsOn` preserves existing records.

## Evidence before decisions
For WORKSTREAM, EPIC, or material uncertainty, complete [discovery.md](discovery.md) first. Bind only verified fact IDs and resolved decisions. Investigate repository/environment facts and constraints before asking a human. Never mark an unknown fact as an assumption.

| Category | Treatment |
|---|---|
| `discoverable_fact` | Inspect scoped source, environment, or authoritative evidence. Inaccessible required evidence can be `blocking` with `agent_selectable` authority; it is not a preference question. |
| `repository_constraint` | Apply the authoritative local rule or compatibility fact. |
| `implementation_choice` | Choose a safe reversible default within authorized scope. |
| `user_decision` | Ask only when outcome, risk, cost, or irreversible scope needs human authority. |
| `external_authority_decision` | Obtain evidence from the actual authority or block. |

Authority is `agent_selectable`, `human_preferred`, `human_required`, or `external_required`. Resolve the first from evidence and safe defaults; use a recorded recommendation/fallback for optional preference. Required authority cannot be assumed from elapsed time or convenience.

## Dependency frontier
Record `dependsOn` IDs when an item relies on another. Resolve prerequisites before dependents. Unknown IDs, cycles, and unresolved prerequisites cannot support a ready dependent. Ask only material `user_decision` items with `human_required` authority on the currently resolvable frontier. Include recommendation, rationale, fallback, and impact. Do not ask downstream questions whose alternatives depend on an unanswered prerequisite. Continue independent investigation while waiting when possible.

After new evidence or an answer, revisit affected descendants. Preserve unrelated verified facts and settled decisions. Do not regenerate discovery, the whole register, or an accepted contract merely because a new turn arrived.

## Record and result
Each material entry carries `id`, `item`, `category`, `authority`, `resolution`, `source`, `confidence`, `reversibility`, `fallback`, `downstreamImpact`, `reviewTrigger`, and `status`. Optional `dependsOn` names prerequisites. `recommendation` and `rationale` are required for `clarification_required`. Keep evidence precise and defaults visibly reversible.

Statuses are `resolved`, `assumed`, `clarification_required`, `blocking`, and `deferred`. Assumptions are authorized choices/fallbacks, never invented facts. Deferred items cannot hide required acceptance prerequisites.

- `READY`: all material items resolved from supplied or discovered evidence.
- `READY_WITH_ASSUMPTIONS`: only visible authorized reversible assumptions/fallbacks remain, with impact and review trigger.
- `CLARIFICATION_REQUIRED`: a material human-required user decision can be answered interactively; ask before executable handoff.
- `BLOCKING_INPUT_REQUIRED`: required authority, evidence, sensitive input, or policy resolution is unavailable without an authorized fallback. Name the exact missing input/owner.

## Handoff
Normal output is one self-contained execution contract. Include binding resolutions, authorized assumptions, fallbacks/review triggers, gates, and evidence IDs. Blocking/clarification entries produce a clearly non-executable planning result.

`--expanded` requests inspection detail; `--persist-record` explicitly copies the canonical record. Keep bot decisions structured, not repeated prose packs. Reject malformed records, invalid enums, duplicate IDs, dependency errors, assumed facts, and readiness contradictions.
