---
name: nsp-prompt-router
description: Routes a request to the least-sufficient execution class, persona, scope, context, and downstream Skill before broad repository work. Use when a new user or agent request needs intent classification, context selection, scope or risk routing, or a handoff to another Skill. Do not use for executing implementation, writing plans, performing review, or answering the request instead of handing off to a downstream Skill.
user-invocable: false
---

# nsp-prompt-router

## Trigger Conditions

Use for intent, scope, persona, docs, validation path, or context-receipt requests.

Must engage before broad context loading when a task involves broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair/hygiene work, project explanation, planning, validation, or evidence collection.

Do not engage for a tiny typo, isolated one-line answer, non-project general question, or narrow local command whose complete answer is already known.

### Delegated entry fast path

If the request contains `NSP_ENTRY_MODE: delegated`, validate the envelope against `.agents/skills/nsp-prompt-router/contracts/delegation-envelope.schema.json` (required: `ROUTING_STATUS: complete`, `EPIC_OWNER`, `TASK_TYPE`, `OBJECTIVE`, `SCOPE`, `EXCLUSIONS`, `EXPECTED_OUTPUT`).

- **Valid envelope:** do **not** re-run top-level genesis, persona discovery, epic/workflow classification, or broad request routing. Trust parent assignments. Load only path-local safety guidance plus `REQUIRED_CONTEXT` / scoped paths (envelope-only context; do not expect or request the parent transcript). Return a bounded result or handoff. Never initiate or own an epic.
- **Invalid/malformed/contradictory envelope:** fail closed — request parent clarification or return a constrained handoff. Do **not** silently fall back to full primary routing.
- **Nested delegation:** allowed only with a fresh valid envelope. `EPIC_OWNER` remains the original primary.

## Owns

- execution-class selection
- concise reason
- next workflow
- intent, persona, scope, risk, and mode awareness
- bounded Context Receipt wording
- discovery-gate applicability and handoff (without inventing repository facts)

## Does Not Own

- decomposition
- harness selection
- model selection
- executor selection
- Work Package creation
- Work Capsule execution
- deterministic validation

## Canonical Classifier Core

```text
Classify the request using the least sufficient execution class:

DIRECT:
One obvious action, no meaningful decomposition, immediate validation.

BOUNDED:
One coherent capability or defect outcome with one acceptance contract.

WORKSTREAM:
Multiple independently verifiable bounded outcomes under one delivery objective.

EPIC:
Multiple phases, durable architectural decisions, multiple workstreams,
migration/rollout sequencing, or roadmap-level resumption.

Do not use file count, estimated duration, token estimates, model capability,
provider identity, or harness identity as classification criteria.

Return the class, one-sentence reason, and next workflow.
```

## Required Start

**Primary entry only.** Skip this section on a valid delegated envelope.

```bash
_nsp context select --target <repo> --request "<summary>" --persona <id> --scope <scope> --limit 8 --receipt-v2
# cite the receipt path; do not paste full JSON unless --format json is required
# for EPIC/WORKSTREAM or material uncertainty, start/validate the run-scoped ledger before planning
_nsp plan-substrate discovery validate --target <repo> --run-id <run-id>
```

## Required Inputs

- The raw user/agent request (or a valid delegated envelope).
- For primary entry: inferred persona, scope, and risk assumptions (stated explicitly before running selection).
- For primary entry: deterministic candidate ranking from `_nsp context select`.
- For delegated entry: the parent envelope fields as the execution contract.
- For gated work: the run-scoped Repository Fact Ledger and its `DISCOVERY_READY`/`DISCOVERY_BLOCKED` result.

## Expected Outputs

- A completed Context Receipt v2 at `.nsp/artifacts/routes/latest-context-receipt.json` with the `semanticInference` block filled (persona, intent, risk, confidence) and left `unreviewed`.
- A downstream skill recommendation from the Handoff Map.
- A bounded read plan: requiredReads, optionalReads, doNotReadUnless.
- A discovery applicability decision: EPIC/WORKSTREAM/material uncertainty requires the Repository Discovery Gate; DIRECT is exempt unless explicitly gated.

## Context Receipt Fields

- `executionClass`
- `classificationReason`
- `nextWorkflow`
- `classificationRevision`

## Validation Gates

- The receipt validates against `contracts/context-receipt-v2.schema.json`.
- `deterministicSelection` facts come from actual CLI output, never invented.
- Over-budget receipts (`contextBudget.overBudget=true`) must narrow requiredReads or justify the expansion.
- The router must not label planning ready when the applicable fact ledger is absent, stale, unsafe, incomplete, or materially unknown/conflicted.

## Handoff Artifacts

- `.nsp/artifacts/routes/latest-context-receipt.json` — the routing contract for the next skill.
- `.nsp/artifacts/runs/<runId>/planning/repository-fact-ledger.json` — the evidence contract for gated planning; decisions remain in the planning readiness register.

## Resume Rules

- Stateless per request. The receipt is the durable handoff. A resuming agent re-reads the latest receipt instead of chat history and re-runs selection when the diff or request changed.

## Concurrency and Run Isolation

- Check active runs before recommending workflow routing: `_nsp run list --target <repo>`.
- Recommend joining an existing run only when workstream, base ref, and claimed paths/scopes are compatible.
- Recommend a new run when the request has a different workstream, ambiguous ownership, or collision risk.
- Warn when requested paths overlap active claims and instruct the agent to claim paths before edits.
- Include the run ID in context receipt wording and tell downstream skills to use run-scoped proof, maintain, review, evidence, and support artifacts.

## Trust Policy

Deterministic NSP facts are authoritative only when validator output exists. Mark recommendations as advisory unless backed by validation or human review. Never claim the CLI performed inference.

## Handoff Map

Prefer the eight public semantic skills for user-facing handoffs. Internal/worker/lifecycle skills remain composable but are not peer user choices.

| User need | Next skill |
|-----------|------------|
| First adoption / establish project knowledge | `nsp-adopt-ezra` |
| Explain / investigate / Domain Language | `nsp-insight-berean` |
| Plan / harden an idea into an executable package | `nsp-plan-genesis` |
| Implement / build | `nsp-build-bezalel` |
| Debug / diagnose defects | `nsp-debug-watchman` |
| Keep alignment on current diff / before PR | `nsp-maintain-steward` |
| Clear ephemeral NSP artifacts from a target repo | `nsp-clean-purify` |
| PR / change review | `nsp-review-discernment` (runs maintain first) |
| Context/code/CCB worker (composed by Steward) | `nsp-context-hygiene` / `nsp-code-hygiene` / `nsp-ccb-hygiene` |
| DIRECT | `direct-execution` via `nsp-build-bezalel` |
| BOUNDED / WORKSTREAM / EPIC execution mechanics | `nsp-build-bezalel` (composes agent-workflow / epic-execution) |
| Post-upgrade alignment | `nsp-upgrade-version` (lifecycle) |

## Discovery Gate Routing

- `EPIC` and `WORKSTREAM` requests require Stage A repository fact extraction,
  Stage B `DISCOVERY_READY`, and Stage C architecture decision registration
  before executable plan synthesis.
- Materially uncertain `BOUNDED` work uses the compact gate; `DIRECT` work is
  exempt unless the request explicitly requires discovery.
- Facts are not decisions. Bind exact paths and commands only to verified fact
  IDs; rejected facts are excluded and unknown/conflicted facts block readiness.

## Routing Rules

- The router classifies and routes. It does not pre-plan the work, select a harness, or select a model.
- Classification ignores harness capability. Harness affects realization later.
- Keep the classification reason concise and bounded to the observed request.
- For BOUNDED/WORKSTREAM/EPIC execution, recommend PIV/Ralph via `nsp-build-bezalel` / `nsp-agent-workflow`, and prefer clean-context delegated sub-agents when the harness supports them (implement/validate separation, envelope-only context, lowest-sufficient effort hints).

## Never Do

- claim `_nsp` invoked a model or agent
- claim validation passed without validator output
- broaden context without evidence
