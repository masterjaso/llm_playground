---
name: nsp-prompt-router
description: Routes a request to the least-sufficient execution class, persona, scope, context, and downstream Skill before broad repository work. Use when a new user or agent request needs intent classification, context selection, scope or risk routing, or a handoff to another Skill. Do not use for executing implementation, writing plans, performing review, or answering the request instead of handing off to a downstream Skill.
user-invocable: false
---

# Prompt router

## Outcome
Choose the least-sufficient execution class, persona, scope, context, and skill for a primary request. A valid delegated envelope already supplies routing: validate it, load only required context and path-local safety, and execute the bounded assignment. Contradictory delegation returns to the parent, without restarting Genesis or owning an epic.
Validate `NSP_ENTRY_MODE: delegated` against `.agents/skills/nsp-prompt-router/contracts/delegation-envelope.schema.json`. The original primary remains epic owner.

## Workflow
1. Infer intent, risk, persona, and scope before broad reads. DIRECT is one action with immediate proof. BOUNDED is one outcome with stable acceptance. WORKSTREAM coordinates independent bounded outcomes. EPIC has durable dependent phases and integration gates. File count, time, provider, and model do not set class.
2. Run `_nsp context select --target <repo> --request "<summary>" --persona <role> --scope <scope> --limit 8 --receipt-v2` and cite its path. Reuse semantic routing for unchanged request/persona/scope/configuration when input checks permit. Load selected required reads within budget. Missing essential safety context or essential overflow needs a precise blocker or narrower expansion. Optional unverifiable graphs do not force repeated rebuilds for doc-only work.
3. Route read-only epic/workflow status, remaining gates, blockers, and closure-sequence reports to Herald (`nsp-workstatus-herald`) before incidental epic, plan, or test-failure words. For reports, skip receipt-producing selection and execution/discovery setup; read existing records only. Actual execution remains with its existing owner; HTTP/environment status is not Herald. Route planning to Genesis, implementation to Build, defects to Debug, explanation to Berean, first adoption to Adopt, diff alignment to Maintain, cleanup to Clean, and final review to Review. These are the ten public entries. Upgrade is lifecycle-triggered; workers, orchestration, prose, and prototype are internal.
4. For WORKSTREAM, EPIC, or material uncertainty, read [discovery.md](../nsp-prompt-router/references/discovery.md) before planning choices. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for coordination/resumption. Read the selected skill once, then retain references and material deltas.

## Evidence
Return objective, class/rationale, persona, scope, receipt path, skill, and unresolved material decisions. The deterministic CLI ranks and validates data; the agent owns semantic routing. Missing substrate permits a narrow disclosed fallback, never invented graph facts.

## Boundaries
Routing is a handoff, not another execution phase. Resume the accepted contract; changed evidence reopens only affected decisions. Do not expose internal skills as menu choices, load every reference, or create Ralph/capsules for clear DIRECT work.
