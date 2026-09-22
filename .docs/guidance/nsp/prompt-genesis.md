<!-- nsp:meta
id: docs.guidance.nsp.prompt.genesis
kind: guidance
scope: guidance
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/prompt-genesis.md
graphTags: docs,nsp,prompt-genesis
validation: context-header-audit,manifest-check,secret-scan
owner: prompt-engineering
lastReviewed: 2026-06-27
replaces: .docs/features/prompt-generation-workflow.md
replacedBy: 
-->

# NSP Prompt Genesis Guidance

Prompt genesis is reusable NSP process guidance, not a target/user-facing feature doc. The canonical execution workflow lives in `nsp-plan-genesis` at `.agents/skills/nsp-plan-genesis/SKILL.md` and is installed into target repositories from the standard skill catalog.

Use `nsp-plan-genesis` to turn vague ideas into one accepted execution contract:

```text
inspect evidence -> resolve authorized choices -> obtain necessary human decisions -> accept contract -> revise invalidated parts
```

In the agent-owned planning readiness pass, classify unknowns, assign decision authority, and emit exactly one of `READY`, `READY_WITH_ASSUMPTIONS`, `CLARIFICATION_REQUIRED`, or `BLOCKING_INPUT_REQUIRED`. Resolve prerequisites before dependents. Investigate discoverable facts and repository constraints; select authorized reversible defaults and record their rationale. Escalate only material preferences, authorization, irreversible tradeoffs, or unavailable human authority. Ask only questions at the unresolved dependency frontier. A blocked discoverable fact needs investigation, not a human preference question.

Normal output is one accepted execution contract with binding decisions, acceptance criteria, and blockers. Use `--expanded` only for legacy inspection detail; do not repeat foundation, draft, hardened, and fallback copies. Reuse accepted decisions and evidence while their dependencies remain valid. Unresolved clarification or blocking input keeps dependent work non-executable; continue independent authorized work.

Quote user-supplied data, avoid broad repository dumps, and challenge assumptions with source, tests, schemas, and counterexamples. Never mark an unknown fact as an assumption. Collect acceptance criteria, validation, and evidence requirements before implementation starts. Stop investigating when the evidence is sufficient for the bounded decision.

For WORKSTREAM and EPIC work, Prompt Genesis may record compact assurance requirements when they materially improve sequencing or acceptance. Keep routine DIRECT work without extra assurance ceremony, distinguish persona from executor assignment, and put only binding independence, fresh-context, and unmet-assurance policy in the hardened prompt. State that realization is agent/harness-owned, require honest Result Receipt assurance, and carry job/ContextRef detail by compact artifact reference.

For long-horizon work, route the execution prompt to `nsp-epic-execution` so Ralph/PIV/ATDD loops, ephemeral roadmap/plan state under `.nsp/artifacts/tmp/ralph/<epic-id>/`, and resumable handoffs are owned by the epic workflow. Never use `.docs/roadmap/` for in-flight epic execution state.

The deterministic, noninteractive CLI helper remains `npm run prompt:workflow`. Optional `--planning-record <path>` input is validated and treated as untrusted data, including optional `dependsOn` prerequisites. Optional `--prompt-id <safe-slug>` places the contract under `.nsp/artifacts/prompts/<prompt-id>/`; only explicit `--persist-record` writes a canonical planning-record copy. The CLI never claims semantic planning judgment. Genesis stops at handoff; the authorized executor uses the same contract in a fresh session or an explicit execution cutover.
