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
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: prompt-engineering
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# NSP Prompt Genesis Guidance

Prompt genesis is reusable NSP process guidance, not a target/user-facing feature doc. The canonical execution workflow lives in `nsp-plan-genesis` at `.agents/skills/nsp-plan-genesis/SKILL.md` and is installed into target repositories from the standard skill catalog.

Use `nsp-plan-genesis` to turn vague ideas into a Prompt Genesis Pack:

```text
idea intake -> planning readiness pass -> strong prompt foundation -> draft implementation plan -> adversarial review -> hardened plan -> hard-mode execution prompt
```

Before Stage A, perform the agent-owned Planning Readiness Pass. Classify unknowns, assign decision authority, and emit exactly one of `READY`, `READY_WITH_ASSUMPTIONS`, `CLARIFICATION_REQUIRED`, or `BLOCKING_INPUT_REQUIRED`. Direct clear requests ask no questions; discoverable facts and repository constraints are investigated; safe reversible defaults are visible; clarification is one question at a time with a recommendation, rationale, fallback, and downstream impact.

Keep Planning Readiness and the Decision and Assumption Register inside Strong Prompt Foundation so the ten top-level sections remain stable. Repeat every binding resolution, assumption, fallback, and blocker in the top Hardened Execution Prompt. Unresolved clarification or blocking input produces a non-executable planning artifact. Session cutover chrome lives only in the pack renderer and `nsp-plan-genesis` output contract (STOP/AFTER banners around the fenced prompt).

The pack must preserve the Stage A through Stage E workflow, quote user-supplied data, avoid broad repository dumps, and collect acceptance criteria, validation, and evidence requirements before implementation starts.

For WORKSTREAM and EPIC work, Prompt Genesis may record compact assurance requirements when they materially improve sequencing or acceptance. Keep routine DIRECT work without extra assurance ceremony, distinguish persona from executor assignment, and put only binding independence, fresh-context, and unmet-assurance policy in the hardened prompt. State that realization is agent/harness-owned, require honest Result Receipt assurance, and carry job/ContextRef detail by compact artifact reference.

For long-horizon work, route the execution prompt to `nsp-epic-execution` so Ralph/PIV/ATDD loops, ephemeral roadmap/plan state under `.nsp/artifacts/tmp/ralph/<epic-id>/`, and resumable handoffs are owned by the epic workflow. Never use `.docs/roadmap/` for in-flight epic execution state.

The deterministic, noninteractive CLI helper remains `npm run prompt:workflow`. Optional `--planning-record <path>` input is validated and treated as untrusted data; optional `--prompt-id <safe-slug>` persists `planning-record.json` and `pack.md` under `.nsp/artifacts/prompts/<prompt-id>/`. The CLI never claims semantic planning judgment.
