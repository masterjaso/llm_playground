---
name: nsp-adopt-ezra
description: Orchestrates first adoption or a major context re-baseline for an existing repository by scanning code and docs, producing durable project context, and coordinating persona and hygiene setup. Use when onboarding NSP to an existing project, following `_nsp setup`, rebuilding context after large undocumented changes, or absorbing existing repository guidance. Do not use for routine diff maintenance, isolated explanations, or implementation execution.
---

# nsp-adopt-ezra

Use this skill as the single user-facing first-run/full-repo documentation-production orchestrator after `_nsp setup`. It uses outputs from `nsp-context-genesis`, then coordinates durable documentation generation plus initial hygiene and CCB readiness.

## Trigger Conditions

- immediately after `_nsp setup`
- first adoption of NSP in an existing repo
- brand-new repo scaffold needing project-specific context
- major re-baseline after large undocumented changes

**Primary entry only.** Delegated sub-agents must not run a full Genesis Sweep. If `NSP_ENTRY_MODE: delegated` is present, return control to the parent.

## Token-Intensive Warning

This is intentionally high-context and high-effort because it performs full repository understanding and documentation production. Do not use it for routine daily maintenance. After Genesis Sweep completes, routine development should use `nsp-maintain-steward` to preserve Genesis-aligned context on changed paths rather than rerunning Genesis.

## Required Start

1. Read `.nsp/artifacts/setup/SETUP_PROMPT.md` and `.nsp/artifacts/setup/GENESIS_SWEEP_RUNBOOK.md` when present.
2. Initialize resume state: copy `.nsp/artifacts/setup/GENESIS_SWEEP_STATE_TEMPLATE.json` to `.nsp/artifacts/tmp/ralph/genesis-sweep/STATE.json` (or rehydrate the existing state).
3. Run `_nsp status --target .`.
4. Run `_nsp context validate --target .` if available and appropriate.
5. Invoke or use `nsp-context-genesis` internally first.
6. Treat `nsp-context-genesis` evidence-backed outputs as the authoritative input to the sweep.

Do not ask the user to run `nsp-context-genesis`, `nsp-code-hygiene`, `nsp-context-hygiene`, or `nsp-ccb-hygiene` as separate first-adoption setup steps. Those are internal capabilities for this sweep when relevant.

## Required Inputs

- The generated setup directive and runbook artifacts from `_nsp setup`.
- `nsp-context-genesis` evidence-backed inventories.
- Existing target-owned docs (preserved unless evidence supports change).

## Owns

- first-adoption sweep orchestration
- durable documentation production
- persona registry updates and guidance normalization
- initial readiness validation coordination

## Does Not Own

- deterministic CLI validation authority
- routine daily maintenance (`nsp-maintain-steward`)
- provider/model configuration (none exists — NSP is model/harness agnostic)

## Document Production

- Generate project-specific `.docs/features/**/*.md` using **feature** prose (task/workflow narrative. CCB appendix last).
- Generate project-specific `.docs/technical/**/*.md` using **technical** prose (contracts, bullets, tables).
- Classify each document with an explicit `documentType` (`feature` | `technical` | `ccb-metadata`) before writing. Follow `.docs/guidance/nsp/documentation-prose-standards.md`. Ambiguous classification fails clearly.
- Preserve the fact that setup-authored feature and technical defaults are only `README.md` and `index.md`.
- Do not invent undocumented features.
- Every promoted durable doc must have repository evidence.
- Absorb useful existing docs instead of duplicating them.
- Preserve existing target-owned docs unless evidence supports update or stale marking.
- Keep NSP reusable guidance under `.docs/guidance/nsp/**`.
- Place target-specific guidance under `.docs/guidance/**` outside `.docs/guidance/nsp/**`.

## Personas

- Preserve supported personas.
- Mark stale unsupported personas with evidence.
- Derive missing personas from `nsp-context-genesis`.
- Avoid persona sprawl.
- Update `.docs/AGENT_PERSONAS.md` as the durable persona registry.

## Internal Hygiene Orchestration

Use internal NSP skills when relevant:

- `nsp-code-hygiene` for code structure or repair findings
- `nsp-context-hygiene` for docs/context alignment
- `nsp-ccb-hygiene` for doc-to-code bridge readiness and repair promotion

Keep these internal to the sweep for first adoption. The initial user-facing setup UX remains two steps: run `_nsp setup`, then copy/paste the generated Genesis Sweep prompt into the local agent/harness.

## Final Validation Expectations

Run deterministic validation where commands are available:

```bash
_nsp context validate --target .
_nsp validate --target .
_nsp graph build --target .
_nsp graph validate --target .
_nsp ccb build --target .
_nsp ccb validate --target . --readiness
```

Also run repository build, typecheck, and test commands when present.

## Expected Outputs

The Genesis output contract (see `.nsp/artifacts/setup/GENESIS_SWEEP_RUNBOOK.md`):

- `.docs/features/**` and `.docs/technical/**` evidence-backed concept docs
- `.docs/AGENT_PERSONAS.md` updates and `.docs/scopes/index.json` coverage
- normalized `.docs/guidance/**`
- CCB initial report, context coverage report, map readiness report
- unresolved risk list and validation summary in the Ralph state

## Validation Gates

- Every promoted durable doc cites repository evidence.
- `_nsp validate --target .` passes. `_nsp ccb validate --target . --tier linked` passes (anchored is the stretch goal).
- `_nsp status --target .` reports readiness `aligned-basic` or better.
- No durable content remains under `.nsp/artifacts/`.

## Handoff Artifacts

- Final report (files created/updated, evidence basis, validation results, hygiene issues completed or deferred, CCB readiness state, remaining risks).
- `.nsp/artifacts/tmp/ralph/genesis-sweep/STATE.json` with all phases `done` and next-action pointer for maintenance.

## Resume Rules

- Rehydrate from `.nsp/artifacts/tmp/ralph/genesis-sweep/STATE.json` at every session start. Verify previously promoted docs still validate before continuing. Never rely on chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- invent undocumented features or promote docs without evidence
- claim setup or the CLI semantically understood the repository
- leave durable context under `.nsp/artifacts/`
- skip validation gates to declare the sweep complete
