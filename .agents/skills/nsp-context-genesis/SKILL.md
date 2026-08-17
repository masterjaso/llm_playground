---
name: nsp-context-genesis
description: Extracts a deep evidence-backed model of an existing repository including features, technical architecture, personas, guidance, and relationships for downstream NSP workflows. Use when a repository needs exhaustive discovery, context generation, re-baselining input, architecture inventory, or evidence-backed project understanding. Do not use for first-adoption orchestration, routine diff alignment, or code repair execution.
user-invocable: false
---

# nsp-context-genesis

Use this skill as an internal capability for deep first-run repository understanding. It is not the initial setup user-facing skill. `nsp-adopt-ezra` invokes or uses it first and treats its evidence-backed outputs as authoritative input for downstream NSP workflows.

## Trigger Conditions

Deep first-run repository understanding, full context re-baseline, or full-repo context recovery — always under `nsp-adopt-ezra` orchestration or an explicit user request for re-baselining.

## Required Start

Deterministic substrate before traversal (the CLI reports facts. This skill owns extraction judgment):

```bash
_nsp status --target <repo>
_nsp map --target <repo>
_nsp context coverage --target <repo>
_nsp frontmatter index --target <repo>
```

## Required Inputs

- Repository working tree at the current commit.
- Existing `.docs/**` context (preserved unless evidence contradicts it).
- Genesis Ralph state when resuming (`.nsp/artifacts/tmp/ralph/genesis-sweep/STATE.json`).

## Owns

- repository traversal and structural understanding
- feature and technical architecture extraction
- persona inference and guidance normalization input

## Does Not Own

- user-facing setup orchestration (`nsp-adopt-ezra`)
- code repair execution (`nsp-code-hygiene`)
- durable sweep closeout validation authority (CLI)

## Relationship To Hygiene Maintain

`nsp-context-genesis` establishes or re-baselines the full repository context model. `nsp-maintain-steward` preserves that model during normal development by applying the same feature, technical, persona, guidance, evidence, grouping, and CCB rules to direct changed paths and bounded blast-radius context only.

## Core Concepts

- **end-user**: the project user who interacts with the system through exposed features such as UI, CLI, APIs, integrations, workflows, or runtime behavior.
- **dev-user**: the developer, maintainer, operator, or technical contributor who interacts with the codebase, architecture, tooling, deployment, tests, and maintenance workflows.
- **features**: externally observable capabilities that end-users interact with. Features must be framed in terms of user-facing behavior and outcomes using the **feature** prose contract (`.docs/guidance/nsp/documentation-prose-standards.md`).
- **technicals**: internal implementation details that dev-users rely on. Write with the **technical** prose contract (compact contracts, bullets, tables — not narrative dumps).
- **personas**: explicit models of distinct actors derived from real usage patterns, including end-users, dev-users, operators, maintainers, external systems, or integration actors. Each persona must have clear goals, responsibilities, interaction surfaces, and evidence.
- **guidance**: normalized, actionable instructions and conventions derived from existing documentation, source, scripts, configs, and observed project patterns that help personas correctly use, extend, validate, or operate the system.

## Document-type classification

Before drafting any concept document, set an explicit `documentType` of `feature`, `technical`, or `ccb-metadata`. Do not rely on destination path alone. If classification is ambiguous, fail clearly and record a gap — do not invent a style.
## Required Exhaustive Scan

Perform a full repository traversal and structural understanding pass. Inspect:

- code
- configs
- package and build scripts
- tests
- docs
- CI/workflows when present
- infra/deployment files when present
- CLI, API, UI, integration, workflow, and runtime entry points
- generated artifacts only when relevant, and never as canonical source over maintained inputs

## Evidence Rules

- Do not invent features.
- Do not infer unsupported personas.
- Do not treat stale docs as authoritative over current source.
- Every promoted finding must cite repository evidence paths.
- Ambiguous findings must be marked as gaps or unknowns, not converted into facts.
- Treat deterministic NSP CLI outputs as bounded evidence, not as semantic inference authority.

## Hygiene And Structure During Generation

Apply documentation hygiene rules while generating outputs:

- enforce naming consistency
- deduplicate overlapping docs
- maintain clear ownership
- use consistent terminology
- avoid orphaned or ambiguously named documents
- ensure generated artifacts conform to expected `.docs` directory conventions from the outset

## Grouping And Organization

Group related documents into coherent feature sets, technical sets, and guidance sets. Use logical folder hierarchy under `.docs` where repository complexity justifies it, such as:

- `.docs/features/auth/`
- `.docs/technical/auth/`
- `.docs/guidance/deployment/`

Colocate documents related to a shared domain or capability. Avoid flat unstructured collections of Markdown files when the repo has meaningful domains. Ensure grouping reflects real system boundaries and usage patterns, not arbitrary categorization, and maintain discoverability across feature, technical, and guidance groupings.

## Feature Extraction

Identify externally observable behaviors and capabilities. Each feature must be something an end-user can directly interact with or experience. Map features to entry points such as APIs, CLI commands, UI surfaces, integrations, workflows, or runtime outputs. Link each feature to supporting code paths, configurations, and tests. Avoid describing internal mechanics as features.

## Technical Extraction

Define module, package, and service boundaries. Construct dependency relationships. Describe runtime flows and execution paths. Document data transformations and state transitions. Identify integration points. Identify validation and testing strategy. Explain how and why the system works for dev-users.

## Persona Extraction

Infer personas from actual usage patterns in code, scripts, configs, tests, docs, and entry points. Explicitly distinguish end-user personas from dev-user, operator, and maintainer personas. Map each persona to workflows, responsibilities, and interaction surfaces. Detect stale, implied, or unsupported personas and flag them. Avoid persona sprawl.

## Guidance Extraction

Absorb existing READMEs, docs, comments, scripts, configs, workflows, and conventions. Extract implicit and explicit guidance used by dev-users, maintainers, operators, and end-users. Normalize guidance into consistent NSP-aligned structures. Align extracted target guidance into `.docs/guidance/**` outside `.docs/guidance/nsp/**`. Preserve existing documentation context and avoid breaking existing links, tooling expectations, or workflows during relocation or normalization. Clearly tie guidance to personas and technical areas. Avoid duplication and conflicting instructions.

## Output Requirements

Produce structured, evidence-backed context for downstream NSP workflows:

- structured feature inventory with evidence
- structured technical architecture inventory with evidence
- persona model with definitions, mappings, and evidence
- guidance inventory with normalization plan and persona alignment
- doc grouping plan
- explicit gaps, ambiguities, unknowns, and recommended follow-up evidence

## Expected Outputs

- The Output Requirements inventories above, staged for `nsp-adopt-ezra` promotion into `.docs/**`.
- Updated Genesis Ralph state phases as extraction proceeds.

## Validation Gates

- Every promoted finding cites repository evidence paths.
- `_nsp context validate --target <repo>` and `_nsp validate --target <repo>` pass after promotion.
- Ambiguities are recorded as gaps, never converted into facts.

## Handoff Artifacts

- Evidence-backed inventories handed to `nsp-adopt-ezra` for promotion.
- `.nsp/artifacts/tmp/ralph/genesis-sweep/STATE.json` phase updates with blockers and next actions.

## Resume Rules

- Resume from the Genesis Ralph state (`.nsp/artifacts/tmp/ralph/genesis-sweep/`). Re-verify previously extracted inventories against the current tree before building on them. Never rely on chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- invent features, personas, or relationships without repository evidence
- treat stale docs as authoritative over current source
- claim the deterministic CLI performed semantic extraction
- promote unreviewed inference as reviewed governance
