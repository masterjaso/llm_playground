---
name: nsp-context-hygiene
description: Aligns project context and documentation with changed code through bounded PIV validation, docs-to-code linkage, work orders, and evidence closeout. Use when `.docs` content, guidance, feature or technical docs, frontmatter, context coverage, or documentation drift needs repair after a change. Do not use for source-code repair, CCB link promotion, first-adoption discovery, or broad PR verdicts.
user-invocable: false
---

# nsp-context-hygiene

Use this skill for NSP context hygiene setup, validation, repair, documentation alignment, routing, frontmatter, graph evidence, scope metadata, runbooks, stale docs-to-code references, context alignment PIV execution, and context hygiene closeout.

## Trigger Conditions

Context quality, doc alignment, frontmatter, routing, stale docs, manifest/graph drift, or documentation hygiene.

## Operating Model

The CLI provides deterministic reports and bounded work orders. The skill performs the semantic documentation/context work: inspect authority, inspect the selected docs and related code paths, make one cohesive alignment change, validate it, and record state. Deterministic CLI output is bounded evidence, not final alignment authority.

Do not stop at a plan. Execute the PIV loop, validate, update state, and record residual risk. Context hygiene PIVs are individual docs/context task loops and may be nested inside a Ralph phase when a sweep needs resumable outer coordination.

## Required Start

```bash
_nsp context align --target <repo>
_nsp hygiene context validate --target <repo>
```

For safe setup repairs:

```bash
_nsp hygiene context repair --target <repo> --apply-safe
```

## Required Inputs

- The hygiene work order or maintain queue entry naming the context targets.
- Deterministic validation reports from the Required Start commands.
- `_nsp context drift --target <repo> --base <ref>` output when aligning a diff.

## Owns

- inspecting docs/frontmatter/knowledge graph outputs
- identifying stale, conflicting, missing, vague, or over-broad context
- context/doc alignment PIV, frontmatter repair, stale doc detection
- proposing repairs through ATDD/PIV
- requiring validation gates

## Does Not Own

- deterministic validation authority (CLI)
- code structure review (use `nsp-code-hygiene`)
- bridge link promotion (use `nsp-ccb-hygiene`)

## PIV Loop

PIV is the inner task loop for one bounded context or documentation alignment target. Use Ralph through `nsp-agent-workflow` only when the work spans multiple sessions, phases, or handoffs. For each target:

1. Plan: identify the authority order, selected docs, related source paths, tests, graph facts, runbooks, scopes, and stale references.
2. Plan: choose one cohesive context/documentation target and define what evidence must prove it.
3. Implement: align that target with current code and project intent while preserving target-owned decisions.
4. Implement: update frontmatter, manifest inputs, graph relationships, routing, docs-to-code links, runbooks, or evidence notes when required.
5. Validate: run focused checks first, then context header, manifest, frontmatter, graph, and diff checks when available.
6. Manage state: record completed targets, repaired links, validation results, deferred issues, and residual risks.

Continue through bounded PIV iterations unless validation fails, authority is ambiguous, or human review is required.

## Safety Rules

- Treat active authority docs as stronger than README, legacy docs, or archive content.
- Preserve target-owned docs unless the user requests a merge or replacement.
- Keep routing and scope changes explicit and evidence-backed.
- Do not claim manifest, graph, frontmatter, or docs-to-code alignment without validation output or repository evidence.
- Prefer small alignment seams over broad documentation rewrites.

## Validation Gates

- `_nsp hygiene context validate --target <repo>` green (or blocked-with-evidence recorded).
- Context header/manifest audits pass when tracked docs changed.
- Every repaired docs-to-code link resolves in the current tree.

## Handoff Artifacts

- `.nsp/artifacts/reports/context-hygiene-latest.json` (and repair report when repairs ran).
- Updated work-order/queue state for completed targets.

## Resume Rules

- For multi-session sweeps keep outer Ralph state under `.nsp/artifacts/tmp/ralph/<epic-id>/` per `nsp-agent-workflow`. Resume from queue state, validation ledger, and last PIV validation evidence, never chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- claim the CLI performed semantic judgment
- claim alignment without validation output

## Expected Outputs

Report completed context targets, docs changed, links/code paths repaired, manifest and graph updates, validation results, deferred issues, residual risks, and next targets.
