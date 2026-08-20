---
name: nsp-ccb-hygiene
description: Maintains the Context-to-Code Bridge by reviewing deterministic coverage candidates, resolving anchors, and promoting only evidence-backed links into the reviewed ledger. Use when CCB coverage reports, stale or missing doc-to-code links, repair candidates, or bridge validation need review and promotion. Do not use for general docs hygiene, source-code repair, or treating CLI-generated candidates as already reviewed.
user-invocable: false
---

# nsp-ccb-hygiene

## Trigger Conditions

Doc-to-code links, bridge staleness, missing mappings, CCB repair promotion, or raising the CCB readiness tier (`structural` → `linked` → `anchored` → `reviewed` → `trusted`).

## Required Start

```bash
_nsp ccb build --target <repo>
_nsp ccb validate --target <repo> --tier trusted
_nsp ccb coverage --target <repo>
```

## Required Inputs

- Deterministic coverage/candidate reports (`.nsp/artifacts/ccb/coverage-latest.json`, `.nsp/artifacts/ccb/repair-candidates.json`).
- The feature docs, technical docs, and anchored code for each candidate under review.

## Owns

- inspecting technical docs and generated CCB index
- finding missing/stale/weak doc-to-code links
- proposing bridge repairs
- reviewing candidate promotions from the coverage report
- promoting only evidence-backed links into the reviewed ledger
- keeping repair guesses unreviewed until human/agent promotion

## Does Not Own

- deterministic bridge build/validate (CLI)
- holistic context or code review (hygiene skills)

## Promotion Workflow (required order)

1. Read the deterministic candidate report: `.nsp/artifacts/ccb/coverage-latest.json` (`candidatePromotions`) and `.nsp/artifacts/ccb/repair-candidates.json`.
2. For each candidate, inspect the actual evidence: open the feature doc, technical doc, and anchored code. Confirm the link is real and current.
3. Promote only evidence-backed links by appending them to `.docs/graph/ccb-reviewed-links.json` (`{ type, source, target, reviewStatus: "reviewed", reviewedBy, reviewedAt, evidence: [paths] }`).
4. Record why each link was promoted or left unreviewed (evidence paths, or the missing-evidence reason).
5. Update Ralph state when the promotion pass is part of long-running work (`.nsp/artifacts/tmp/ralph/<epic-id>/`).
6. Rerun `_nsp ccb validate --target <repo> --tier reviewed` (and `--tier trusted` after graph rebuild) and record the output as evidence.

## Expected Outputs

- Promoted, evidence-backed entries in `.docs/graph/ccb-reviewed-links.json`.
- A promotion record explaining why each candidate was promoted or left unreviewed.

## Validation Gates

- `_nsp ccb validate --target <repo> --tier reviewed` after promotion (and `--tier trusted` after graph rebuild), output recorded as evidence.
- Every promoted anchor resolves (`repo://` path, range, symbol, or test exists).

## Handoff Artifacts

- `.docs/graph/ccb-reviewed-links.json` (durable ledger).
- Refreshed `.nsp/artifacts/ccb/coverage-latest.{json,md}` after the promotion pass.

## Resume Rules

- For long promotion passes keep Ralph state under `.nsp/artifacts/tmp/ralph/<epic-id>/`. Resume from the coverage report and ledger diff, never chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- mark CLI repair candidates as reviewed without human/agent review of the actual evidence
- promote a link whose code anchor does not resolve
- claim the CLI inferred correct bridge links
- edit `.nsp/artifacts/**` as if it were the durable ledger (the ledger lives at `.docs/graph/ccb-reviewed-links.json`)
