---
name: nsp-review-discernment
description: Performs exhaustive evidence-backed review of a pull request, merge request, patch, commit range, or changed-file set across correctness, security, tests, hygiene, knowledge, and release readiness. Use when a user asks for PR review, code review, merge review, release-readiness assessment, or severity-ranked findings. Do not use for implementing fixes, planning a feature, or replacing Maintain's diff-alignment pass.
---

# Pull Request Review Workflow

Use this skill to perform a precise, evidence-based review of a pull request, merge request, patch, commit range, changed-file list, or proposed code change. Stay project-agnostic and language-agnostic: adapt to the repository's stack, size, architecture, conventions, and review guidance instead of assuming any particular framework, tool, file path, company process, or coding style.

## Trigger Conditions

A request to review a pull request, merge request, patch, commit range, changed-file list, or proposed code change.

## Owns

- PR scope review and severity-ranked findings
- hygiene and CCB impact assessment for the diff
- release-readiness verdict

## Does Not Own

- merge authority (human/repo policy)
- fix implementation (hand off to hygiene skills)
- deterministic validation authority (CLI substrate only)

## Required Inputs

- The resolved review base artifact (`.nsp/artifacts/reviews/review-base-latest.json`).
- The maintain work order, queue state, and review manifest.
- Proof-of-read record and context drift report for the comparison range.

## Required Start

**Step 0a — Resolve and verify review base (mandatory before any verdict):**

Every review must compare **head** (current branch / PR head) against a verified **parent or integration branch** (e.g. `origin/main`, PR base branch). Do not approve without this comparison.

```bash
_nsp review-manifest resolve-base --target <repo> [--base <ref>]
```

Read `.nsp/artifacts/reviews/review-base-latest.json`. Required fields:

- `ok: true` and non-empty `comparisonRange` (e.g. `origin/main...HEAD`)
- `hasParentBranch: true` for PR merge-target reviews — when `false`, the CLI used **prior-commit fallback** (`HEAD~1`). Call this out prominently and use verdict **`Not enough information to approve`** unless the user explicitly requested single-commit review
- `changedFileCount` and `changedFiles` — inspect these files. An empty diff means nothing was compared

When reviewing a GitHub PR, prefer `--base` matching the PR base ref (e.g. `main` or `origin/main`). If `gh pr view` is available, confirm the resolved base matches `baseRefName`.

**Step 0b — Build maintain work order (always after base is verified):**

```bash
_nsp hygiene maintain --target <repo> [--base <ref>]
_nsp hygiene code minimize-review --target <repo> --base <ref>
```

Read `.nsp/artifacts/reports/hygiene-maintain-latest.json` and `.nsp/artifacts/tmp/maintain-queue/STATE.json`:

- **direct_targets** — changed paths in the diff. **one PIV queue item each**
- **related_targets** / **phaseRelatedContext** — blast-radius **context** for each direct PIV (not separate queue items)
- **Excluded:** `tests/**`, `.nsp/skills/**`, `README.md` — not queued (`not_applicable` in manifest)

When `--base` is omitted, maintain auto-resolves the same parent/integration branch as `review-manifest resolve-base`.

**Step 0c — Execute maintain PIV queue (mandatory — complete every item):**

Read **`nsp-maintain-steward`** and **`nsp-epic-execution`**. You **must** execute the maintain queue until `maintain_ready=true` before a final approve verdict.

- One PIV per **direct changed path**. Use blast-radius context during that PIV (excluding `tests/**` from context)
- Verify maintain completed Genesis-aligned delta review for changed paths: affected features, technicals, personas, guidance, evidence, and CCB links were checked or explicitly marked not applicable
- **No file-count limit** — large PRs need more iterations, not scope reduction or skipping
- **Every queued item must be recorded** — partial progress (e.g. 8/N) is failure to close out. Continue the loop
- Resume across sessions with Ralph artifacts: `.nsp/artifacts/tmp/ralph/maintain-pr-<id>/` plus maintain-queue STATE
- **Do not** use `Not enough information to approve` because the queue is large — that means **continue the loop**, not stop
- **Do not** emit sections 1–7 of this review format until `maintain_ready=true` unless the output is explicitly labeled **INTERIM PROGRESS ONLY**

```bash
_nsp hygiene maintain next --target <repo>
# → delegated skill PIV on direct path + blastRadiusContext
_nsp hygiene maintain record --target <repo> --phase <code|context|ccb> --path <direct-file> --status complete
_nsp hygiene maintain status --target <repo>   # must exit 0 before approve
```

**Step 0d — Review substrate (after maintain queue complete or for interim status):**

```bash
_nsp review-manifest --target <repo> [--base <ref>]
_nsp context drift --target <repo> --base <ref>
_nsp validate --target <repo>
```

Use deterministic CLI outputs, **review-base artifact**, maintain report, drift report, and **changed file list from the resolved comparison** as substrate for review findings.

**Step 0e — Verify proof-of-read (blocker when missing):**

Read `.nsp/artifacts/routes/proof-of-read-latest.json` (written by `nsp-maintain-steward`). It must record: the context receipt path used, docs actually opened/read, files inspected, skipped context with reasons, deviations from the receipt, and validation run after deviations. When the record is missing or does not cover the direct changed files, treat it as a review blocker: the maintain pass cannot prove which context informed the changes. `_nsp context drift` reports the same gap deterministically.

## Review Principles

- **Never approve** if review base resolution failed, if no parent/integration comparison was established for a PR review, if **`maintain_ready=false`**, if **`reviewComplete=false`**, if any direct changed path was not PIV-aligned and recorded, or if you skipped Step 0c execution.
- **Never** treat blast-radius `related_targets` as a reason to multiply queue items — they are **context** consumed during each direct PIV.
- **Never** use `Not enough information to approve` because the maintain queue has many direct items — execute via `nsp-epic-execution` across sessions until `maintain_ready=true`.
- Large PRs may be phased for execution depth, but the **file inventory and coverage accounting must be exhaustive**. Use `.nsp/artifacts/reviews/review-manifest-latest.json` — every changed file must appear with `reviewed` or explicit `not_reviewed` status.
- **Focus suggestions** (ranked inspection areas) are advisory only — they do not count as review coverage.
- Do not say "approved," "success," "complete," or "reviewed" unless every changed behavioral/config/security/data/API/test/doc file is accounted for in the manifest and maintain queue.

- Review only what is actually available: diff, files, PR metadata, CI output, tests, docs, configuration, maintain report, and repository guidance.
- Consult repository-specific guidance when present, such as agent instructions, contribution/review docs, architecture docs, CI config, PR templates, security docs, release docs, or ownership docs.
- Prefer concrete, actionable findings over broad advice. Do not invent issues, inspected files, successful commands, test results, or project rules.
- Prioritize correctness, security, data integrity, runtime behavior, API contracts, compatibility, deployment/configuration risk, test gaps, maintainability, and scope creep.
- Acknowledge files or areas with no issues. Avoid style nitpicks unless they affect maintainability, consistency, or documented repository rules.
- Label uncertainty clearly and recommend the smallest useful validation step.
- Do not approve if critical or high correctness, security, data, data migration, or deployment blockers remain.
- Do not re-run full-repo hygiene epics unless maintain returned `PARTIAL` with an explicit escalation note.

### Assurance review

Treat independence and fresh-context as evidence-backed invariants, not role labels. Prefer rather than require independence for normal review; require it only for material security, destructive migration, data-loss, or release-blocking risk. Do not claim independence without bounded evidence. Surface only material unmet assurance in the existing seven-section review.

### Prediction evidence review

Treat unresolved material contradictions, unclassified mismatches, invalid validation that has not been repaired and rerun, counterexamples pending replan, scope discovery pending plan/context/claim updates, and unjustified `benign-deviation` classifications as findings and blockers. Relabeling a mismatch without evidence is a finding; do not silently accept it. A benign deviation is justified only when evidence shows the difference is understood, material invariants hold, scope did not materially expand, and the hypothesis remains usable.

These conditions block PR readiness even if acceptance tests otherwise pass. The CLI may validate Prediction Result structure and lifecycle, but the reviewer owns the semantic assessment and must not attribute classifier judgment to deterministic tooling.

## Review Workflow

1. Run **Step 0a** (`review-manifest resolve-base`) and record comparison range, parent branch status, and changed-file count in Review Scope.
2. Run **Steps 0b–0c** (`hygiene maintain` + **execute the direct-item PIV queue** via epic-execution resumability). Record maintain artifact path, queue progress, and Ralph state path in Review Scope.
3. Run **Step 0d** (`review-manifest`, `context drift`, `validate`) for deterministic substrate after maintain work (or for interim status while queue in progress).
4. Establish the available review scope: PR number/title if available, **resolved base/head refs and comparison range**, commit range, changed-file count, diff source, visible test/CI evidence, maintain queue progress, and repository guidance consulted.
5. Inspect **every changed file** in the exhaustive manifest. For large PRs, group trivial/generated/lockfile/vendor files for summary prose but still record each file's review status (`reviewed` / `not_reviewed`) — never drop files from the inventory.
6. Compare changes against stated PR intent and repository rules. Look for inconsistent updates across public APIs, schemas, interfaces, docs, tests, migrations, configuration, and deployment prerequisites.
7. Classify findings by severity. Use `Critical` for likely severe security/data-loss/outage issues, `High` for likely blockers, `Medium` for important but non-blocking correctness or maintainability issues, `Low` for minor risks, and `Suggestions` for optional improvements.
8. Build a dynamic compliance checklist from both generic review criteria and any repository-specific guidance discovered.
9. End with a release-readiness verdict that distinguishes code blockers from release/deployment confirmations.

## Required Output Format

Produce a structured markdown review with the following sections in order (**sections 1–7 only** — this is the complete normal output).

**Do not** include command-trace appendices or NSP-internal process-validation sections in normal PR reviews. That misleads developers. If NSP maintainers need process validation, capture it in maintainer notes outside developer-facing PR review output.

### 1. Review Scope

State what was reviewed, including available PR number or title, **resolved base ref, head ref, comparison range (`base...head`), whether a parent/integration branch was established or prior-commit fallback was used**, changed-file count, commit range, diff source, **hygiene maintain report path and phase statuses**, and repository guidance consulted. Explicitly state limitations when the full diff, full files, test output, CI status, project docs, or runtime context were unavailable. **Never imply files, commands, or tests were checked unless evidence was available.** If `hasParentBranch` is false, state that PR merge-target comparison was not established.

### 2. File Review Summary Table

Include a table with these columns:

| File | Review Status | Type | Purpose | Key Changes | Issues Found | Severity |
|---|---|---|---|---|---|---|

Cover **every changed file** from the review manifest (`reviewed` / `not_reviewed` / `N/A`). For large PRs, you may summarize trivial rows in prose but the manifest must still list all files — never silently truncate the inventory.

### 3. Detailed Findings

Group findings under these headings:

- Critical
- High
- Medium
- Low
- Suggestions

For each finding, use this format:

```markdown
- Location: file path and function/class/area if available
  What is wrong:
  Why it matters:
  Suggested fix:
  Confidence: High / Medium / Low
```

If there are no findings in a severity category, write: `none identified in the reviewed diff.`

### 4. Compliance Checklist

Use `PASS`, `FAIL`, `PARTIAL`, or `N/A` with a short evidence note for every item. Prefer `PASS` and `FAIL` for clear compliance intent. Use `PARTIAL` only when evidence shows mixed compliance. Include generic checks plus repository-specific checks discovered from available guidance. Do not hard-code language, framework, tool, company, ticketing, or file-path assumptions.

Generic checks to consider:

| Check                                                                                                    | Status | Evidence |
| -------------------------------------------------------------------------------------------------------- | ------ | -------- |
| Pre-PR hygiene maintain completed or N/A with evidence (`.nsp/artifacts/reports/hygiene-maintain-latest.md`) |        |          |
| Maintain PIV queue complete (`maintain_ready=true`. One PIV per **direct** changed path. Related paths used as context only) |        |          |
| Genesis-aligned context delta reviewed: affected features, technicals, personas, guidance, evidence, and CCB were checked for changed paths |        |          |
| Review base resolved and diff compared against parent/integration branch (`.nsp/artifacts/reviews/review-base-latest.json`) |        |          |
| Review manifest exhaustive (`reviewComplete=true`, `.nsp/artifacts/reviews/review-manifest-latest.json`) |        |          |
| Every changed file accounted for (`unreviewedFileCount=0`) |        |          |
| Proof-of-read recorded and covers direct changed files (`.nsp/artifacts/routes/proof-of-read-latest.json`) |        |          |
| Context drift findings addressed or explicitly accepted (`_nsp context drift --base <ref>`) |        |          |
| Does the change match the stated PR scope?                                                               |        |          |
| Are public APIs, contracts, schemas, migrations, or interfaces updated consistently?                     |        |          |
| Are tests added or updated for new or changed behavior?                                                  |        |          |
| Are docs, changelog, release notes, or user-facing guidance updated when needed?                         |        |          |
| Are new dependencies justified and safe?                                                                 |        |          |
| Are environment variables, secrets, permissions, infrastructure, or deployment prerequisites documented? |        |          |
| Are error handling, logging, observability, and failure modes considered?                                |        |          |
| Are security, auth, privacy, and data handling impacts reviewed?                                         |        |          |
| Are backward compatibility and rollback concerns addressed?                                              |        |          |
| Are generated files, lockfiles, build config, and compiler/runtime config changes intentional?           |        |          |
| Are large files, broad refactors, or cross-cutting changes justified by the PR scope?                    |        |          |

Add repository-specific rows when applicable and cite the guidance source in the evidence note.

### 5. Test and Validation Review

Summarize:

* Tests claimed in the PR
* Tests or CI evidence actually visible
* Missing tests or validation gaps
* Manual smoke tests that should be performed before merge
* Deployment/runtime checks needed before release

### 6. PR Hygiene Review

Assess whether the PR description clearly includes:

* Summary
* Motivation, linked issue, or ticket when applicable
* Type of change
* Dependencies
* Configuration or environment changes
* Testing evidence
* Rollout or rollback notes when relevant
* Screenshots or examples for UI or user-facing changes when relevant
* Hygiene maintain summary or link to maintain report when behavior/docs/code anchors changed

### 7. Release Readiness Verdict

End with exactly one concise verdict:

* `Approved / approvable`
* `Approvable after listed confirmations`
* `Request changes`
* `Not enough information to approve`

Name the blockers or confirmations required before merge. Distinguish code blockers from release/deployment confirmations. **Use `Not enough information to approve` when review base resolution failed, parent branch was not established for a PR review, or maintain execution was not started.** **Do not** use this verdict because the direct-item queue is large — continue execution instead. Use **`Request changes`** or interim **`Approvable after listed confirmations`** only when maintain is complete but specific blockers remain.

## Validation Gates

- `maintain_ready=true` and `reviewComplete=true` before any approve verdict.
- Every changed file accounted for in the review manifest (`unreviewedFileCount=0`).
- `_nsp validate --target <repo>` output recorded, not asserted.
- No unclassified mismatch, unresolved material contradiction, invalid validation, pending counterexample/scope-discovery action, or unjustified benign deviation remains.

## Handoff Artifacts

- The sections 1–7 review output.
- `.nsp/artifacts/reviews/review-manifest-latest.json` and `.nsp/artifacts/reviews/latest-review.md` as deterministic backing evidence.

## Resume Rules

- Interim outputs must be labeled **INTERIM PROGRESS ONLY**. Resume from the review manifest, maintain queue, and Ralph state (`.nsp/artifacts/tmp/ralph/maintain-pr-<id>/`), never chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Review Axes

Independent exhaustive axes (do not reduce rigor for small diffs):
1. Intent / Spec
2. Architecture / Standards
3. Runtime / Tests
4. Risk / Security / Data
5. Knowledge / Domain / CCB
6. Verification

## Verification-Before-Claim

No claim such as fixed, complete, passing, ready, or resolved without fresh supporting evidence. Applies even to DIRECT work.

## Never Do

- approve while `maintain_ready=false`, `reviewComplete=false`, or the proof-of-read is missing
- invent inspected files, successful commands, or test results
- treat focus suggestions as review coverage
- claim the deterministic CLI performed the review judgment
