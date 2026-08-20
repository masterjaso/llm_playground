---
name: nsp-maintain-steward
description: Keeps changed code, context, Domain Language, and Context-to-Code Bridge artifacts aligned through diff-scoped hygiene before handoff. Use when a task is complete and the current diff needs alignment, before or during PR preparation, after docs or code changes, or when a user asks to reconcile drift or run a maintain pass. Do not use for full-repository adoption, isolated code repair, bridge-only promotion, or merge verdicts.
---

# nsp-maintain-steward

Use this skill to align **only what changed** on the current branch or PR diff. It orchestrates bounded subsets of the hygiene skills — not full-repo sweeps.

## Trigger Conditions

- Before opening or updating a pull request
- After implementing a task when docs, code, or bridge links may have drifted
- Standalone "keep alignment" pass on the current diff
- Automatically invoked at the start of `nsp-review-discernment`

Do not use for cold-start setup, post-upgrade epics, or full-repo drift recovery — use `nsp-epic-execution` with the full `nsp-code-hygiene` / `nsp-context-hygiene` / `nsp-ccb-hygiene` skills instead.

## Required Start

Resolve the review comparison base first (same rules as `nsp-review-discernment` Step 0a):

```bash
_nsp review-manifest resolve-base --target <repo> [--base <ref>]
_nsp hygiene maintain --target <repo> [--base <ref>]
_nsp hygiene code minimize-review --target <repo> --base <ref>
```

When `--base` is omitted, NSP auto-resolves the parent/integration branch (`origin/main`, `main`, …) or falls back to `HEAD~1` with an explicit **no parent branch** warning.

Read the work order at `.nsp/artifacts/reports/hygiene-maintain-latest.json` before selecting phases. Confirm `comparisonRange`, `hasParentBranch`, and `changedFiles` match the diff you will align.

## Required Inputs

- The resolved comparison range and changed-file list from the work order.
- The maintain queue (`.nsp/artifacts/tmp/maintain-queue/STATE.json`).
- The latest context receipt when one was produced for this work.

## Owns

- diff classification (`trivial`, `docs_only`, `code_only`, `governance`, `mixed`)
- phase selection: code → context → ccb (only enabled phases from the work order)
- delegating **one PIV per direct changed path**, using blast-radius as context
- writing the maintain report artifact with phase statuses and residual risks

## Does Not Own

- merge or release verdict (use `nsp-review-discernment`)
- full-repo hygiene sweeps (use individual hygiene skills or `nsp-epic-execution`)
- deterministic validation authority (CLI)

## Genesis-Aligned Delta Maintenance

`nsp-maintain-steward` applies the same context model established by `nsp-context-genesis`, but only to the current diff and bounded blast-radius context. It inherits Genesis definitions for:

- **features**: externally observable end-user behavior.
- **technicals**: implementation, architecture, runtime, APIs, modules, data flow, validation, and dev-user understanding.
- **personas**: end-users, dev-users, maintainers, operators, integrations, or other actors affected by the change.
- **guidance**: project conventions, workflows, setup, validation, deployment, operations, and agent/human instructions.
- **CCB**: Feature → Technical → Code alignment.
- **evidence**: repository-backed proof for promoted context changes.

Maintain applies Genesis rules per direct changed path. It does not perform a full repository traversal, does not replace `nsp-context-genesis`, and does not replace `nsp-adopt-ezra`. Use `nsp-context-genesis` / `nsp-adopt-ezra` for first adoption, full re-baseline, or full-repo context recovery.

Related targets are bounded blast-radius context only. They inform the direct changed path PIV, but they are not separate PIV queue items.

### Assurance guidance

Carry compact assurance only when a direct-path PIV needs independence, fresh-context, or blocking unmet policy. Record assurance on the Result Receipt. Do not multiply queue items solely for role labels.

## Direct Path Context Decision Matrix

For each direct changed path, decide:

| Question | Context action |
|---|---|
| Does this change externally observable behavior? | Check or update feature docs |
| Does this change implementation architecture, runtime flow, APIs, modules, data, validation, or operational behavior? | Check or update technical docs |
| Does this change who uses, operates, maintains, or integrates with the system? | Check or update persona registry |
| Does this change commands, workflows, project conventions, setup, deployment, validation, or agent instructions? | Check or update guidance |
| Does this break or improve Feature → Technical → Code traceability? | Check or update CCB links |
| Is there no existing durable context for the affected behavior or technical concern? | Record a gap or create a candidate with evidence |
| Is existing context contradicted by source? | Mark stale, repair, or defer with evidence |

Do not create full-repo queues from this matrix. Apply it per direct changed path. Related targets are context only.

## Phase Order (per-PR)

Always run enabled phases in this order:

1. **Code** — when work order enables `code`
2. **Context** — when work order enables `context`
3. **CCB** — when work order enables `ccb`

## Phase Delegation

For each enabled phase:

- **direct_targets** — files changed in the comparison range (`base...head`). **One maintain queue item per direct path = one PIV unit.**
- **related_targets** — blast-radius paths (graph impact, CCB seams, scopes). **Context only** — load and review during each direct PIV. **do not** create separate queue items or PIV loops for related paths.

There is **no file-count cap** on direct changed paths. Large PRs require more PIV iterations, not scope reduction.

### Excluded from maintain PIV (diff inventory only)

These paths may appear in `changedFiles` but are **not** queued and **not** in blast-radius context:

- **`tests/**`** — test provenance. Not context/knowledge hygiene for this workflow
- **`.nsp/skills/**`** — generated compatibility index. Authoritative skill content lives under `.agents/skills/**`
- **`README.md`** — user-facing prose without NSP frontmatter

They appear as `not_applicable` in the review manifest. Do not create queue items for them.

### Code phase

Read `nsp-code-hygiene`. For each direct code path, PIV-align the change using `phaseRelatedContext.code` and `_nsp impact --base <ref>` as traversal evidence. Apply Minimum Code Gate. Add or confirm tests before risky edits.

### Context phase

Read `nsp-context-hygiene`. For each direct context path, align using `phaseRelatedContext.context` (affected docs, scopes). Regenerate manifest when tracked docs changed.

### CCB phase

Read `nsp-ccb-hygiene`. For each direct CCB path, align using `phaseRelatedContext.ccb`. Run `_nsp ccb repair --dry-run` when candidates exist. Promote only reviewed links.

## Maintain PIV queue (mandatory — execute, do not skip)

`_nsp hygiene maintain` writes `.nsp/artifacts/tmp/maintain-queue/STATE.json`:

- **items[]** — one entry per **direct changed path** per enabled phase
- **phaseRelatedContext** — blast-radius paths for that phase (not queue items)

**Do not mark maintain complete or allow PR approval until `maintain_ready=true`.**

**You must execute every queued direct item.** Partial progress (e.g. 8/N) is **not** closeout. Do not emit a final `nsp-review-discernment` verdict or approval until the maintain loop finishes. If the session ends early, resume from queue STATE — never treat partial progress as done.

When classification is `trivial`, phases short-circuit to `N/A` and `maintain_ready=true` without a queue.

### Prediction readiness

For an affected direct-path PIV with an activated Prediction Contract, require a current-run Prediction Result and contradiction state before readiness. Set `maintain_ready=false` when the result is missing or unclassified, validation is invalid and not repaired/rerun, a counterexample is pending hypothesis revision and replanning, scope discovery is pending plan/context/claim/validation updates, or a material contradiction remains unresolved. `benign-deviation` is non-blocking only when evidence shows the difference is understood, material invariants hold, scope did not materially expand, and the hypothesis remains usable.

This semantic check does not change the deterministic queue shape. Do not invent child queue items for related context or blast-radius paths; record the blocker on the affected direct-path PIV and resume it through its existing queue item. The CLI reports structural state only, while the skill/agent owns the classifier and readiness judgment.

### Execution loop (use `nsp-epic-execution` for resumability across sessions)

Initialize Ralph state for large PRs:

```text
.nsp/artifacts/tmp/ralph/maintain-pr-<pr-or-branch>/
  STATE.md
  STATE.json
```

Loop until `maintain status` exits **0**:

```bash
_nsp hygiene maintain next --target <repo>
# → PIV the direct path. Read blastRadiusContext for impact/context
_nsp hygiene maintain record --target <repo> --phase <code|context|ccb> --path <direct-changed-file> --status complete
_nsp hygiene maintain status --target <repo>
```

Update Ralph STATE after each recorded item. Resume across sessions until complete.

**Never** treat a large queue as a reason to skip work or approve without executing the loop.

## Closeout

After **all direct queue items** are `complete` or justified `skipped`/`blocked` with evidence:

```bash
_nsp ccb build --target <repo>          # if ccb phase ran
_nsp graph build --target <repo>        # if docs or routing changed
_nsp hygiene context validate --target <repo>
_nsp hygiene code validate --target <repo>
_nsp hygiene maintain status --target <repo>   # must exit 0
```

## Proof-of-read (required record)

Before closeout, write `.nsp/artifacts/routes/proof-of-read-latest.json` recording what context was actually used:

```json
{
  "schemaVersion": 1,
  "recordedBy": "nsp-maintain-steward",
  "contextReceiptPath": ".nsp/artifacts/routes/latest-context-receipt.json",
  "docsRead": ["<paths actually opened>"],
  "filesInspected": ["<code paths actually inspected>"],
  "skippedContext": [{ "path": "<path>", "reason": "<why it was safe to skip>" }],
  "deviationsFromReceipt": ["<any reads outside the receipt and why>"],
  "validationAfterDeviations": ["<commands rerun after deviating>"]
}
```

`_nsp context drift` reports a finding when this record is missing, and `nsp-review-discernment` treats a missing or stale proof-of-read as a review blocker.

## Required Output

- `.nsp/artifacts/reports/hygiene-maintain-latest.md` — work order + `maintain_ready`
- `.nsp/artifacts/tmp/maintain-queue/STATE.json` — direct-item PIV progress + phaseRelatedContext
- `.nsp/artifacts/routes/proof-of-read-latest.json` — proof-of-read record

## Validation Gates

- `_nsp hygiene maintain status --target <repo>` exits 0 (`maintain_ready=true`) before any final verdict.
- No affected direct-path PIV has blocked prediction state or an unresolved material contradiction.
- Closeout validators green: `_nsp hygiene context validate`, `_nsp hygiene code validate`, plus `_nsp ccb build` / `_nsp graph build` when those phases ran.
- Proof-of-read written and covering every direct changed path.

## Handoff Artifacts

- The Required Output artifacts above, consumed by `nsp-review-discernment`.
- Ralph state (`.nsp/artifacts/tmp/ralph/maintain-pr-<id>/`) for queues spanning sessions.

## Resume Rules

- Resume from the maintain queue STATE and Ralph state. Verify recorded items against the current diff before continuing. Never treat partial progress as done and never rely on chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- approve or claim PASS while `maintain_ready=false`
- stop mid-queue and emit a final review verdict (interim reports must say **INTERIM ONLY**)
- create separate PIV queue items for blast-radius related paths
- queue or PIV `tests/**`, `.nsp/skills/**`, or `README.md` (excluded from maintain scope)
- skip direct items because the PR is large — use epic-execution resumability instead
- claim semantic alignment without executing the queue loop
- claim validation passed without CLI output
- claim docs were read without recording them in proof-of-read
- substitute maintain for merge approval
