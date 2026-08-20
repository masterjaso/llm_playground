---
name: nsp-clean-purify
description: Inspects and safely clears ephemeral NSP artifact storage in a target repository through guarded deterministic cleanup and final rescan. Use when a user asks to clean, purge, reset, or clear stale `.nsp/artifacts` output, remove noisy runtime evidence after completed work, or inspect cleanup blockers. Do not use for durable `.docs` cleanup, installed-skill reconciliation, run closure, or arbitrary file deletion.
---

# nsp-clean-purify

Clean Purify guards the boundary between NSP's temporary runtime store and
the target repository's durable project knowledge. It coordinates inspection,
preview, explicit cleanup, and post-cleanup verification. The `_nsp` CLI is
deterministic substrate only; this skill interprets its facts and owns the
operator-facing safety decision.

## Trigger Conditions

- A user asks to clear, purge, reset, or clean stale NSP artifacts.
- Completed or abandoned work left `.nsp/artifacts/**` large, noisy, or
  misleading for the next agent.
- An operator needs a bounded artifact inventory before deciding whether to
  remove runtime output.
- A cleanup attempt returned `blocked` or `partial` and needs an evidence-backed
  next action.

Do not use this skill to repair durable `.docs/**` context, reconcile installed
skills, close someone else's run, or delete arbitrary `artifacts/` directories.

## Required Start

Resolve the exact target and inspect it before any mutation:

```bash
_nsp status --target <repo> --json
_nsp artifacts --target <repo> --json
_nsp project cleanup --target <repo> --json
```

`project cleanup` is preview-by-default. The preview must be read before an
apply decision. Always pass `--target`; never let the current working directory
choose a cleanup target implicitly.

## Owns

- Bounded cleanup of `<repo>/.nsp/artifacts/**` through the guarded CLI surface.
- Sequencing inspection → dry-run preview → explicit apply → final rescan.
- Interpreting deterministic status, counts, blockers, errors, and remaining
  entries for the operator.
- Requiring explicit authorization before destructive apply or lifecycle
  override.

## Does Not Own

- The deterministic cleanup implementation or its validation authority.
- `.nsp/project.json`, `.nsp/metrics.json`, `.nsp/capabilities.json`,
  `.nsp/skills/**`, `.docs/**`, `.agents/**`, source files, or arbitrary
  repository artifacts.
- Closing runs, releasing claims, completing Ralph phases, or deciding whether
  evidence is safe to discard.
- Semantic claims that are not supported by the CLI result and a final rescan.

## Cleanup Workflow

### 1. Establish the boundary

Confirm the requested target is the intended repository. Treat only
`.nsp/artifacts/**` as the deletion scope. The cleanup substrate preserves the
artifact root's `.gitkeep` placeholder and does not remove adjacent durable
`.nsp` state.

### 2. Inspect and preview

Read the JSON from `artifacts` and `project cleanup`. Record:

- `artifactRoot`, `planned`, and `remainingEntries`;
- `open-run`, `active-claim`, and `incomplete-ralph` diagnostics;
- non-overridable safety or integrity blockers;
- `nspProvenance.semanticJudgmentByCli` (it must be `false`).

If the preview is `blocked`, stop and report the exact blocker. Do not infer
that an old artifact is disposable merely because it looks stale.

### 3. Apply only with explicit intent

When the user has explicitly authorized removal after seeing the preview, run:

```bash
_nsp project cleanup --target <repo> --apply --json
```

The command's exit status and JSON are authoritative for mechanical cleanup. A
normal apply must finish with `status: "cleaned"`, an empty `errors` array, and
an empty `remainingEntries` array.

### 4. Handle lifecycle blockers deliberately

Normal cleanup refuses to remove artifacts that contain open runs, active
claims, or incomplete Ralph state. Prefer handing those blockers to the owning
workflow so it can close or preserve them. If the operator explicitly accepts
discarding that lifecycle state, name the blockers and run:

```bash
_nsp project cleanup --target <repo> --apply --force --json
```

`--force` overrides only those lifecycle blockers. It never bypasses target or
artifact-root symlink checks, containment checks, read-integrity failures,
mutex contention, identity rechecks, or final-rescan failures.

### 5. Verify the result

After apply, require all of the following before reporting success:

```bash
_nsp artifacts --target <repo> --show-tree --depth 1 --max-entries 50 --json
```

- the result status is `cleaned`;
- `errors` and `remainingEntries` are empty;
- `.nsp/artifacts/.gitkeep` exists; and
- the final inspection shows no artifact entries other than `.gitkeep`.

If the result is `partial`, preserve the report, list the residual entries and
errors, and stop. Never rerun with `--force` just to make the status green.

## Required Inputs

- The exact target repository path.
- A user request that establishes whether this is inspection, preview, or
  removal; apply and `--force` require explicit intent.
- Fresh JSON output from the deterministic inspection and cleanup commands.
- Run ownership or lifecycle context when the preview reports blockers.

## Expected Outputs

- A bounded inventory or cleanup summary with target, scope, planned/removed
  counts, status, blockers, errors, and residual entries.
- A clear operator decision: preview-only, cleaned, blocked, or partial.
- A final inspection proving the survivor set when cleanup succeeds.
- No new durable project documentation and no second cleanup implementation.

## Validation Gates

- Cleanup commands always include an explicit `--target`.
- Inspection and dry-run precede a mutating apply.
- The CLI result carries deterministic provenance with
  `semanticJudgmentByCli: false`.
- Success requires `cleaned`, no errors, no remaining entries, and the root
  `.gitkeep` survivor.
- `blocked` and `partial` results remain non-success and include their exact
  evidence in the handoff.
- Non-overridable safety and integrity blockers are never bypassed.

## Handoff Artifacts

- The JSON inspection, preview, apply, and final-rescan results captured by the
  current agent or harness are the cleanup evidence.
- Successful cleanup intentionally leaves only
  `<repo>/.nsp/artifacts/.gitkeep`; do not create a new report inside the tree
  being purged.
- If lifecycle blockers remain, hand the exact run IDs, claims, Ralph paths, and
  next owning workflow to the operator rather than mutating them here.

## Resume Rules

This is a bounded operation, not a long-running Ralph workflow. If interrupted,
reinspect the target and run a fresh dry-run; do not resume from chat memory or
from a stale cleanup plan. A `partial` result is resumed only after its errors
and remaining entries have been understood or the operator changes the scope.

## Never Do

- Never use `rm`, recursive shell deletion, `find -delete`, or a custom script to
  remove NSP artifacts.
- Never omit `--target`, assume the repository from `cwd`, or broaden the scope
  beyond `.nsp/artifacts/**`.
- Never use `--force` automatically or treat it as a safety bypass.
- Never delete durable context, installed skills, project beacons, metrics,
  capabilities, or another workflow's source files.
- Never claim cleanup succeeded from assistant prose, an exit code alone, or a
  dry-run; require the fresh JSON result and final rescan.
- Never claim the deterministic CLI made a semantic judgment.
