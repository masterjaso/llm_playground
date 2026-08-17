---
name: nsp-upgrade-version
description: Reconciles target-owned NSP guidance, skills, and tool surfaces after a deterministic `_nsp update-version` or alignment run, preserving customizations and recording residual risks. Use when NSP-managed assets were refreshed, a target is upgrading versions, or an upgrade report and backups need review. Do not use for executing the upgrade itself, ordinary feature changes, or application behavior work unrelated to upgrade integration.
user-invocable: false
---

# NSP Upgrade Version

Use this skill when a target repository upgrades NSP and needs its NSP-specific guidance, skills, and workflow instructions reviewed in place.

## Trigger Conditions

After `_nsp update-version` refreshes NSP-managed assets in a target repository, or when a target upgrade left conflicts, customizations, or stale guidance to reconcile.

## Owns

- upgrade alignment review of refreshed NSP-managed files
- customization reconciliation (target-owned edits vs new canonical guidance)

## Does Not Own

- deterministic upgrade execution (`_nsp update-version` / `_nsp align target`)
- application behavior changes beyond upgrade-required integrations

## Required Inputs

- The deterministic upgrade report (`.nsp/artifacts/reports/nsp-version-upgrade-latest.md`).
- Backups under `.nsp/artifacts/upgrades/` for changed files.

## Required Start

```bash
_nsp update-version --target <repo>
# Optional: force provider-native skill refresh for a detected harness adapter
_nsp update-version --target <repo> --provider <harness-id>
_nsp align target --target <repo> --dry-run
_nsp align target --target <repo> --apply-safe
_nsp status --target <repo>
_nsp hygiene code validate --target <repo>
_nsp hygiene context validate --target <repo>
```

Use `--dry-run` first when the target has heavy local customization. The command is deterministic and bounded to NSP-managed files. It writes backups under `.nsp/artifacts/upgrades/` before replacing existing NSP-managed guidance. From 2.2.0 it refreshes all setup `.docs/guidance/nsp/**` guidance (guidance language), prose standards, and auto-detected provider-native skill trees for harness adapters that declare native skill roots.

## Upgrade Review

1. Read `.nsp/artifacts/reports/nsp-version-upgrade-latest.md`.
2. Compare refreshed files with backups when the report shows `updated`.
3. Preserve target-specific rules that remain valid, but do not keep stale NSP guidance that conflicts with the current package.
4. Apply the Minimum Code Gate before adding compatibility code for the upgrade.
5. Validate with focused checks first, then run target governance and hygiene validation.
6. Review the reconciled `.agents/skills/` catalog and run alignment skills (`nsp-context-hygiene`, `nsp-code-hygiene`, `nsp-ccb-hygiene`) when docs, code, or bridge files changed.

## Safety Rules

- Do not edit application behavior unless the upgrade report identifies a target-owned integration that requires it.
- Do not add runtime dependencies to the target repository for NSP guidance upgrades.
- Do not remove target-specific safety, security, accessibility, migration, or validation rules without human review.
- Treat deterministic upgrade output as evidence, not semantic authority.
- Record residual risks and files requiring human review.

## Expected Outputs

Report files refreshed, backups reviewed, target-specific merges made, validation commands run, residual risks, and whether human review is required before merge.

## Validation Gates

- `_nsp status --target <repo>` healthy after reconciliation.
- Target governance/hygiene validation green (`_nsp hygiene code validate`, `_nsp hygiene context validate`).
- Every preserved customization and every dropped stale rule recorded with a reason.

## Handoff Artifacts

- The upgrade reconciliation report (files refreshed, merges, risks, human-review list).
- Backups retained under `.nsp/artifacts/upgrades/` for reviewer comparison.
- Ralph state for resumable upgrades under `.nsp/artifacts/tmp/ralph/upgrade-version-<target-or-branch>/`.

## Resume Rules

- Resume from the upgrade report and backups. Re-run `_nsp align target --dry-run` to re-establish the remaining delta. Never rely on chat history.

## Concurrency and Run Isolation

- Start a new run or join an existing compatible run before long-running work: `_nsp run start --target <repo> --workstream <id>` or `_nsp run list --target <repo>`.
- Claim direct paths, scopes, or shared artifacts before editing or writing state: `_nsp run claim --target <repo> --run-id <id> --phase <phase> --paths <csv>`.
- Heartbeat active claims during long work and release them at handoff or closeout: `_nsp run heartbeat ...` then `_nsp run release ...`.
- Write workflow evidence under `.nsp/artifacts/runs/<runId>/...`. Treat legacy latest paths as compatibility pointers, not canonical evidence.
- Never use proof, maintain queue items, review manifests, or support bundles from another run as evidence for the current run.
- Never complete another run queue item or release another run claim unless explicitly performing stale-lock recovery with event-log evidence.
- Every handoff must include the run ID, active/released claims, canonical artifact paths, and collision/blocker state.

## Never Do

- remove target-specific safety, security, accessibility, migration, or validation rules without human review
- add runtime dependencies to the target repository for NSP guidance upgrades
- claim the deterministic upgrade command performed semantic reconciliation
