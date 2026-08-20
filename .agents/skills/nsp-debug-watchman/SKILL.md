---
name: nsp-debug-watchman
description: Reproduces, diagnoses, minimally repairs, and verifies software defects with falsifiable hypotheses and fresh regression evidence. Use when a user reports a failing test, bug, regression, crash, incorrect behavior, flaky result, or performance or reliability defect and wants root-cause proof. Do not use for new feature implementation, vague planning, or final PR review.
---

# nsp-debug-watchman

## Trigger Conditions

Defects, failing tests, unexpected runtime behavior, regressions, or repair requests where diagnosis must precede speculative fixes.

## Owns

- reproduction and minimization
- ranked falsifiable hypotheses
- instrumentation and targeted tests
- root-cause identification
- smallest justified repair
- regression proof and fresh verification

## Does Not Own

- feature implementation without a defect loop (`nsp-build-bezalel`)
- PR review verdicts (`nsp-review-discernment`)
- deterministic CLI validation authority
- semantic Domain definition (`nsp-insight-berean`)

## Required Start

```bash
_nsp status --target <repo>
_nsp context select --target <repo> --request "<defect summary>" --limit 8 --format json --receipt-v2
_nsp impact --target <repo> <suspect-path-or-symbol>
```

Collect failing command output or reproduction steps before editing.

## Required Inputs

- Reproduction steps or failing command
- Observed vs expected behavior
- Scoped evidence (logs, tests, graph impact, Domain terms when relevant)

## Expected Outputs

- Established reproduction
- Ranked hypotheses with what would falsify each
- Root cause statement with evidence
- Minimal repair
- Fresh regression proof (tests or deterministic checks)

## Validation Gates

- Do not apply repeated speculative fixes without an established feedback loop
- Root cause before broad repair
- Verification-before-claim: no "fixed" without fresh evidence
- Prefer project graphs, Domain Language, impact evidence, and tests to narrow diagnosis

## Handoff Artifacts

- Reproduction notes and hypothesis log under `.nsp/artifacts/runs/<runId>/...` or Ralph debug folder when multi-session
- Validation command output proving the regression

## Resume Rules

Resume from the last reproduction command, open hypotheses, and failing evidence — not from chat memory alone.

## Verification-Before-Claim

No claim such as fixed, complete, passing, ready, or resolved without fresh supporting evidence. Applies even to DIRECT work.

## Never Do

- Guess-and-patch loops without reproduction
- Claim fixed without re-running the failing proof
- Expand scope into unrelated refactors
- Treat assistant prose as validation

## Flow

```text
reproduce -> minimize -> ranked falsifiable hypotheses -> instrument/test
-> root cause -> smallest justified repair -> regression proof -> fresh verification
```

## Concurrency and Run Isolation

Use run claims for multi-file repairs. Ralph path when multi-session: `.nsp/artifacts/tmp/ralph/debug-<id>/`.
