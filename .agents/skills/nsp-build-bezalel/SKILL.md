---
name: nsp-build-bezalel
description: Orchestrates implementation of an accepted bounded objective or Work Package with right-sized PIV, ATDD, verification-before-claim, and adaptive test-first discipline. Use when a user asks to implement, build, modify, add, integrate, or safely change code or docs from an accepted scope. Do not use for vague planning, defect diagnosis without reproduction, PR review, or routine diff alignment.
---

# nsp-build-bezalel

## Trigger Conditions

Implementation requests, feature work, bug fixes with a known approach, multi-step delivery under one outcome, or EPIC/WORKSTREAM execution that needs structured engineering discipline.

## Owns

- execution-class selection follow-through after routing
- Work Package / Work Capsule orchestration
- PIV / ATDD / Ralph execution for implementation
- verification-before-claim and adaptive test-first discipline
- fresh-context Capsule handoffs when supported
- composing internal `nsp-agent-workflow` and `nsp-epic-execution` primitives

## Does Not Own

- persona/request routing (internal router / control plane)
- PR review verdicts (`nsp-review-discernment`)
- defect diagnosis without reproduction (`nsp-debug-watchman`)
- deterministic CLI validation authority

## Required Start

```bash
_nsp status --target <repo>
_nsp context select --target <repo> --request "<summary>" --limit 8 --format json --receipt-v2
_nsp run list --target <repo>
```

For EPIC/WORKSTREAM work, validate the run-scoped Repository Fact Ledger before planning. Use `_nsp work` substrate for packages/capsules/results when material.

## Required Inputs

- Bounded objective and acceptance criteria
- Context Receipt / selected docs
- Applicable Domain Language terms when present
- Assurance requirements when material

## Expected Outputs

- Implemented change within claimed scope
- Fresh validation evidence before any complete/fixed/ready claim
- Ralph/handoff artifacts for multi-session work
- Result Receipt / closeout notes when using work substrate

## Validation Gates

- No completion claim without fresh supporting evidence
- Small work stays small; escalate structure only when warranted
- Tests/gates proportional to behavioral risk (adaptive test-first, not ceremonial TDD for docs-only/mechanical changes)
- Deterministic CLI remains non-semantic

## Handoff Artifacts

- `.nsp/artifacts/tmp/ralph/<workstream>/` when long-running
- `.nsp/artifacts/runs/<runId>/...` evidence
- Context Receipt and work artifacts as applicable

## Resume Rules

Resume from Ralph state, run claims, and validation ledger — never chat memory alone.

## Verification-Before-Claim

No claim such as fixed, complete, passing, ready, or resolved without fresh supporting evidence. Applies even to DIRECT work.

## Adaptive Test-First

Use test-first/change-sensitive proof when behavioral correctness materially benefits. Do not force TDD ceremony onto docs-only, mechanical migrations, or formatting-only changes.

## Never Do

- Claim fixed/complete/ready without fresh evidence
- Force EPIC ceremony onto DIRECT work
- Bind implementation to a specific model/provider brand
- Call the CLI for semantic judgment

## Internal composition

- Read and follow `.agents/skills/nsp-agent-workflow/SKILL.md` for PIV/ATDD/Ralph mechanics, including clean-context sub-agent preference when available.
- Read and follow `.agents/skills/nsp-epic-execution/SKILL.md` when execution class is EPIC.
- Keep those skills internal; do not present them as peer user choices.
- Prefer delegated envelopes for Implement vs Validate slices over one agent carrying the full parent transcript.

## Concurrency and Run Isolation

- Start or join a run before long-running edits: `_nsp run start|join --target <repo> ...`
- Claim paths before editing; heartbeat and release at handoff.
- Write evidence under `.nsp/artifacts/runs/<runId>/...`.
- Ralph state for multi-phase work: `.nsp/artifacts/tmp/ralph/<id>/`
