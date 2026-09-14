---
name: nsp-prototype
description: Runs a bounded disposable technical probe to resolve one internal implementation uncertainty. Use when a cheap isolated experiment can distinguish competing mechanisms before production changes. Do not use for user-facing prototypes, production implementation, open-ended research, or external side effects.
user-invocable: false
---

# Internal prototype

## Outcome
Answer one material technical question with a disposable probe. The parent owns
the accepted scope and decision; this internal skill is not a public menu.

## Workflow
State competing hypotheses, discriminating observation, bounded time or cost,
and stop condition before running anything. Reuse current evidence first.
Create owned scratch only under
`.nsp/artifacts/runs/<runId>/prototypes/<probe-id>/` or the task's approved
temporary root. Keep fixtures minimal and non-sensitive.

Run the smallest test of the mechanism. Read production interfaces or call a
side-effect-free public seam if needed, but keep probe code and dependencies
isolated: no production imports of scratch files, root dependency changes, live
external side effects, or source edits disguised as an experiment. Stop when
the distinguishing result is available or the declared bound is reached.

## Evidence
Return `supported`, `refuted`, or `inconclusive` with question/decision ID,
command, input identity, observation, evidence path, and next decision.
The deterministic CLI validates records; the agent interprets the experiment.
Inconclusive results preserve uncertainty and never count as acceptance.

## Boundaries
Do not promote prototype files into production. Implement the selected design
through the owning Build/Debug work with regression and acceptance checks.
Use existing artifact retention/guarded cleanup for owned scratch; do not add a
recursive deletion mechanism. A bot-only probe needs no epic, new planning pack,
or human report.
