---
name: nsp-clean-purify
description: Inspects and safely clears ephemeral NSP artifact storage in a target repository through guarded deterministic cleanup and final rescan. Use when a user asks to clean, purge, reset, or clear stale `.nsp/artifacts` output, remove noisy runtime evidence after completed work, or inspect cleanup blockers. Do not use for durable `.docs` cleanup, installed-skill reconciliation, run closure, or arbitrary file deletion.
---

# Clean Purify

## Outcome
Inspect and clear authorized ephemeral NSP artifacts in the exact target through guarded deterministic cleanup.

## Workflow
Inspect artifacts, preview target cleanup, and assess blockers/active lifecycle ownership. Use supported project cleanup CLI. An explicit scoped cleanup request authorizes ordinary guarded cleanup; do not repeat permission already supplied. Broader deletion or force acceptance needs its own authority.

Apply only the verified scope. Explicit force can accept documented lifecycle consequences, never bypass containment, symlink/reparse protections, read integrity, or mutex failures. Finish with rescan and required `.gitkeep` placeholders retained.

## Evidence
Return target, preview/apply results, final counts, and blockers. The deterministic CLI enforces deletion guards; the agent checks scope. A blocked cleanup is not a completed purge.

## Boundaries
No homemade recursive delete, arbitrary targets, `.docs` cleanup, skill reconciliation, foreign-run closure, or guessed force authority. Do not expand cleanup into a new workflow run.
