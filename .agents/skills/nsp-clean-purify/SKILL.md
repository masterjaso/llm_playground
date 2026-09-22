---
name: nsp-clean-purify
description: Inspects and safely clears ephemeral NSP artifact storage in a target repository through guarded deterministic cleanup and final rescan. Use when a user asks to clean, purge, reset, or clear stale `.nsp/artifacts` output, remove noisy runtime evidence after completed work, or inspect cleanup blockers. Do not use for durable `.docs` cleanup, installed-skill reconciliation, run closure, or arbitrary file deletion.
---

# Clean Purify

## Outcome
Inspect and clear authorized ephemeral NSP artifacts in the exact target through guarded deterministic cleanup.

## Workflow
Inspect artifacts, preview target cleanup, and assess blockers/active lifecycle ownership. Use supported project cleanup CLI. An explicit scoped cleanup request authorizes ordinary guarded cleanup; do not repeat permission already supplied. Broader deletion or force acceptance needs its own authority.

Selection is the exact target's entire `.nsp/artifacts/**` tree except root `.gitkeep`, not only stale entries. `project cleanup --target <repo>` previews; adding `--force` still only previews. Use `--apply` to delete. `--apply --dry-run` is rejected. Legacy `artifacts --clean` remains an explicit mutation alias.

Explicit `--apply --force` deletes selected open runs, active claims, incomplete Ralph state, pins, references and recent retained evidence. It does not close or abandon work as a substitute, or mark deleted work successful. Report these consequences and overridden blockers before applying authorized force.

Filesystem/read integrity, containment, target/root/ancestor symlink boundaries and real mutation mutex contention remain hard blockers. Selected symlink leaves are unlinked without touching their external targets. Never kill workers or steal locks; stop and retry once writers are idle. The mutex serializes cooperating mutations but does not revoke worker handles or prevent every stale writer from recreating artifacts after release. Finish with the same policy and options in the final rescan and preserve root `.gitkeep`.

## Evidence
Return target, preview/apply results, final counts, overridden lifecycle protections, hard blockers and residual writer risks. The deterministic CLI enforces deletion guards; the agent checks scope. A blocked cleanup is not a completed purge.

## Boundaries
No homemade recursive delete, arbitrary targets, `.docs` cleanup, skill reconciliation, foreign-run closure, or guessed force authority. Do not expand cleanup into a new workflow run.
