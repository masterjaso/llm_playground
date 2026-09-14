# Bounded execution

Use this reference when implementing or repairing behavior. An accepted objective
and its current evidence are the starting point. Do not regenerate Genesis or
reclassify an unchanged contract at each PIV.

1. Plan one observable behavior with an exact acceptance check. For risky existing
   behavior, capture the regression or characterization before changing it.
2. Implement one vertical slice through the necessary layers. Run its focused
   checks before starting the next behavior; avoid separate batches of all tests,
   all interfaces, and all implementations.
3. Validate with fresh commands appropriate to the change. Evidence includes the
   command, result, relevant input identity, and bounded output reference. Reuse
   declared unchanged inputs only where the owning validator supports it; never
   reuse an old PASS as a generic validation cache.

DIRECT work needs its action and proof, with no capsule, Ralph, prediction, or
placeholder record by default. BOUNDED work needs one acceptance contract; add a
Work Capsule only for dispatch, handoff, or resumption. WORKSTREAM adds a Work
Package with bounded children and integration gates. EPIC uses the epic skill.
File count, time estimates, provider, model, and harness do not determine class.

Apply the Minimum Code Gate: reuse repository behavior, then platform/runtime or
installed dependencies, before adding code. A new layer must hide meaningful
complexity or reduce caller knowledge. Preserve trust-boundary checks, public
behavior, recoverability, accessibility, and required validation. Prefer readable
deep modules and local ownership to arbitrary file splitting. Comments explain
non-obvious intent, constraints, and invariants; remove comments that restate code.

For uncertain or risky mechanisms, read [prediction.md](prediction.md). A cheap
discriminating probe can use the hidden
[prototype skill](../../nsp-prototype/SKILL.md); it never authorizes production
edits. For actual continuation or delegation, read [run-state.md](run-state.md).
Neither reference is a default read for clear DIRECT work.

Required independence or fresh-context assurance needs evidence and blocks
acceptance when unmet. Preferred assurance permits disclosed reduction. Role
labels do not prove independence or assign a provider. The deterministic CLI
validates structure and records; the agent owns design and semantic judgment.

Handoff with changed behavior, exact validation and residual risk. Keep bot state
as compact fields and references; render human prose only for a human audience.
