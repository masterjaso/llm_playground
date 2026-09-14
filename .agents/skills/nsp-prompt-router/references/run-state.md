# Continuation and delegation

Read only when work needs run coordination, dispatch, or resumption. Reuse the
existing compatible run and accepted contract; do not create another run merely
because another prompt arrived. Check conflicts with `_nsp run list --target <repo>`.
Claim affected paths before edits, heartbeat long claims, and release only your
claims at handoff. Never accept another run's receipt as this run's result or
complete its queue items. Reuse validation evidence only with verified dependency
identities and originating provenance; shared logs can serve multiple gates.

The original primary remains `EPIC_OWNER`. A delegated child receives a complete
`NSP_ENTRY_MODE: delegated` envelope with `ROUTING_STATUS: complete`, objective,
scope, exclusions, required context, expected output, task type, and epic owner.
Validate against `contracts/delegation-envelope.schema.json`. An invalid envelope
returns to the parent; it never triggers broad routing. Children use envelope-only
context, cannot initiate an epic, and return bounded evidence. Delegate only when
authorized and the slice can run independently. Fresh critique can improve
assurance, but do not claim independence without evidence.

Keep run evidence under `.nsp/artifacts/runs/<runId>/`. Ralph continuation remains
under `.nsp/artifacts/tmp/ralph/<id>/`; `.docs/roadmap/**` is never in-flight state.
Use the existing supported compact record format. Keep one canonical bot state
and small references; human-readable views are optional projections. Preserve v1
read compatibility. Do not hand-migrate persistent formats or accepted contracts.

Accepted work contracts are immutable. Resume existing packages/capsules and
update only authorized progress/results. Capsule completion is distinct from run
closure. Shared evidence is valid only when all required acceptance, ownership,
input identity, and provenance checks hold. Do not close an entire run for one
completed child or release another agent's claim.

Rehydrate the current phase, contract IDs, claims, latest validations, unresolved
decisions, and next exact command. Verify relevant inputs before relying on prior
evidence. Revise only invalidated dependencies. Fully elaborate only the active
phase; future phase objectives remain coarse. In-flight failed checks, claims,
and blockers are evidence, not disposable prose.

Runtime owns run `STATE.json`. Never replace it with an epic summary. Use `_nsp run closeout` after satisfying claims and gates; Ralph completion alone cannot close a run.

Use capability-based realization when dispatch is needed. No workflow depends
on a provider brand, model, or mandatory `nsp-agent` runtime. The deterministic CLI
does not execute semantic reasoning.
