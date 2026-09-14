# Repository discovery gate

Use for EPIC/WORKSTREAM planning, material uncertainty, or an explicit discovery
requirement. Clear DIRECT and low-risk BOUNDED work remain exempt. An unchanged
accepted contract resumes from its evidence instead of repeating full discovery.

1. Extract scoped repository facts from producers, consumers, tests, schemas,
   commands, packaging/install surfaces, and public compatibility contracts.
2. Validate the fact ledger with `_nsp plan-substrate discovery validate --target
   <repo> --run-id <run-id>`. `DISCOVERY_READY` requires verified material facts,
   valid references, relevant coverage, current target/run identity, and no
   material unknown/conflicted facts. Otherwise report `DISCOVERY_BLOCKED` and the
   exact missing evidence, impact, and bounded next investigation.
3. Register architecture decisions separately before binding an executable plan.

Canonical ledger: `.nsp/artifacts/runs/<runId>/planning/repository-fact-ledger.json`;
use the existing prompt fallback only without a run. Keep its existing schema.
Each material fact has an ID, claim/status, bounded evidence references,
confidence, impact if wrong, and review trigger. Status is `verified`, `rejected`,
`unknown`, or `conflicted`. Absent evidence is unknown, never an assumption.
Rejected facts remain traceable but cannot authorize paths or commands.

The CLI checks declared structure/identity/coverage; it does not prove semantic
truth. Bind plans only to verified fact IDs and resolved decision IDs. Inspect
discoverable facts before asking humans. Current source outranks stale diagrams.
Optional graph provenance that is unavailable does not force a rebuild or block
a question answered by current bounded source evidence.

On relevant content, inventory, configuration, scope, or prediction changes,
invalidate the affected facts and dependent decisions. Unrelated source changes
do not reopen the whole plan. Respect required-context budgets: retain applicable
safety guidance and report essential overflow for narrowing or explicit expansion.
