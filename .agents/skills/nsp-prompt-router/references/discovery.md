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

For each verified or rejected fact, review its evidence and explicitly capture
its bounded dependencies with `_nsp plan-substrate discovery capture --target
<repo> --paths-json '["src/relevant.ts","package.json"]'`. Include every referenced
path, relevant configuration, and directories whose inventory supports the claim.
Copy the returned `dependencies` object into that fact only after review. Capture
prints fingerprints; it does not write a ledger or verify a claim. Use optional
`--branch` or `--commit` bindings only when the claim depends on that identity;
commit evidence requires a commit binding. Recheck evidence if it changes between
review and capture. Narrow scopes that exceed capture limits (256 paths per fact,
10,000 entries or 32 MiB across one capture). Verification shares those entry and
byte budgets across the ledger and reuses identical path reads. File content,
executable bits, missing paths, and directory inventory affect fingerprints.
Symlinks, special nodes, and observed concurrent changes are rejected. Streaming
reads bound memory use; metadata checks do not provide an atomic filesystem
snapshot or containment against concurrent ancestor replacement. Capture from a
stable, trusted checkout rather than a concurrently modified untrusted tree.

Validation/readiness never refresh fingerprints. Legacy facts without dependency
coverage require bounded review and explicit capture, not full rediscovery.
Content and scoped inventory changes block affected facts even when Git's dirty
flag is unchanged. Unrelated Git changes do not invalidate unchanged dependencies.
Historical accepted completion remains distinct from current discovery readiness.

New capsules can declare `discoveryDependencies`: one `{gate, factIds}` binding
for every acceptance gate. Gate names must be unique and match the capsule;
each binding lists nonempty, unique verified fact IDs. Include all facts used by
that gate's decisions. Resume reports affected fact and gate IDs without erasing
historical acceptance. Unrelated stale facts do not block a bound capsule, but
missing/rejected bound facts, malformed ledgers, and target/run identity failures
do. Capsules without bindings retain conservative whole-ledger checks. Never
rewrite an accepted capsule to add bindings; prepare an explicit revision.

The CLI checks declared structure/identity/coverage; it does not prove semantic
truth. Bind plans only to verified fact IDs and resolved decision IDs. Inspect
discoverable facts before asking humans. Current source outranks stale diagrams.
Optional graph provenance that is unavailable does not force a rebuild or block
a question answered by current bounded source evidence.

On relevant content, inventory, configuration, scope, or prediction changes,
invalidate the affected facts and dependent decisions. Unrelated source changes
do not reopen the whole plan. Respect required-context budgets: retain applicable
safety guidance and report essential overflow for narrowing or explicit expansion.
