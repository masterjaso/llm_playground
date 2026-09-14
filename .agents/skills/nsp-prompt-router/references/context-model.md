# Evidence-backed project context

Features are externally observable end-user capabilities. Technicals explain
implementation, runtime, APIs, state, architecture, tests, and operations for
dev-users. Personas represent evidenced end-users, developers, operators,
maintainers, and integration actors. Guidance records the conventions each actor
needs. CCB links Feature -> Technical -> Code with resolvable evidence.

Before writing a concept document classify `documentType` as `feature`,
`technical`, or `ccb-metadata`; use the target's
`.docs/guidance/nsp/documentation-prose-standards.md`.
Feature prose follows user tasks/outcomes; technical prose describes precise
contracts. Setup defaults are README.md and index.md, not invented feature docs.

Preserve useful target-owned docs and decisions; repair only contradictions backed
by current evidence. Group by real domain boundaries, maintain ownership and
consistent Domain Language, and avoid duplicate docs/persona sprawl. Every promoted
feature, persona, technical claim, or reviewed CCB link needs repository evidence.
Unresolved claims stay gaps or unknowns.

Durable project knowledge lives in `.docs/features/**`, `.docs/technical/**`,
`.docs/AGENT_PERSONAS.md`, and target guidance under `.docs/guidance/**` outside
`.docs/guidance/nsp/**`. Reusable NSP guidance remains in that reserved subtree.
Bot inventories and progress remain ephemeral structured records, not human prose.

Adoption may inspect all relevant entrypoints, source, configs, build scripts,
tests, CI, infra, and existing docs. Routine Maintain applies this model only to
direct changes and bounded related context; it never restarts full Genesis.
