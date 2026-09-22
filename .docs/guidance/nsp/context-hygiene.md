<!-- nsp:meta
id: docs.guidance.nsp.context.hygiene
kind: guidance
scope: context-hygiene
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/context-hygiene.md
graphTags: docs,context-hygiene
validation: manifest-check,frontmatter-audit,secret-scan
owner: context-hygiene
lastReviewed: 2026-06-27
replaces:
replacedBy:
-->

# Context hygiene

Context hygiene keeps agent-facing documentation, metadata, graph artifacts, routing, evidence, and stewardship context healthy, current, bounded, and safe.

Context hygiene is different from code hygiene:

- Context hygiene governs what agents and humans read before acting.
- Code hygiene governs how implementation behavior is organized and tested.
- Context hygiene reduces token waste and wrong assumptions.
- Code hygiene reduces implementation entropy and unsafe changes.

They reinforce each other. Context hygiene explains ownership, scope, routing, evidence, and review status. Code hygiene keeps the implementation modular enough for those routed agents to make local, testable changes.

## Setup, Validation, Repair

Setup creates or verifies target-owned context hygiene guidance and skill discovery:

```bash
_nsp hygiene context setup --target <repo>
```

Validation composes existing NSP checks where available:

```bash
_nsp hygiene context validate --target <repo>
```

Repair produces a conservative plan first:

```bash
_nsp hygiene context repair --target <repo> --dry-run
```

Context hygiene validation should inspect missing or stale NSP frontmatter, stale graph artifacts, missing ownership/persona/scope metadata, unreviewed model-derived content, unsafe broad context dumping, missing evidence before PR handoff, broken target-owned guidance, graph/context mismatch, and missing or stale hygiene skill files.

Safe context repair may create missing scaffolded guidance, missing hygiene docs, missing skill files, missing policy files, and graph rebuild recommendations. It should not overwrite target-owned docs without an explicit force or merge decision.
