<!-- nsp:meta
id: docs.guidance.nsp.context.code.bridge
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/context-code-bridge.md
graphTags: context,docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Context-to-Code Bridge Guidance

The Context-to-Code Bridge (CCB) connects:

```text
Feature docs → Technical docs → Code anchors
```

- **Feature docs** (`.docs/features/`) tell end-users what exists and how to use it.
- **Technical docs** (`.docs/technical/`) tell developers how and why it is built.
- **Code anchors** (`repo://` paths under `## Code anchors`) prove where implementation lives.

Feature docs link to technical docs under `## Technical implementation`. Relationship fields do not belong in frontmatter.

Descriptions are short one-line summaries for indexes, routing, and previews.

```bash
_nsp ccb build --target .
_nsp ccb validate --target . --readiness
_nsp ccb explain --target . --feature feature.<slug>
_nsp ccb repair --target . --dry-run
```
