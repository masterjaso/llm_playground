<!-- nsp:meta
id: docs.guidance.nsp.context.maintenance
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/context-maintenance.md
graphTags: context,docs,nsp
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-06-27
replaces: .docs/guidance/context-maintenance.md
replacedBy: 
-->

# NSP Context Maintenance Guidance

Use this guidance for smaller updates after initial context establishment. Routine context maintenance preserves the Genesis-established context model on changed paths.

Evaluate changed files against feature docs, technical docs, personas, guidance, and CCB links. Do not rerun full Genesis unless the user requests a re-baseline or repository context is broadly stale.

## Maintenance Loop

1. Identify changed files or use a provided changed-file list.
2. Store temporary changed-file lists or scratch notes under `.nsp/artifacts/` only.
3. Read only affected feature docs, technical docs, personas, graph relationship inputs, and guidance.
4. Infer whether feature docs, technical docs, doc-to-code links, tests, or personas drifted.
5. Preserve unaffected context.
6. Promote accepted durable context updates outside `.nsp/artifacts/` into `.docs/...`.
7. Run deterministic validation.
8. Clean up temporary maintenance artifacts when appropriate.

A typical changed-file list can be staged temporarily with:

```bash
git diff --name-only main...HEAD > .nsp/artifacts/changed-files.txt
```

The deterministic `_nsp` CLI does not call a model for maintenance. The external agent/model that receives this guidance performs the reasoning externally.
