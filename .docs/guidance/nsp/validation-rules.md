<!-- nsp:meta
id: docs.guidance.nsp.validation.rules
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/validation-rules.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# NSP Validation Rules

NSP validates structure, path safety, required fields, artifact boundary rules, and graph consistency. It does not claim semantic inference authority.

## Required Checks

- `.nsp/artifacts/` exists and is ignored except for `.gitkeep`.
- Durable context, personas, guidance, and graph material live outside `.nsp/artifacts/`.
- Feature docs include identity, audience, status, short description, resource, and links to technical docs in the body.
- Technical docs include identity, audience, status, short description, resource, implementation rationale, validation notes, and repo:// code anchors in the body.
- Personas include responsibility, rationale, related docs, evidence, confidence, and status.
- Doc-to-code mappings reference existing relative paths or explicitly mark missing tests with a reason.
- Stale or prune decisions include evidence.

Temporary proposals under `.nsp/artifacts/` may be inspected during work, but final validation targets durable canonical files.
