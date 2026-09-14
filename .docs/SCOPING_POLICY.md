<!-- nsp:meta
id: docs.scoping.policy
kind: guidance
scope: guidance
persona: platform-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/SCOPING_POLICY.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: platform-engineering
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Scoping policy

- Machine-readable scopes live in `.docs/scopes/index.json`.
- Prefer narrow scopes over whole-repo context.
- Keep user-facing docs in `.docs/features/` and maintainer docs in `.docs/technical/`.
- Regenerate the manifest intentionally; validation should fail on drift.
