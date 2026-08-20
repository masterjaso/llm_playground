<!-- nsp:meta
id: docs.guidance.nsp.stewardship
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/stewardship.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# NSP Stewardship Guidance

NSP is external standalone tooling for agent-first software development through Context Management and Knowledge Discovery.

## Ownership Boundary

Target repositories own their `.docs/...` context, generated graph artifacts when committed by convention, guidance, personas, and validation decisions. NSP tooling stays external and must not add provider dependencies, API key requirements, or target runtime dependencies.

General repository folders such as `docs/`, `spec/`, and `test/` are repository-owned. NSP setup must not treat them as NSP-managed scaffold locations.
