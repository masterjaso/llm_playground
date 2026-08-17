<!-- nsp:meta
id: docs.guidance.nsp.ccb.hygiene
kind: guidance
scope: context-hygiene
persona: qa-validation
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/ccb-hygiene.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: qa-validation
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# CCB hygiene

Use `nsp-ccb-hygiene` for agentic Context-to-Code Bridge upkeep and review. The deterministic `_nsp ccb` commands build, validate, explain, and propose bridge repairs; the skill performs evidence-backed review and promotion decisions.

## Ownership

- Code hygiene execution stays with `nsp-code-hygiene`.
- Context and documentation alignment stays with `nsp-context-hygiene`.
- CCB bridge readiness, stale link detection, and repair promotion stay with `nsp-ccb-hygiene`.
- First adoption orchestration is user-facing through `nsp-adopt-ezra`; this guidance is internal support material for that sweep or later maintenance.

## Evidence Rules

Every accepted Feature -> Technical -> Code relationship needs repository evidence. Missing technical links, missing code anchors, missing tests, and stale relationships should be recorded as gaps or repair candidates instead of converted into facts.

## Validation

```bash
_nsp ccb build --target .
_nsp ccb validate --target . --readiness
_nsp ccb explain --target . --feature <feature-id>
_nsp ccb repair --target . --dry-run
```
