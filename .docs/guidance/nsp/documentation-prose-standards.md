<!-- nsp:meta
id: docs.guidance.nsp.documentation.prose.standards
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/documentation-prose-standards.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Genesis documentation prose standards

Authoritative prose contract for Genesis and CCB concept documentation. Classify the document type **before** selecting generation instructions. Do not rely on destination path alone.

| Document type | Audience | Primary style |
|---------------|----------|---------------|
| **feature** | End users and workflow consumers | Task-oriented workflow narrative |
| **technical** | Engineers and maintainers | Compact contracts, bullets, tables, schemas |
| **ccb-metadata** | Guidance/automation | Short structured appendix / concept metadata |

## Classification

Set an explicit `documentType` of `feature`, `technical`, or `ccb-metadata`. If classification is ambiguous, **fail clearly** and ask for clarification — do not guess from path alone.

## Feature documentation

- Plain, accessible language.
- Explain through tasks and workflows: when and why to use the capability.
- Focus on observable behavior and outcomes; prefer concrete examples.
- Avoid class names, function names, algorithms, file paths, and storage mechanics unless the user must type them.
- Required CCB / concept sections stay at the **bottom** and remain compact so they do not dominate the narrative.

## Technical documentation

- Purpose and boundaries; inputs and outputs; contracts; invariants; failure behavior; integration points.
- Prefer bullets, tables, schemas, and diagrams.
- Link or cite code instead of copying implementation.
- No line-by-line narration; no duplicated overview/design/implementation prose.

## CCB metadata

Short structured appendix only — provenance, validation, and change-control fields. Never the main body of a feature narrative.

## Deterministic vs semantic checks

Deterministic tooling may enforce section order, CCB-last placement, duplicate H2 detection, and banned implementation-token patterns in feature fixtures. Semantic qualities (tone, accessibility) remain model-evaluated with documented limits — not brittle word-count gates.
