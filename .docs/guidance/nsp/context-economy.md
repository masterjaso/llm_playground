<!-- nsp:meta
id: docs.guidance.nsp.context.economy
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/context-economy.md
graphTags: context,docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# NSP Context Economy Guidance

NSP treats context as a managed resource. Context should be clean, tight, evidence-backed, and locally useful.

## Context Dedupe Principle

Avoid repeated long-form guidance across markdown files. Prefer canonical ownership plus short local reminders. Duplicate only when execution locality, safety, offline skill usability, or setup isolation requires it.

## Allowed Duplication

Duplication is allowed when it preserves:

- standalone skill execution
- safety-critical gates
- setup prompt isolation
- provider/harness compatibility
- short reminders that prevent misuse

## Wasteful Duplication

Duplication should be removed or replaced with a reference when it is:

- a repeated long-form explanation
- a copied checklist with no local variation
- stale guidance at a legacy path
- duplicated setup instructions already owned by a canonical guide
- repeated narrative that does not alter execution behavior

## Preferred Pattern

Use:

- one canonical source for detailed guidance
- short local summaries where needed
- explicit links or path references to the canonical source
- self-contained skills for execution-critical rules
- artifact-backed details instead of repeated prompt text

## Review Rule

Before removing duplicated text, classify whether it is intentional locality/safety redundancy or wasteful duplication. Do not over-dedupe skills.
