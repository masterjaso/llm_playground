<!-- nsp:meta
id: docs.guidance.nsp.hygiene.review.checklist
kind: guidance
scope: code-hygiene
persona: qa-validation
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/hygiene-review-checklist.md
graphTags: docs,review
validation: manifest-check,frontmatter-audit,secret-scan
owner: qa-validation
lastReviewed: 2026-06-27
replaces:
replacedBy:
-->

# Hygiene review checklist

Use this checklist when reviewing human or AI-generated changes.

## Code Hygiene

- Was the Minimum Code Gate applied before accepting new source surface?
- Can the behavior be avoided, reused, configured, or provided by the runtime/platform before adding code?
- Does an installed dependency already cover the behavior without new dependency ownership?
- Is the change the smallest safe behavior-preserving change, with the safety floor and validation evidence intact?
- Does the change preserve or improve module depth?
- Are callers using small intent-based interfaces instead of coordinating implementation steps?
- Are related rules local to the module that owns them?
- Are seams explicit where behavior crosses infrastructure, UI, CLI, network, storage, or process boundaries?
- Are adapters protecting domain behavior from infrastructure details?
- Are large files reviewed in context instead of automatically condemned?
- Are generated, vendored, build, lock, bundle, and data-heavy files excluded or policy-classified?
- Are there parallel implementations or duplicated business rules?
- Does UI code contain reusable business logic that belongs behind a seam?
- Is there a harness before risky refactoring?
- Is there golden-path coverage for user-critical behavior?
- Does the smallest proposed diff increase future change amplification, caller coordination, duplicated knowledge, or hidden coupling?
- Are costly or destructive decisions appropriately reversible, recoverable, compatible, or preservative?
- Does every new layer hide or simplify meaningful complexity?
- Can expected absence, repetition, retries, or already-completed state be represented idempotently rather than as avoidable failures?
- For durable or high-risk design decisions, was the draft design subjected to NSP adversarial review and hardened before implementation?
- Were all must-fix adversarial findings resolved before execution?
- Does the selected abstraction represent demonstrated domain needs without becoming either a one-off special case or a speculative framework?

These are semantic review prompts, not universal deterministic failures. Do not convert them into brittle lint based on exact phrases, word counts, file-size thresholds alone, subjective keyword matching, or a required prose template.

## Context Hygiene

- Did the agent route before broad reads?
- Are NSP frontmatter, scopes, personas, graph artifacts, and evidence current enough for the claim?
- Are model-derived or heuristic graph facts marked with confidence and review status?
- Are target-owned docs preserved unless force or merge was explicit?
- Are hygiene reports included in evidence when present?
- Are secrets redacted from reports, graph artifacts, and handoff output?

## Repair Safety

Refuse automatic repair and output a plan only when behavior is ambiguous, tests are missing, the module boundary is unclear, source files are generated or vendored, changes cross multiple ownership areas, or user-visible workflows lack golden-path coverage.
