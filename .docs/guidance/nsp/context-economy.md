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

## Bot-only state

Use concise structured values, readable field names, stable IDs, evidence references, and exact next actions. One record owns each responsibility: run ownership, accepted plan, Ralph continuation, capsule contract, result, and validation evidence. Ralph references these records instead of repeating plans or gate definitions.

Do not write empty placeholders, copied transcripts, repeated full prompts, or automatic JSON/Markdown pairs. Render expanded human reports only when requested. Preserve failure evidence, concurrency events, accepted contract identity, and provenance. Closing a run does not make failed work successful.

Reuse routing and completed evidence only while relevant identities remain valid. Unknown or external validation dependencies require fresh checks. An unrelated edit does not justify whole-project discovery. Repeated setup without progress requires a blocking-operation diagnosis, not another preparation sequence.

## Selection and retention

Context selection packs relevant reads inside the configured budget. The default remains 24,000 recommended tokens, estimated from UTF-8 bytes divided by four; `--max-tokens` changes that bound. Essential instructions that cannot fit produce a budget blocker and a bounded expansion path; never silently drop them. Content and inventory identities detect edits, additions, deletion, malformed inputs, and dirty-to-dirty changes. Private reuse metadata preserves Context Receipt v2 compatibility and current agent-owned routing decisions.

Successful closeout applies retention using `closedAt`: retain the newest ten completed runs or any completed within thirty days. Active work, pins, active Atlas sessions, and transitive evidence remain protected. Unknown ownership and malformed timestamps are not automatic deletion candidates. Preview with `_nsp run gc --target . --dry-run`; override completed-run policy with `--keep-runs` and `--keep-days`. Protection can exceed the policy's storage estimate.

## Review Rule

Before removing duplicated text, classify whether it is intentional locality/safety redundancy or wasteful duplication. Do not over-dedupe skills.
