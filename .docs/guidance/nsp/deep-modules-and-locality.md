<!-- nsp:meta
id: docs.guidance.nsp.deep.modules.and.locality
kind: guidance
scope: code-hygiene
persona: platform-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/deep-modules-and-locality.md
graphTags: docs,architecture
validation: manifest-check,frontmatter-audit,secret-scan
owner: platform-engineering
lastReviewed: 2026-06-27
replaces:
replacedBy:
-->

# Deep modules and locality

A deep module gives callers a small, stable, intent-based interface while hiding meaningful implementation complexity. A shallow module exposes helpers, flags, modes, intermediate data, or infrastructure details that callers must coordinate.

Deep modules improve leverage because callers ask for outcomes instead of orchestrating steps. They improve locality because related changes stay inside the module and its tests. Shallow modules increase software entropy because rules drift into many callers, duplicated concepts appear, and future changes require broad edits.

Minimum viable code supports deep modules when it keeps behavior inside the correct owner and avoids speculative surface. Reuse existing repository patterns, platform/runtime behavior, or installed dependencies before adding new abstractions. Do not collapse necessary seams, adapters, or validation just to reduce line count.

Prefer descriptive names and implemented behavior over narrative comments. Remove commented-out implementations, redundant explanations, and placeholders used instead of code. Keep concise comments for invariants, compatibility, contracts, and rationale that code cannot express. Put broader design in maintained technical docs; do not scatter it across callers.

## Review Signals

Review a module when you see:

- many public entrypoints with overlapping purpose
- scattered business rules across UI, CLI, adapters, and tests
- parallel implementations with similar names
- direct filesystem, database, network, process, or storage access from domain code
- UI components containing reusable business logic
- large files or modules combined with unrelated responsibilities
- many options, flags, or modes that callers must coordinate
- missing tests around important seams

Large size alone is not condemnation. Some generated, orchestration-heavy, engine, or data modules are intentionally large. A large file becomes a stronger repair candidate when it also has weak locality, duplicated behavior, infrastructure leakage, missing harnesses, or missing golden-path coverage.

## Safe Refactoring

Refactor messy areas in this order:

1. Identify the user or caller behavior that must remain true.
2. Add a golden-path user behavior test if the behavior crosses integration boundaries.
3. Add characterization or seam-level tests around current behavior.
4. Introduce the desired seam.
5. Move implementation details behind the seam.
6. Replace one caller and validate.
7. Replace remaining callers.
8. Delete duplicated logic.
9. Re-run hygiene validation and update context docs or graph artifacts when architecture changed.
