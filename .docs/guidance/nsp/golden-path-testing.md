<!-- nsp:meta
id: docs.guidance.nsp.golden.path.testing
kind: guidance
scope: code-hygiene
persona: qa-validation
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/golden-path-testing.md
graphTags: docs,testing
validation: manifest-check,frontmatter-audit,secret-scan
owner: qa-validation
lastReviewed: 2026-06-27
replaces:
replacedBy:
-->

# Golden-path testing

Golden-path user behavior testing means:

> A test validates the primary successful user journey that the product or feature exists to support, from the user-visible entrypoint through the important integration seams, using realistic inputs and asserting meaningful observable outcomes.

It is not just a unit test and not just a brittle UI click script.

Golden-path tests answer:

- What does the user need to accomplish?
- What is the normal successful path?
- Which boundaries must work together for this to be true?
- What output, state change, event, saved data, graph artifact, report, screen, CLI result, or file proves success?
- Which failure would make the feature unusable even if isolated units still pass?

Use golden-path tests for e2e integrations where multiple modules must work together. Back them with smaller seam-level tests, contract tests, characterization tests, and golden output tests where appropriate.

For NSP-style repositories, good golden paths include:

- `_nsp setup --target <repo>` creates expected target-owned guidance without overwriting user-owned docs.
- `_nsp hygiene setup --target <repo>` creates hygiene policy, docs, skills, and report folders.
- `_nsp hygiene code validate --target <repo>` scans a fixture repo and writes a bounded report.
- `_nsp hygiene code repair --target <repo> --dry-run` emits a safe repair plan without mutating source.
- `_nsp hygiene context validate --target <repo>` composes context checks and writes a report.
- `_nsp prepare-pr --target <repo> --base main` includes hygiene status in evidence when reports exist.

Golden-path assertions should use realistic fixture repos, public CLI or UI entrypoints, real file output where relevant, bounded report assertions, preservation checks for existing files, no secret leakage, and proof that no NSP scripts or runtime dependency were added to the target.
