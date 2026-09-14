<!-- nsp:meta
id: docs.guidance.nsp.code.hygiene
kind: guidance
scope: code-hygiene
persona: qa-validation
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/code-hygiene.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: qa-validation
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Code hygiene

Code hygiene is the target repository's discipline for keeping source code readable, maintainable, testable, and resistant to software entropy. It is the code-side counterpart to context hygiene.

The central standard:

> A good module gives callers maximum useful behavior through the smallest reasonable interface, while keeping related changes local and testable.

Code hygiene is not formatting, tidiness, naming taste, or personal style. It is architecture that makes future changes safer. It asks whether modules are deep enough, whether interfaces are small and intent-based, whether implementation details are hidden, whether important seams have test harnesses, and whether user-critical workflows have golden-path user behavior tests.

## Minimum Code Gate

Apply this gate before semantic code repair, refactoring, or generation work:

1. Can the request be satisfied without new source code?
2. Does the target repository already have a helper, module, scene, adapter, command, resource, or pattern that should be reused?
3. Does the language runtime, standard library, framework, game engine, or platform already provide this behavior?
4. Does an already-installed dependency cover the behavior without new dependency ownership?
5. Can the change be expressed as one focused function, one narrow seam, or a small local edit?
6. What must not be simplified away: validation at trust boundaries, data-loss prevention, security, accessibility, public behavior, golden-path user behavior, migration safety, and required tests/checks?
7. What evidence will prove the smaller implementation is correct?

The gate favors minimum viable code, reuse-before-new-code, native/platform-first implementation, abstraction restraint, dependency restraint, and the smallest safe behavior-preserving change. It does not weaken deep modules, locality, seams, adapters, or validation evidence.

Run the deterministic advisory review before PR handoff or repair planning when a base ref exists:

```bash
_nsp hygiene code minimize-review --target . --base main
_nsp hygiene code minimize-review --target . --base main --format json
_nsp hygiene code repair-packet --target .
```

Minimum-code findings are bounded review triggers. If the git base ref is missing, rerun after fetching or creating it. If no diff exists, the review reports no changed surface. If the code graph is unavailable, NSP continues with diff-only findings. GDScript/Godot hints are conservative and do not claim full engine semantic analysis.

## Design Pressure Checks

1. **Minimum code is not automatically the smallest diff.** Reject a locally tiny patch when it materially increases change amplification, caller coordination, duplicated knowledge, hidden coupling, cognitive load, or unknown change impact. The smallest safe behavior-preserving change may be larger when it removes duplicated behavior, restores ownership, reduces caller knowledge, moves related rules behind the correct seam, or prevents future scattered changes. Keep the change cohesive, evidence-backed, and bounded; strategic design does not authorize unrelated refactoring.

2. **Preserve reversibility and recoverability.** Isolate costly or difficult-to-reverse decisions behind seams. Destructive changes, migrations, public contracts, persistent formats, vendor integrations, and protocols need a proportional rollback, forward-recovery, compatibility, or preservation strategy. Do not invent an artificial rollback when forward recovery is safer or more realistic.

3. **Require layers to earn their existence.** A layer must hide, normalize, combine, translate, enforce, or isolate meaningful behavior. A wrapper that only renames or forwards a call without reducing caller complexity is a review signal. Seams, adapters, modules, and service boundaries must provide real leverage, locality, replaceability, validation value, or infrastructure isolation—not merely satisfy a pattern.

4. **Design avoidable errors out of existence.** Prefer idempotent operations and explicit representations for expected absence, repetition, retries, already-completed state, and safe convergence. Normalize conditions the owned API can safely represent; do not add avoidable failure handling. Unexpected data-loss, security, validation, corruption, permission, and operational failures must remain visible and evidenced.

5. **Harden durable design through the existing NSP workflow.** For durable public interfaces, persistent data contracts, major module boundaries, cross-system protocols, or other high-cost choices, use:

   ```text
   draft design -> adversarial review -> resolve every must-fix finding
     -> hardened design -> ATDD/Ralph/PIV implementation
   ```

   Challenge ownership boundaries, interface size, caller knowledge, locality, change amplification, failure behavior, reversibility, compatibility, testability, migration risk, operational risk, and agentic execution risk. Do not introduce an independent design methodology, and do not begin implementation while must-fix findings remain unresolved.

6. **Balance abstraction restraint with durable generality.** Prefer the smallest stable domain abstraction that represents the underlying concept, serves demonstrated callers, preserves ownership, avoids duplicated knowledge, and keeps future changes local. Reject both one-off APIs shaped around an accidental use case and speculative frameworks for hypothetical consumers. Generalize when code represents the same owned knowledge or behavior—not merely because fragments look similar.

## Vocabulary

- A module is a cohesive unit that changes together. It may be a directory, package, service, CLI command group, UI feature, adapter group, or domain area.
- An interface is the public surface callers use. A small interface can still provide high leverage.
- An implementation is the private mechanism hidden behind the interface.
- A deep module provides substantial useful behavior behind a small interface.
- A shallow module exposes many details or tiny helpers while forcing callers to coordinate behavior.
- Depth is the ratio between useful behavior and interface size.
- A dependency graph shows what code depends on what other code.
- A seam is a boundary where behavior can be tested, replaced, or adapted.
- An adapter protects domain code from infrastructure such as filesystems, databases, networks, process state, storage, or UI frameworks.
- Locality means related changes remain near each other.
- Leverage means callers get more value with less coordination.
- Software entropy is the drift toward scattered rules, parallel implementations, and hard-to-change code.
- A harness is test coverage that makes refactoring observable and safer.

## Setup, Validation, Repair

Setup establishes target-owned policy, docs, skills, and report locations:

```bash
_nsp hygiene code setup --target <repo>
```

Validation checks objective signals and conservative heuristics:

```bash
_nsp hygiene code validate --target <repo>
```

Repair starts with a dry-run plan:

```bash
_nsp hygiene code repair --target <repo> --dry-run
```

Validation separates deterministic failures, deterministic warnings, heuristic review triggers, informational findings, and excluded files. A large file is usually a review trigger, not proof of bad design. Generated, vendored, build, lock, bundled, and intentionally data-heavy files should be excluded or policy-classified.

Repair should refuse automatic edits when behavior is unclear, tests are missing, boundaries are ambiguous, or infrastructure access is mixed into domain behavior. The safe default is: validate findings, produce a repair plan, recommend harnesses and golden-path coverage, then make no destructive edits.
