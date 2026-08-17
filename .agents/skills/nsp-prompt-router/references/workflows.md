# NSP Workflows

## Auto Engagement

Must engage NSP when the task involves broad feature work, code generation or source modification, architecture or public interface changes, security-sensitive changes, PR review, repair or hygiene work, project explanation, impact analysis, planning, validation, or evidence collection.

Should not engage NSP for a tiny typo, isolated one-line answer, non-project general question, or narrow local command whose complete answer is already known.

When engagement is required, start with `nsp-prompt-router` plus `_nsp context select --target . --request "<task summary>"` before broad reads. Add `_nsp status --target .` and `_nsp map --target .` when deterministic substrate freshness matters.

## Status And Map

1. Run `_nsp status --target .`.
2. Run `_nsp map --target .` when status reports stale or missing map artifacts.
3. Treat stale map/manifest/graph outputs as drift until repaired or explicitly documented.

## Ephemeral Artifact Cleanup

1. Use `nsp-clean-purify` when `.nsp/artifacts/**` is stale, noisy, or safe to
   discard after completed work.
2. Inspect with `_nsp artifacts --target . --json`, then preview with
   `_nsp project cleanup --target . --json`.
3. Apply only after explicit operator intent with
   `_nsp project cleanup --target . --apply --json`; use `--force` only for
   explicitly accepted open-run, active-claim, or incomplete-Ralph blockers.
4. Require a `cleaned` result and a final inspection showing only
   `.nsp/artifacts/.gitkeep`. `partial` is a failure requiring investigation.

## Ask

1. Use `nsp-insight-berean` for the explanation or question.
2. Run `_nsp status --target .` and refresh `_nsp map --target .` when substrate is missing or stale.
3. Use `_nsp ask-anchors --target . "<question>"` or `_nsp explain-facts --target . <anchor>` as bounded evidence.
4. Read only returned anchors unless scope expansion is justified.

## Route

1. Use `nsp-prompt-router` to interpret the request and select context.
2. Load returned docs/scopes first.
3. Preserve do-not-do directives.
4. Validate before handoff.

## Tool Output Economy

All NSP workflows should compose tool calls with token economy in mind. Use compact commands first, expand only when evidence requires it, and prefer artifact-backed details over console dumps.

## Context Economy

Use canonical guidance plus short local reminders. Preserve safety-critical and execution-local instructions inside skills, but avoid repeated long-form guidance across workflow docs, setup prompts, and reports.

## Pre-PR Maintain

1. Run `_nsp hygiene maintain --target . --base main` to classify the diff and emit a work order.
2. Follow `nsp-maintain-steward` for bounded code → context → ccb alignment on changed paths only.
3. Record `.nsp/artifacts/reports/hygiene-maintain-latest.md` before PR handoff or review.

`nsp-maintain-steward` is Genesis-aligned delta maintenance. It keeps feature docs, technical docs, personas, guidance, and CCB links aligned with changed paths. It does not rerun full `nsp-context-genesis`.

Use standalone `/nsp-maintain-steward` anytime to keep alignment between epics. Use full hygiene skills or `nsp-epic-execution` for repo-wide sweeps.

## Hygiene

1. Run `_nsp hygiene setup --target .` when hygiene policy, docs, skills, or report directories are missing.
2. Run `_nsp hygiene code validate --target .` before architecture-sensitive source changes or before PR handoff when guidance docs, graph artifacts, routing, or evidence changed.
3. Invoke the target-owned `nsp-code-hygiene` skill for architectural source repair. Treat heuristic findings as review triggers, not proof, and preserve generated/vendor/build exclusions and target-owned docs.

## Review

1. Use `nsp-review-discernment` to review the changes.
2. Check affected scopes, ownership, risk, drift, validation, maintain report, and evidence.
3. Report blockers before summaries.

## Knowledge Discovery

1. Use `nsp-insight-berean` for a semantic answer, lesson, comparison, impact
   explanation, or visual tour.
2. Retrieve bounded substrate with `_nsp ask-anchors`, `_nsp explain-facts`,
   or the Phase 02 `_nsp knowledge atlas` / local MCP operations.
3. Cite current Atlas evidence, label Fact, Inference, Decision, Conflict,
   Stale evidence, and Unknown, and abstain when the evidence does not support
   a conclusion.
4. For a material visual handoff, validate an advisory `AtlasViewSpec` and
   optional `GuidedTour`; return the exact local command when opening is not
   available in the current harness.

## Repair

1. Use `_nsp hygiene code repair --target . --dry-run` and `_nsp hygiene code repair-packet --target .` to emit deterministic repair substrate.
2. Use `nsp-code-hygiene` for semantic source repair and `nsp-context-hygiene` for semantic documentation repair.
3. Keep model-suggested changes advisory until reviewed and validated.
