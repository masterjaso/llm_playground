<!-- nsp:meta
id: technical.flashmini-validation-operations
kind: technical-doc
audience: developer
status: active
description: Map FlashMini gate policy, evidence-producing checks, and release readiness decisions.
resource: ezra://technical/flashmini-validation-operations
scope: technical
persona: qa-validation
source: model
confidence: high
reviewStatus: unreviewed
graphNode: technical:flashmini-validation-operations
graphTags: flashmini,validation,gates,release
validation: context-header-audit,manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# FlashMini validation and operations

## Purpose

Define how implementation checks, experiment gates, and production-data audits
become reproducible readiness evidence. The policy distinguishes a passing
local probe from a release decision and keeps unresolved evidence explicit.

## Feature links

- [Validation and release gates](../features/flashmini-validation-release.md)
- [Matched FlashMini experiments](../features/flashmini-experiment-gates.md)
- [Production training-data releases](../features/flashmini-production-data.md)

## Architecture / design

- `gate_policy.py` loads and validates the registered thresholds, exact token and
  update budgets, evaluation slices, and verdict mapping.
- `gates.py` records structured decisions and relative comparisons; training and
  evaluation modules provide finite loss, clipping, routing, and quality inputs.
- The v3 validation scripts and JSON reports cover upstream differential checks,
  collision/causality probes, data identity, runtime health, and budget checks.
- Production readiness is a separate contract over source locks, recipe totals,
  exact tokenizer counts, remote checksums, decontamination, materialization,
  deterministic sampling, and private-evaluation evidence.

## Invariants

- Unknown checks or malformed policy fields fail closed rather than inheriting a
  permissive default.
- Gate evidence carries the candidate, configuration, data identity, and report
  path needed to reproduce the decision.
- A structural context probe cannot be promoted to long-context capability, and a
  green implementation suite cannot replace matched fresh-corpus runs.
- Production status remains false until every required readiness gate passes;
  paused or blocked work is recorded with its concrete reason.

## Code anchors

- repo://src/flashmini/gate_policy.py
- repo://src/flashmini/gates.py
- repo://src/flashmini/evaluation.py
- repo://src/flashmini/eval.py
- repo://src/flashmini/metrics_reconcile.py
- repo://scripts/flashmini_v3_validate.py
- repo://scripts/flashmini_v3_upstream_check.py
- repo://scripts/flashmini_v3_simulate.py
- repo://scripts/flashmini_v3_execute.py
- repo://configs/flashmini/v3_gate_policy.json
- repo://docs/validation/flashmini-v3-runtime.json
- repo://docs/validation/flashmini-v3-data.json
- repo://training_data/production_release_status.json

## Validation

- `python -m pytest tests/flashmini/test_gate_policy.py tests/flashmini/test_gates.py`
- `python -m pytest tests/flashmini/test_evaluation_slices.py tests/flashmini/test_v3_probes.py`
- `python scripts/flashmini_v3_validate.py --help`
- `python scripts/flashmini_v3_upstream_check.py --help`
- `_nsp validate --target .` validates the NSP evidence substrate; project tests
  and release reports provide the domain evidence.

## Risks

- The repository records strong implementation and short-pilot evidence, but
  official scaling and long-context gates remain open.
- Remote pilot integrity findings and missing private-evaluation exports block a
  final data-release claim even when local code checks pass.
- Generated checkpoints and large corpus artifacts are run evidence, not durable
  source claims; only referenced, validated reports should be promoted.
