---
name: nsp-debug-watchman
description: Reproduces, diagnoses, minimally repairs, and verifies software defects with falsifiable hypotheses and fresh regression evidence. Use when a user reports a failing test, bug, regression, crash, incorrect behavior, flaky result, or performance or reliability defect and wants root-cause proof. Do not use for new feature implementation, vague planning, or final PR review.
---

# Debug Watchman

## Outcome
Reproduce the defect, identify its cause, repair it within scope, and prove the original failure is gone.

## Workflow
1. Start with observed failure, bounded environment, current receipt, and expected behavior. Preserve the smallest useful failing input and command.
2. Form falsifiable hypotheses and inspect relevant paths. Instrument or use an isolated [prototype](../nsp-prototype/SKILL.md) when a cheap probe distinguishes causes. Unreproduced behavior remains uncertain.
3. Read [execution.md](../nsp-prompt-router/references/execution.md), capture a meaningful regression before repair, change one mechanism, and rerun the original case plus affected checks. Iterate vertically for further necessary slices.

## Evidence
Return root cause, reproduction, regression result, validation, and remaining uncertainty. The deterministic CLI checks records; diagnosis remains agent-owned. Read [prediction.md](../nsp-prompt-router/references/prediction.md) only for material uncertainty and [run-state.md](../nsp-prompt-router/references/run-state.md) only for continuation/dispatch.

## Boundaries
No speculative patch presented as proven, unrelated cleanup, generic validation cache, or completion while the failure persists. Resume the existing scope/evidence instead of rebuilding the plan each turn.
