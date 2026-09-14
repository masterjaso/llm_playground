---
name: nsp-upgrade-version
description: Reconciles target-owned NSP guidance, skills, and tool surfaces after a deterministic `_nsp update-version` or alignment run, preserving customizations and recording residual risks. Use when NSP-managed assets were refreshed, a target is upgrading versions, or an upgrade report and backups need review. Do not use for executing the upgrade itself, ordinary feature changes, or application behavior work unrelated to upgrade integration.
user-invocable: false
---

# Upgrade reconciliation

## Outcome
Reconcile target-owned guidance and skills after deterministic update/alignment, preserving customizations and reporting residual risks.

## Workflow
Read upgrade report, backups, marker, target instructions, and changed surfaces. Do not repeat `_nsp update-version` when the intended version already applied. Reconcile assigned drift and safety conflicts with target ownership intact.

Run affected skill, manifest, context, and setup validations. Use Maintain for diff alignment and Adopt only for a justified missing baseline. Healthy `aligned-basic` context needs no repeated full Genesis. Read [run-state.md](../nsp-prompt-router/references/run-state.md) only for continuation.

## Evidence
Use `upgrade-reconciliation-report` for customization, decisions, checks, and blockers. Set `agenticReconciliationVersion`, `agenticReconciliationAt`, and `agenticReconciliationReport` only after gates pass. `appliedNspVersion` and `appliedAt` remain deterministic CLI-owned. Validation is not semantic reconciliation.

## Boundaries
No unrelated application repair, dependency changes, blind overwrite, fabricated markers, or duplicate human reports. Missing substrate needs its exact limitation, never invented PASS.
