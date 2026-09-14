---
name: nsp-adopt-ezra
description: Orchestrates first adoption or a major context re-baseline for an existing repository by scanning code and docs, producing durable project context, and coordinating persona and hygiene setup. Use when onboarding NSP to an existing project, following `_nsp setup`, rebuilding context after large undocumented changes, or absorbing existing repository guidance. Do not use for routine diff maintenance, isolated explanations, or implementation execution.
---

# Adopt Ezra

## Outcome
Establish evidence-backed context on first adoption or explicit major re-baseline. Primary orchestration owns adoption; delegated workers keep their assigned slice.

## Workflow
Read supplied setup prompt/runbook, preserve target-owned guidance, and resume `genesis-sweep` state. Use [Context Genesis](../nsp-context-genesis/SKILL.md) for discovery, then context/code/CCB workers for bounded repairs. Read [context-model.md](../nsp-prompt-router/references/context-model.md) for coverage and [run-state.md](../nsp-prompt-router/references/run-state.md) only for coordination/resumption.

Inventory entrypoints, runtime/configuration, tests, delivery, features, mechanisms, personas, guidance, and gaps. Produce project-specific `.docs` context supported by source. Preserve useful docs/customization; reserved NSP guidance is not for target-specific rules.

## Evidence
Validate affected frontmatter, manifest, context, and graph with available deterministic commands. `aligned-basic` is the baseline; linked CCB is minimum when enabled, with anchored/reviewed quality pursued from evidence. Record unavailable commands and remaining gaps. CLI output does not replace semantic discovery.

## Boundaries
No full sweep for routine diffs, undocumented features stated as fact, automatic candidate promotion, or mandatory exhaustive anchoring before useful adoption. Preserve original epic ownership in delegation. Durable knowledge goes in `.docs`; bot inventories/state remain ephemeral.
