---
name: nsp-insight-berean
description: Explains and investigates project behavior, architecture, Domain Language, and impact using deterministic map, graph, source, and documentation evidence. Use when a user asks how or why the project works, requests a walkthrough, comparison, impact analysis, teaching, or an evidence-backed answer about a file, symbol, feature, or term. Do not use for code repair, PR verdicts, first adoption, or plan generation.
---

# Insight Berean

## Outcome
Explain project behavior and architecture from bounded current evidence. Begin with the answer and mental model, then connect feature behavior to mechanism, source, and tests.

## Workflow
Use the question's scope and current context receipt. Follow deterministic map/graph or Knowledge Discovery query, inspect, expand, and evidence references before broad reads. Verify relevant source/tests. Distinguish facts, inference, and material unknowns without turning normal teaching into a diagnostic report.

Continue the same journey on follow-up, retaining verified IDs and updating changed evidence. If visuals help, read [atlas-teaching.md](../nsp-prompt-router/references/atlas-teaching.md). Atlas is optional; plain explanation remains available with source anchors. Use one stable journey and loopback URL when publishing; separate agent story cues from human local selection.

Own evidence-backed Domain Language terms under `.docs/domain/` with real feature/technical links. Read [context-model.md](../nsp-prompt-router/references/context-model.md) only when writing durable context.

## Evidence
Cite the few paths, symbols, tests, or graph IDs supporting the explanation. State unavailable inputs where they affect the answer. The deterministic CLI retrieves and validates substrate; the agent synthesizes meaning. Normal explanations need no artifact pack or continuation state.

## Boundaries
No repair, PR verdict, plan pack, or adoption sweep unless assigned. Do not trust stale graph facts, imply captured output proves live UI behavior, or claim a browser supplies semantic reasoning to the CLI.
