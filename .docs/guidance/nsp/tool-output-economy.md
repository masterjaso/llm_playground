<!-- nsp:meta
id: docs.guidance.nsp.tool.output.economy
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/tool-output-economy.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# NSP Tool Output Economy Guidance

NSP treats tool output as context. Tool calls must be token-efficient by default.

Default CLI and agent-composed tool calls should return only:

- meaningful results
- actionable warnings
- actionable errors
- short summaries
- artifact paths for deeper inspection

Verbose output is opt-in only through explicit need or flags such as `--verbose`, `--debug`, `--json`, `--full`, or `--details`.

Full logs and large outputs should be written to ignored artifacts and referenced by path in concise console output.

Agents must avoid broad, noisy commands when a scoped command can answer the question. Prefer exact paths, bounded searches, line ranges, filters, summaries, and artifact-backed detail.

## Tool Output Economy

Tool Output Economy means every command result is treated as part of the active context budget. Default output should be summary-first, compact by default, and limited to meaningful results with actionable warnings or errors.

## Token-Efficient Tool Use

Token-Efficient Tool Use preserves analysis depth without dumping full logs, generated files, lockfiles, dependency trees, graph JSON, or large reports into the active session. Inspect concise summaries first, then expand only when evidence requires it.

## Summary-First Output

Command output should start with the result, counts, verdict, or next action. Detailed evidence belongs in ignored artifacts with printed paths.

## Compact By Default

Default command behavior should be minimal by default. Do not make normal tool output noisy to prove work happened.

## Verbose Opt-In

Verbose, debug, full, details, and JSON output are explicit modes for humans, automation, or failure investigation. They should not be the default human-readable console path. Keep verbose opt-in as the default contract for expanded output. `_nsp context select` is summary-first by default; full selection JSON is `--format json`.

## Artifact-Backed Details

Full logs, reports, machine-readable JSON, and large validation details should go under `.nsp/artifacts/**` or `artifacts/**` and be referenced by path from concise console output.

## Context and job projections

Keep ContextRefs and job registries rich at rest and sparse in flight. Store job/result detail under `.nsp/artifacts/runs/<run-id>/`; inject only inspect/peek slices and compact assurance summaries. Do not dump parent transcripts, role taxonomies, or unbounded context into child, verifier, or handoff prompts.

## Actionable Warnings/Errors Only

Warnings and errors should explain what failed, why it matters, and the smallest next action. Avoid broad warning dumps when a summary plus report path is enough.

## Agent-Composed Tool Calls

Agent-composed command economy means agents should build shell and CLI calls that answer the immediate question without dumping unrelated context.

When composing shell or CLI commands in-flight, agents should:

- prefer `rg` over broad `grep -R`
- use exact paths or scoped globs
- use `--glob`, `--max-count`, `--files-with-matches`, `--count`, or equivalent when useful
- use `sed -n`, `head`, `tail`, `jq`, or targeted filters instead of dumping whole files
- redirect full noisy output to ignored artifacts when detail is needed
- inspect summaries first, then expand only when needed
- avoid dumping generated files, lockfiles, dependency trees, build logs, or full test logs unless explicitly needed

## CLI Output Contract

NSP commands should prefer:

- result summaries on stdout
- actionable warnings/errors on stderr
- full details in `.nsp/artifacts/**` or `artifacts/**`
- paths to full logs/reports
- `--verbose` or `--debug` for expanded human-readable output
- `--json` for machine-readable output
- non-zero exits only for true failure states
