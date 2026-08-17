<!-- nsp:meta
id: agents
kind: template
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:AGENTS.md
graphTags: routing
validation: manifest-check,secret-scan
owner: context-hygiene
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Agent instructions

NSP is external standalone tooling for agent-first software development through Context Management and Knowledge Discovery. Durable NSP context lives under root `.docs/`; `.nsp/skills/` holds assistant-facing NSP contracts, and `.nsp/artifacts/` is purgeable runtime output. Do not vendor NSP scripts into this repository and do not add NSP as a runtime dependency.

## Entry modes

- **Primary** (default): infer intent/persona/scope/risk, use `nsp-prompt-router`, then bounded context loading.
- **Delegated**: if the prompt contains `NSP_ENTRY_MODE: delegated` with a valid envelope (`ROUTING_STATUS: complete`, `EPIC_OWNER`, `TASK_TYPE`, `OBJECTIVE`, `SCOPE`, `EXCLUSIONS`, `EXPECTED_OUTPUT`), skip top-level genesis, broad routing, persona discovery, and epic classification. Execute only the bounded assignment from envelope-only context (do not expect the parent transcript). Malformed envelopes fail closed (ask parent; do not re-run full routing). Nested delegates may spawn further delegates only with a new valid envelope and must never own an epic. Prefer clean-context sub-agents for Implement/Validate slices when available (`nsp-agent-workflow`). Schema: `.agents/skills/nsp-prompt-router/contracts/delegation-envelope.schema.json`.

Before broad context loading on **primary** entry, first infer task intent, primary persona, supporting personas, likely scopes, and risk flags from the user request.

For multi-session or epic work, keep in-flight state under `.nsp/artifacts/tmp/ralph/<epic-id>/` only. Do not write or resume in-flight epic plans, status, or ledgers from `.docs/roadmap/**`.

If available, use the `nsp-prompt-router` skill to infer the proper persona and in-scope context for the request. When deterministic context facts are useful, run:

```bash
_nsp status --target .
_nsp context select --target . --persona <persona> --scope <scope> --tags <tag,list> --intent "<short summary>" --limit 8 --format json
_nsp hygiene setup --target .
```

If `nsp-prompt-router` or `_nsp` is unavailable, continue in NSP-passive mode without asking the user to install it. Prefer `.nsp/artifacts/frontmatter/index.json` when present, then cheaply inspect fixed metadata blocks:

```bash
rg -n -A 16 '^<!-- nsp:meta$' \
  AGENTS.md README*.md .docs/**/*.md docs/**/*.md architecture/**/*.md adr/**/*.md handbook/**/*.md \
  --glob '!node_modules/**' \
  --glob '!dist/**' \
  --glob '!coverage/**' \
  --glob '!artifacts/**'
```

Fallback without `rg`:

```bash
grep -R -n -A 16 '^<!-- nsp:meta$' \
  AGENTS.md README*.md .docs docs architecture adr handbook 2>/dev/null
```

Read only fixed metadata blocks first, rank candidate docs by metadata, and open only selected documents after ranking. Do not scan all docs or the full repo into prompt context.

Maintain a compact context receipt across turns: routing mode, task intent, personas, scopes, loaded context with reasons, not-loaded areas, and reload triggers such as scope changes, new path hints, validation failures, or frontmatter/index changes.

Use the NSP Map first when available. Repair missing, malformed, or stale governed facts before relying on affected docs.

## Auto-engagement rules

Must engage NSP when the task involves broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair or hygiene work, project explanation, planning, validation, or evidence collection.

Should not engage NSP for a tiny typo, isolated one-line answer, non-project general question, or narrow local command whose complete answer is already known.

Failure behavior: if `_nsp` is missing, the map is stale, the graph is invalid, context selection is ambiguous, or model support is unavailable, use passive metadata fallback where possible, keep scope narrow, mark stale or model-inferred facts as advisory, and do not claim validation without validator output. Mention missing NSP only when the user asks for NSP-powered validation, map refresh, graph/code-graph analysis, hygiene execution, or deterministic evidence.

Use bounded substrate commands when the matching skill needs deterministic facts:

```bash
_nsp ask-anchors --target . "<question>"
_nsp hygiene code validate --target .
_nsp hygiene context validate --target .
_nsp explain-facts --target . <file-or-node>
_nsp review-manifest --target . --base main
_nsp evidence collect --target .
```

Use `nsp-insight-berean` for explanation/teaching and Domain Language curation, `nsp-build-bezalel` / `nsp-debug-watchman` for implementation and defect work, `nsp-clean-purify` for guarded ephemeral artifact cleanup, `nsp-code-hygiene` for semantic repair over repair packets, and `nsp-review-discernment` for PR review reasoning over those outputs.

Do not dump the repo. Do not expose secrets.

`.agents/skills/` is the committed NSP assistant-contract surface (full skill bodies). Cline/Claude may install thin discovery wrappers elsewhere that point here; other harnesses discover `.agents/skills/**` directly. `.nsp/artifacts/` is purgeable runtime storage and must not hold durable passive-mode guidance.
