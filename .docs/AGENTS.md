<!-- nsp:meta
id: docs.agents
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/AGENTS.md
graphTags: docs,routing
validation: context-header-audit,manifest-check,secret-scan
owner: context-hygiene
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Canonical execution router

This router governs agent work inside this repository. Route before load and keep context bounded.

## Authority order

1. Safety invariants — no secrets, no silent audit bypasses.
2. Context hygiene — route before loading broad context.
3. Correctness / evidence — validation must match claims.
4. This router.
5. Persona registry and scope map.

## Entry modes (primary vs delegated)

| Mode | When | Required behavior |
|------|------|-------------------|
| **Primary** | Original user/top-level request; no valid delegated envelope | Full genesis/classification/routing/persona/decomposition and epic ownership as needed |
| **Delegated** | Valid `NSP_ENTRY_MODE: delegated` envelope from a parent | Trust parent routing; execute only the bounded objective with envelope-only context (no parent chat dump); skip top-level genesis, broad routing, persona discovery, and epic classification; escalate missing/contradictory context to the parent |

Required envelope fields: `NSP_ENTRY_MODE`, `ROUTING_STATUS: complete`, `EPIC_OWNER`, `TASK_TYPE`, `OBJECTIVE`, `SCOPE`, `EXCLUSIONS`, `EXPECTED_OUTPUT`. Optional harness hints: `CONTEXT_POLICY: envelope-only`, `EFFORT_CLASS`, `SEPARATION`. Schema: `.agents/skills/nsp-prompt-router/contracts/delegation-envelope.schema.json`. Malformed or contradictory envelopes **fail closed** — do not silently fall back to primary routing. Nested delegation requires a fresh valid envelope; nested agents must not initiate epics; the original primary remains `EPIC_OWNER`.

When the harness supports sub-agents, prefer clean-context delegated slices for Implement/Validate work (see `nsp-agent-workflow`).

## Required preflight (primary entry)

```bash
_nsp context select --target . --persona <persona> --scope <scope> --tags <tag,list> --intent "<short summary>" --limit 8 --format json
```

Then load only the selected docs and relevant scopes. Use map/graph explain for bounded context and evidence collection before PR handoff.

For multi-session or epic work, keep in-flight state under `.nsp/artifacts/tmp/ralph/<epic-id>/` only. Do not write or resume in-flight epic plans, status, or ledgers from `.docs/roadmap/**`.

## Auto-engagement rules

Must engage NSP for broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair/hygiene work, project explanation, planning, validation, and evidence collection. Should not engage NSP for tiny typos, isolated one-line answers, non-project general questions, or narrow local commands whose complete answer is already known. If NSP is missing, stale, invalid, ambiguous, or model support is unavailable, use deterministic fallback where possible, keep scope narrow, and do not claim validation without validator output.

Repair missing or malformed frontmatter before relying on affected docs. Do not dump the repo and do not expose secrets.
