<!-- nsp:meta
id: docs.nsp.usage
kind: runbook
scope: guidance
persona: context-hygiene
status: active
source: model
confidence: high
reviewStatus: unreviewed
graphNode: document:.docs/NSP_USAGE.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-09-22
replaces:
replacedBy:
-->

# NSP usage

NSP is installed or run externally and pointed at this repository with `--target`. Use it to support agent-first software development: Context Management keeps work bounded, token-efficient, and evidence-backed; Knowledge Discovery helps humans inspect project relationships before carrying the context manually. Target repositories do not take a runtime dependency on NSP.

## Modes

- **NSP-active:** `_nsp` is available or explicitly requested. Infer intent/persona/scope first, use `nsp-prompt-router` for the reasoning layer, then use `_nsp context select` to emit deterministic context-selection facts when needed. Use map, graph, code graph, hygiene, validation, and evidence commands only when their deterministic facts matter.
- **NSP-passive:** committed guidance and `.nsp/**` artifacts are available but `_nsp` is not installed or not intentionally invoked. Use `AGENTS.md`, `.docs/**`, `.agents/skills/**`, `.nsp/**`, NSP frontmatter blocks, and context receipts. Do not ask the user to install NSP by default.
- **NSP-absent:** neither `_nsp` nor committed NSP artifacts are available. Behave normally, make no NSP assumptions, and do not claim NSP validation or evidence.

## Entry modes

- **Primary** (default): full classification, persona, routing, and epic ownership as needed.
- **Delegated**: valid `NSP_ENTRY_MODE: delegated` envelope — execute only the bounded objective; skip top-level genesis/routing/epic ownership. Schema: `.agents/skills/nsp-prompt-router/contracts/delegation-envelope.schema.json`. Malformed envelopes fail closed.

Typical flow:

```bash
_nsp status --target .
_nsp project validate --target .
_nsp knowledge build --target .
_nsp knowledge view --lens architecture --query "Atlas architecture" --target .
_nsp knowledge view domain --focus <term-slug> --target .   # Domain Language exact term
_nsp map --target .  # compatibility alias
_nsp context select --target . --persona <persona> --scope <scope> --tags <tag,list> --intent "<short summary>" --limit 8 --format json
_nsp ask-anchors --target . "<question>"
_nsp explain-facts --target . <file-or-node>
_nsp review-manifest --target . --base main
_nsp evidence collect --target .
```

Public skills (peer discovery): `nsp-adopt-ezra`, `nsp-insight-berean`, `nsp-plan-genesis`, `nsp-build-bezalel`, `nsp-debug-watchman`, `nsp-maintain-steward`, `nsp-clean-purify`, `nsp-review-discernment`, `nsp-workstatus-herald`, `nsp-acceptance-prover`. Internal/worker/lifecycle composables include `nsp-context-hygiene`, `nsp-code-hygiene`, and `nsp-ccb-hygiene`. Use `nsp-prompt-router` for preflight routing. The commands above are substrate, not CLI-side inference.

**Skill surfaces:** full skill bodies live only under `.agents/skills/**`. Cline/Claude may also have thin discovery wrappers under `.cline/skills/**` or `.claude/skills/**` that point back to `.agents`. Codex, Cursor, Copilot, Continue, Roo, and generic agents use `.agents/skills/**` (and/or their command/rules adapters) — they do not get a separate full skill-tree copy.

Domain Language terms (when present) live at `.docs/domain/<slug>.md`. Retrieve with `_nsp knowledge view domain --focus <slug>` or `nsp-insight-berean`; do not treat the CLI as the semantic owner of meaning.

Auto-engage NSP for broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair/hygiene work, project explanation, planning, validation, and evidence collection. Do not auto-engage it for tiny typos, isolated one-line answers, non-project general questions, or a narrow local command whose complete answer is already known.

If `_nsp` is missing, continue in NSP-passive mode when committed NSP guidance or `.nsp/**` artifacts exist. Keep context narrow, mark stale or model-inferred facts as advisory, document missing NSP only when it limits a requested NSP-powered validation/evidence outcome, and do not claim validation without validator output.

Passive cheap scan recipe:

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

Read only fixed metadata blocks first, prefer `.nsp/frontmatter/index.json` when present, rank candidate docs by metadata, and open only selected documents after ranking. Do not scan all docs or the full repo into prompt context.

Maintain a compact context receipt across turns with routing mode, task intent, primary and supporting personas, primary scopes, loaded context with reasons, not-loaded areas, and reload triggers. Re-check scope cheaply each turn and reload selected docs only when the task scope, path hints, validation failures, conversation context, or frontmatter/index changes require it.

Advanced graph/frontmatter commands remain available for deterministic layer maintenance, but ordinary human learning should start with chat/Berean and one Atlas; `/nsp` and substrate commands remain available to the routed workflow.

## Predictive PIV

Meaningful work still uses ATDD and Plan → Implement → Validate. During Plan, select prediction depth adaptively: `none` for direct or already-clear low-risk work, `compact` for uncertain/risky/high-impact work, and `expanded` only when alternatives or diagnostic probes improve a decision. Acceptance criteria define the destination; a Prediction Contract describes the expected path and is not a reasoning transcript.

Activate a contract before implementation. During Validate, record exactly one result classifier (`expected-match`, `benign-deviation`, `scope-discovery`, `counterexample`, or `invalid-validation`) in the same run. Material contradictions require re-planning or a revised/abandoned hypothesis and block completion until resolved. Resume from compact current-run evidence rather than chat history; temporary predictions are not durable project truth.

## Safe artifact handling

Inspect before cleanup:

```bash
_nsp artifacts --target .
_nsp artifacts --target . --json
_nsp artifacts --clean --target . --dry-run --json
```

Real cleanup requires an explicit target: `_nsp artifacts --clean --target .`. Active work, retained runs, pins, Atlas sessions and referenced evidence are protected by default. Explicit `--force` can override typed lifecycle-policy protections, including selected active work or retained evidence; it never bypasses hard integrity or live-mutex safety. Do not use force as a routine way around a blocker. Preview completed-run retention with `_nsp run gc --target . --dry-run`; `--keep-runs` and `--keep-days` change retention without bypassing protection. Successful full cleanup leaves `.gitkeep`; `partial` reports remaining entries and exits nonzero.

The target repository owns `AGENTS.md`, durable `.docs/**` context and policy, `.agents/skills/**` (installed skill surface), `.nsp/skills/**` (deprecated compatibility index only), and the `.nsp/artifacts/` runtime workspace. NSP owns executable tooling. Do not copy NSP scripts here, do not add NSP to `package.json`, and do not create broad repository dumps.

`.nsp` is limited to `.nsp/skills/` (compatibility index) and `.nsp/artifacts/`. `.nsp/artifacts/` is purgeable temporary storage; do not place durable passive-mode guidance there.
