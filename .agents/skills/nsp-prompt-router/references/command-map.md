# NSP Command Map

NSP commands support agent-first software development through two product features:

- Context Management: status, context selection, hygiene, review, repair, validate, evidence, prepare-pr for bounded work, token efficiency, context hygiene, and code hygiene governance.
- Knowledge Discovery: map, code-map, impact, ask, explain, local viewer/report, relationship visualization, and lower-level graph/code-graph diagnostics when needed.

Deterministic CLI utilities remain directly usable for setup, maps, validation, graph layers, reports, and bounded evidence. Inference-bearing workflows are skill-owned and use CLI output as substrate.

Direct deterministic utilities:

```bash
_nsp setup --target .
_nsp status --target .
_nsp map --target .
_nsp code-map --target . --no-open
_nsp impact --target . <file-or-symbol> --depth 2
_nsp context coverage --target .
_nsp context drift --target . --base main
_nsp ccb validate --target . --tier trusted
_nsp ccb coverage --target .
_nsp hygiene setup --target .
_nsp hygiene code validate --target .
_nsp hygiene maintain --target . --base main
_nsp hygiene code repair-packet --target .
_nsp hygiene context validate --target .
_nsp validate --target .
_nsp evidence collect --target .
_nsp support bundle --target . --redacted
_nsp artifacts --target . --json
_nsp artifacts --clean --target . --dry-run --json
_nsp project cleanup --target . --json
_nsp scaffold skill --target . --provider generic
```

Inference-bearing workflows:

| Workflow | Skill owner | Deterministic substrate |
|----------|-------------|-------------------------|
| `/nsp context <task>` | `nsp-prompt-router` | `_nsp context select --target . --request "<task summary>" --receipt-v2` |
| `/nsp ask <question>` | `nsp-insight-berean` | `_nsp ask-anchors --target . "<question>"`, then bounded Atlas query/inspect/evidence as needed |
| `/nsp explain <node-or-anchor>` | `nsp-insight-berean` | `_nsp explain-facts --target . <node-or-anchor>`, then bounded Atlas inspection as needed |
| `/nsp plan <milestone>` | `nsp-plan-genesis` or `nsp-epic-execution` | `_nsp plan-substrate --target . --milestone "<milestone>" [--execution-class <class>] [--fact-ledger <path>] [--require-discovery\|--material-uncertainty]` |
| `/nsp plan discovery` | `nsp-plan-genesis` / `nsp-epic-execution` | `_nsp plan-substrate discovery <seed\|validate\|readiness> --target . --run-id <id>` |
| `/nsp review` | `nsp-review-discernment` | `_nsp review-manifest --target . --base main` |
| `/nsp repair` | `nsp-code-hygiene` or `nsp-context-hygiene` | `_nsp repair-plan --target .` |
| `/nsp hygiene code repair` | `nsp-code-hygiene` | `_nsp hygiene code repair --target . --dry-run` and `_nsp hygiene code repair-packet --target .` |
| `/nsp cleanup` | `nsp-clean-purify` | `_nsp project cleanup --target . --json`, then explicit `--apply` after preview |
| `/nsp ccb promote` | `nsp-ccb-hygiene` | `_nsp ccb repair --target . --dry-run` and `_nsp ccb coverage --target .` |
| Predictive PIV | `nsp-agent-workflow` | `_nsp work prediction <validate\|record\|resume\|complete> --target . --run-id <id> --mode <none\|compact\|expanded>` |

`_nsp project cleanup` is the preferred product cleanup surface and previews by
default. `nsp-clean-purify` inspects and interprets that output before an
explicit apply. `_nsp artifacts --clean --target <repo>` remains the low-level
deterministic alias; `--force` overrides lifecycle blockers only.

Deprecated root aliases (`_nsp ask`, `_nsp explain`, `_nsp plan`, `_nsp review`, `_nsp repair`) keep working for one release, emit a deterministic-boundary warning, and point to the canonical command above.

For a material visual request, `nsp-insight-berean` validates an advisory
`AtlasViewSpec` and optional `GuidedTour` against the captured snapshot before
opening a local view. If a harness cannot open it, return the exact
`_nsp knowledge atlas atlas_open_view --view-spec-json <json>` handoff instead.
The deterministic CLI retrieves and validates evidence; it does not semantically
answer, author lessons, or author tours.

Advanced internals remain available under `_nsp graph ...`, `_nsp code-graph ...`, and `_nsp frontmatter ...` for debugging and deterministic layer maintenance.

<!-- nsp:managed-section start id="nsp-command-map-ccb" template="skill-catalog/references/command-map.md" -->
- /nsp ccb build → _nsp ccb build --target .
- /nsp ccb validate → _nsp ccb validate --target . --readiness
- /nsp ccb explain → _nsp ccb explain --target . --feature <feature-id>
- /nsp ccb repair → _nsp ccb repair --target . --dry-run
<!-- nsp:managed-section end -->
