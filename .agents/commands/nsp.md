# NSP Command

Interpret `/nsp <verb> <args>` as a request to run the NSP workflow against the current repository. Deterministic verbs map directly to `_nsp`; inference-bearing verbs invoke the named skill and use `_nsp` only as deterministic substrate.

Ordinary natural-language questions about project architecture, behavior,
relationships, tests, decisions, or change impact automatically engage
`nsp-insight-berean`; `/nsp ask` is an optional explicit form, not the human's
required entry point. Gather deterministic evidence internally and teach in
chat with contextual follow-ups.

Canonical mappings:

- `/nsp setup` -> `_nsp setup --target .`
- `/nsp status` -> `_nsp status --target .`
- `/nsp map` -> `_nsp knowledge build --target .` (or compatibility `_nsp map --target .`)
- `/nsp knowledge view` -> internally validate the current `AtlasViewSpec`, then run `_nsp knowledge view --serve --view-spec-json <json> --target . --json` and return its clickable loopback URL
- `/nsp code-map` -> `_nsp code-map --target . --no-open`
- `/nsp impact <file-or-symbol>` -> `_nsp impact --target . <file-or-symbol> --depth 2`
- `/nsp ask <question>` -> use `nsp-insight-berean` with `_nsp ask-anchors --target . "<question>"` substrate
- `/nsp context <task>` -> use `nsp-prompt-router`, then `_nsp context select --target . --request "<task>"`
- `/nsp hygiene setup` -> `_nsp hygiene setup --target .`
- `/nsp hygiene code validate` -> `_nsp hygiene code validate --target .`
- `/nsp hygiene code repair` -> use `nsp-code-hygiene` with `_nsp hygiene code repair --target . --dry-run` substrate
- `/nsp hygiene code repair-packet` -> `_nsp hygiene code repair-packet --target .`
- `/nsp hygiene code minimize-review` -> `_nsp hygiene code minimize-review --target . --base main`
- `/nsp hygiene maintain` -> `_nsp hygiene maintain --target . --base main`
- `/nsp hygiene context validate` -> `_nsp hygiene context validate --target .`
- `/nsp cleanup` -> use `nsp-clean-purify` with `_nsp project cleanup --target . --json` preview, then explicit `--apply`
- `/nsp explain <node-or-anchor>` -> use `nsp-insight-berean` with `_nsp explain-facts --target . <node-or-anchor>` substrate
- `/nsp review` -> use `nsp-review-discernment` with `_nsp review-manifest --target . --base main` substrate
- `/nsp repair` -> use `nsp-code-hygiene` or `nsp-context-hygiene` with `_nsp repair-plan --target .` substrate
- `/nsp validate` -> `_nsp validate --target .`
- `/nsp plan <milestone>` -> use `nsp-plan-genesis` or `nsp-epic-execution` with `_nsp plan-substrate --target . "<milestone>"` substrate

For an Atlas visual request, `nsp-insight-berean` first gathers bounded
evidence with `_nsp knowledge atlas` (or local MCP), then returns a validated
`AtlasViewSpec` and optional `GuidedTour`, starts the read-only loopback view
internally, and returns one clickable
`http://127.0.0.1:<port>/atlas/<session-id>` link for those same stable IDs. Do
not make the human operate the CLI to open the primary journey. No `nsp-agent`
runtime is required.

The canonical loop is **harness ask → Berean → Atlas evidence → Berean teaching → Atlas visualization**. `_nsp knowledge view` is the primary visual surface; deterministic CLI/MCP operations are evidence substrate, not semantic conversation.

Use the NSP response format and safety rules from `.agents/skills/nsp-prompt-router/references/`.

Auto-engagement rule: use `/nsp` automatically for broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair/hygiene work, project explanation, planning, validation, and evidence collection. Do not use `/nsp` for tiny typos, isolated one-line answers, non-project general questions, or a narrow local command whose complete answer is already known.

Failure behavior: when `_nsp` is missing, continue in passive mode when committed `AGENTS.md`, `.docs/**`, `.agents/skills/**`, and NSP frontmatter exist; otherwise behave normally without NSP assumptions. Mention missing NSP only when it limits requested deterministic validation/evidence, keep context narrow, mark stale/model-inferred facts as advisory, and do not claim validation without validator output.
