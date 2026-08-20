# NSP Safety Rules

- Use `_nsp` as the deterministic engine.
- Prefer `/nsp` workflows over broad repository reads.
- Run status/map before relying on stale artifacts.
- Auto-engage NSP for broad feature work, code generation, architecture changes, security-sensitive changes, PR review, repair/hygiene work, project explanation, planning, validation, and evidence collection.
- Do not auto-engage NSP for tiny typos, isolated one-line answers, non-project general questions, or narrow local commands whose complete answer is already known.
- If `_nsp` is missing, maps are stale, graphs are invalid, routes are ambiguous, or model support is unavailable, keep scope narrow and use deterministic fallback where possible.
- Never dump the repository.
- Never expose secrets.
- Never mark model inference as reviewed.
- Never use model output to lower risk or satisfy validation.
- Never bypass validation.
- Keep evidence ephemeral unless explicitly promoted.

## Tool Output Economy

All NSP workflows should compose tool calls with token economy in mind. Use compact commands first, expand only when evidence requires it, and prefer artifact-backed details over console dumps.

## Context Economy

Use canonical guidance plus short local reminders. Preserve safety-critical and execution-local instructions inside skills, but avoid repeated long-form guidance across workflow docs, setup prompts, and reports.
