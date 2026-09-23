<!-- nsp:meta
id: docs.agent.personas
kind: guidance
scope: guidance
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: unreviewed
graphNode: document:.docs/AGENT_PERSONAS.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: platform-engineering
lastReviewed: 2026-09-23
replaces: 
replacedBy: 
-->

# Agent persona registry

| `role_id` | Mission | Primary scope keys | Validation expectations |
|-----------|---------|--------------------|-------------------------|
| `context-hygiene` | Keep routing, docs, and scope boundaries coherent | `governance`, `features`, `technical` | Header + manifest audits pass |
| `prompt-engineering` | Keep prompts and planning workflows aligned | `features`, `governance` | Prompt workflow docs and packs stay coherent |
| `platform-engineering` | Maintain Python package, scripts, tooling, and deterministic runs | `source`, `scripts`, `tooling` | `python -m pytest tests/flashmini` and targeted checks succeed |
| `security-engineering` | Treat secrets and unsafe autonomy as blocking | `security` | Secret scan and security review pass |
| `qa-validation` | Provide correctness evidence and review discipline | `tests`, `technical` | Required checks are explicit and green |
| `release-operations` | Ship data and package changes safely | `root`, `training-data`, `tooling` | Release flow uses immutable identities and deterministic evidence |
| `governance-package` | Keep stewardship package coherence end-to-end | `governance`, `scripts`, `root` | Target validation and evidence stay aligned |
| `model-research` | Run matched architecture experiments and interpret registered gates | `source`, `configs`, `technical`, `tests` | Matched identities, checkpoints, and gate evidence remain reproducible |
| `data-operations` | Build bounded, provenance-preserving training-data releases | `training-data`, `source`, `technical` | Source locks, checksums, dedupe, and release blockers stay explicit |
