<!-- nsp:meta
id: docs.agent.personas
kind: guidance
scope: guidance
persona: platform-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/AGENT_PERSONAS.md
graphTags: docs
validation: context-header-audit,manifest-check,secret-scan
owner: platform-engineering
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Agent persona registry

| `role_id` | Mission | Primary scope keys | Validation expectations |
|-----------|---------|--------------------|-------------------------|
| `context-hygiene` | Keep routing, docs, and scope boundaries coherent | `governance`, `features`, `technical` | Header + manifest audits pass |
| `prompt-engineering` | Keep prompts and planning workflows aligned | `features`, `governance` | Prompt workflow docs and packs stay coherent |
| `platform-engineering` | Maintain scripts, tooling, and CI determinism | `scripts`, `tooling`, `ci` | `npm run validate:governance` succeeds |
| `security-engineering` | Treat secrets and unsafe autonomy as blocking | `security` | Secret scan and security review pass |
| `qa-validation` | Provide correctness evidence and review discipline | `tests`, `technical` | Required checks are explicit and green |
| `release-operations` | Ship guidance and package changes safely | `root`, `ci` | Release flow uses deterministic evidence |
| `governance-package` | Keep stewardship package coherence end-to-end | `governance`, `scripts`, `root` | Target validation and evidence stay aligned |
