<!-- nsp:meta
id: docs.guidance.nsp.context.establishment
kind: guidance
scope: guidance
persona: context-hygiene
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:.docs/guidance/nsp/context-establishment.md
graphTags: context,docs,nsp
validation: context-header-audit,manifest-check,secret-scan
owner: guidance
lastReviewed: 2026-06-27
replaces: 
replacedBy: 
-->

# NSP Context Establishment Guidance

This guidance is durable repository-owned NSP guidance. It is model-agnostic and provider-agnostic.

The user copies the Genesis Sweep prompt from `_nsp setup` into a local agent or harness. That external agent performs the token-intensive first-run repository understanding pass. NSP itself scaffolds, validates, builds graphs, and writes deterministic reports; it does not call an LLM, require API keys, or manage model configuration.

## Durable Locations

- Target feature docs: `.docs/features/`
- Target technical docs: `.docs/technical/`
- Persona registry: `.docs/AGENT_PERSONAS.md`
- Scope map: `.docs/scopes/index.json`
- NSP reusable/default guidance: `.docs/guidance/nsp/`
- Target-specific guidance: `.docs/guidance/` outside `nsp/`
- Durable graph relationship guidance and accepted doc-to-code mappings: `.docs/graph/`

Fresh setup writes only README and index files into `.docs/features/` and `.docs/technical/`. Project-specific feature and technical docs are produced by the Genesis Sweep after repository evidence is inspected. Do not create a competing durable `.nsp/context/` source of truth.

## Artifact Boundary

`.nsp/artifacts/` is temporary, purgeable, ignored scratch space. Use it for setup prompts, context marshalling, progress notes, temporary reports, proposed outputs, and graph previews. Accepted durable outputs must be promoted out of `.nsp/artifacts/` before completion.

## Evidence Rules

Every inferred feature, technical concern, persona, doc-to-code mapping, test relationship, stale decision, or prune decision needs repository evidence. Unsupported claims stay draft or are marked stale with reasons.

## Validation

Run deterministic validation after promotion:

```bash
_nsp context validate --target .
_nsp validate --target .
```
