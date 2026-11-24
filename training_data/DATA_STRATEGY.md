<!-- nsp:meta
id: docs.flashmini-data-strategy
kind: document
scope: technical
persona: release-operations
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:training_data/DATA_STRATEGY.md
graphTags: flashmini,data,strategy
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-11-24
replaces:
replacedBy:
-->

# FlashMini data strategy (locked 2026-11-24)

## Decision

**Default training data path is stream-from-upstream**, not growing a
self-hosted mirror on Hugging Face.

Any machine that clones this repository should:

1. Authenticate to the Hub with a read token (`HF_TOKEN` or
   `HUGGING_FACE_API_KEY` / `.env`, never committed).
2. Use the pinned identities in
   `training_data/registry/source_snapshot.lock.json` and
   `training_data/registry/sources.yaml`.
3. Apply domain weights from the selected recipe under
   `training_data/recipes/`.
4. Treat `mjaso/flashmini-v3-frozen-corpus` as the **locked PoC corpus** for
   v3 A/B/C/D reproduction (exact memmap + manifest
   `b06f559f65e6969a9dae36d392873610330b8a2425fe27f47518c4d200d42dab`).
5. Treat `mjaso/flashmini-data-v1` as a **partial historical lake** (paused).
   Do not resume multi-terabyte Hub mirroring unless a new written decision
   re-opens it.

Machine-readable copy of this decision:
`training_data/data_strategy.json`.

## Why

The production upload toward 8T reached only about **1.35B** exact tokens
(~0.017% of the 8T recipe) before pause. Code and reasoning domains were
empty; Nemotron sources remain gated; private-eval exports were missing; local
resume state does not travel with the git checkout. Re-hosting the full mix
is storage- and ops-heavy. The upstream datasets already exist on Hugging Face.

## Operator defaults

| Goal | Do this |
| --- | --- |
| Reproduce v3 PoC | Download / verify `mjaso/flashmini-v3-frozen-corpus` into `data/fineweb_v3_2b` (see `docs/flashmini-v3-data.md`) |
| Plan a 1B/50B mix | `flashmini-data plan --recipe training_data/recipes/<recipe>.yaml` against locked sources |
| Bounded build without Hub publish | `flashmini-data build --recipe ... --no-publish` (local staging only) |
| Resume Hub lake mirroring | **Not default.** Requires an explicit new decision and spend/storage plan |

## Upstream sources (public)

See `SOURCES.md` and `registry/sources.yaml`. Primary open families:

- general web: FineWeb / FineWeb-Edu
- books: Common Pile pre-1929 books
- synthetic edu: Cosmopedia v2
- math: FineMath, Open-Web-Math
- code: Stack v2 Edu (stream / review_required; do not mirror by default)
- reasoning: OpenThoughts (review_required)
- gated NVIDIA Nemotron sets: recipe-only until terms are accepted

Optional ready-made mixes (SmolLM-Corpus and similar) may be used for smoke
work. They do not replace the pinned FlashMini source lock unless adopted in
a recipe revision.

## What is frozen vs paused

| Artifact | Status |
| --- | --- |
| `mjaso/flashmini-v3-frozen-corpus` | Locked PoC data |
| Recipes + `source_snapshot.lock.json` | Locked mix identity for streaming |
| `mjaso/flashmini-data-v1` production prefixes | Partial; Hub mirroring paused |
| `mjaso/flashmini-eval-v1` | Empty; final decontam freeze still blocked |
