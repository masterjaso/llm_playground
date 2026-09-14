<!-- nsp:meta
id: roadmap-epic-flashmini-poc
kind: roadmap
scope: root
persona: governance-package
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:roadmap-epic-flashmini-poc
graphTags: epic,flashmini,roadmap
validation: manifest-check
owner: root
lastReviewed: 2026-09-07
-->

# Epic: FlashMini — Qwen3.8-Flash-Next Architecture PoC → 1B GO/NO-GO

## Goal

Build and execute a **fail-fast local research program** that determines whether a
consumer-optimized derivative of the Qwen3.8-Flash-Next architecture is sufficiently
promising to justify renting datacenter GPUs for a future ~18–24B-main-parameter model.

**Do NOT train the final large model.** The experiment ends at the ~1B decision package.

## Execution class

`EPIC` — multi-phase, resumable, ATDD-gated, PIV loops, Ralph state.

## Hardware

- 2× NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120), CUDA 13.3
- 10 CPU cores, 50 GB RAM, ~579 GB free NVMe
- Python 3.13.14, `uv` available

## Phase list (progressive elaboration — only active phase is fully elaborated)

| Phase | Objective | Status |
|-------|-----------|--------|
| phase-00 | Discovery + Harness (package, hardware doctor, param/FLOP accounting, data prep, checkpoint/resume, eval runner, metrics/gate engine, micro-overfit) | not-started |
| phase-01 | Smoke model (~120–180M main, Flash+PLE, 25–50M tokens) | not-started |
| phase-02 | ~400M Architecture PoC (A control / B Flash / C Flash+PLE, 250M→500M tokens) | not-started |
| phase-03 | ~1B Scaling Confirmation (A1 control / W1 winner, 1B→2B tokens) | not-started |
| phase-04 | Final GO/NO-GO decision package + report | not-started |

## Architecture under test

- 3 × GatedDeltaNet → 1 × global/full attention, repeated through decoder (pre-QSA shape)
- Gated Residual / HyperConnection for Flash variants
- Conventional causal global attention for attention layers (no QSA indexer training)
- MoE sparse top-k routing, 16–32 experts, shared expert if supported
- PLE / Engram n-gram capacity (off-GPU-capable)
- MTP OFF, QSA OFF during this epic

## Tokenizer

- One established, ungated tokenizer, ~32K–50K vocab (prefer low end)
- Frozen revision across all A/B/C and scale runs
- Identical BOS/EOS/document-boundary behavior everywhere

## Data

- FineWeb-Edu primary frozen corpus, ~2–3B usable tokens + fixed held-out validation slice
- Deterministic packing, no leakage, memory-mapped/sharded local representation
- All variants consume the exact same token sequence

## Decision gates (summary)

- **PoC GATE 1 (100M)**: catastrophic-failure gate → `NO_GO_ARCHITECTURE_AT_POC`
- **PoC GATE 2 (250M)**: quality/efficiency/balanced win → select Flash winner
- **PLE GATE (C vs B)**: `PLE_PASS` / `PLE_UNPROVEN` / `PLE_FAIL`
- **1B FAIL-FAST (250–500M)**: `NO_GO_SCALING_REVERSAL`
- **1B FINAL**: advantage survived scale → GO
- **Final**: `GO_FINAL_SCALE` / `GO_FLASH_PLE_DEFERRED` / `NO_GO` / `BLOCKED`

## Ralph state

`.nsp/artifacts/tmp/ralph/epic-flashmini-poc/`

## Run

`flashmini-poc-20260907t061908z-9675de3c`