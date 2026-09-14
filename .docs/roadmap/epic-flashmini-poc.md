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
lastReviewed: 2026-09-14
replaces:
replacedBy:
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
- Runtime at v3 validation: 31 GiB visible RAM; approximately 381 GiB free disk
- Python 3.13.14, `uv` available

## Phase list (progressive elaboration — only active phase is fully elaborated)

| Phase | Objective | Status |
|-------|-----------|--------|
| phase-00 | Harness and v3 validity correction | verified: regression suite, independent review, pinned upstream mechanisms and 2B frozen corpus; see validation handoff |
| phase-01 | Smoke and micro-overfit | v3 A/B/C micro-overfit, two-GPU steps, resume and structural context probes passed; not a quality result |
| phase-02 | Matched v3 Architecture PoC (A control / B Flash / C Flash+PLE, 100M→250M tokens) | short 32,768-token A/B/C pilots passed; official runs not started; old outcomes do not satisfy this gate |
| phase-03 | ~1B Scaling Confirmation (A1 control / W1 winner, 1B→2B tokens) | not-started |
| phase-04 | Final GO/NO-GO decision package + report | not-started |

## Architecture under test

- 3 × GatedDeltaNet → 1 × global/full attention, repeated through decoder (pre-QSA shape)
- Four-stream dynamic Gated Residual for every v3 control, including A
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
- **1B FINAL**: advantage survived scale → eligible for final review, not automatic GO
- **Final**: `GO_FINAL_SCALE` / `GO_FLASH_PLE_DEFERRED` / `NO_GO` / `BLOCKED`

The [v3 contract](../../docs/flashmini-v3.md) defines the active experiment.
The [validation handoff](../../docs/flashmini-v3-validation.md) records the completed
implementation checks and their limits.
V1 quality claims are withdrawn. V2 was repeated-corpus screening with known
validity limitations; old artifacts were permanently deleted with explicit user
authorization. Source history remains in Git, not a retained runtime baseline.
V3 requires fresh checkpoints and matched A/B/C controls. Sequence 256 supports
local screening only; a 1024-token causal probe proves execution, not long-context
quality. Final GO remains blocked on genuine long-context evaluation, scaling,
and at least three full matched seeds when a small effect could reverse with seed.

## Ralph state

`.nsp/artifacts/tmp/ralph/epic-flashmini-poc/`

## Run

`flashmini-poc-20260907t061908z-9675de3c`
