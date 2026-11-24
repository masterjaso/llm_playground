<!-- nsp:meta
id: docs.flashmini-qsa-decision
kind: document
scope: technical
persona: platform-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-qsa-decision.md
graphTags: flashmini,qsa,decision
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-11-24
replaces:
replacedBy:
-->

# QSA exploration decision (rejected for now)

**Status:** rejected for the current FlashMini track (2026-11-24).  
**Does not change:** v3 PoC contract. QSA and MTP remain off in A/B/C/D.

## What we explored

Qwen Sparse Attention (QSA) selects a budget of past 4-token blocks with a
learned indexer, then runs ordinary attention only over those tokens. It is
attractive for long-context inference on consumer GPUs. Upstream references
appear in Qwen3.8-Flash-Next / Qwen4-Exp writeups and a draft Transformers
implementation (not a released FlashMini dependency).

We considered a new treatment **E** = treatment C plus QSA on the 1-in-4
attention layers, compared against A/B/C at matched batch size.

## Why it was rejected for now

1. **PoC sequence length cannot prove QSA.** Official v3 runs use `max_seq_len`
   256. A Qwen-scale 2,048-token budget then selects every block, so E collapses
   to dense attention plus indexer cost.
2. **Indexer training is non-trivial.** The draft Qwen4-Exp modeling path uses
   hard top-k selection into an attention mask. That path does not expose a
   clear indexer training loss. Bolting QSA on without a verified indexer
   objective would leave the indexer near random init and bias the comparison
   against QSA for the wrong reason.
3. **Fair comparison needs matched reruns.** Changing batch size for E alone
   changes optimizer-step count and loss curves. Proving E requires at least
   matched C and E (and preferably A/B) at the same batch and sequence policy.
4. **Cost vs evidence.** Spot G4 smoke (2026-11-24) showed treatment C at about
   21k–31k tok/s depending on batch. A meaningful QSA screen still needs
   implementation, a reduced-budget long-context or synthetic selection test,
   and matched 250M-token runs. That is more engineering and GPU spend than the
   current epic's fail-fast 1B GO/NO-GO path justifies.

## What remains true

- GDN + full attention (pre-QSA shape) stays the architecture under test.
- Inference-oriented sparse attention can be revisited later with: (a) a
  verified indexer training recipe, (b) sequence lengths where the token budget
  is a real sparsity ratio, and (c) an explicit spend approval.
- Treatment **D** remains 4-bit KVC on top of C, not QSA.

## Evidence touched

- Epic contract: `.docs/roadmap/epic-flashmini-poc.md` (QSA OFF).
- Spot smoke artifacts (local): `.artifacts/flashmini-spot/` on the builder
  workstation (not committed).
- Upstream draft modeling review: Qwen4-Exp QSA indexer / top-k mask path
  (Transformers PR branch; not vendored here).
