<!-- nsp:meta
id: docs.flashmini-v3
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-v3.md
graphTags: flashmini,v3,training
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-14
replaces:
replacedBy:
-->

# FlashMini v3 experimental contract

V3 starts fresh matched A/B/C runs. It does not rehabilitate old quality claims
or complete the epic from code correctness alone.

| Run | Mixer | PLE |
| --- | --- | --- |
| A3 | Conventional causal full attention throughout | Off |
| B3 | Three GDN layers then one full-attention layer, repeated | Off |
| C3 | Identical to B3 | On |

All use architecture version 3, GPT-2 vocabulary, width 768, ten layers,
four-stream dynamic Gated Residual, the same MoE topology, sequence length 256,
and matched data, seed, batching and optimizer schedule. Name-keyed initialization
gives shared same-shaped parameters identical values despite differing parameter
allocation. Tests enforce shared initialization and B/C config matching.
V2 retains its old semantics and cannot resume as v3.

See [upstream semantics and deviations](flashmini-v3-upstream.md). QSA and MTP
remain off. FlashMini is not a full Qwen reproduction.

## Validity controls

Shared backbone, PLE dense and PLE sparse gradients are clipped independently.
Metrics record each preclip norm, coefficient, clipping event and cumulative
fraction. Sparse gradients remain compatible with SparseAdam; its moments still
require dense CPU memory.

Training validates actual dataset/config sequence length before writing output.
V3 uses seeded epoch permutations without replacement and checkpoints preserve
position, optimizer, RNG and clipping counters. Decisive runs require verified
v3 data, immutable provenance, sufficient frozen scored tokens and no corpus
reuse. The explicit reuse override marks the run non-decisive.

Comparison requires explicit PLE-off baseline and PLE-on candidate, matching
architecture/backbone, manifest, tokenizer, seed, tokens, batching, optimizer,
schedule and source. It verifies actual dataset files. B/C measures treatment;
C-on/off measures within-model reliance. Holdout block uncertainty is not
training-seed variance.

The planned PLE/hash settings and optimizer recipe must match even for the
PLE-off control. Effective auxiliary-loss coefficient, precision and clipping
policy are recorded and enforced on comparison/resume. Dataset EOS must match
the configured document boundary. The comparator automatically excludes the
recorded periodic-validation prefix and rejects an explicit overlapping slice.
Tuning performed outside the harness must reserve additional holdout explicitly.

Use the [frozen-data reference](flashmini-v3-data.md) and
[run/resume commands](flashmini-resume.md). Large generated artifacts remain
untracked. Only the newest validated checkpoint per run is retained.

## Historical evidence

V1 had future-token leakage and incorrect context hashing; its quality conclusions
are withdrawn. V2 corrected causality but used repeated ~29.4M-token data, lacked
a matched A2, coupled clipping, and omitted reference GR and PLE convolution.
It was seq256 mechanism screening, not fresh-token or long-context evidence.
The user explicitly superseded the original preservation requirement and
authorized permanent deletion of old runtime artifacts and corpora. No old
checkpoints, metrics or collision artifacts are claimed preserved. Historical
source remains in Git.

## Decision sequence

Implementation checks, micro-overfit and short fresh-corpus pilots passed;
see the [validation handoff](flashmini-v3-validation.md). Official 100M/250M
runs have not started. These short checks do not establish an architecture winner.

| Gate | Evidence required |
| --- | --- |
| 0: implementation | Tests, upstream differential and independent review |
| 1: micro-overfit | A/B/C finite gradients and decreasing loss |
| 2: fresh pilot | Stable short updates on the new frozen corpus |
| 3: ~100M | Matched fresh-corpus A/B/C screening |
| 4: 250M | Quality, stability, routing, clipping, reliance and resource costs |
| 5: ~1B scaling | Matched larger models and stronger evaluation |
| Final | Genuine long-context quality; multiple matched seeds for small effects |

Seed 17 is acceptable for screening. At least three full matched seeds are needed
when seed variance could reverse a small effect. `final_go_readiness` blocks
missing conventional-control, data, scaling and long-context evidence. The
1024-token causal probe is structural only, not 192K/256K capability evidence.
