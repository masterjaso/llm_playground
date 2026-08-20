<!-- nsp:meta
id: docs.quality.gates
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/QUALITY_GATES.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Quality gates

Thresholds are fixed for the initial p8/top-2 experiments and remain the
layer-level gates for the p16 proof track and p32 product targets.

| Gate | Green | Yellow | Use |
|---|---:|---:|---|
| all-expert reconstruction MSE | <= 1e-8 | — | exact partition contract |
| holdout normalized output MSE | <= 0.05 | <= 0.10 | layer quality |
| holdout cosine similarity | >= 0.98 | >= 0.95 | layer quality |
| router load coefficient of variation | <= 0.50 | — | collapse guard |
| dead routed experts | 0 | — | collapse guard |
| oracle regret | <= 0.10 | — | router ceiling |
| repeated-run metric variation | <= 5% | — | reproducibility |
| end-to-end perplexity increase | <= 5% | <= 10% | model quality |
| mean token KL | <= 0.10 | <= 0.20 | model quality |
| teacher top-1 agreement | >= 85% | — | model quality |

Synthetic fixtures validate software contracts only.  They are labelled in
reports and cannot satisfy the real-model quality gates.

The historical layer-0 Gaussian-input pilot and ablation are classified as
`RANDOM_INPUT_FROZEN_SLICE_DIAGNOSTIC`.  They preserve useful regression
evidence, but are not eligible for these gates, are not teacher-activation
measurements, and must not be interpreted as a hard ceiling on a trainable
student.  Real activation reports must identify the frozen simplex,
non-negative, and train-fit global-scale methods separately.

## Quality/load oracle gate

An unconstrained per-token oracle is not sufficient evidence for a production
router: it can obtain excellent reconstruction by dispatching nearly every
token to the same experts.  Before selector training, each p16/top4,
p32/top5, and p32/top4 basis should report:

- global NMSE and mean cosine;
- hardest residual-quartile cosine;
- expert usage, dead experts, and load CV;
- a priced load-aware assignment Pareto curve; and
- whether any assignment reaches cosine `>= 0.98`, global NMSE `<= 0.05`, and
  load CV `<= 0.50` simultaneously.

`dense2moe.partition.frozen_slice_load_aware_oracle` performs this bounded
Lagrangian diagnostic.  p16/top4 uses exhaustive candidate sets when its
`C(16,4)=1820` combinations fit the configured bound.  p32/top4 and p32/top5
use a deterministic correlation-ranked candidate pool and must be labelled as
bounded rather than exhaustive.  Candidate scores are streamed through
float32 Gram/correlation blocks; score matrices and p32 candidate IDs may be
memory-mapped, and the receipt records the candidate count, storage mode,
coefficient solver, and assurance level.  Feasible points are ordered by
cosine first (then NMSE and load CV), while the reported Pareto frontier keeps
all three dimensions: maximize cosine, minimize global NMSE, and minimize load
CV.  A bounded p32 result below the cosine target is screening evidence, not a
proof of impossibility.

The `trainable_student_proxy` interpretation is intentionally separate: a
trained student is constrained by the distillation data and fixed split, the
expert parameterization/shared capacity, router expressivity and load
balancing, and the available optimization, memory, and convergence budget.
Those constraints—not the unchanged frozen slices alone—determine whether a
jointly trained MoE can improve on a frozen-slice diagnostic.
