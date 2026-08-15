# Quality gates

Thresholds are fixed for the initial p8/top-2 experiments.

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

The `trainable_student_proxy` interpretation is intentionally separate: a
trained student is constrained by the distillation data and fixed split, the
expert parameterization/shared capacity, router expressivity and load
balancing, and the available optimization, memory, and convergence budget.
Those constraints—not the unchanged frozen slices alone—determine whether a
jointly trained MoE can improve on a frozen-slice diagnostic.
