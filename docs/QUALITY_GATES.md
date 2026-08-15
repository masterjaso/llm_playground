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
