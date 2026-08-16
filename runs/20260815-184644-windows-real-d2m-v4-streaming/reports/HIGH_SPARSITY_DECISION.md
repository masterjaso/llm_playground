# High-sparsity layer-0 decision

Status: finalist holdout confirmation complete; representative/full replay remains paused.

Selection used the deterministic 16,384-row TRAIN/dev subset (`5a7739c...d8b8e1c`).
Only the frozen finalist set was evaluated on the full holdout.

| candidate | reduction | dev training NMSE / cosine | holdout NMSE / cosine | dead | holdout load CV |
|---|---:|---:|---:|---:|---:|
| p16/top4 independent-positive (schedule A) | 70.59% | 0.04550 / 0.96725 | 0.04427 / 0.96092 | 0 | 0.50080 |
| p16/top3 independent-positive | 76.47% | 0.04403 / 0.96358 | 0.04729 / 0.95692 | 0 | 0.76087 |
| p32/top6 independent-positive | 76.47% | 0.04509 / 0.96474 | 0.04697 / 0.95771 | 0 | 0.69416 |

No candidate clears the required `cosine >= 0.98` gate. p16/top4 is closest on
load balance but is just over the `load_cv <= 0.50` threshold on holdout;
p16/top3 and p32/top6 are more imbalanced. Trained students improve NMSE over
the frozen positive-oracle partitions, yet the remaining angular error means
the basis/router problem is not solved.

The p8/top6 independent-positive result remains a trainability/router-quality
control, not a production choice. The p8s14/top2 result remains a dense-ish
quality upper bound. Do not resume the durable layer-29 replay checkpoint or
start 64-layer execution until basis/routing work produces a green >=70%
candidate.

Evidence:

- `high-sparsity-partition-search.json`
- `p16-top4-schedule-ablations.json`
- `high-sparsity-equal-compute-finalists.json`
- `high-sparsity-finalist-holdout-confirmation.json`
