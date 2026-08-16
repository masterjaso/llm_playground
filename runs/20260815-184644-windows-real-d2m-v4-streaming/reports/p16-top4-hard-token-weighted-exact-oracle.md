# Exact p16/top4 validation oracle

- Scope: `all_validation` (16,384 tokens); validation identity `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Code commit: `1e4617398c7472b631a236fea2e83364891ae06f`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.035822 | 0.082247 | 0.974862 | 0.2987 | 0 |
| bounded_oracle | 0.017091 | 0.037682 | 0.981166 | 0.6924 | 0 |
| exact_oracle | 0.016960 | 0.037553 | 0.981230 | 0.6653 | 0 |

Decision: **capacity_sufficient_focus_router**.
