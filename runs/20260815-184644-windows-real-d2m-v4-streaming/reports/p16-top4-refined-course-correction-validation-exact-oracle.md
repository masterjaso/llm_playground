# Exact p16/top4 validation oracle

- Scope: `all_validation` (16,384 tokens); validation identity `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Code commit: `1f3915dbeaa9de5bf234d730e5adcc3349c085b9`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.022887 | 0.048865 | 0.981599 | 0.4421 | 0 |
| bounded_oracle | 0.014087 | 0.030059 | 0.984797 | 0.6217 | 0 |
| exact_oracle | 0.013984 | 0.029907 | 0.984875 | 0.5933 | 0 |

Decision: **capacity_sufficient_focus_router**.
