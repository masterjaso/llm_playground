# Exact p16/top4 validation oracle

- Scope: `full_holdout` (16,598 tokens); validation identity `46b278a85b4cb31a8fe659be2dd146bd9206eb491268220e25440981d06c0a03`.
- Code commit: `1f3915dbeaa9de5bf234d730e5adcc3349c085b9`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.027239 | 0.055476 | 0.977044 | 0.4713 | 0 |
| bounded_oracle | 0.015519 | 0.034847 | 0.982124 | 0.6924 | 0 |
| exact_oracle | 0.015445 | 0.034729 | 0.982185 | 0.6674 | 0 |

Decision: **capacity_sufficient_focus_router**.
