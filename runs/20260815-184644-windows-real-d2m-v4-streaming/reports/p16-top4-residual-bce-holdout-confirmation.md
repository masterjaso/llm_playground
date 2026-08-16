# Exact p16/top4 validation oracle

- Scope: `full_holdout` (16,598 tokens); validation identity `46b278a85b4cb31a8fe659be2dd146bd9206eb491268220e25440981d06c0a03`.
- Code commit: `e2de5273aefd74e4b9e02278e4348c91d8d1fe32`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.027123 | 0.054474 | 0.977197 | 0.5003 | 0 |
| bounded_oracle | 0.015519 | 0.034847 | 0.982124 | 0.6924 | 0 |
| exact_oracle | 0.015445 | 0.034729 | 0.982185 | 0.6674 | 0 |

Decision: **capacity_sufficient_focus_router**.
