# Exact p16/top4 validation oracle

- Scope: `all_validation` (16,384 tokens); validation identity `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Code commit: `a7f7668f885aa39703bfd1e5196267235c3c8ab2`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.042510 | 0.104641 | 0.972971 | 0.2978 | 0 |
| bounded_oracle | 0.018989 | 0.042435 | 0.979286 | 0.6705 | 0 |
| exact_oracle | 0.018836 | 0.042268 | 0.979361 | 0.6381 | 0 |

Decision: **capacity_not_green_move_to_basis_topology**.
