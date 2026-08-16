# Exact p16/top4 validation oracle

- Scope: `hardest_residual_norm_quartile` (4,096 tokens); validation identity `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Code commit: `a7f7668f885aa39703bfd1e5196267235c3c8ab2`; holdout opened: **no**.
- Exact oracle: all 1,820 four-expert sets with non-negative active-face solves.

| variant | global NMSE | mean token relative MSE | mean cosine | load CV | dead experts |
|---|---:|---:|---:|---:|---:|
| student | 0.038655 | 0.202874 | 0.941689 | 0.4955 | 0 |
| bounded_oracle | 0.016050 | 0.078924 | 0.960521 | 0.5030 | 0 |
| exact_oracle | 0.015875 | 0.078594 | 0.960690 | 0.4770 | 0 |

Decision: **capacity_not_green_move_to_basis_topology**.
