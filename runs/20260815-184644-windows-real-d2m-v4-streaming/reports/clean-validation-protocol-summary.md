# Clean validation protocol summary

The architecture-dev rows are now a true validation split. FIT uses 115,124
of the 131,508 train rows and excludes the deterministic 16,384-row validation
set (identity hash `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`).
The 16,598-row holdout remained closed for every run below.

| candidate | active width / reduction | dispatches | validation NMSE | validation cosine | dead | load CV | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| p16/top4 independent-positive, repeated CE | 5120 / 70.59% | 4 | 0.04251 | 0.97297 | 0 | 0.2978 | clean baseline; gate fallback |
| p16/top4 independent-positive, unordered BCE | 5120 / 70.59% | 4 | 0.04549 | 0.97276 | 0 | 0.2447 | balance improved, quality worse |
| p32/top5 independent-positive, repeated CE | 3584 / 79.41% | 5 | 0.04485 | 0.96729 | 0 | 0.5366 | misses cosine and load gates |

Checkpoint selection is gate-aware at every epoch: feasible means NMSE ≤ 0.05,
cosine ≥ 0.98, dead experts = 0, and load CV ≤ 0.50; feasible points maximize
cosine, then minimize NMSE and load CV. When no point is feasible, the full
Pareto trajectory is retained and the highest-cosine fallback is selected.

Validation diagnostics for the p16/top4 baseline show mean selector recall
against the residual-correlation positive oracle of 0.719 (exact set match
23.4%). Conditional reconstruction is cosine 0.9793 with oracle IDs and
oracle amplitudes versus 0.9730 with student IDs and amplitudes. The hardest
residual quartile reaches only oracle cosine 0.9605, so the remaining blocker
is routing/conditional reconstruction on difficult tokens, not dead-expert
collapse. See `p16-top4-clean-validation-diagnostics.json` for the complete
diagnostic receipt.

The p8 k=1…6 positive oracle curve remains the capacity control: validation
cosine rises from 0.8906 (k=1) to 0.9890 (k=6), with k=4 at 0.9720 and k=5 at
0.9816. No representative-layer or full replay was resumed.

Recommendation: keep p16/top4 as the primary ≥70%-reduction research
candidate, do not promote p32/top5, and improve selector/amplitude learning
before any 64-layer commitment. Holdout confirmation remains gated on a frozen
finalist.
