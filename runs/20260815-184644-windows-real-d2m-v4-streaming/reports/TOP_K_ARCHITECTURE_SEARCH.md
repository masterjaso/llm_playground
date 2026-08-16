# Layer-0 top-k architecture search

Status: complete. Selection used only a deterministic TRAIN architecture-dev subset; the full holdout was read once for the frozen finalists.

- Architecture-dev rows: **16384** / 131508 (seed 20260815)
- Architecture-dev row-key hash: `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`
- Holdout confirmation rows: **16598**
- p8 k=1..6: exact all-combination active-face simplex and positive oracles.
- p16/p32: deterministic residual-correlation candidate pools with bounded beam search and exact final positive/simplex coefficient solves; not an exhaustive p32 combination claim.
- Finalist training used `routing_mode: independent_positive`; normalized-softmax remains a baseline mode only.

## p8 exact curve (TRAIN/dev)

| k | positive NMSE | learned-scale NMSE | cosine | active width |
|---:|---:|---:|---:|---:|
| 1 | 0.106807 | 0.106807 | 0.8906 | 3072 |
| 2 | 0.056880 | 0.056880 | 0.9379 | 5120 |
| 3 | 0.037279 | 0.037279 | 0.9590 | 7168 |
| 4 | 0.025402 | 0.025402 | 0.9720 | 9216 |
| 5 | 0.016874 | 0.016873 | 0.9816 | 11264 |
| 6 | 0.010126 | 0.010126 | 0.9890 | 13312 |

Quality/compute elbow: **p8/k4** is the first strong knee at width 9216 (NMSE gain k5 over k4: 0.008528; k6 over k5: 0.006747). k5/k6 remain diagnostic quality endpoints with increasingly dense active width.

## Frozen finalists

| profile | k | dev NMSE | dev cosine | holdout NMSE | holdout cosine | active width | dispatches/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| p8 | 4 | 0.025402 | 0.9720 | 0.022181 | 0.9736 | 9216 | 4 |
| p8 | 6 | 0.010126 | 0.9890 | 0.008752 | 0.9898 | 13312 | 6 |
| p16 | 4 | 0.049836 | 0.9454 | 0.044366 | 0.9477 | 5120 | 4 |

p8/k5 remains a dev-selected reserve point (NMSE 0.016874, active width 11264) and was not trained under the three-finalist cap.

Current provisional recommendation: p8/k6 independent-positive is the trained sparse leader, but representative-layer replay remains paused until its oracle regret is resolved with an equal-budget extension; p8s14 is not eligible.

The p8s14/top2 checkpoint is classified `DENSEISH_QUALITY_UPPER_BOUND` (NMSE 0.002087, cosine 0.997600) because it retains 15,104/17,408 active FFN width (86.8%); it is a control, not the production candidate. No deeper representative replay was started after the safe layer-29 checkpoint.

## Independent-positive finalist training

| profile | routing mode | status | trained holdout NMSE | cosine | oracle holdout NMSE |
|---|---|---|---:|---:|---:|
| p8/k4 | independent_positive | research candidate | 0.041034 | 0.9682 | 0.022181 |
| p8/k6 | independent_positive | trained validated | 0.018899 | 0.9834 | 0.008752 |
| p16/k4 | independent_positive | validation failed | 0.068389 | 0.9458 | 0.044366 |

Independent-positive routing materially closes the normalized-router gap. p8/k6 is the current sparse leader; its remaining gap to the oracle is an optimization-budget question, not a forced coefficient-sum limitation.
