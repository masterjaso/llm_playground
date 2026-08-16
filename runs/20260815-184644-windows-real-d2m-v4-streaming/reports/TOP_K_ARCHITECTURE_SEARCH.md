# Layer-0 top-k architecture search

Status: complete. Selection used only a deterministic TRAIN architecture-dev subset; the full holdout was read once for the frozen finalists.

- Architecture-dev rows: **16384** / 131508 (seed 20260815)
- Architecture-dev row-key hash: `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`
- Holdout confirmation rows: **16598**
- p8 k=1..6: exact all-combination active-face simplex and positive oracles.
- p16/p32: deterministic norm-ranked selection with exact coefficients on the selected set; not an exhaustive p32 combination claim.

## p8 exact curve (TRAIN/dev)

| k | positive NMSE | learned-scale NMSE | cosine | active width |
|---:|---:|---:|---:|---:|
| 1 | 0.106807 | 0.106807 | 0.8906 | 3072 |
| 2 | 0.056880 | 0.056880 | 0.9379 | 5120 |
| 3 | 0.037279 | 0.037279 | 0.9590 | 7168 |
| 4 | 0.025402 | 0.025402 | 0.9720 | 9216 |
| 5 | 0.016874 | 0.016873 | 0.9816 | 11264 |
| 6 | 0.010126 | 0.010126 | 0.9890 | 13312 |

## Frozen finalists

| profile | k | dev positive NMSE | holdout positive NMSE | active width |
|---|---:|---:|---:|---:|
| p8 | 4 | 0.025402 | 0.022181 | 9216 |
| p8 | 6 | 0.010126 | 0.008752 | 13312 |
| p16 | 4 | 0.051106 | 0.045592 | 5120 |

The p8/top2 trained checkpoint remains the historical baseline; no deeper representative replay was started after the safe layer-29 checkpoint.

## Identical layer-0 finalist budget

All three frozen finalists were trained for one epoch with microbatch 512, learning rate 1e-4, seed 17, and the same streaming train/holdout manifests:

| profile | training status | trained holdout NMSE | cosine | dead experts | load CV |
|---|---|---:|---:|---:|---:|
| p8/k4 | research candidate | 0.082823 | 0.9660 | 0 | 0.217 |
| p8/k6 | research candidate | 0.081370 | 0.9776 | 0 | 0.271 |
| p16/k4 | validation failed | 0.108993 | 0.9397 | 0 | 0.292 |

The frozen oracle ceilings are substantially better than the one-epoch trained values, so these results measure the bounded training budget rather than a final architecture ceiling.
