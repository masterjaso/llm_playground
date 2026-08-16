# Handoff for `20260815-184644-windows-real-d2m-v4-streaming`

- Current status: `independent-positive-finalist-training-complete`
- Current phase: `finalist-selection`
- Last completed gate: `INDEPENDENT_POSITIVE_FINALIST_TRAINING_COMPLETE`
- Active blocker: `none`
- Exact next command: `review runs/20260815-184644-windows-real-d2m-v4-streaming/reports/TOP_K_ARCHITECTURE_SEARCH.md and reports/architecture-finalist-training.json`
- Expected output: retain or extend p8/k6 independent-positive training; do not resume representative-layer replay yet
- Code commit: `964dcc6d3c90f3af7b60c11ddb9e7f5827320ee5`
- Relevant log: `runs\20260815-184644-windows-real-d2m-v4-streaming\events.jsonl`
- Safe replay checkpoint: train rolling replay is durable through layer 29 (stage-0030 manifest); layer 30 was intentionally stopped after steering.
- Architecture-dev subset: 16,384 deterministic TRAIN rows, hash `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Full holdout was reserved for and read once for the frozen finalists p8/k4, p8/k6, and p16/k4.
- Finalists were trained with `routing_mode: independent_positive`; p8/k6 is `TRAINED_VALIDATED`, p8/k4 is a research candidate, and p16/k4 failed the current gate.
- Do not restart teacher capture, discard activation artifacts, or resume deep representative replay before reviewing the finalist training report.
