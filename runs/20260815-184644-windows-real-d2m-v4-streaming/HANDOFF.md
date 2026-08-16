# Handoff for `20260815-184644-windows-real-d2m-v4-streaming`

- Current status: `HIGH_SPARSITY_FINALIST_HOLDOUT_COMPLETE_REPLAY_PAUSED`
- Current phase: `high-sparsity-finalist-holdout-confirmation`
- Last completed gate: `HIGH_SPARSITY_FINALIST_HOLDOUT_CONFIRMATION_COMPLETE`
- Active blocker: no >=70% reduction candidate meets the green gate (`NMSE <= 0.05`, `cosine >= 0.98`, `dead_experts = 0`, `load_cv <= 0.50`).
- Exact next command: investigate sparse basis/angular error and routing load; do not start representative or 64-layer replay.
- Code commit: `7e375c4c9d92646ab9b15d65b12b9ffc8373b225`
- Safe replay checkpoint: train rolling replay is durable through layer 29 (stage-0030 manifest); layer 30 was intentionally stopped after steering.

## Completed high-sparsity evidence

The deterministic 16,384-row TRAIN/dev subset is identified by hash
`5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
Partition/oracle selection used TRAIN/dev only. The full holdout was read once,
after finalist training, by `reports/high-sparsity-finalist-holdout-confirmation.json`.

| candidate | active width | reduction | holdout NMSE | holdout cosine | dead | load CV | role |
|---|---:|---:|---:|---:|---:|---:|---|
| p16/top4 independent-positive | 5120 | 70.59% | 0.04427 | 0.96092 | 0 | 0.501 | primary |
| p16/top3 independent-positive | 4096 | 76.47% | 0.04729 | 0.95692 | 0 | 0.761 | aggressive |
| p32/top6 independent-positive | 4096 | 76.47% | 0.04697 | 0.95771 | 0 | 0.694 | aggressive |

The p16/top4 schedule-A dev winner was NMSE `0.04550`, cosine `0.96725`,
dead `0`, load-CV `0.456`. The p16/top3 and p32/top6 equal-compute runs
used the same schedule, seed, optimizer budget, and dev selector. Trained
students improved NMSE relative to their frozen positive oracles, but angular
fidelity remains below the `0.98` gate. The required basis investigation is
therefore still open; no deeper replay is authorized.

`qwen38_p8s1_top6` remains a `TRAINABILITY_ROUTER_QUALITY_CONTROL` only
(holdout NMSE `0.01890`, cosine `0.98338`). The prior p8s14/top2 artifact is
preserved as `DENSEISH_QUALITY_UPPER_BOUND`, not a production candidate.

Do not discard captured activations, checkpoints, or reports. Do not resume
teacher capture or the 64-layer replay until a >=70% candidate clears the
green gate after basis/routing work.
