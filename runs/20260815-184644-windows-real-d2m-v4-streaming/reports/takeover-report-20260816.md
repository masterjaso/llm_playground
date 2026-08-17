# Dense2MoE takeover report (2026-08-16)

Status: `TAKEOVER_COMPLETE_BOUNDED_REFINEMENT_CONTINUING`

The checkpoint-aware contribution path is proven against both active topologies. Both fresh-A diagnoses classify the blocker as `BASIS_QUALITY`; bounded basis-only continuations are published and remain below the green gate.

| Target | Oracle cosine before → after | Oracle NMSE before → after | Oracle CV before → after | Blocker | State |
|---|---:|---:|---:|---|---|
| p16/top4 | 0.921769 → 0.922827 | 0.131389 → 0.129890 | 0.712610 → 0.695971 | `BASIS_QUALITY` | `REFINING` |
| p32/top5 | 0.921811 → 0.922495 | 0.133187 → 0.132286 | 1.140175 → 1.191638 | `BASIS_QUALITY` | `REFINING` |

Infrastructure receipts, split hashes, checkpoint fingerprints, and the complete JSON report are adjacent to this file. `HOLDOUT_OPENED = false`, `REPRESENTATIVE_REPLAY_STARTED = false`, and `FULL64_REPLAY_STARTED = false`.
