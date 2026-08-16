# Handoff for `20260815-184644-windows-real-d2m-v4-streaming`

- Current status: `HIGH_SPARSITY_RESEARCH_COMPLETE_REPLAY_BLOCKED`
- Current phase: `selector-generalization-blocked`
- Last completed gate: `p16-top4-refined-course-correction-holdout-confirmation`
- Active blocker: the best >=70% candidate fails the full green gate on holdout (cosine `0.977044` < `0.98`), despite a sufficient exact holdout oracle (`0.982185`).
- Exact next command: develop a more robust selector using FIT/validation only; do not tune on the opened holdout and do not start representative or 64-layer replay.
- Current code commit at handoff: `11098c0f02296cce5825a8433d9caf600737afdc`
- Safe replay checkpoint: train rolling replay remains durable through layer 29 (stage-0030 manifest); layer 30 was intentionally stopped.

## Fixed protocol

- Source: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- FFN-only conversion; attention, RoPE, norms, residual topology, embeddings,
  LM head, tokenizer, special tokens, and chat/reasoning behavior remain out
  of scope for modification.
- FIT rows: `115124`.
- Validation rows: `16384`, identity hash
  `5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c`.
- Holdout rows: `16598`. It was opened only for post-selection finalist
  confirmations and a non-gradient exact capacity diagnostic; it was never
  used for optimizer updates.
- Green gate: normalized MSE `<=0.05`, cosine `>=0.98`, dead experts `=0`,
  load CV `<=0.50`.

## Best >=70% candidate

`qwen38_p16s1_top4`: 16 experts, width 1024, shared width 1024, top-4,
active width `5120`, reduction `70.5882%`, independent-positive routing.

Checkpoint:
`layer-checkpoints/clean-validation/p16-top4-refined-course-correction-continue`

| split | NMSE | cosine | dead | load CV |
|---|---:|---:|---:|---:|
| validation | 0.022887 | 0.981599 | 0 | 0.4421 |
| holdout | 0.027239 | 0.977044 | 0 | 0.4713 |

The validation checkpoint was strictly reloaded and was selected with the
gate-aware rule. The finalist holdout confirmations are recorded in
`reports/p16-top4-refined-course-correction-holdout-confirmation.json`.

## Capacity and selector evidence

The exact all-1,820-set positive oracle on the finalist basis gives holdout
cosine `0.982185` (bounded residual oracle `0.982124`), so p16/top4 capacity
is not the blocker. Learned-vs-exact selector top-k recall is `0.834694` on
holdout (`0.863358` on validation).

The following bounded experiments are recorded and rejected or retained as
diagnostics: shared-neuron promotion, shared scalar fitting, low-rank SiLU
routers (hidden 128 and 512), exact-set margin/listwise objectives, and a
differentiable soft top-k surrogate. None produced a superior holdout-safe
green candidate. See the consolidated receipt:
`reports/p16-top4-research-synthesis.json`.

## Replay decision

Representative-layer replay and full 64-layer replay remain **blocked**. The
p8/top6 result remains a trainability control only; its active width is
`13312` (23.53% reduction), so it cannot redefine the product objective.
Do not discard captures, checkpoints, or reports. A future selector experiment
must keep validation excluded from gradient updates and must not use the
opened holdout for tuning.

## Follow-up evidence (2026-08-16)

The prior handoff commit line above predates the guarded-command and oracle
hardening work. The current source commits are `408675e` (native PowerShell
guard supervision), `53fdc5a` (mmap contribution-store inputs and bounded
input windows), `09d6acd` (exact selected-route active-face refit plus FIT-only
calibration), and `edc824e` (reproducible FIT contribution-store writer).

- Native Windows PowerShell 5.1 guarded-command smoke is complete and recorded
  in `docs/windows-guarded-command-smoke-receipt.json`; success, failure,
  timeout-descendant cleanup, heartbeat, and git-integrity cases all passed.
- The current-head FIT search (`reports/high-sparsity-partition-search.json`)
  selected p16/top4 residual-swap refinement at normalized MSE `0.04236924`,
  cosine `0.94904102`. The same search gives p32/top5 `0.05537351` / `0.93400606`
  (load CV `1.4957`) and p32/top4 `0.06408007` / `0.92364422` (load CV
  `1.6382`); neither high-sparsity product target is green.
- Capacity-exact FIT-only p32 plans were materialized as
  `partitions/current-head-p32-top5-product.json` and
  `partitions/current-head-p32-top4-product.json`. They are explicitly marked
  `PARTITION_READY_P32_PRODUCT_TARGET_FIT_ONLY`; no replay was started.
- `reports/streaming-solver-calibration-store-p16-1024.json` is a deterministic
  1,024-row stratified train calibration. The streaming candidate sets match
  the exhaustive 1,820-set reference exactly. After the selected-route exact
  refit, cosine delta is `-2.96e-08` and global-NMSE delta is `-1.34e-10`; the
  candidate score itself is also within `2.35e-07` cosine and `4.32e-11` NMSE.
- The real mmap CLI receipt is
  `reports/load-aware-real-store-p16-fit-1024.json`. It opened only the
  generated FIT store (holdout opened: false), used `npy_memmap` arrays and a
  726-token input batch under a 256 MiB cap, and spilled candidate errors to a
  memmap. Its p16/top4 result is cosine `0.94680261`, NMSE `0.04575238`, dead
  experts `0`, load CV `1.25233`; the target load CV `0.50` and green gate are
  therefore false on this bounded FIT diagnostic.

These follow-up artifacts validate the bounded solver and input path; they do
not change the selector-generalization blocker or authorize representative or
64-layer replay. The 1,024-row store is a FIT calibration, not a full
validation/holdout result.

## Takeover decision boundary (2026-08-16)

The authoritative continuation receipt is
`reports/takeover-decision-20260816.json`. It records every hypothesis,
falsifier, budget, split identity, partition hash, and decision from this
takeover. The source code at the boundary is `b190014c1476ac30775651d0fe7bde317f2098b5`
and the pinned source revision remains
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The p16/top4 residual-BCE selector was green on FIT/validation-A+B:
NMSE `0.02022360`, cosine `0.98615631`, load CV `0.47347065`, dead `0`;
validation-B was NMSE `0.01763082`, cosine `0.99060470`, load CV
`0.47750669`, dead `0`. Its authorized post-selection holdout confirmation
was not green: student NMSE `0.02712321`, cosine `0.97719723`, load CV
`0.50026573`, dead `0`. The exact holdout oracle reached cosine `0.98218541`
but load CV `0.66740332`, confirming capacity while leaving selector/load
generalization as the blocker. FIT-only load-balance sweeps at `0.35` and
`0.50` selected the identical tensor SHA-256
`6428f0d405eb67abf07913d1216675e55e704800d9417590444be011f7cd5c1f` and
therefore did not change the decision.

The topology-specific p32 finalists remain research-only. p32/top5 validation
was NMSE `0.04484424`, cosine `0.96728188`, load CV `0.53659621`; p32/top4
was NMSE `0.04630313`, cosine `0.96485071`, load CV `0.67825704` (dead `0`
for both). The bounded router-only load-aware refinements were rejected: top5
ended at load CV `0.94037224` and top4 at `1.10319876`. Pool-16 bounded
load-aware oracle screens also stayed below cosine `0.98` and above load CV
`0.50`; these are bounded screens, not impossibility proofs.

Decision: `SELECTOR_GENERALIZATION_BLOCKED_REPLAY_PAUSED`. Do not resume
representative or 64-layer replay, and do not tune the opened holdout. If a
future candidate clears the gate, the representative set must be exactly
layers `0/1/2/3`, `28/29/30/31`, and `60/61/62/63`, labeled respectively
`LINEAR_A/B/C/FULL_ATTENTION`; only SwiGLU FFNs may be replaced. No reasoning
effort optimization or reasoning post-training was performed.
