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
