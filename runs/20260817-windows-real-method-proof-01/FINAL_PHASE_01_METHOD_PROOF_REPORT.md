# Phase 01 real Qwen layer-0 method-proof report

Date: 2026-08-17
Host: native Windows 11 / PowerShell 5.1
Scientific execution commit: `b8524095fc4d078dbd0a6494a07dceae42422699`
Final repository head: `6ff187966ab92553418d58338234ae3b4cbe2839`

## Provenance gates

- Runtime lock: `runs/windows-runtime-lock.json`
  (`5f376d096b6e44074c1b98318cb552805d9647579ffbb15ef4978b598e06416b`).
- METHOD_PROOF_ONLY receipt:
  `runs/20260817-windows-real-method-proof-00b/method-proof/receipt.json`
  (`e4302c76a5e4e38753364215a7a8bd8548dd01379811be46d0199f3dc94ad27d`).
- Capture receipt:
  `capture/real-qwen-layer0-receipt.json`
  (`c88f08eda79a2d01fe46b3e20c861e1759d67088d7790d802a242b0f6160afa1`).
- Source: Qwen/Qwen3.8-27B, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, model type `qwen3_5_text`;
  layer 0, 64 layers, hidden 5120, dense intermediate 17408.
- Capture: 36,939 FIT-TRAIN tokens, 20 BF16 safetensors shards; both `X`
  (`ffn_input`) and `Y` (`dense_ffn_target`) are `[36939, 5120]`.
- Evaluation/benchmark contamination: rejected by receipt gates; official
  holdout remained closed.

## Stage results

All stages use p16/top4, selector-independent exhaustive-set projected-positive
oracle assignments (`candidate_count=1820`, `exact=false`), and checkpoint
reload/hash validation PASS.

| stage | oracle NMSE before → after | cosine before → after | loadCV before → after | dead experts | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 2k | 0.11582 → 0.04386 | 0.93918 → 0.97532 | 0.82153 → 0.67899 | 0 | YELLOW |
| 4k | 0.11514 → 0.03749 | 0.93826 → 0.97859 | 0.83850 → 0.73463 | 0 | YELLOW |
| 32k | 0.11402 → 0.03920 | 0.93727 → 0.97914 | 0.72917 → 0.65825 | 0 | YELLOW |

Ordinary selector-routed metrics were recorded separately; the selector and
amplitude router remained frozen throughout. The method signal is the oracle,
not the unresolved learned selector.

## Decision and boundary

**Phase 01 method-proof decision: YELLOW.** The real-data improvement repeats,
all metrics are finite, and the oracle has no dead experts. NMSE is within the
historical `<=0.05` gate, but 32k cosine (`0.97914`) is just below `0.98` and
loadCV (`0.65825`) is above `0.50`. This is not a product-green declaration;
`production_promotion_eligible=false`.

The first 32k command with a 2,048-row window was blocked before model/optimizer
steps because the candidate-memory guard required 167,772,160 bytes. The
successful run used 512-row windows and retained the guarded heartbeat.

No production Corpus V2.2 run, selector optimization, p32 transfer, historical
holdout, historical replay, representative-layer run, or full64 conversion was
performed. Candidate training and selection remain blocked pending a separately
accepted follow-up plan.

Final bounded-maintenance closeout: 16/16 direct paths complete with
`maintain_ready=true`; code and context deterministic failure counts are zero.
