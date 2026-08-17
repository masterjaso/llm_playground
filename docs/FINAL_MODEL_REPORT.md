<!-- nsp:meta
id: docs.final.model.report
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/FINAL_MODEL_REPORT.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Continuation report: dense-to-MoE recovery

Run: `20260815-145505-windows-real-d2m`
Parent evidence: `20260815-030931-windows`
Profile under study: `qwen38_p8s1_top2`

## Verified facts

- Source identity is `Qwen/Qwen3.8-27B`, resolved revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- The text configuration is Qwen 3.5 (`qwen3_5_text`), 64 layers, hidden size
  5120, dense intermediate size 17,408, and mixed linear/full attention.
- The canonical run remains a separate parent run; this continuation does not
  overwrite its artifacts.
- Control-plane tests cover resumable `BLOCKED`, monotonic fact merging,
  binary activation-shard resume, strict layer schema, and placeholder
  assembly rejection.
- The local SwiGLU MoE spike strictly saves/reloads and reconstructs the dense
  FFN in all-expert mode.

## Measured results

- Real source layer-0 all-expert reconstruction MSE: approximately
  `5e-13`.
- Initial untrained p8/top-2 relative MSE: `0.6102` on the structural pilot.
- Exact p8/top-2 oracle normalized MSE: `0.6007` for the contiguous partition
  in the bounded ablation.
- Best bounded fallback (contribution-signature partition) oracle normalized
  MSE: `0.2758`, cosine `0.9839`; this is still red against the fixed `0.10`
  yellow ceiling.
- Native Transformers 5.15 contains Qwen 3.5 mixed-attention dense classes,
  but its config exposes no routed/shared MoE expert plan; the local target
  module is therefore the explicit fallback.

## Inferences

The p8 capacity-preserving decomposition is the current limiting factor on
layer 0.  Router optimization cannot recover the residual that remains after
the exact oracle ceiling.  Contribution-aware grouping is materially better
than contiguous grouping, but not sufficient for the fixed quality gate.

## Unresolved risks and blockers

- No approved, representative calibration corpus was supplied for a real
  teacher activation capture.  The capture command refuses to create an empty
  or JSON activation artifact.
- A full 64-layer Hugging Face checkpoint has not been assembled; no quality,
  perplexity, benchmark, throughput, or GGUF success claim is made.
- The next scientific experiment is a bounded expanded-capacity/dense-upcycling
  study (or an explicitly approved change to the p8 target), followed by a
  real train/holdout corpus and teacher-relative evaluation.

## Recommended next command

```text
d2m prepare-data --run-dir runs/20260815-145505-windows-real-d2m --corpus-manifest <approved-jsonl>
```
