# Real dense-to-MoE continuation plan

This plan records the continuation of `20260815-030931-windows` without
mutating that historical run.  The source is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its text backbone is Qwen 3.5
(`qwen3_5_text`, 64 layers, hidden size 5120, dense intermediate size 17408).
The first production profile is `qwen38_p8s1_top2`.

## Phases and falsifiable gates

1. **Control plane** — hypothesis: a run can resume from a non-terminal
   `BLOCKED` state and weaker environment observations cannot erase pinned
   facts.  Falsifier: a regression test shows an early return or fact loss.
2. **Target architecture** — hypothesis: a versioned SwiGLU MoE module can
   strictly save/reload and reconstruct a dense FFN when all experts are used.
   Falsifier: missing/unexpected keys, non-finite tensors, or reconstruction
   MSE above `1e-8` on the tiny integration fixture.
3. **Calibration and capture** — hypothesis: fixed, non-overlapping local
   token splits can be represented by hashed binary shards and resumed at
   shard granularity.  Falsifier: hash/overlap mismatch or a resumed capture
   rewriting a validated shard.
4. **Oracle ceiling** — hypothesis: contribution-aware p8/top-2 routing is
   sufficiently expressive before router optimization.  Falsifier: oracle
   normalized MSE is above `0.10` on representative real layer samples.
5. **One-layer vertical slice** — hypothesis: router warm-up plus bounded
   joint distillation improves over the oracle baseline on a fixed holdout.
   Falsifier: no finite improvement or dead/collapsed experts.
6. **Assembly/evaluation** — hypothesis: validated layer safetensors can be
   assembled into a strict, reloadable target checkpoint.  Falsifier: any
   inventory, fingerprint, shape, or reload gate fails.

Full 64-layer training is explicitly gated on the representative-layer and
quality evidence above.  GGUF is a separate, downstream track and cannot
   turn a structural smoke file into a model-quality claim.

## Resource strategy

The canonical source snapshot is used read-only.  Derived artifacts are
streamed, sharded, and hashed; source weights and activation corpora are never
committed.  CPU execution is the portable fallback.  If PyTorch/Transformers
support for the exact Qwen 3.5 hybrid block is unavailable, the target spike
remains a self-contained text FFN/MoE contract and the HF milestone is marked
research-candidate rather than pretending multimodal support.

## Corpus strategy

`prepare-data` requires a manifest (local JSONL/JSON/TSV or a declared public
dataset) and writes deterministic train/holdout IDs, tokenizer identity,
sequence length, token counts, and SHA-256 hashes.  The default pilot uses at
least 131,072 train tokens and 16,384 holdout tokens when the supplied corpus
can satisfy those limits; no private data is inferred.

## Fallback matrix

If a gate fails, run only one bounded change at a time: normalized top-k
scaling, contribution/activation partitioning, router initialization,
additional calibration data, longer warm-up, joint loss, wider shared expert,
top-3 diagnostic, progressive correction, low-rank router/expert correction,
modestly expanded capacity, then dense upcycling.  Thresholds are immutable
unless a decision-register entry records the old/new values and rerun scope.
