<!-- nsp:meta
id: docs.real.d2m.plan
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/REAL_D2M_PLAN.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Real dense-to-MoE continuation plan

This plan records the continuation of `20260815-030931-windows` without
mutating that historical run.  The source is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its text backbone is Qwen 3.5
(`qwen3_5_text`, 64 layers, hidden size 5120, dense intermediate size 17408).
The current product profiles are `qwen38_p16s1_top4` (safe fallback) and
`qwen38_p32s1_top5` (preferred product). `p32/top4` is inactive. The
canonical execution graph is phase 00A runtime lock, 00B method-proof data,
01 p16 method proof, 02 production Corpus V2.2/p16, 03 selector/generalization,
04 p32 transfer, 05 method lock, 06 representative layers, 07 full64 p16,
08 BF16/whole-model validation, and 09 quantization.

The bounded method-proof data gate is independent of production Corpus V2.2:

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_method_proof_data.py `
  --corpus-manifest data\public_v21\corpus-v2.1.jsonl `
  --output <phase-00b-run-dir>\method-proof `
  --min-tokens 32768 --json

& .\.venv\Scripts\python.exe scripts\capture_real_qwen_layer0.py `
  --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json `
  --source-snapshot <pinned-qwen-source> `
  --run-dir <phase-01-run-dir> `
  --runtime-lock runs\windows-runtime-lock.json `
  --shard-tokens 2048 --resume --json

& .\.venv\Scripts\python.exe scripts\run_real_oracle_routed_basis_refinement.py `
  --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json `
  --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json `
  --runtime-lock runs\windows-runtime-lock.json `
  --source-snapshot <pinned-qwen-source> `
  --topology p16/top4 --max-tokens 2048 --epochs 1 --device cuda:0 `
  --batch-rows 256 --learning-rate 0.0001 `
  --result-receipt <phase-01-run-dir>\metrics\p16-2k.json `
  --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-2k --json

# Stage 2: repeat on 4k captured tokens (rows are bounded batch windows).
& .\.venv\Scripts\python.exe scripts\run_real_oracle_routed_basis_refinement.py `
  --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json `
  --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json `
  --runtime-lock runs\windows-runtime-lock.json `
  --source-snapshot <pinned-qwen-source> `
  --topology p16/top4 --max-tokens 4096 --epochs 1 --device cuda:0 `
  --batch-rows 512 --learning-rate 0.0001 `
  --result-receipt <phase-01-run-dir>\metrics\p16-4k.json `
  --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-4k --json

# Stage 3: bounded scientific method proof on at least 32k captured tokens.
& .\.venv\Scripts\python.exe scripts\run_real_oracle_routed_basis_refinement.py `
  --method-proof-receipt <phase-00b-run-dir>\method-proof\receipt.json `
  --capture-receipt <phase-01-run-dir>\capture\real-qwen-layer0-receipt.json `
  --runtime-lock runs\windows-runtime-lock.json `
  --source-snapshot <pinned-qwen-source> `
  --topology p16/top4 --max-tokens 32768 --epochs 1 --device cuda:0 `
  --batch-rows 2048 --learning-rate 0.0001 `
  --result-receipt <phase-01-run-dir>\metrics\p16-32k.json `
  --checkpoint-dir <phase-01-run-dir>\checkpoints\p16-32k --json

For long native-Windows jobs, wrap each runner command with
`scripts\Invoke-GuardedCommand.ps1` and retain its heartbeat/termination
receipt.  `--max-tokens` is a captured-token limit; `--batch-rows` is only the
in-memory window size.  A result records a method-proof decision of
`GREEN`, `YELLOW`, or `FAILED` against the historical product gates, while
`production_promotion_eligible` remains false in all cases.
```

## Synthetic smoke versus real method proof

`scripts/run_oracle_routed_basis_smoke.py` is the only supported entrypoint
for the tiny random SwiGLU fixture.  It reports
`ORACLE_ROUTED_BASIS_SYNTHETIC_SMOKE_GREEN` with
`evidence_class=synthetic-smoke`, and both scientific and production
promotion flags are false.  `--samples 32768` still means random fixture
rows; it never means captured tokens and cannot satisfy Phase 01.

The real method proof is the explicit
`run_real_oracle_routed_basis_refinement.py` path.  It requires a validated
`dense2moe-method-proof-data` receipt, a
`dense2moe-real-qwen-layer-capture` receipt, the exact p16/top4 topology, the
approved current Windows runtime lock, and `--max-tokens`.  Its preflight
checks the pinned Qwen source identity, layer-0 geometry, shard hashes,
selected-record identity, benchmark/evaluation exclusion, and runtime before
constructing an optimizer.  Any failed gate reports `BLOCKED` with zero
optimizer steps.

Phase 01 is a method proof only.  A 32k decision reports GREEN/YELLOW/FAILED
against the historical NMSE/cosine/dead-expert/loadCV gates but does not open
the official holdout or promote a production candidate.  Corpus V2.2,
selector optimization, p32 transfer, representative layers, and full64
conversion remain later, separately gated phases.

## Phases and falsifiable gates

1. **Control plane** — hypothesis: a run can resume from a non-terminal
   `BLOCKED` state and weaker environment observations cannot erase pinned
   facts.  Falsifier: a regression test shows an early return or fact loss.
2. **Target architecture** — hypothesis: a versioned SwiGLU MoE module can
   strictly save/reload and reconstruct a dense FFN when all experts are used.
   Falsifier: missing/unexpected keys, non-finite tensors, or reconstruction
   MSE above `1e-8` on the tiny integration fixture.
3. **Calibration and capture** — hypothesis: frozen METHOD_PROOF_ONLY records
   replay through the exact layer-major Qwen teacher into hashed layer-0 X/Y
   shards.  Falsifier: any source, split, target-hook, or shard-hash mismatch.
4. **Oracle ceiling** — hypothesis: exhaustive p16/top-4 oracle routing is
   sufficiently expressive before router optimization on real Qwen samples.
   Falsifier: no repeatable improvement or expert collapse at 2k, 4k, and 32k.
5. **One-layer vertical slice** — hypothesis: selector-independent basis
   refinement is numerically stable on real captured data.  The selector stays
   frozen in this phase; no holdout is opened and no production claim follows.
6. **Assembly/evaluation** — hypothesis: validated layer safetensors can be
   assembled into a strict, reloadable target checkpoint.  Falsifier: any
   inventory, fingerprint, shape, or reload gate fails.

Full 64-layer training is explicitly gated on the representative-layer and
quality evidence above.  GGUF is a separate, downstream track and cannot
   turn a structural smoke file into a model-quality claim.

## Resource strategy

The canonical source snapshot is used read-only.  Derived artifacts are
streamed, sharded, and hashed; source weights and activation corpora are never
committed. The scientific pipeline is native-Windows only: WSL/Linux is a
policy violation and cannot be used as a CPU fallback for D2M claims. If the
approved Windows runtime is unavailable, the phase remains blocked and the
historical environment receipt is advisory only.

## Corpus strategy

`dense2moe.cli prepare-data` requires a manifest (local JSONL/JSON/TSV or a declared public
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
