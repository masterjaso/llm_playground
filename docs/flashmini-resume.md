<!-- nsp:meta
id: docs.flashmini-resume
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-resume.md
graphTags: flashmini,v3,training
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-14
replaces:
replacedBy:
-->

# FlashMini v3 run and resume reference

Old A/B/C and C2/B2 assets were permanently deleted with explicit user
authorization. They cannot be resumed. Historical configs remain recoverable
from Git but are removed from the active directory to prevent accidental reuse.
Do not migrate v2 checkpoints into v3.

Use the [v3 contract](flashmini-v3.md) and verified frozen corpus. Run A3, B3 and
C3 sequentially: each uses both GPUs. The short validation passed; see the
[validation handoff](flashmini-v3-validation.md) before starting official runs.

## Matched screening recipe

From the repository root, set `variant=a`, then repeat with `b` and `c`. Each
uses the same 250M-token schedule, even when pausing at the 100,663,296-token
gate. The PLE LR multiplier has no effect without PLE.

```bash
variant=a
env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m flashmini.cli train \
  --config "configs/flashmini/poc_${variant}_v3.yaml" \
  --data-dir data/fineweb_v3_2b \
  --run-dir "runs/flashmini/poc_${variant}_v3_seed17" \
  --model-parallel-gpus 1,0 --gpu-memory-gib 15 \
  --tokens 250000000 --stop-after-tokens 100663296 \
  --batch-size 16 --grad-accum 1 --seed 17 \
  --lr 3e-4 --ple-lr-multiplier 5 --warmup-tokens 524288 \
  --cosine-decay --min-lr-ratio 0.1 \
  --eval-every-tokens 2097152 --eval-max-batches 128 \
  --checkpoint-every-tokens 4194304 --log-every 10
```

These are starting hyperparameters, not tuned winners. Batch 32 passed an
unrestricted single-step probe but failed the multi-step preflight under the
15 GiB budget. The matched recipe uses batch 16. Other processes can still
reduce available memory. Gradient accumulation is not implemented.
If batch size must decrease, validate and apply the same change to all three
runs before starting.

## Resume

Repeat the same command and add `--resume` with the one `step_*.pt` file in that
run's `checkpoints/` directory. Do not change seed, source, config, batching,
corpus or schedule. To continue past screening, omit `--stop-after-tokens` but
leave `--tokens 250000000` unchanged. The final update may contain fewer rows;
token positions round up to a complete sequence.
Intermediate pause gates must align to complete optimizer batches.
Weight loading for evaluation permits changing only PLE storage placement.
Exact training resume is stricter: retain the original config hash, including
storage placement, and the recorded execution policy.

Checkpoint writes retain the previous checkpoint until the replacement has
been read back, synced and atomically installed. Older numbered checkpoints are
then deleted. Allow space for two full checkpoints temporarily. Optimizer and
RNG states remain intact for exact continuation. No milestone copies are made.
Evaluate all three at the screening gate before continuing any run: the next
checkpoint retires that gate's weights.

## B3/C3 comparison

```bash
.venv/bin/python -m flashmini.compare_ple \
  --baseline runs/flashmini/poc_b_v3_seed17 \
  --candidate runs/flashmini/poc_c_v3_seed17 \
  --data-dir data/fineweb_v3_2b \
  --skip-sequences 1024 --block-sequences 64 \
  --out runs/flashmini/b3_c3_screening.json
```

The skipped prefix excludes the 128 single-sequence batches used by periodic validation,
with additional separation from that tuning prefix.
Keep further tuning off this holdout. The comparator verifies actual data bytes
and matched checkpoint provenance. C-off is within-model memory reliance, not B.
Evaluate A3 on the same holdout with `flashmini.cli eval --skip-sequences 1024`
and its matching `--config`, `--checkpoint`, `--data-dir` and `--run-dir`.
The PLE comparator correctly rejects A/B backbone mismatches. One seed is
screening only. Small effects need at least
three full matched seeds. Final rental GO remains blocked on scaling and
genuine long-context quality, not merely seq256 or structural execution.
