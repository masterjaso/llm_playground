<!-- nsp:meta
id: docs.flashmini.ple
kind: document
scope: features
persona: platform-engineering
status: active
source: agent
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-ple.md
graphTags: flashmini,ple,training
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-09-14
replaces: 
replacedBy: 
-->

# FlashMini conditional memory

FlashMini v2 adds trainable bigram/trigram lookup memory to a small causal hybrid
language model. The table can stay in CPU RAM while the backbone and retrieved
values run on the GPU. This increases stored capacity with a small accelerator
working set; it does not guarantee an accuracy gain.

The historical PoC B/C models contained future-token leakage and a hash that
discarded prior tokens. Their quality conclusions are withdrawn. Version 2
requires fresh training; checkpoint loading rejects legacy versions and changes
to model semantics. Only changing PLE storage between CPU and GPU is permitted
when loading otherwise identical configurations.

## Memory behavior

Each hash head owns a distinct prime-sized table. Heads alternate between
bigrams and trigrams, with separate polynomial bases and a sentinel distinct
from token zero. EOS resets the lookup context. A context-conditioned gate
controls the memory residual before the second block, after the first block
has mixed the hidden state. The residual scale starts small and remains bounded.

`num_heads * head_dim` is the retrieved width per token. `table_size` is the
approximate row count **per head**. Increasing row counts expands total storage
without increasing the per-token retrieved width. Collisions still occur within
individual tables; multiple heads reduce identical combined signatures.

`offload: cpu` keeps the table in CPU RAM even when the model is moved to CUDA.
Only unique requested rows are copied to the GPU. Sparse gradients and
SparseAdam update selected rows, with zero table weight decay. Its two moment
arrays are dense and remain on CPU: budget roughly three copies of the table
for training, plus temporary buffers. Training savings include these moments;
inference savings include only the table weights.

`offload: gpu` uses the same lookup and optimizer semantics with GPU storage.
CPU/GPU lookup, gradient, optimizer update and checkpoint tests check parity.
GPU caching, overlapped prefetch and NVMe are not implemented. Unsupported
offload/cache settings fail explicitly. This module draws on conditional-memory
ideas; it is not a full implementation of Qwen3.8-Flash-Next or Engram.

## Run a matched pilot

Prepare a new frozen corpus once. The document limit must be large enough to
reach the desired token budget. The manifest reports actual deduplicated token
counts and hashes. Complete documents are deduplicated and separated before
packing; padding labels are ignored.

```bash
.venv/bin/python -m flashmini.prepare_fineweb \
  --out-dir data/fineweb_v2_30m --max-tokens 30000000 --max-docs 50000 \
  --seq-len 256 --val-fraction 0.02
```

Use separate new run directories and matching seeds, batches, backbone LR and
token budgets for `ple_v2_b_pilot.yaml` and `ple_v2_c_pilot.yaml`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv/bin/python -m flashmini.cli train \
  --config configs/flashmini/ple_v2_c_pilot.yaml --data-dir data/fineweb_v2_30m \
  --run-dir runs/flashmini/my_c2_run --tokens 2097152 --batch-size 4 \
  --lr 3e-4 --ple-lr-multiplier 5 --seed 17 --warmup-tokens 65536 \
  --eval-every-tokens 524288 --eval-max-batches 128 \
  --checkpoint-every-tokens 524288
```

The 5x table LR is an experimental setting, not a universal default. The
backbone optimizer remains AdamW. Table LR and weight decay are independent.
The training loop records LR, finite losses/gradients, routing, PLE scale and
norm ratio, periodic memory-on/off validation, exact RNG state and counters.
Gradient accumulation above one is rejected until implemented correctly.

`poc_b_v2.yaml` and `poc_c_v2.yaml` retain the larger 10-block backbone shapes
for subsequent confirmation. They currently use sequence length 256 for the
new dataset and are not matched to the historical sequence-length-2048 runs.
Use fresh paired training and measure memory again when changing sequence length.

## Run full-sized C2 across two GPUs

From the repository root, start a fresh 30M-token confirmation run:

```bash
env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m flashmini.cli train \
  --config configs/flashmini/poc_c_v2.yaml \
  --data-dir data/fineweb_v2_30m \
  --run-dir runs/flashmini/ple_v2/c2_dual_gpu_seed17 \
  --model-parallel-gpus 1,0 --gpu-memory-gib 15 \
  --tokens 30000000 --batch-size 32 --seed 17 \
  --lr 3e-4 --ple-lr-multiplier 5 --warmup-tokens 524288 \
  --eval-every-tokens 2097152 --eval-max-batches 128 \
  --checkpoint-every-tokens 4194304 --log-every 10
```

This uses sequential layer sharding, not DDP or overlapping pipeline stages.
The device list is a placement order: with `CUDA_VISIBLE_DEVICES=0,1`, `1,0`
puts physical GPU 1 first, so it owns the tied embedding/head, the first block
shard and the PLE injection block; physical GPU 0 receives the later block
shard. Sharding is static (five blocks per device), not demand-driven
"fill-then-overflow" allocation.
PLE tables and sparse optimizer moments remain on CPU; tied embeddings/head
stay on the first GPU. Larger batches exploit the extra memory but change the
number of updates per token: match batch size and schedule in the B2 control.
The LR is inherited from pilots, not a completed full-sized LR search.

The memory option sets a PyTorch allocator limit after subtracting existing
device use and reserving at least 1 GiB free. It is not a system-wide hard cap;
driver allocations and other processes can change consumption. Reduce batch
size if the guard raises an out-of-memory error. Start with a new run directory;
old checkpoints are not silently resumed.

On two RTX 5060 Ti cards, the 262,144-token batch-32 smoke run achieved about
8,431 tokens/sec, with peak reserved memory 12.10/5.70 GiB (GPU 0/1), excluding
other processes and driver memory. This short test includes a tiny validation
pass but no checkpoint writes; it does not establish long-run speed or quality.
Evidence: `runs/flashmini/ple_v2/dual_gpu_probe_b32/summary.json`.

## Evaluate accuracy and VRAM separately

Training keeps only the latest routine `checkpoints/step_*.pt` file. A save is
read back, flushed, and atomically published before older steps are removed.
Allow temporary disk space for two checkpoints (about 12.5 GiB for full C2)
and extra host RAM for read-back. Optimizer and RNG state retain full precision.
Use one training process per run directory. Evaluation milestones belong in a
separate `milestones/` directory so routine retention cannot remove them.

The September 14 cleanup retained C2's 100,663,296-token milestone and latest
113,246,208-token resume state, the five corrected pilot final checkpoints, and
small historical/result artifacts. Legacy weights were deleted. The retention
audit under `runs/flashmini/ple_v2/retention_20260914/` records deleted files and
the verified storage-only source provenance migration for the resume checkpoint.
See `docs/flashmini-resume.md` for the matched continuation commands.

Use `python -m flashmini.compare_ple` with the baseline run first. By default it
excludes the first 128 validation sequences used for pilot LR screening. It
reports NLL, top-1 token accuracy, PLE-off ablation and a paired block bootstrap.
That interval describes the fixed holdout; repeat training seeds before drawing
broader conclusions. No downstream reasoning benchmark is currently connected.

Use `python -m flashmini.benchmark_ple` in a separate process per storage mode.
It reports CUDA peak allocation/reservation, optimizer placement, host peak RSS
and short-step throughput. This is training/prefill measurement, not cached
autoregressive decoding. A correctness pass, memory saving and quality gain are
three separate acceptance conditions.

Research context: [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
and [Engram paper](https://arxiv.org/html/2601.07372v1).
