# FlashMini corrected PoC continuation

Run from `/home/jason/workspace/llm_playground`. Run C2 and B2 sequentially:
each command uses both GPUs. These commands target the 250M-token PoC budget;
training repeatedly samples the existing approximately 29.4M-token corpus.
This is not 250M unique training tokens and does not complete the epic's 1B
scaling or conventional-attention baseline requirements.

## Resume C2 to 250M tokens

```bash
env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m flashmini.cli train \
  --config configs/flashmini/poc_c_v2.yaml \
  --data-dir data/fineweb_v2_30m \
  --run-dir runs/flashmini/ple_v2/c2_dual_gpu_seed17 \
  --resume runs/flashmini/ple_v2/c2_dual_gpu_seed17/checkpoints/step_13824.pt \
  --model-parallel-gpus 1,0 --gpu-memory-gib 15 \
  --tokens 250000000 --batch-size 32 --seed 17 \
  --lr 3e-4 --ple-lr-multiplier 5 --warmup-tokens 524288 \
  --eval-every-tokens 2097152 --eval-max-batches 128 \
  --checkpoint-every-tokens 4194304 --log-every 10
```

The checkpoint contains 113,246,208 token positions. Interrupted progress after
that point is recomputed. If interrupted again, use the single newer checkpoint
filename in `checkpoints/`; the old resume filename will have been retired.

## Start B2 and stop at the matching early gate

```bash
env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m flashmini.cli train \
  --config configs/flashmini/poc_b_v2.yaml \
  --data-dir data/fineweb_v2_30m \
  --run-dir runs/flashmini/ple_v2/b2_dual_gpu_seed17 \
  --model-parallel-gpus 1,0 --gpu-memory-gib 15 \
  --tokens 100663296 --batch-size 32 --seed 17 \
  --lr 3e-4 --warmup-tokens 524288 \
  --eval-every-tokens 2097152 --eval-max-batches 128 \
  --checkpoint-every-tokens 4194304 --log-every 10
```

Compare this checkpoint against C2's retained
`milestones/step_12288.pt` at the same token count. Preserve the gate evaluation
and move or copy any checkpoint needed for later evaluation into `milestones/`
before continuing. C2's milestone retains its original source provenance;
evaluation loading is supported, but resuming that milestone would require a
separately verified provenance migration.

After the early evaluation permits continuation, run the B2 command again with
`--tokens 250000000` and add
`--resume runs/flashmini/ple_v2/b2_dual_gpu_seed17/checkpoints/step_12288.pt`.
If that file was moved, use its milestone path instead.

Compare final B2/C2 on the same held-out slice, including C2 memory-on/off,
training health, resource costs and correctness. B2/C2 isolates the PLE effect.
The original epic additionally requires a matched conventional-attention
control, scale confirmation if PoC gates pass, and a decision report. Historical
POC-A outcomes remain available as context; they are not matched controls for
these new training conditions. Finishing C2 alone does not establish a GO.
