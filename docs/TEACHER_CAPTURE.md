<!-- nsp:meta
id: docs.teacher.capture
kind: document
scope: features
persona: prompt-engineering
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/TEACHER_CAPTURE.md
graphTags: docs
validation: manifest-check,secret-scan
owner: features
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# Native teacher activation capture

`dense2moe.cli capture` can run a real text-to-teacher path when a pinned, local
Transformers snapshot is available:

```text
& .\.venv\Scripts\python.exe -m dense2moe.cli capture `
  --run-dir runs\<child-run> `
  --dataset-manifest runs\<child-run>\capture\data-plan.json `
  --source-dir C:\path\to\qwen-snapshot `
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 `
  --layers 0,16,32,48,63 `
  --split both `
  --device-map auto `
  --microbatch 1 `
  --resume
```

The loader uses `local_files_only=True` and `trust_remote_code=False`.  It
does not download a model or substitute a different tokenizer.  If the source
snapshot, tokenizer runtime, model runtime, corpus source record, or pinned
revision is unavailable, the command records a `BLOCKED` result with a
specific `blocker_code` and creates no fabricated activation payload.

The teacher's actual named modules are searched for `layers[N].mlp` modules
that expose `gate_proj`, `up_proj`, and `down_proj`.  Every selected hook is
experimentally checked by comparing the teacher output with
`down_proj(silu(gate_proj(x)) * up_proj(x))`; the default normalized-MSE gate
is `1e-7`.

The general capture command writes MLP inputs for diagnostic or downstream
calibration work.  Phase 01 uses the separate
`scripts/capture_real_qwen_layer0.py` command, which selects only the frozen
`METHOD_PROOF_ONLY` FIT-TRAIN records and reuses the same layer-major loader.
For layer 0 it writes paired safetensors tensors:

* `ffn_input` — the real layer-0 dense SwiGLU input `X`, shape `[N, 5120]`;
* `dense_ffn_target` — the real dense `gate_proj/up_proj/SiLU/down_proj`
  output `Y`, shape `[N, 5120]`.

The resulting `real-qwen-layer0-receipt.json` has evidence class
`real-qwen-layer-capture`, source/config/tokenizer/runtime-lock hashes,
selected record/token identities, exclusion proof, and every shard SHA-256.
It is the only capture receipt accepted by the Phase 01 method-proof runner.
Synthetic fixture smoke output and legacy `mlp_input`-only manifests are not
scientific method-proof evidence.  Resume validates shard hashes and the
fixed split identity before reusing an artifact; it never fabricates a target
from an input-only shard.
