# Target architecture

The target is text-only Qwen 3.5 with the source embedding, mixed
linear/full-attention blocks, RMSNorm, output head, tokenizer, and chat
template preserved byte-for-byte where the runtime permits.  Only each dense
SwiGLU MLP is replaced.

For a dense hidden input `x`, the teacher MLP is

```text
h = SiLU(x @ gate_proj.T) * (x @ up_proj.T)
y = h @ down_proj.T
```

The p8 profile partitions the 17,408 intermediate neurons into one shared
expert of width 1,024 and eight routed experts of width 2,048.  Each expert
stores `gate_proj`, `up_proj`, and `down_proj` tensors.  The partition is
disjoint and exhaustive, so all-expert mode is an exact reconstruction oracle.
Sparse mode selects normalized top-2 router weights, never drops tokens, and
records load-balancing metrics.

The installed Transformers 5.15 source contains native Qwen 3.5 text
linear/full attention and dense MLP classes, but its Qwen 3.5 config advertises
`base_model_ep_plan = None` and no routed/shared expert fields.  Therefore the
native model is the teacher/runtime reference, while the local
`Qwen35SwiGLUMoE` implementation is the versioned target fallback.  It does
not depend on remote code.  Its state-dict inventory is generated from the
module itself; strict save/reload is a prerequisite for assembly.  The module
is a text FFN/MoE building block, not a claim that a full 27B hybrid model has
been converted or that llama.cpp already supports this target.
