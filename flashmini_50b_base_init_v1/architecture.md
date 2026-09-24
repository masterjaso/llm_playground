# FlashMini-50B-Base architecture v4

`FlashMini-50B-Base` / `FlashMini-50B-Base-Init-v1`. The canonical YAML
(`flashmini_50b_base_init_v1.yaml`) is the source of truth; `architecture_sha256`
(93db9ff8ae480999afaa98e6893f9be0978e87683bab8c52eb0228c9313aba70) covers every geometry field and excludes
training, inference, storage and provenance metadata.

- vocab 131072, d_model 2048, 48 layers (38 Gated DeltaNet, 10 attention at 1-based 4,8,12,16,20,24,29,34,39,44)
- attention: 16 Q / 2 KV heads, head_dim 256, partial RoPE 64/256, context 262144, theta 1e7; five KV-cache pairs (reuse layers have no K/V tensors)
- MoE: 80 routed + 1 sigmoid-gated shared SwiGLU-1280 expert, top-6; balancing is global/logical-batch
- hyper-connections: 4 streams, low-rank 256
- PLE: injection at 0-based layer 1; 8 bigram + 8 trigram hash heads over segment-local `(x_t, x_(t-1), x_(t-2))`, EOS reset; tables on host
- MTP: one layer, recursive depths 1..3 (window base + 3), shares embedding and LM head
- untied input embedding and LM head; base count excludes MTP

Parameters: base 50,276,673,408; MTP 685,511,168; total 50,962,184,576; active base per token 4,894,079,872.

GDN short convolution follows Qwen3-Next semantics: causal depthwise convolution followed by SiLU, with no additive residual around the convolution.
