<!-- nsp:meta
id: docs.flashmini-v3-upstream
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-v3-upstream.md
graphTags: flashmini,v3,training
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-14
replaces:
replacedBy:
-->

# FlashMini v3 upstream semantics and deviations

Verification uses Transformers commit `bd15bc95a89e728bbc1224084eb3b5829428c353`,
[modeling_qwen4_exp.py](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
Source SHA-256:
`2a44aeadb215acbb5c75939fcc97e9f14bccff5a51c232826427594993f6a760`.
The official model configuration was inspected at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Downloaded upstream implementation code is not vendored into this repository.

## Verified mechanisms

GR carries four streams, normalizes and dynamically reads them into a mixer,
then dynamically writes its output into all streams. Mixer and MoE have separate
GR units; final GR combines streams. V3 has no extra block LayerNorm or internal
GDN residual. Rank 96 scales as d/8 at width 768 rather than copying rank 320.

PLE has eight heads per n-gram order, grouped bigram then trigram, independent
prime tables and reference splitmix64/XOR hashes. Single-layer hash index is
zero. EOS resets subsequent context. Normalized context/key gating, value
projection and causal depthwise convolution use kernel four, dilation three,
SiLU and a direct residual. Convolution initializes at zero.

`scripts/flashmini_v3_upstream_check.py` verifies the pinned file hash and compares
reference/local GR outputs and input gradients, PLE hashes including EOS, and
nonzero-convolution PLE outputs/input gradients. Results are in
`validation/flashmini-v3-upstream.json`.

## Intentional deviations

- Local PLE convolution resets after EOS; upstream hashes reset but its inspected
  convolution does not segment documents. This stronger isolation is tested
  separately from one-document numerical parity. Segment discovery synchronizes
  GPU token positions to the host; measured throughput includes that cost.
- GDN retains a simplified single-state recurrence with scalar decay, not full
  upstream multihead GDN. Chunked/sequential numerical and gradient tests remain.
- MoE retains local GELU two-projection experts, not upstream gated-SiLU experts
  and shared-expert gating. All A/B/C use this same implementation.
- Conventional causal SDPA uses full-head RoPE, theta 10,000, not QSA or upstream
  partial-RoPE long-context configuration. MTP is absent.
- Table capacity and embedding width are reduced for local hardware. CPU tables
  have sparse gradients but dense SparseAdam moments. No GPU caching, overlapped
  prefetch or NVMe paging is implemented.
- V3 has no legacy learned scalar PLE residual scale. Its reported scale is fixed
  at one; learned gates and output norms measure treatment behavior.

Conclusions apply to these reduced-scale mechanisms, not the complete Qwen model.
