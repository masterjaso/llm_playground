# Dense-to-MoE p16/top6 negative result (2026-08)

This is the compact durable record for the p16/top6 geometry with a 50% active-width target: six selected experts per token, with the active computation constrained to half of the dense width.

## Frozen comparison

| Variant | Cosine | NMSE |
|---|---:|---:|
| Frozen oracle | `0.969319582` | `0.060663771` |
| Trained standard | `0.900795996` | `0.309793144` |
| Trained-basis oracle control | `0.927752376` | `0.159798950` |

The candidate failed the immutable quality gates. In particular, the layer-quality green gates require cosine `>= 0.98` and NMSE `<= 0.05` (with load CV `<= 0.50` for the load-aware oracle); none of these comparisons establishes a passing candidate. The frozen oracle is already below the target, so router optimization cannot make this fixed basis pass. The trained-basis control shows that basis degradation is primary; routing and amplitude errors are secondary contributors.

## Reusable implementation

- Model and expert implementation: `src/dense2moe/models/torch_moe.py`
- Distillation/training implementation: `src/dense2moe/training/torch_distill.py`
- p16/top6 selection runner: `scripts/run_p16_top6_50_selection.py`
- Decomposition diagnostic: `scripts/diagnose_p16_top6_50_decomposition.py`
- Regression coverage: `tests/test_p16_top6_50_selection.py`
- Related quality-gate contract: `docs/QUALITY_GATES.md`

Raw checkpoints, activation arrays, capture shards, and runtime evidence from the abandoned experiments were intentionally purged to reclaim storage. This archive is text-only and contains no large arrays, checkpoint tensors, raw logs, secrets, or copied artifact dumps.
