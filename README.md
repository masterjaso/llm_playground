<!-- nsp:meta
id: readme
kind: readme
scope: root
persona: governance-package
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:README.md
graphTags: 
validation: manifest-check,secret-scan
owner: root
lastReviewed: 2026-09-22
replaces: 
replacedBy: 
-->

# flashmini

`flashmini` is the primary package in this repository: the FlashMini v3
experimental LLM training track. It trains a compact decoder-only transformer
with a pre-registered gate policy on a frozen FineWeb-EDU corpus.

- **Architecture**: MoE (16 experts, top-2, shared expert) + GDN
  (d_state 128, short_conv) + HyperConnections (x4), full attention at blocks
  3 and 7, PLE n-gram memory (sparse, CPU offload), 4-bit KVC (E2M1/E4M3, QAT).
- **v3 PoC results** (250M tokens, seed 17, frozen corpus manifest
  `b06f559f...`): treatment C (MoE+GDN+PLE) reached 29.82 ppl / 0.381 top-1;
  treatment D adds 4-bit KVC at quality parity (30.02 ppl / 0.379 top-1).
- **Data**: `HuggingFaceFW/fineweb-edu` train split, GPT-2 tokenizer,
  manifest-hashed and identical across all treatments.

## Usage

All commands run from the repository root with the project virtualenv:

```bash
uv sync --extra dev
.venv/bin/pip install -e .
.venv/bin/flashmini --help
.venv/bin/python -m pytest tests/flashmini
```

## Documentation

- [docs/flashmini-v3.md](docs/flashmini-v3.md) — v3 experiment design and PoC results
- [docs/flashmini-ple.md](docs/flashmini-ple.md) — PLE n-gram memory
- [docs/flashmini-resume.md](docs/flashmini-resume.md) — checkpoint resume policy
- [docs/flashmini-v3-data.md](docs/flashmini-v3-data.md) — frozen corpus contract
- [docs/flashmini-v3-upstream.md](docs/flashmini-v3-upstream.md) — upstream checks
- [docs/flashmini-v3-validation.md](docs/flashmini-v3-validation.md) — validation evidence

## Good resources

https://github.com/FareedKhan-dev/train-llm-from-scratch

