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
lastReviewed: 2026-05-23
replaces: 
replacedBy: 
-->

# dense2moe

`dense2moe` is a Windows-first, resumable control plane for experiments that
convert a dense feed-forward network into a sparse top-k mixture-of-experts
checkpoint.  The implementation keeps the source checkpoint immutable, writes
run state atomically, and records evidence for every command.

The project is intentionally useful without CUDA or a model download: the
partition, routing, state, queue, checkpoint, and tiny-model tests use small
synthetic fixtures.  Optional PyTorch, Transformers, safetensors, and Hugging
Face Hub integrations are detected at runtime and are never silently assumed.

All commands are run from PowerShell on Windows, for example:

```powershell
python -m pip install -e ".[runtime,dev]"
d2m doctor --run-dir runs\<run-id> --json
d2m status --run-dir runs\<run-id> --json
```

The repository does not upload models or modify Windows drivers.  A real model
run must pass discovery, source, structural, quality, and export gates before
it can be marked `SUCCEEDED`.

The separate [FlashMini v3 experiment](docs/flashmini-v3.md)
tests matched attention/hybrid controls with optional CPU-resident n-gram memory.
V3 requires fresh checkpoints and a verified frozen corpus; historical v1/v2
outcomes do not satisfy its decision gates.
