<!-- nsp:meta
id: docs.flashmini.unsloth
kind: document
documentType: technical
scope: flashmini
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: unreviewed
graphNode: document:docs/flashmini-unsloth.md
graphTags: flashmini,unsloth,training,huggingface
validation: pytest,ruff,integration-smoke
owner: flashmini
lastReviewed: 2026-10-10
-->

# Train FlashMini with Unsloth Desktop

The `flashmini_unsloth` branch adds a Hugging Face wrapper around the existing
FlashMini v2/v3 model. The native trainer and Kaggle production commands remain
available. The adapter keeps native parameter names behind an HF `model.` prefix,
tied embeddings, MoE auxiliary loss, PLE memory, and KVC sharing. It shifts HF
labels internally; pass ordinary unshifted token IDs as labels.

The first supported Desktop path is **full fine-tuning with BF16 weights**. Fresh
weights support learning from scratch; this does not supply a pretrained model.
LoRA, QLoRA, Unsloth's architecture-specific speedups, and distributed Desktop
training are not validated for FlashMini. The frozen TPU production schedule,
data identity, and exact-resume contract are separate from a Desktop run.

## Install the adapter in Desktop's environment

Quit an active training run before changing its environment. From this repository
on `flashmini_unsloth`, use Desktop's Python, preserving its dependency versions:

```powershell
$desktopPython = 'D:\AI\Unsloth\Backend\unsloth_studio\Scripts\python.exe'
& $desktopPython -m pip install --no-deps --no-build-isolation --editable C:\work\llm_playground
```

The model export includes the HF wrapper's Python file and an `auto_map`. It
depends on the installed `flashmini` package for the native model implementation.
Keep that package on this branch and re-export after changing adapter code.
Restart Desktop after installing or updating the package. Do not rename FlashMini
to a built-in architecture to make a loader accept it.

## Prepare fresh 1B weights

The existing `configs/flashmini/flashmini_1b_v1.yaml` is the starting configuration.
It has 1,001,859,456 learned parameters, including its expert bank and PLE table.
Desktop rejects GPT-2's EOS-as-padding policy. `--desktop-pad-token` adds a dedicated
`<|flashmini_pad|>` token and builds a separate fresh model with 50,258 vocabulary
entries and **1,001,860,224 parameters**. Its native config and PLE hash metadata
follow the new vocabulary. The frozen YAML and production tokenizer stay intact.

The base tokenizer uses the production contract's immutable GPT-2 revision:

```powershell
& $desktopPython -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', cache_dir='D:/AI/Unsloth/Models/hub'); t.pad_token=t.eos_token; t.padding_side='right'; t.save_pretrained('D:/AI/Unsloth/Backend/cache/flashmini-gpt2-tokenizer')"

& $desktopPython scripts/flashmini_hf_export.py `
  --config configs/flashmini/flashmini_1b_v1.yaml `
  --tokenizer D:/AI/Unsloth/Backend/cache/flashmini-gpt2-tokenizer `
  --dtype bfloat16 --seed 17 --desktop-pad-token `
  --output D:/AI/Unsloth/Backend/models/flashmini-1b-desktop-fresh
```

Use a new destination for each export. The export contains random initial weights;
the name `fresh` is deliberate. This Desktop variant has its own tokenizer and run
identity; it is not a frozen TPU production artifact. The native trainer can use
its exported native config and weights too. Exporting allocates the model on CPU
before casting and saving. This does not start a training job.

## Start a Desktop pilot

1. Open Desktop's training workflow and select the local folder
   `D:\AI\Unsloth\Backend\models\flashmini-1b-desktop-fresh`. Refresh the local inventory
   or add its parent as a scan folder if needed.
2. Select **Full Fine-tuning**, with **4-bit loading off** and BF16/16-bit loading.
3. Enable **Trust Remote Code** for this local export. Follow Desktop's normal
   review prompt if it presents one; the adapter does not disable security checks.
4. Import `configs/unsloth/flashmini_1b_desktop.yaml` as the hyperparameter preset.
   Check that **gradient checkpointing is off**, **packing is off**, and the
   optimizer is **adamw_torch**. The preset does not select the model, training
   method, dataset, or remote-code option.
5. Select a plain-text training dataset with a `text` field and use raw-text
   formatting. Start with the preset's two updates, batch size one, and 128-token
   context. Inspect loss, gradients, memory, and saved weights before increasing
   the context length and run duration.

The 128-token preset is an integration probe. It is not an adequate context length
or token budget for producing a useful 1B model. Measure progressively larger
contexts on the actual dataset; the model's 2,048-token capacity is not a guarantee
that every training setting fits a 16 GB GPU.

## Reproduce the backend integration check

The probe uses the installed Desktop loader, trainer, security checks, and output
path checks. It tests BF16 loading, causal training, optimizer updates, and saving.
It uses the first physical GPU by default and does not start the frontend.

```powershell
# Offline tiny v3 model with GDN, MoE, PLE, and KVC enabled.
& $desktopPython scripts/flashmini_unsloth_smoke.py `
  --output D:/AI/Unsloth/Backend/outputs/flashmini-tiny-probe --gpu 0

# Bounded run using an already exported 1B model.
& $desktopPython scripts/flashmini_unsloth_smoke.py `
  --model D:/AI/Unsloth/Backend/models/flashmini-1b-desktop-fresh `
  --output D:/AI/Unsloth/Backend/outputs/flashmini-1b-probe --gpu 0 --sequence-length 128
```

Use new output directories under Desktop's `outputs` root. Each probe writes
`report.json` with actual step count, weight-change evidence, losses, and peak CUDA
allocation. A successful short probe establishes training integration, not model
quality or long-run stability.

### Validation on this Windows deployment

On 2026-10-10, the installed Desktop backend (`unsloth 2026.10.3`, Transformers
5.5.0, PyTorch 2.11.0+cu130) completed two full-training updates of the Desktop 1B
variant on one RTX 5060 Ti 16 GB. Both examples were truncated to 128 tokens:
256 input tokens total, finite losses 10.8214 and 10.7347, changed embedding weights,
and saved full model plus optimizer checkpoint. Peak PyTorch CUDA allocation was
**8.673 GiB**. This excludes unrelated processes and is not a long-context memory
estimate. The fresh export was preserved separately from the probe's trained copy.

Evidence: `D:\AI\Unsloth\Backend\outputs\flashmini-1b-desktop-probe128-20261010\report.json`.
The 200,816-parameter v3 fixture also completed two Desktop updates with GDN, MoE,
PLE, and KVC enabled.

```powershell
.venv-win/Scripts/python.exe -m pytest tests/test_hf_adapter.py
# 10 passed
.venv-win/Scripts/python.exe -m pytest tests/flashmini/test_v3_architecture.py tests/flashmini/test_v3_training.py tests/flashmini/test_training_integrity.py
# 43 passed, 1 skipped
.venv-win/Scripts/python.exe -m ruff check src/flashmini/hf_adapter.py scripts/flashmini_hf_export.py scripts/flashmini_unsloth_smoke.py tests/test_hf_adapter.py
# Passed
```

`_nsp context validate --target .` passed. Full `_nsp validate` did not pass because
the repository map was missing/stale. Attempting to rebuild it hit the existing
artifact-directory safety check (`unsafe artifact directory: C:\work`). That
guard was preserved; no full governance-validation pass is claimed. The Desktop
frontend interaction and long-run training quality were not tested.

## Transfer weights between training methods

Add `--checkpoint <native-file-or-production-directory>` to the export command to
warm-start HF training. Supply `--config` when the checkpoint does not contain its
native model config. A full production checkpoint directory should be complete
and verified before use. Conversion transfers model weights; it does not convert
optimizer state, scheduler, RNG, data cursor, or production lineage.
`--desktop-pad-token` is restricted to fresh initialization and cannot be combined
with checkpoint import: adding a vocabulary entry changes the native PLE hash
contract, so it must not silently reinterpret an existing checkpoint. An ordinary
export preserves the original tokenizer, but GPT-2's EOS-as-pad export currently
cannot pass Desktop's loader. Checkpoint imports with an already distinct padding
token can use the standard Desktop workflow.

HF-trained weights can be loaded back into the native model for a **new** native
run:

```python
from flashmini.hf_adapter import FlashMiniForCausalLM
from flashmini.models import FlashMiniModel

hf = FlashMiniForCausalLM.from_pretrained("path/to/hf-checkpoint")
native = FlashMiniModel(hf.config.to_native_config())
native.load_state_dict(hf.model.state_dict(), strict=True)
# Construct a fresh native optimizer and run identity before continuing training.
```

`save_native_checkpoint()` also writes a native weights-only envelope. The native
v3 resume guard correctly rejects that envelope when asked to restore an optimizer.
Do not treat a weights transfer as exact continuation of the native production run.

## Supported input and runtime limits

- Contiguous right padding is supported. Padded examples run at their true lengths
  so padding cannot enter recurrence or routing statistics. Left padding, interior
  mask holes, reset position IDs, and arbitrary attention masks fail explicitly.
- Sparse PLE gradients are rejected by this dense-optimizer adapter. The frozen 1B
  config already uses dense accelerator-resident PLE. Native sparse CPU PLE training
  remains available through the native trainer.
- Gradient checkpointing, KV-cache generation, and token-vocabulary resizing are
  unsupported. Keep tokenizer size and token IDs aligned with the native config.
- The expert dispatch casts its float32 routing contribution to the dispatch
  buffer dtype. This fixes BF16-loaded training and leaves float32 dispatch math
  unchanged. The HF loader also restores derived native buffers after meta loading.

The adapter follows [Hugging Face's custom-model contract](https://huggingface.co/docs/transformers/custom_models).
Desktop's general local-model workflow is described in the
[Unsloth Studio guide](https://unsloth.ai/docs/new/studio/start).
