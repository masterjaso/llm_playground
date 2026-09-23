<!-- nsp:meta
id: technical.flashmini-training-runtime
kind: technical-doc
audience: developer
status: active
description: Describe the FlashMini model, training, checkpoint, and matched-comparison runtime contracts.
resource: ezra://technical/flashmini-training-runtime
scope: technical
persona: platform-engineering
source: model
confidence: high
reviewStatus: unreviewed
graphNode: technical:flashmini-training-runtime
graphTags: flashmini,architecture,training
validation: context-header-audit,manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# FlashMini training runtime

## Purpose

Define the runtime boundary for the FlashMini v3 experimental track: model
construction, deterministic training steps, checkpoint/resume, evaluation, and
matched treatment comparison. The runtime consumes an already validated corpus
and does not decide whether production data is release-ready.

## Feature links

- [Matched FlashMini experiments](../features/flashmini-experiment-gates.md)
- [Validation and release gates](../features/flashmini-validation-release.md)

## Architecture / design

- `FlashMiniConfig` is the source of truth for architecture version, width,
  layer pattern, sequence length, MoE routing, GDN, PLE, KVC, optimizer, and
  execution policy.
- Model blocks combine four-stream HyperConnections, a 16-expert top-2 MoE
  with a shared expert, GDN mixer layers, full attention at the registered
  positions, optional PLE n-gram memory, and optional 4-bit KVC.
- `training.py` owns token accounting, epoch sampling, learning-rate schedule,
  grouped gradient clipping, validation cadence, and run metadata. The pipeline
  engines provide monolithic, serial microbatch, and overlapped two-device
  execution while preserving one logical optimizer update.
- Checkpoints carry model/config identity, dataset identity, optimizer and RNG
  state, sampler position, clipping counters, and execution-policy metadata.
- Comparison helpers require matched backbone, data, tokenizer, seed, schedule,
  batching, optimizer, and PLE specification before producing a treatment result.

## Invariants

- Causality and document-boundary resets are enforced by tests and the data
  contract; a suffix cannot change a prefix result.
- A resume is rejected when architecture, data, optimizer, or policy identity
  differs unless the explicit pipeline-transition authorization is present.
- PLE-off controls and PLE-on candidates keep shared name-keyed initialization
  so allocation changes do not become an uncontrolled treatment difference.
- Shared, PLE dense, and PLE sparse gradients are clipped and reported by group;
  non-finite values fail closed.

## Code anchors

- repo://src/flashmini/config.py
- repo://src/flashmini/models/moe.py
- repo://src/flashmini/models/gated_delta_net.py
- repo://src/flashmini/models/ple.py
- repo://src/flashmini/models/hyperconnection.py
- repo://src/flashmini/training.py
- repo://src/flashmini/pipeline.py
- repo://src/flashmini/checkpoint.py
- repo://src/flashmini/comparison.py
- repo://src/flashmini/evaluation.py
- repo://src/flashmini/cli.py
- repo://configs/flashmini/poc_a_v3.yaml
- repo://configs/flashmini/poc_b_v3.yaml
- repo://configs/flashmini/poc_c_v3.yaml
- repo://configs/flashmini/poc_d_v3.yaml

## Validation

- `python -m pytest tests/flashmini/test_v3_architecture.py tests/flashmini/test_v3_training.py`
- `python -m pytest tests/flashmini/test_checkpoint.py tests/flashmini/test_training_integrity.py`
- `python -m pytest tests/flashmini/test_generic_comparator.py tests/flashmini/test_v3_clipping.py`
- `repo://tests/flashmini/test_v3_probes.py` covers structural context and
  deterministic evaluation probes.

## Risks

- Official 100M/250M matched runs and long-context evidence remain incomplete;
  passing implementation probes are not a winner claim.
- Sparse PLE optimizer state is CPU-resident and can dominate memory.
- Pipeline transitions require explicit authorization and are not equivalent to
  proving the two execution schedules have identical performance.
