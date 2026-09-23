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

The repository also contains a separate production-preview boundary for the
frozen FlashMini-1B Kaggle TPU trajectory. That path owns the production
freeze manifest, virtual remote-shard data cursor, XLA worker, durable
checkpoint protocol, cumulative metrics, and workstation-only controller. It
must not be treated as evidence that the TPU, quota, source authorization, or
25M/100M gates have passed.

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

## Production preview extension

- `configs/flashmini/flashmini_1b_v1.yaml` and
  `training_data/manifests/releases/flashmini-1b-foundation-v1.freeze.json`
  freeze the 1B architecture, tokenizer, source lock, virtual data view,
  trajectory, and checkpoint contract.
- `src/flashmini/production.py` owns the freeze, parameter, trajectory, and
  source-fingerprint identities; `src/flashmini/data_v4/virtual.py` owns the
  deterministic source/document/token cursor and exact token accounting.
- `src/flashmini/tpu_backend.py` and `src/flashmini/xla_training.py` own the
  static-shape TPU loop, XLA cache identity, session budget, scheduler, and
  status/checkpoint callbacks.
- `src/flashmini/production_checkpoint.py` and
  `src/flashmini/production_metrics.py` own full-state durable checkpoint
  promotion/readback and append-only cumulative metrics. The official worker
  refuses to run without a runnable source freeze, XLA FSDP, and either the
  private versioned `FLASHMINI_REMOTE_CHECKPOINT_DATASET` backend or a durable
  `FLASHMINI_REMOTE_CHECKPOINT_DIR`.
- `scripts/flashmini_1b_kaggle_controller.py` is workstation-only; it prepares,
  submits, and observes the Kaggle worker but never trains locally. The
  production document records the current queued-worker and blocked-source
  states without claiming TPU results.

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
- repo://configs/flashmini/flashmini_1b_v1.yaml
- repo://src/flashmini/production.py
- repo://src/flashmini/data_v4/virtual.py
- repo://src/flashmini/production_checkpoint.py
- repo://src/flashmini/production_metrics.py
- repo://src/flashmini/tpu_backend.py
- repo://src/flashmini/xla_training.py
- repo://scripts/flashmini_1b_kaggle_controller.py
- repo://scripts/flashmini_1b_kaggle_worker.py

## Validation

- `python -m pytest tests/flashmini/test_v3_architecture.py tests/flashmini/test_v3_training.py`
- `python -m pytest tests/flashmini/test_checkpoint.py tests/flashmini/test_training_integrity.py`
- `python -m pytest tests/flashmini/test_generic_comparator.py tests/flashmini/test_v3_clipping.py`
- `python -m pytest tests/flashmini/test_production_preview.py`
- `repo://tests/flashmini/test_v3_probes.py` covers structural context and
  deterministic evaluation probes.
- `repo://training_data/manifests/releases/flashmini-1b-equivalence-preview.json`
  records the bounded local CPU signature and explicitly does not close the
  TPU equivalence gate.

## Risks

- Official 100M/250M matched runs and long-context evidence remain incomplete;
  passing implementation probes are not a winner claim.
- Sparse PLE optimizer state is CPU-resident and can dominate memory.
- Pipeline transitions require explicit authorization and are not equivalent to
  proving the two execution schedules have identical performance.
- The production preview retires the metadata-only pinned `stack_edu` source
  and uses the immutable, content-bearing `stackv2_edu` revision upstream-only;
  no unpinned substitution or content mirroring is allowed.
- Kaggle queue state, TPU topology, sustained throughput, and durable remote
  checkpoint behavior require live worker evidence and are not established by
  local CPU tests.
