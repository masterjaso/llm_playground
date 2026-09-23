<!-- nsp:meta
id: feature.flashmini-experiment-gates
kind: feature-doc
audience: end-user
status: active
description: Run matched FlashMini architecture experiments with reproducible checkpoints and decision gates.
resource: ezra://feature/flashmini-experiment-gates
scope: features
persona: platform-engineering
source: model
confidence: high
reviewStatus: unreviewed
graphNode: feature:flashmini-experiment-gates
graphTags: flashmini,experiments,training
validation: context-header-audit,manifest-check,secret-scan
owner: features
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# Matched FlashMini experiments

## What this feature does

This capability lets an operator compare the registered FlashMini treatments on
the same frozen data, seed, schedule, and evaluation contract. The experiment
track distinguishes the conventional control, the hybrid mixer, and the
hybrid-plus-memory treatment, with an optional quantized continuation.

## Who uses it

Use it when you are screening model changes, reproducing the v3 pilot,
or deciding whether a treatment has enough evidence to advance to a larger
run. Researchers own the comparison decision; operators own the run directory,
checkpoint, and resume evidence.

## How to use it

Prepare the project environment, choose the registered treatment, and launch
the documented v3 run command from the repository root. Keep the control and
candidate settings matched, and use the resume command when a run pauses at a
token gate. Inspect the validation handoff before starting an official run.

## Expected behavior

The runner records the data identity, configuration, seed, optimizer policy,
token counts, clipping observations, and checkpoint state. A resumed run must
continue from the same dataset and policy identity. Comparisons reject unequal
backbones, tokenizers, manifests, schedules, or treatment labels instead of
producing an ambiguous result.

## Limits / known constraints

The implementation and short pilots are evidence gates, not a model
winner. Official 100M and 250M matched runs are not complete, long-context
quality is still an open gate, and historical v1/v2 claims are not reused.
The quantized treatment is a continuation of the hybrid-plus-memory setup, not
a substitute for the matched control.

## Technical implementation

See [the training runtime technical reference](../technical/flashmini-training-runtime.md).
