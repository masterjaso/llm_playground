<!-- nsp:meta
id: docs.flashmini-1b-kaggle-production
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-1b-kaggle-production.md
graphTags: flashmini,production,kaggle,tpu
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-23
replaces:
replacedBy:
-->

# FlashMini-1B Kaggle TPU production preview

The repository now contains a frozen, resumable production path for the first
10B tokens of the declared 100B FlashMini trajectory. The workstation only
prepares and monitors a Kaggle worker; training state and metrics belong to the
worker's durable run directory.

## Frozen contract

- Architecture: accepted PoC-D semantics — GDN hybrid, top-2 16-expert MoE,
  PLE, 4-stream HyperConnections, and KVC QAT (E2M1 values, E4M3 group
  scales, group 16, cross-layer share 2).
- Config: [`flashmini_1b_v1.yaml`](../configs/flashmini/flashmini_1b_v1.yaml).
- Learned parameters: 1,001,859,456 total; 238,181,504 active per token.
  The freeze manifest reports embedding, attention, GDN, MoE, PLE, dense/shared,
  and active-expert counts separately.
- Frozen breakdown: dense/shared 24,800,256; embedding/head 38,597,376;
  attention 11,796,480; GDN 13,296,000; all MoE experts 802,406,400;
  active routed/shared MoE per token 141,803,520; PLE 110,962,944; KVC
  parameter contribution 0 (QAT transform).
- Shape: static 2,048 tokens; BF16; target topology TPU v5e-8 (eight devices).
- PLE: the table is TPU-resident and uses dense shard-local updates on XLA;
  sparse CPU-table updates remain a PoC-only CUDA option.
- Tokenizer: GPT-2, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`, vocabulary 50,257, EOS 50,256.
- Curriculum: foundation 80B, quality 15B, long/coherent 5B. The 10B preview
  is a prefix of foundation, not a schedule endpoint. Full checkpoints cross
  every 25M exact-token threshold and allow at most one logical-batch overshoot.
- Data: a virtual remote-shard view with pinned source revisions, bounded host
  cache, deterministic document selection, exact GPT-2 token accounting, and a
  source/document/token cursor. The full 100B corpus is not materialized.

The checked-in freeze is [`flashmini-1b-foundation-v1.freeze.json`](../training_data/manifests/releases/flashmini-1b-foundation-v1.freeze.json).

## Operator commands

```bash
python scripts/flashmini_1b_kaggle_controller.py prepare
python scripts/flashmini_1b_kaggle_controller.py run --smoke
python scripts/flashmini_1b_kaggle_controller.py run
python scripts/flashmini_1b_kaggle_controller.py status
python scripts/flashmini_1b_kaggle_controller.py status --json
python scripts/flashmini_1b_kaggle_controller.py watch
python scripts/flashmini_1b_kaggle_controller.py logs --tail 100
python scripts/flashmini_1b_kaggle_controller.py metrics --tail 25
python scripts/flashmini_1b_kaggle_controller.py verify-latest
```

The bounded local semantic probe is also reproducible before a quota window:

```bash
PYTHONPATH=src python scripts/flashmini_1b_equivalence_probe.py --allow-unavailable
```

The checked-in preview result is
[`flashmini-1b-equivalence-preview.json`](../training_data/manifests/releases/flashmini-1b-equivalence-preview.json).
It records the finite CPU loss/gradient/router signature and explicitly marks
XLA as unavailable locally; it is not a TPU equivalence pass.

`prepare` writes `runs/flashmini/1b_kaggle/`. `run` submits the thin Kaggle
kernel and never trains on the workstation. The worker bundles the tested
package, validates the freeze, initializes the persistent XLA cache, runs a
two-update reduced model smoke, and emits `FLASHMINI_STATUS` heartbeats. An
official worker reconstructs the pinned virtual HF stream, restores the
newest verified full checkpoint, and trains through the tested XLA loop only
when the source audit is runnable, FSDP is available, and
either `FLASHMINI_REMOTE_CHECKPOINT_DATASET` names the private versioned
Kaggle checkpoint dataset or `FLASHMINI_REMOTE_CHECKPOINT_DIR` names a
genuinely durable mounted backend. The default worker bundle uses
`masterjaso/flashmini-1b-checkpoints`; the dataset is initialized with an
empty `LATEST.json` pointer before the first checkpoint.

## Durability and observability

`flashmini.production_checkpoint` writes a full-state candidate (model,
optimizer, scheduler, RNG, cursor, identities, lineage), reads it back,
checksums it, and only then promotes it. A provider-neutral filesystem backend
implements remote publish/verify/pointer/retention semantics for tests and
mounted durable storage. `KaggleDatasetRemoteBackend` uploads each verified
checkpoint and append-only metric snapshot as a new private Kaggle Dataset
version, and downloads/verifies the latest version on resume. The official
worker refuses to use Kaggle scratch as the only recovery copy.
`MetricsLedger` keeps `metrics.jsonl` and cumulative 25M milestone rows
append-only. `StatusLogger` atomically maintains `status/heartbeat.json`,
`status/progress.json`, and `run_status.json`. `Watchdog` distinguishes XLA
compile waits, data/network stalls, checkpoint uploads, and hard training
stalls.

## Validation evidence

The local suite passes with the isolated project environment:

```text
pytest -q                         100% passed
ruff check <changed files>        All checks passed
worker --smoke --allow-local-cpu  SMOKE_READY
```

The source audit retires the pinned `stack_edu` Python configuration because it
exposes metadata without a content field. The production recipes now use the
already pinned, content-bearing `stackv2_edu` source for the same code-domain
weights. It remains upstream-only (`review_required`), so content is streamed
from the immutable revision and is never mirrored into our corpus. The audit
records a bounded 256-record, non-empty-text/token probe and does not claim
full upstream capacity or corpus materialization.

The Kaggle API accepted the smoke submission, but no 8-device topology, XLA
version, throughput, checkpoint upload, 25M gate, or 100M gate is claimed until
the worker executes and writes its result manifest. The private checkpoint
dataset's empty pointer and live upload/download round-trip are verified; a
real TPU checkpoint still requires the worker to execute. The current external
state is `BLOCKED_KAGGLE_WORKER_QUEUED`, not a training success.
