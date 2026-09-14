<!-- nsp:meta
id: docs.flashmini-v3-validation
kind: document
scope: technical
persona: platform-engineering
status: active
source: model
confidence: high
reviewStatus: reviewed
graphNode: document:docs/flashmini-v3-validation.md
graphTags: flashmini,v3,validation
validation: manifest-check,secret-scan
owner: technical
lastReviewed: 2026-09-14
replaces:
replacedBy:
-->

# FlashMini v3 validation handoff

Implementation validity, micro-overfit and short fresh-corpus pilots passed.
This is permission to begin matched screening, not evidence that Flash or PLE
outperforms A. No official 100M/250M training was launched during this task.
The integration base on `flash-mini` was
`fa35b6bff56b63b562e9380b2aed16fd3ba86366`.

## Reproduce the checks

Run from the repository root. The full suite covers architecture causality,
chunked/sequential GDN gradients, four-stream GR, PLE convolution/EOS/hash
semantics, shared initialization, clipping isolation, data integrity, comparison
rejection, deterministic evaluation and exact checkpoint/resume:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest tests/flashmini -o addopts='' -q
```

Result: **125 tests and 3 subtests passed**, with no skipped tests. The deliberate
huge-PLE-gradient acceptance test is
`tests/flashmini/test_v3_clipping.py::test_huge_ple_gradient_cannot_change_shared_clipping_coefficient`.
The six `test_v3_*.py` modules contain the new regression coverage; the existing
FlashMini tests remain part of this command.

Run the short runtime validator separately, with both GPUs otherwise idle:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/flashmini_v3_validate.py \
  --official-shape --official-batch-size 16 --gpu-memory-gib 15 \
  --data-dir data/fineweb_v3_2b --out /tmp/flashmini-v3-final-runtime.json
```

The output path must not already exist. The validator uses disk-backed temporary
storage under `runs/flashmini/.validation`, overridable with `--scratch-dir`.
It removes its temporary datasets, logs and checkpoints on normal completion.

Result: **PASS** for A/B/C micro-overfit (200 steps each), tiny save/resume,
1024-token causal probes, three consecutive official-shape optimizer updates,
and fresh official-shape pilots of 32,768 tokens/eight updates each. The real
model-parallel path used `cuda:1,cuda:0`, with C's sparse PLE table on CPU.
Peak allocated GPU memory in the three-update probe was at most 9.67 GiB on
either GPU, under the same 15 GiB budget guard used by the CLI. Allocated memory
is not total device usage. Micro-overfit losses fell approximately 85%, 91% and
95% for A/B/C. The short pilots had finite losses and clipping/PLE metrics;
their held-out losses are not useful architecture-quality conclusions.
See [runtime evidence](validation/flashmini-v3-runtime.json), including source
and validation-script hashes, metrics and data-integrity output.

Pinned upstream differential check:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/flashmini_v3_upstream_check.py \
  --source /tmp/flashmini-v3-upstream.py --out /tmp/flashmini-v3-upstream-handoff.json
```

Obtain the source separately from the pinned URL in the
[upstream reference](flashmini-v3-upstream.md); the checker requires its exact
SHA-256 before evaluating allowlisted definitions. Result: **PASS**, including
GR gates/input gradients, EOS hash keys and one-document PLE convolution/input
gradients; maximum PLE output difference was zero. The stronger local
convolution reset after EOS is intentionally tested separately.
See [differential evidence](validation/flashmini-v3-upstream.json).

The official-size A/B/C models also passed this structural probe, independently
of the tiny-model runtime probes (run once per variant):

```bash
variant=a
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m flashmini.context_probe \
  --config "configs/flashmini/poc_${variant}_v3.yaml" \
  --seq-len 1024 --device cuda:1 --seed 17
```

Maximum prefix-logit differences were 2.03e-6, 2.26e-6 and 2.15e-6, within the
1e-5 tolerance. These were untrained models, not long-context quality tests.
See [structural evidence](validation/flashmini-v3-context1024.json).

Data verification and collision audit:

```bash
.venv/bin/python -c 'from pathlib import Path; from flashmini.data import verify_dataset_integrity; print(verify_dataset_integrity(Path("data/fineweb_v3_2b")))'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m flashmini.audit_ple \
  --config configs/flashmini/poc_c_v3.yaml --data-dir data/fineweb_v3_2b \
  --sequences 2048 --out docs/validation/flashmini-v3-collisions.json
```

Full integrity verification passed, including shard/database hashes, counts,
shapes and nonempty label rows. See [data identities and pass counts](flashmini-v3-data.md)
and the [manifest evidence](validation/flashmini-v3-data.json).
The audit sampled 524,288 validation positions in 2,048 evenly spaced rows,
observing 738 EOS tokens, 269,772 distinct bigrams and 444,724 distinct trigrams.
Individual-head collision fractions were approximately 81.4% and 88.7%.
No combined eight-head signature collision was observed for either order.
This is a sample measurement, not proof of collision freedom or a quality
threshold. See [collision evidence](validation/flashmini-v3-collisions.json).

## Failures and review

Batch 32 exhausted GPU memory in the multi-step 15 GiB-budget check, despite
an earlier unrestricted single-step pass. The matched recipe is now batch 16.
The first complete batch-16 validator exhausted the 16 GiB `/tmp` tmpfs while
saving C after A/B passed. Moving temporary checkpoints to project disk resolved
that validation failure; the complete rerun passed and removed its scratch data.
See [failure/resolution evidence](validation/flashmini-v3-batch-budget.json).

The data preparer completed its files but stalled in upstream HTTP teardown;
it was terminated after independent integrity verification, not recorded as a
clean process exit. No corpus files were removed or altered to suppress errors.

Two independent fresh-context, read-only reviewers found resume, provenance,
packing, EOS, holdout and RNG loopholes. Their valid findings were repaired and
rechecked before handoff; see [review scope and results](validation/flashmini-v3-review.json).
Changed Python files pass Ruff. A broader FlashMini scan still reports 13
pre-existing findings in untouched `hardware.py`, `lr_probe.py`, `metrics.py`
and `models/moe.py`; those are not represented as a clean repository-wide lint.

## Storage and next gates

Old runtime evidence and corpora were permanently deleted under the user's
explicit amendment to the original preservation requirement. Old C2/B2 runs
cannot resume, and no old metrics/checkpoints are claimed preserved. Historical
source remains in Git. The new corpus occupies approximately 15.3 GiB; the
project filesystem has approximately 381 GiB free after validation cleanup.

Each official run retains only its newest validated checkpoint. A write needs
temporary room for both old and replacement checkpoints. Optimizer and RNG
state are preserved; this policy limits checkpoint count, not checkpoint size.
The validation task leaves no pilot checkpoints to confuse with official runs.

Follow the [matched run/resume reference](flashmini-resume.md). Evaluate all
three screening checkpoints before proceeding: later checkpoints replace them.
Final epic GO remains blocked on matched 100M/250M outcomes, scaling confirmation,
genuine long-context quality and at least three matched seeds for small effects.
QSA/MTP remain off; simplified GDN/MoE and the other documented upstream
deviations limit the scope of any eventual architectural claim.
