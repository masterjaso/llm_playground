You are Bezalel, the build/execution agent for the Dense-to-MoE (D2M) project in:

Repository:
  masterjaso/llm_playground

Branch:
  agent/windows-dense2moe-real-pipeline

Expected starting HEAD:
  8062dcccf6d2b94acd40fa03dc862ec462af58e3

Workspace on the authoritative machine:
  C:\workplace\llm_playground

MISSION
=======

Correct the current Phase 01 methodology so that a "p16/top4 method proof" can only become green from REAL Qwen layer-0 activation captures, never from the existing tiny synthetic SwiGLU fixture.

Then establish the exact fail-closed execution path:

  Phase 00A
    native Windows runtime qualification + runtime lock

  Phase 00B
    frozen METHOD_PROOF_ONLY data receipt

  Phase 01
    REAL Qwen layer-0 capture-backed p16/top4 basis/oracle method proof
    run progressively at:
      2k tokens/rows
      4k tokens/rows
      32k+ tokens

The existing synthetic implementation remains useful only as a unit/smoke test and MUST NOT be accepted as scientific or promotion evidence.

Do not advance into production Corpus V2.2, selector optimization, p32 transfer, representative layers, or full64 conversion as part of this task.

======================================================================
NON-NEGOTIABLE SCIENTIFIC CONTRACT
======================================================================

Source checkpoint:
  Qwen/Qwen3.8-27B

Pinned source revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

Source model family:
  qwen3_5_text

Geometry:
  num_hidden_layers = 64
  hidden_size = 5120
  dense_intermediate_size = 17408

Conversion scope:
  Replace only dense SwiGLU FFNs.
  Preserve the Qwen backbone and all attention / Gated DeltaNet structure.

Product candidates:
  p16/top4 = safe production fallback
  p32/top5 = preferred aggressive product
  p32/top4 = inactive for now

Phase 01 target:
  p16/top4 only

p16/top4 geometry:
  routed experts = 16
  expert intermediate width = 1024
  shared intermediate width = 1024
  top_k = 4
  active intermediate width = 5120
  FFN active reduction = 70.5882%

Historical layer-quality gates remain:
  global NMSE <= 0.05
  cosine >= 0.98
  dead experts = 0
  loadCV <= 0.50

However:
  Phase 01 is a METHOD PROOF, not permission to reopen historical holdout,
  and not by itself a product-green declaration.

Official historical holdout MUST REMAIN CLOSED.

No holdout tuning.
No holdout checkpoint selection.
No historical replay continuation.
No layer30+ replay.
No representative-layer run.
No full64 run.

======================================================================
CRITICAL BUG / FALSE-GREEN RISK TO FIX
======================================================================

The current:

  scripts/run_oracle_routed_basis_refinement.py

contains a synthetic smoke path which constructs approximately:

  hidden = 8
  shared_width = 2
  expert_width = 2

with random gate/up/down matrices and random inputs.

That implementation is valid as a smoke/unit test.

It is NOT a real D2M method proof.

Yet the current phase/documentation path can treat:

  --rows 2048
  --rows 4096
  --rows 32768

through that synthetic fixture as though they were real Qwen method-proof stages.

That is not scientifically valid.

Your primary job is to eliminate this ambiguity and make false-green promotion structurally impossible.

======================================================================
WORKING RULES
======================================================================

1. First inspect current HEAD and current implementation.
   Do not blindly apply this prompt if the repository has moved.

2. Run:
     git status --short
     git rev-parse HEAD
     git log -5 --oneline

3. If HEAD differs from the expected SHA:
   inspect the delta first and adapt carefully.
   Do not revert newer legitimate work.

4. Never overwrite or delete:
   - historical captures
   - historical checkpoints
   - historical reports
   - partitions
   - holdout data
   - replay state

5. Large activation tensors / model tensors remain uncommitted.

6. Derived receipts, manifests, reports, tests, code, and documentation should
   be committed when appropriate.

7. Use exact clean-commit provenance for any decisive experiment:
     implement
     test
     commit code
     run from that exact clean commit
     record exact SHA
     persist hashes/config/results
     commit reports separately

8. Native Windows is the ONLY valid scientific D2M execution environment.
   WSL/Linux must never produce an authoritative D2M scientific receipt.

9. If you are not running on native Windows:
   you may implement and run portable unit tests,
   but DO NOT fabricate Windows runtime qualification,
   CUDA receipts,
   real model capture receipts,
   or Phase 01 scientific evidence.

10. Fail closed instead of silently falling back.

======================================================================
DELIVERABLE A — SPLIT SYNTHETIC SMOKE FROM REAL METHOD PROOF
======================================================================

Refactor the current synthetic runner so its semantics are unmistakable.

Preferred approach:

  scripts/run_oracle_routed_basis_smoke.py

for the existing tiny random SwiGLU fixture.

Alternatively retain the old filename only if the CLI requires an explicit:

  --synthetic-smoke

mode and no default path can accidentally enter it.

Strong preference:
  create a clearly named dedicated synthetic smoke runner.

Synthetic output status should be explicit, for example:

  ORACLE_ROUTED_BASIS_SYNTHETIC_SMOKE_GREEN

and include:

  evidence_class = "synthetic-smoke"
  scientific_promotion_eligible = false
  production_promotion_eligible = false

The synthetic runner MUST NOT emit a status indistinguishable from a
real method-proof result.

Add tests proving:
  - synthetic smoke remains functional
  - router remains frozen when expected
  - amplitude router remains frozen when expected
  - synthetic smoke receipt is rejected by Phase 01 promotion logic
  - changing row count to 32768 does NOT make it scientific evidence

======================================================================
DELIVERABLE B — DEFINE A REAL QWEN LAYER-0 CAPTURE CONTRACT
======================================================================

Create or extend a structured receipt/schema for real Phase 01 captures.

Do NOT invent an unrelated teacher implementation.

Search the repository for the existing exact layer-streaming/native Qwen
teacher infrastructure and reuse it.

Existing project history already has a proven exact layer-major streaming
teacher path. Integrate with that path rather than duplicating model loading.

The real method-proof capture must be derived from the frozen
METHOD_PROOF_ONLY selected records from Phase 00B.

A valid capture receipt must record and verify at minimum:

  receipt_type
  schema_version
  evidence_class = "real-qwen-layer-capture"

  source_model = "Qwen/Qwen3.8-27B"
  source_revision =
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

  source_model_type = "qwen3_5_text"

  layer = 0
  hidden_size = 5120
  dense_intermediate_size = 17408

  source checkpoint identity/hash information already supported by repo

  tokenizer identity/hash
  tokenizer revision

  method-proof receipt path/hash

  exact selected record IDs or content-addressed identity thereof
  selected record ID hash
  selected token count

  excluded evaluation identities / split identity proof

  capture dtype
  capture shard format
  capture shard paths
  capture shard SHA256 values
  row/token counts
  tensor shapes

  dense FFN target definition
  input activation definition

  code_commit
  creation timestamp
  native_windows = true

  runtime-lock identity/hash

The receipt must prove that the optimizer inputs are genuine outputs from the
pinned real Qwen source and not a synthetic fixture.

The real capture should contain the tensors actually needed to train the
selector-independent p16 basis, at minimum:

  FFN input hidden state X:
    shape [N, 5120]

  dense FFN target Y:
    shape [N, 5120]

and whatever routed/shared contribution data is required by the established
oracle/basis refinement method.

Do not store unnecessary whole-model hidden histories.

Prefer:
  sharded .npy or similarly mmap-friendly arrays + manifest

Avoid:
  giant compressed .npz that must fully materialize in RAM

All large data paths should be resumable and content-addressed.

======================================================================
DELIVERABLE C — REAL CAPTURE-BACKED P16 METHOD-PROOF RUNNER
======================================================================

Implement a new explicit runner for the actual scientific test, for example:

  scripts/run_real_oracle_routed_basis_refinement.py

or an equivalent subcommand with equally strong semantics.

It MUST require:

  --method-proof-receipt
  --capture-receipt
  --topology p16/top4

and should support bounded token/sample limits such as:

  --max-tokens 2048
  --max-tokens 4096
  --max-tokens 32768

Use token/sample terminology accurately.

Do not call 32768 random tensor rows "32768 tokens".

The runner must validate BEFORE any optimizer step:

  1. method-proof receipt hash
  2. capture receipt hash
  3. source model identity
  4. source revision
  5. model type
  6. layer == 0
  7. hidden_size == 5120
  8. dense intermediate == 17408
  9. native Windows proof
  10. approved runtime-lock identity
  11. capture shard hashes
  12. selected record/token identities
  13. no benchmark-derived material
  14. no validation/holdout contamination
  15. no synthetic evidence class
  16. topology exactly p16/top4
  17. sufficient requested token/sample count exists

Any mismatch:
  return BLOCKED
  perform zero optimizer steps

======================================================================
DELIVERABLE D — PHASE ORCHESTRATION MUST FAIL CLOSED
======================================================================

Inspect:

  src/dense2moe/phase.py

and all relevant phase-state / receipt / promotion code.

Change Phase 01 so that it cannot become method-proof green from:

  evidence_class = synthetic-smoke

or old synthetic status names.

Phase 01 scientific acceptance must require a validated:

  evidence_class = real-qwen-layer-capture

plus the real method-proof training/result receipt.

Add explicit states such as:

  PHASE_01_BLOCKED_NO_REAL_CAPTURE
  PHASE_01_BLOCKED_INVALID_CAPTURE
  PHASE_01_REAL_METHOD_PROOF_RUNNING
  PHASE_01_REAL_METHOD_PROOF_GREEN
  PHASE_01_REAL_METHOD_PROOF_FAILED

Exact naming may follow repo conventions, but semantics must be obvious.

A synthetic smoke result should be informational only.

Do not allow:
  synthetic rows >= threshold
to satisfy:
  real sample/token threshold

Add regression tests that specifically attempt to pass synthetic smoke receipts
into Phase 01 and assert fail-closed behavior.

======================================================================
DELIVERABLE E — PHASE 00A WINDOWS QUALIFICATION
======================================================================

Preserve and use the new runtime-lock system.

Inspect:
  src/dense2moe/hardware.py
  scripts/Invoke-GuardedCommand-Smoke.ps1
  environment doctor / runtime lock commands

On native Windows, before real scientific capture:

1. Verify clean git state.

2. Run the guarded-command smoke suite.

Require the expected cases to be green:
  success
  nonzero failure
  timeout + descendant kill
  heartbeat/progress
  git-log command

3. Run the environment doctor.

Required capability probes must include the established set such as:
  native_windows
  python
  torch_import
  torch_version
  cuda_runtime
  cuda_available
  expected_gpu
  bf16_tensor
  small_cuda_gemm
  safetensors_load
  d2m_checkpoint_load
  p16_forward
  dense_teacher_ffn_forward
  oracle_module_import
  source_checkpoint_readable

4. Use ONLY:
     C:\workplace\llm_playground\.venv\Scripts\python.exe

for authoritative D2M execution.

5. Create the approved runtime lock only when the doctor is green.

6. Persist:
  runtime fingerprint
  Python version
  Torch version
  compiled CUDA
  driver
  GPU inventory
  selected training GPUs
  BF16 capability
  package versions
  receipt hashes
  exact code commit

7. Do not use the historical recovery pin as current proof.
   It is recovery guidance only.

If the runtime differs after lock creation:
  stop with WINDOWS_RUNTIME_DRIFT

======================================================================
DELIVERABLE F — PHASE 00B METHOD_PROOF_ONLY DATA
======================================================================

Preserve the existing strict METHOD_PROOF_ONLY policy.

Use:

  scripts/prepare_method_proof_data.py

and existing:
  src/dense2moe/method_proof.py

The method-proof subset must remain:

  FIT-TRAIN only
  benchmark-free
  provenance retained
  evaluation identities excluded
  overlap checked
  byte/hash verified

Minimum:
  32768 tokens

Require meaningful code + technical diversity under the current policy.

Do NOT falsely promote this subset to production Corpus V2.2.

It is only:
  METHOD_PROOF_ONLY

Persist:
  manifest
  selected IDs
  selected rows
  selected tokens
  domain/diversity summary
  source manifest hash
  receipt SHA
  exclusion hashes

======================================================================
DELIVERABLE G — REAL LAYER-0 CAPTURE
======================================================================

After Phase 00A and 00B are green on native Windows:

Capture ONLY what is required for layer 0.

Do not run uncontrolled 64-layer replay.

Use the established exact Qwen layer-streaming teacher path.

Capture from the exact frozen METHOD_PROOF_ONLY records.

The source layer-0 dense FFN must be the real Qwen FFN:

  gate_proj
  up_proj
  SiLU/SwiGLU
  down_proj

Generate real:
  X = layer-0 FFN inputs
  Y = dense layer-0 FFN outputs

and any contribution tensors required by the p16 basis method.

Validate the dense teacher FFN computation against the already established
native/reference path before optimizing.

Persist all hashes and shape metadata.

======================================================================
DELIVERABLE H — 2K -> 4K -> 32K REAL P16 METHOD PROOF
======================================================================

Only after the real capture receipt is validated:

Run three bounded stages.

Stage 1:
  approximately 2048 real captured tokens/samples

Purpose:
  implementation/numerical sanity only

Require:
  finite loss/metrics
  no NaN/Inf
  selector remains frozen
  router parameters remain unchanged where oracle-routed basis refinement
  requires that
  basis parameters actually change
  receipt hashes verify
  output checkpoint reloads
  deterministic/equivalent rerun where appropriate

Stage 2:
  approximately 4096 real captured tokens/samples

Purpose:
  verify improvement direction repeats

Compare:
  initial reconstruction
  post-refinement reconstruction
  oracle quality
  load statistics

Stage 3:
  >= 32768 real method-proof tokens/samples

Purpose:
  bounded scientific method proof

For every stage record at minimum:

  token/sample count
  source record identity hash
  topology
  layer
  source revision
  checkpoint hash
  partition hash
  code commit

  initial:
    global NMSE
    cosine
    mean-token-relative-MSE
    dead experts
    loadCV

  final:
    global NMSE
    cosine
    mean-token-relative-MSE
    dead experts
    loadCV

  oracle:
    assurance level
    candidate-set strategy
    candidate count
    global NMSE
    cosine
    loadCV
    whether coefficient fitting is exact or projected/approximate

  optimization:
    epochs
    steps
    learning rate
    assignment refresh
    candidate pool settings
    wall time
    peak RAM if measurable
    peak VRAM if measurable

  telemetry:
    expert utilization
    routing entropy/margin if applicable
    shared output norm
    routed output norm
    shared:routed ratio
    reconstruction norm
    target norm

Do not label projected-positive coefficient fitting as "exact".

Use honest names:
  exhaustive-set projected-positive oracle
  bounded screening oracle
  unconstrained oracle
as applicable.

======================================================================
SCIENTIFIC DECISION FOR PHASE 01
======================================================================

The central question is:

  Does selector-independent p16/top4 basis refinement on REAL Qwen layer-0
  activations produce a credible and repeatable reconstruction improvement,
  and is the resulting basis/oracle ceiling plausibly compatible with the
  eventual product gates?

Do not require the final learned selector to be solved in Phase 01.

Do require:
  real-data evidence
  repeated direction of improvement
  no numerical instability
  no expert collapse
  clear oracle/basis headroom measurements
  provenance-safe results

At 32k, report both:

A. Method-proof decision:
  GREEN / YELLOW / FAILED

B. Product-gate comparison:
  NMSE vs 0.05
  cosine vs 0.98
  dead experts vs 0
  loadCV vs 0.50

Do not call the overall D2M product green based on Phase 01.

======================================================================
PRODUCTION CORPUS REMAINS BLOCKED
======================================================================

Do NOT weaken Corpus V2.1/V2.2 requirements to make Phase 02 pass.

Current known V2.1 deficiencies include:
  insufficient independent non-benchmark agent tasks
  acquisition/rebalancing requirements

The method-proof path exists specifically so algorithm work can continue while
production-corpus acquisition remains blocked.

Phase 01 success must not erase Phase 02 corpus requirements.

======================================================================
DO NOT DO THESE THINGS
======================================================================

Do NOT:
  - tune on historical holdout
  - open historical holdout
  - use validation B/C as optimizer data
  - resume rolling replay
  - touch layer30+ replay
  - run representative 12 layers
  - run full64 conversion
  - train p32/top5
  - reactivate p32/top4
  - change source checkpoint/revision
  - alter attention
  - optimize reasoning length
  - modify generation policy
  - claim synthetic evidence is scientific evidence
  - fabricate Windows/CUDA receipts
  - silently use Linux/WSL as authoritative fallback

======================================================================
TEST REQUIREMENTS
======================================================================

Add focused unit/integration tests covering at least:

1. synthetic smoke runs
2. synthetic smoke is marked non-promotable
3. Phase 01 rejects synthetic receipt
4. Phase 01 rejects wrong model
5. Phase 01 rejects wrong revision
6. Phase 01 rejects wrong model type
7. Phase 01 rejects layer != 0
8. Phase 01 rejects hidden != 5120
9. Phase 01 rejects intermediate != 17408
10. Phase 01 rejects invalid shard hash
11. Phase 01 rejects missing runtime lock
12. Phase 01 rejects Windows runtime drift
13. Phase 01 rejects evaluation/benchmark contamination
14. real capture receipt accepts a deterministic small fixture
15. mmap/sharded input loading does not eagerly materialize entire dataset
16. optimizer does zero steps on any failed preflight gate
17. real runner distinguishes tokens/samples from raw rows
18. checkpoint/result receipt reload and hash verification
19. method-proof receipt remains FIT-TRAIN-only
20. phase state machine cannot skip required prerequisites

Keep tests lightweight by using explicit fixture receipts and tiny tensor
fixtures where real Qwen execution is unnecessary.

Real Qwen/CUDA tests should be separately marked/invoked.

======================================================================
DOCUMENTATION UPDATES
======================================================================

Update:
  docs/REAL_D2M_PLAN.md

and any other affected docs.

Clearly distinguish:

  SYNTHETIC SMOKE
    implementation-only
    not scientific evidence

from:

  REAL METHOD PROOF
    actual Qwen layer-0 capture-backed evidence

Document canonical Windows commands for:

  Phase 00A runtime qualification
  Phase 00B method-proof data preparation
  real layer-0 capture
  p16 2k
  p16 4k
  p16 32k

Remove or correct any example implying that:

  run_oracle_routed_basis_refinement.py --rows 32768

on the tiny fixture is a real p16 method proof.

======================================================================
IMPLEMENTATION QUALITY
======================================================================

Prefer reuse over parallel infrastructure.

Before adding new code, inspect:
  scripts/
  src/dense2moe/
  src/dense2moe/training/
  existing streaming_teacher / teacher capture implementation
  existing capture manifests
  existing provenance utilities
  existing hash/receipt utilities
  existing partition loaders
  existing mmap contribution-store code

Avoid duplicate:
  hashing
  receipt validation
  runtime checking
  model geometry constants
  teacher FFN math

Centralize contracts when reasonable.

Use atomic receipt writes where existing utilities support them.

Use mmap/streaming for large arrays.

Do not introduce a "quick workaround" that creates another ambiguous path.

======================================================================
COMMIT PLAN
======================================================================

Keep the work reviewable.

Suggested commits:

Commit 1:
  "Separate synthetic oracle smoke from method proof"

Contents:
  runner rename/refactor
  synthetic evidence class
  phase rejection
  tests

Commit 2:
  "Add real Qwen layer0 method-proof capture contract"

Contents:
  capture receipt schema
  validation
  streaming/mmap integration
  tests

Commit 3:
  "Add capture-backed p16 method-proof runner"

Contents:
  real runner
  preflight gate
  metrics/receipts
  tests

Commit 4:
  "Wire real method proof into phase orchestration"

Contents:
  phase states
  prerequisite validation
  docs
  tests

Do not squash these unless there is a strong reason.

After each commit:
  git status --short
  run focused tests

Before scientific execution:
  full relevant test suite
  clean tree
  record git SHA

======================================================================
NATIVE WINDOWS EXECUTION SEQUENCE
======================================================================

Once implementation is committed and tests are green, and ONLY if actually
running on native Windows with the source checkpoint/captures available:

1. Confirm clean state:
     git status --short
     git rev-parse HEAD

2. Run guarded-command Windows smoke.

3. Run environment doctor.

4. Create/verify approved runtime lock.

5. Prepare METHOD_PROOF_ONLY data:
     & .\.venv\Scripts\python.exe scripts\prepare_method_proof_data.py `
       --corpus-manifest data\public_v21\corpus-v2.1.jsonl `
       --output <NEW_PHASE_00B_RUN>\method-proof `
       --min-tokens 32768 `
       --json

6. Verify receipt hashes.

7. Capture REAL Qwen layer-0 activations from those exact records using the
   new/reused capture path.

8. Verify capture receipt/shards.

9. Run real p16/top4 2k method proof.

10. Review metrics.
    If numerically broken or directionally bad:
      STOP and diagnose.

11. Run real p16/top4 4k method proof.

12. Review 2k vs 4k.
    If improvement fails to repeat:
      STOP and diagnose.

13. Run real p16/top4 >=32k method proof.

14. Produce a final Phase 01 method-proof synthesis report.

15. DO NOT continue to Phase 02 automatically.

======================================================================
FINAL REPORT REQUIRED
======================================================================

When finished, provide a concise but complete handoff containing:

1. Final git SHA
2. Commit list created
3. Files changed
4. Tests run and results
5. Whether execution environment was native Windows
6. Runtime lock status/hash
7. Method-proof data receipt/hash
8. Real layer-0 capture receipt/hash
9. Capture tensor shapes/counts
10. 2k metrics
11. 4k metrics
12. 32k metrics
13. Before/after reconstruction comparison
14. Oracle ceiling comparison
15. Expert load/dead-expert summary
16. Any resource/runtime observations
17. Phase 01 decision:
      GREEN
      YELLOW
      FAILED
      or BLOCKED
18. Exact blocker if not green
19. Explicit confirmation:
      historical holdout was NOT opened
      historical replay was NOT resumed
      p32 was NOT run
      representative layers were NOT run
      full64 was NOT run

======================================================================
AUTONOMY
======================================================================

Proceed autonomously through bounded implementation and tests.

Do not ask for permission between normal coding/test steps.

Pause only for:
  - destructive action
  - source/provenance ambiguity
  - unexpected historical artifact mutation
  - inability to identify the existing exact teacher path
  - native Windows/runtime-lock failure
  - missing pinned source checkpoint
  - major GPU/resource blocker
  - scientific result that invalidates the planned next stage

The priority is not to make the pipeline appear green.

The priority is to make it impossible for synthetic evidence to masquerade as
real Qwen evidence, then obtain the first provenance-safe, capture-backed,
real layer-0 p16/top4 method proof.