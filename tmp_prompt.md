D2M / QWEN3.8 MOE — FEATURE-COMPLETION EPIC
GENERALIZATION-FIRST DISTILLATION → ROBUST BF16 MOE → QUANTIZED MODEL

NSP execution:
- Use /nsp-build-bezalel as the primary implementation/execution skill.
- Use /nsp-plan-genesis only when material discovery invalidates this plan or requires the execution contract to be regenerated.
- This is EPIC-scale work.
- Use .agents/skills/nsp-epic-execution/SKILL.md for EPIC Ralph/PIV/ATDD execution.
- Rehydrate from repository and Ralph artifacts, never from chat history or the Prompt Genesis conversation.
- Do not begin implementation or edit source until Ralph state and explicit active-phase ATDD gates with exact validation commands exist.

Repository:
    C:\workplace\llm_playground

Branch:
    agent/windows-dense2moe-real-pipeline

Observed HEAD when this prompt was authored:
    3d7dd69fcaf2f885169de35a20d05b2e43cc524e

Do not assume that SHA is still current.
Fetch/pull and record actual HEAD first.

======================================================================
0. PRIMARY OUTCOME
======================================================================

Drive D2M from its current research state to a reproducible feature-complete
pipeline capable of producing:

    dense Qwen3.8 source
        ->
    sparse FFN-distilled MoE
        ->
    validated BF16 MoE checkpoint
        ->
    quantized MoE artifact
        ->
    measured production candidate

while first proving that the distillation/training method learns a
GENERALIZABLE sparse basis rather than overfitting another activation corpus.

This epic must produce progress toward an actual trained model.

Infrastructure work is allowed only when it removes a concrete blocker from:

    corpus
    teacher capture
    distillation
    training
    assembly
    evaluation
    quantization

Do not allow the project to return to open-ended diagnostic research.

======================================================================
1. PRODUCT TARGETS
======================================================================

Exactly TWO sparse topologies remain active.

SAFE FALLBACK:

    p16/top4
    16 routed experts
    routed width 1024
    shared width 1024
    top_k = 4
    active FFN width = 5120
    ~70.59% FFN reduction

PRIMARY PRODUCT TARGET:

    p32/top5
    32 routed experts
    routed width 512
    shared width 1024
    top_k = 5
    active FFN width = 3584
    ~79.41% FFN reduction

Do NOT resume research on p32/top4.

p16/top4 is the minimum viable completion path.

p32/top5 is the preferred product target.

CRITICAL DELIVERY RULE:

    p32/top5 MUST NOT indefinitely block creation of a complete p16/top4
    BF16 + quantized model.

Once p16/top4 is robustly viable, proceed toward full-model p16 conversion
while p32/top5 research continues if needed.

======================================================================
2. PRESERVE QWEN BACKBONE
======================================================================

The project remains an FFN-only Dense-to-MoE conversion.

Preserve:

    attention
    Gated DeltaNet / linear attention
    full-attention blocks
    RoPE
    norms
    residual topology
    embeddings
    LM head
    tokenizer
    special tokens
    chat template
    generation semantics

Replace only dense SwiGLU FFNs.

Do not introduce reasoning-efficiency post-training.

Do not train a shorter-reasoning policy.

======================================================================
3. PRODUCTION QUALITY GATES
======================================================================

Layer-level green remains:

    cosine >= 0.98
    global NMSE <= 0.05
    load CV <= 0.50
    dead experts = 0

Do not weaken these because a new corpus is harder.

Whole-model evaluation later must preserve the established project gates:

    perplexity increase:
        green <= 5%
        yellow <= 10%

    token KL:
        green <= 0.10
        yellow <= 0.20

    top-1 agreement:
        >= 85%

Exact conversion / equivalence sanity where applicable:

    MSE <= 1e-8

Quantization must be evaluated INCREMENTALLY against the validated BF16 MoE,
in addition to evaluating final quality against the dense source model.

======================================================================
4. NSP REQUIRED START / DISCOVERY GATE
======================================================================

Before broad repository loading or edits, run the NSP substrate.

Required commands:

    _nsp status --target C:\workplace\llm_playground

    _nsp context select \
        --target C:\workplace\llm_playground \
        --request "Drive D2M Qwen3.8 MoE from Corpus V2 through generalizable oracle-routed distillation, p16/p32 training, full-model assembly, validation, and quantization" \
        --limit 8 \
        --format json \
        --receipt-v2

    _nsp run list --target C:\workplace\llm_playground

    _nsp plan-substrate \
        --target C:\workplace\llm_playground \
        --milestone "feature-complete robust Qwen3.8 Dense-to-MoE conversion and quantized candidate"

For this EPIC, start or join an NSP run and record RUN_ID.

Then:

    _nsp plan-substrate discovery seed \
        --target C:\workplace\llm_playground \
        --run-id <RUN_ID>

Persist the canonical Repository Fact Ledger under:

    .nsp/artifacts/runs/<RUN_ID>/planning/repository-fact-ledger.json

Then validate:

    _nsp plan-substrate discovery validate \
        --target C:\workplace\llm_playground \
        --path .nsp/artifacts/runs/<RUN_ID>/planning/repository-fact-ledger.json

Do not proceed into source edits until agent-owned semantic review concludes:

    DISCOVERY_READY

Required fact coverage includes at minimum:

    actual branch HEAD
    current run state
    current corpus hashes
    current split hashes
    current teacher-capture implementation
    current oracle-refinement implementation
    current p16 checkpoint
    current p32 checkpoint
    current source checkpoint/revision
    Python/Torch/CUDA environment used by prior successful Windows runs
    model assembly path
    model export/inference path
    quantization-capable runtime/backend candidates
    applicable tests
    current holdout/replay closure

Rejected or stale facts must not silently authorize execution.

If material facts are unknown/conflicted:

    DISCOVERY_BLOCKED

and resolve them before continuing.

======================================================================
5. EPIC / RALPH OWNERSHIP
======================================================================

Epic id:

    epic-d2m-qwen38-moe

Ralph state:

    .nsp/artifacts/tmp/ralph/epic-d2m-qwen38-moe/

This Ralph state is the in-flight source of truth.

Never use chat memory as execution state.

Never write in-flight epic state under .docs/roadmap/**.

Every phase must contain:

    objective
    exact acceptance gates
    exact validation commands
    prediction depth
    expected artifacts
    actual observations
    Prediction Result classifier
    residual risks
    next-phase eligibility

Use clean-context implementation and validation envelopes where supported.

Do not mark any phase complete without fresh evidence.

======================================================================
6. KNOWN CURRENT STATE TO VERIFY
======================================================================

Treat these as expected facts requiring repository verification:

- Corpus V2 exists and is frozen.
- It currently contains 676 records.
- A balanced activation plan targets ~750k tokens.
- Planned capture mixture is approximately:

      code                         44%
      agentic SWE                 28%
      SWE natural language        12%
      structured/tool              6%
      general                     10%

- teacher capture for Corpus V2 has NOT started.
- oracle-routed basis refinement has been implemented.
- p16 can use exhaustive C(16,4)=1820 route search.
- p32 uses a bounded candidate-pool oracle.
- current ML execution is blocked because the environment used by the last
  agent could not import usable PyTorch.
- official holdout remains closed for the new run.
- representative replay remains blocked.
- full64 replay remains blocked.

Also verify/fix known provenance drift:

    child-run state/HANDOFF may record an older source commit than actual HEAD.

No decisive experiment may start with stale provenance.

======================================================================
7. PHASE 0 — CORPUS V2.1 + ENVIRONMENT RECOVERY
======================================================================

Objective:

    create a final production-oriented training/evaluation corpus and restore
    a deterministic native-Windows ML environment.

This is the final planned corpus revision before real distillation unless a
later independent gate demonstrates a concrete coverage failure.

--------------------------------------------------
7A. CORPUS V2.1
--------------------------------------------------

Do NOT mutate frozen Corpus V2 in place.

Derive:

    Corpus V2.1

Preserve V2 hashes and receipts.

A. QUARANTINE BENCHMARK-DERIVED TASKS

Current corpus evidence includes SWE-Bench-style / SWE-rebench-derived tasks.

Move all benchmark-derived records outside:

    FIT-TRAIN
    FIT-DEV
    GATE-A
    SHADOW-B
    SHADOW-C

Use an explicit bucket such as:

    BENCHMARK-CANARY-EXCLUDED

They must never participate in:

    gradients
    checkpoint selection
    shadow promotion

Keep provenance for diagnostic use only.

B. INCREASE INDEPENDENT AGENT TASK DIVERSITY

Current V2 contains too few complete agent trajectories for the importance
assigned to that domain.

Target:

    >= 96 independent non-benchmark agent tasks

Prefer:

    ~128-200 tasks

if acquisition is straightforward.

Do not endlessly grow the corpus once diversity requirements are met.

Favor:

    more independent tasks
    more repositories
    more languages/frameworks

over:

    more tokens from the same task

Suggested activation cap per agent task:

    approximately 2k-4k sampled tokens

unless evidence supports a different bounded cap.

Hard concentration checks:

    no single task dominates agentic activation sampling
    no single repository dominates production sampling
    no source family overwhelms the balanced sampler

Retain complete raw visible trajectories for provenance; sample bounded windows
for activation capture.

C. REPOSITORY/TASK/DOCUMENT DISJOINTNESS

Maintain:

    FIT-TRAIN
    FIT-DEV
    GATE-A
    SHADOW-B
    SHADOW-C
    PRESERVATION-CANARY

Require zero forbidden overlap at:

    repository
    task/issue
    document

level.

Do not use random token splitting as generalization evidence.

D. PRODUCTION COVERAGE

Maintain production-weighted diversity:

    code
    agentic coding
    tool calls/results
    shell
    git
    compiler/test output
    structured formats
    technical docs
    broad/general preservation

Preserve the old Wiki/Gutenberg/general corpus only as an OOD/regression
canary, not as primary optimization data.

E. FREEZE

Produce immutable:

    corpus manifest
    splits
    source/provenance receipt
    benchmark exclusion receipt
    exact tokenizer audit
    activation sampling plan

with hashes.

--------------------------------------------------
7B. WINDOWS ML ENVIRONMENT
--------------------------------------------------

Do NOT blindly install whatever current torch package pip selects.

First identify the exact native-Windows Python/Torch/CUDA environment or
dependency combination that previously produced successful D2M GPU runs.

Recover/reuse it where possible.

Then pin enough information to reproduce it.

Create or extend an environment doctor that proves:

    Python executable/version
    torch import
    torch version
    CUDA runtime
    torch.cuda.is_available() == True
    expected GPU visible
    BF16 tensor operation
    small CUDA GEMM
    safetensors load
    existing D2M checkpoint load
    one p16 forward
    one dense-teacher FFN forward
    oracle module import

Run complete relevant ML tests after recovery.

--------------------------------------------------
PHASE 0 ATDD EXIT
--------------------------------------------------

Required:

    Corpus V2.1 frozen
    benchmark-derived tasks excluded from optimization/promotion
    >=96 independent non-benchmark agent tasks unless a documented acquisition
      blocker is accepted by the epic owner
    overlap audits green
    tokenizer audit green
    balanced capture plan green
    environment doctor green
    complete ML test collection green
    actual HEAD/provenance reconciled

No teacher training campaign before this gate.

======================================================================
8. PHASE 1 — DISTILLATION METHOD PROOF
======================================================================

Objective:

    prove oracle-routed basis training works scientifically before committing
    large compute.

Use p16/top4 as the method-proof topology because its top4 routing can be
exhaustively evaluated over all 1,820 expert sets.

--------------------------------------------------
8A. TEACHER CAPTURE
--------------------------------------------------

Start small.

Capture balanced V2.1 teacher activations at approximately:

    2k-4k states

Verify:

    source revision
    tokenizer
    hidden geometry
    dense FFN target
    capture hashes
    split identity

Teacher output reconstruction/equivalence must satisfy existing strict
numerical sanity requirements.

Do not generate the source corpus autoregressively with the dense model.

External fixed trajectories supply contexts.

Dense Qwen supplies:

    hidden states
    dense FFN outputs
    reconstruction target

--------------------------------------------------
8B. ORACLE METHOD VERIFICATION
--------------------------------------------------

Prove that:

    learned selector is never consulted for E-step assignments
    selector parameters receive no basis-training gradients
    p16 E-step is exhaustive
    route coefficients obey intended constraints
    M-step uses frozen E-step assignments
    assignments can refresh as basis changes
    checkpoint save/reload preserves results

Add change-sensitive tests if absent.

--------------------------------------------------
8C. SMALL REAL PILOT
--------------------------------------------------

Run:

    ~2k-4k p16 oracle-routed smoke

Then:

    ~32k p16 production-balanced GPU pilot

Use:

    shared-foundation
        ->
    routed-residual
        ->
    joint refreshable basis refinement

Primary measurement:

    trained-basis UNCONSTRAINED oracle quality

Selector is telemetry only.

--------------------------------------------------
METHOD-PROOF FALSIFIER
--------------------------------------------------

Do not accept another +0.0004-style result as evidence of success.

Method proof is green if either:

A. the initial V2.1 basis is already near capacity:
       oracle cosine >= ~0.97
       and NMSE trajectory is healthy

OR

B. the 32k pilot produces material improvement such as:
       approximately +0.01 absolute oracle cosine
       AND meaningful NMSE reduction

with no material collapse on source-disjoint A sampling or preservation
canaries.

If neither occurs:

    STOP SCALING.

Diagnose:

    shared branch
    residual target
    partition initialization
    oracle assignment quality
    route coefficient fitting
    optimizer/LR
    gradient normalization
    expert specialization
    candidate search

One bounded hypothesis at a time.

Do not spend 750k-state compute on an unproven method.

======================================================================
9. PHASE 2 — P16/TOP4 ROBUST TRAINING
======================================================================

Objective:

    produce the safe-fallback layer-0 candidate and LOCK the training recipe.

Progression:

    32k pilot
        ->
    ~128k serious run
        ->
    larger balanced FIT / up to planned ~750k
        only while trajectory remains useful

Do not blindly consume the largest dataset.

Use hard-token mining only from FIT.

--------------------------------------------------
9A. QUALITY FIRST
--------------------------------------------------

First solve unconstrained reconstruction.

Target:

    oracle cosine >= .98
    oracle NMSE <= .05

Do NOT heavily optimize load balance while oracle cosine remains far below
~.97.

--------------------------------------------------
9B. JOINT QUALITY/LOAD
--------------------------------------------------

Once reconstruction is close:

introduce load-friendly basis pressure / balanced oracle assignment.

Require simultaneously:

    cosine >= .98
    NMSE <= .05
    CV <= .50
    dead experts = 0

Load metrics must be measured on meaningful sample counts, not 16 tokens.

--------------------------------------------------
9C. SELECTOR
--------------------------------------------------

Only after oracle joint capacity is proven:

    freeze basis

Cache strong oracle labels on diverse FIT:

    route IDs
    coefficients
    reconstruction regret
    cosine regret

Train selector against the FINAL basis.

Weight routing mistakes by actual output damage.

FIT-DEV:
    frequent feedback allowed

GATE-A:
    milestone selection

SHADOW-B:
    confirmation only

SHADOW-C:
    final independent confirmation

If B or C causes a method change:

    that shadow is now diagnostic
    it is no longer untouched evidence

Use another still-untouched shadow for final promotion.

--------------------------------------------------
9D. GENERALIZATION
--------------------------------------------------

Report aggregate and per:

    source family
    task family
    language
    repository group

Track:

    worst-domain cosine
    domain spread
    FIT-DEV -> A gap
    A -> B gap
    B -> C gap

Flag:

    independent-cohort cosine drop > ~.01
    important production domain < ~.975 when aggregate is near green

Do not call the model "close" because one familiar split is green.

--------------------------------------------------
P16 ROBUST-GREEN
--------------------------------------------------

P16 is robust-green only when:

    oracle capacity green
    load green
    selector green
    independent A/B/C evidence supports generalization
    preservation canary shows no unexplained catastrophic collapse

Only then may p16 advance toward representative-layer transfer.

======================================================================
10. PHASE 3 — P32/TOP5 PRIMARY PRODUCT
======================================================================

Do not make p32 rediscover the successful p16 recipe from scratch.

After p16 methodology is proven:

transfer:

    shared 1024 branch

Investigate structured initialization:

    each p16 1024-wide routed expert
        ->
    two p32 512-wide experts

Prefer contribution/neuron-aware splitting over arbitrary halves.

Then run p32-specific:

    top5 oracle assignment
    shared/residual/joint refinement
    load geometry
    selector refinement

Because p32 uses bounded oracle search:

    audit candidate-pool adequacy on small samples

Periodically expand candidate pools where practical.

Do not interpret a bounded-search failure as architecture impossibility
without a coverage check.

--------------------------------------------------
P32 DECISION
--------------------------------------------------

If:

    oracle >= .98 / <= .05

then proceed aggressively.

If:

    .975-.98

continue bounded refinement.

If:

    remains ~.92-.94

after:

    successful p16-derived initialization
    meaningful production-balanced GPU training
    candidate-pool adequacy checks

then document credible topology-capacity concern.

p32 research may continue, but p16 completion must proceed.

======================================================================
11. PHASE 4 — TRAINING METHOD LOCK
======================================================================

When p16 is robust-green and p32 status is understood, freeze a reproducible
training specification.

Lock:

    corpus version
    split hashes
    activation sampler
    per-domain caps
    partition initialization
    shared/residual/joint stage schedule
    E-step refresh schedule
    coefficient solver
    optimizer
    LR
    losses
    oracle configuration
    load policy
    selector architecture
    selector loss
    checkpoint-selection rule
    validation cadence
    quality gates

Publish:

    TRAINING_METHOD_LOCK

After this point:

    representative layers validate TRANSFERABILITY

They are not a new architecture-search playground.

No arbitrary layer-by-layer hyperparameter invention.

======================================================================
12. PHASE 5 — REPRESENTATIVE LAYER MATRIX
======================================================================

Once p16 training method is locked, test exactly:

    0  1  2  3
    28 29 30 31
    60 61 62 63

Record:

    attention_type
    cycle_position = layer % 4
    cycle_index = layer // 4

Cycle mapping:

    mod4 0 = LINEAR_A
    mod4 1 = LINEAR_B
    mod4 2 = LINEAR_C
    mod4 3 = FULL_ATTENTION

Require the locked method to work across:

    early
    middle
    late
    all attention-cycle classes

Prefer:

    one uniform topology

or at most:

    simple attention-cycle-class decisions justified by evidence

Avoid bespoke per-layer designs.

--------------------------------------------------
REPRESENTATIVE EXIT
--------------------------------------------------

The p16 fallback advances if representative evidence demonstrates that the
locked training method transfers adequately across the 12-layer matrix.

Do not require p32 to block p16 full conversion.

======================================================================
13. PHASE 6 — FULL 64-LAYER P16 CONVERSION
======================================================================

This is the minimum viable model-completion path.

Once representative p16 is green:

    START FULL64 P16.

Do not automatically train every layer on 750k states.

Use representative experiments to determine the SMALLEST sufficient
production-balanced training budget.

Suggested strategy:

    normal layer budget ~64k-128k
    hard/problem layers escalate selectively
    hard-token continuation only where needed

Avoid wasting full 750k-scale optimization on easy layers.

Use the existing resumable layer-major teacher/capture architecture.

Maintain:

    clean commit
        ->
    guarded execution
        ->
    receipt
        ->
    checkpoint hash
        ->
    report commit

for decisive runs.

No science solely in an uncommitted tree.

======================================================================
14. PHASE 7 — BF16 MODEL ASSEMBLY
======================================================================

Assemble a complete unquantized sparse model first.

Preserve all non-FFN tensors exactly unless required format conversion is
proven equivalent.

Verify:

    all 64 FFNs replaced as intended
    topology inventory
    tensor inventory
    model config
    tokenizer/chat template
    generation config
    checkpoint load
    representative forward tests
    exact preserved tensor hashes where practical

Produce a canonical:

    BF16_SPARSE_MASTER

Do not quantize an unvalidated research checkpoint.

======================================================================
15. PHASE 8 — WHOLE-MODEL VALIDATION
======================================================================

Evaluate BF16_SPARSE_MASTER against dense source.

Include:

A. MODEL-DISTRIBUTION METRICS

    perplexity delta
    token KL
    top1 agreement

B. CODING / AGENTIC BEHAVIOR

Use tasks/repositories that are absent from Corpus V2.1 optimization splits.

Cover:

    code generation
    bug fixing
    repository navigation
    multi-file edit
    tests
    compiler/runtime failure
    retry/recovery
    terminal interaction
    JSON/tool calls
    long-context code
    multi-turn agent sequences

C. GENERAL PRESERVATION

Use general/STEM/OOD canaries.

Do not optimize against the official holdout.

Official holdout becomes eligible only after robust pre-holdout evidence.

Use it once for finalist confirmation under the established project policy.

======================================================================
16. PHASE 9 — QUANTIZATION READINESS IN PARALLEL
======================================================================

Do not wait until the final day to discover that the target runtime cannot
represent the new MoE.

Once PHASE 2 method proof is green, start a PARALLEL NON-BLOCKING engineering
track for:

    sparse model serialization
    runtime support
    quantization backend compatibility
    expert tensor layout
    router precision requirements
    shared-expert precision requirements

Do NOT spend large compute quantizing research candidates.

This track should answer:

    what inference runtime will load this sparse architecture?
    what quantized formats can represent the topology?
    what tensors must remain BF16/FP16?
    what tooling changes are needed?

Avoid binding the project to GGUF, GPTQ, AWQ, EXL2, or another format before
repository/runtime discovery establishes compatibility.

======================================================================
17. PHASE 10 — QUANTIZATION
======================================================================

Once BF16_SPARSE_MASTER passes whole-model gates:

freeze it.

Quantize from that exact checkpoint.

Start conservatively:

    higher precision / Q8-like baseline if supported

then test more aggressive expert-weight quantization.

Prefer keeping numerically sensitive components higher precision initially:

    router
    routing coefficients/scales
    norms
    other tiny/high-sensitivity tensors

Measure BOTH:

    Dense -> BF16 MoE degradation

and

    BF16 MoE -> Quantized MoE incremental degradation

Do not hide distillation loss inside quantization loss.

Produce at least one practical quantized candidate that remains inside the
accepted whole-model quality envelope.

======================================================================
18. FEATURE-COMPLETION DEFINITION
======================================================================

D2M is FEATURE-COMPLETE when a clean checkout can reproducibly execute the
full path:

    source model verification
        ->
    corpus verification
        ->
    balanced teacher capture
        ->
    oracle-routed basis refinement
        ->
    load refinement
        ->
    selector training
        ->
    representative validation
        ->
    full64 conversion
        ->
    sparse model assembly
        ->
    BF16 validation
        ->
    quantization
        ->
    quantized validation

without inventing a new ad-hoc script for every layer/phase.

Integrate with the project's existing control plane where appropriate rather
than creating unnecessary parallel orchestration.

All major stages must be:

    resumable
    receipt-bearing
    hash/fingerprint aware
    failure-safe
    provenance aware

======================================================================
19. RESEARCH-BUDGET / DRIFT RULES
======================================================================

Every research run must do at least one:

    materially improve a candidate
    falsify a specific hypothesis
    remove a blocker on the path to the finished model

Do not perform repetitive no-op refinements.

A failed experiment must record:

    hypothesis
    prediction
    result
    falsifier outcome
    next decision

Maximum two materially similar retries without a new hypothesis.

Do not respond to a failed shadow by repeatedly tuning on that shadow.

Do not let infrastructure polishing consume multiple passes once environment
and corpus gates are green.

EXPECTED PACING:

Pass 1:
    V2.1 + environment + tests + small capture/oracle smoke

Pass 2:
    real 32k p16 method-proof experiment

Pass 3:
    128k p16 if green trajectory

Pass 4:
    larger p16 run / joint-load work

Then:
    selector + robust promotion
    p32 transfer
    representative layers
    full64

If execution diverges materially from that pacing, document why.

======================================================================
20. PARALLELISM
======================================================================

Once a phase has stable contracts, use independent work packages where useful.

Potential parallel tracks:

A. DISTILLATION SCIENCE
    p16 / p32 basis + selector

B. PRODUCTIZATION
    assembly / serialization / runtime compatibility

C. EVALUATION
    clean coding-agent and whole-model harness

D. QUANTIZATION READINESS
    format/runtime spike

Do not let parallel productization mutate the scientific training contract.

Use path claims/run isolation under NSP.

======================================================================
21. REQUIRED CLOSEOUT ARTIFACTS PER PHASE
======================================================================

Every phase closeout records:

    actual code HEAD
    clean/dirty status
    commands
    tests
    data hashes
    checkpoint hashes
    metrics
    prediction result
    gate decision
    shadow sets opened
    holdout opened
    replay started
    residual risks
    exact next phase

Do not claim:

    fixed
    complete
    green
    ready

without fresh evidence.

======================================================================
22. HOLDOUT / REPLAY RULES
======================================================================

Until explicitly promoted:

    official holdout = CLOSED
    historical opened holdout = IMMUTABLE / NO TUNING
    representative replay = BLOCKED
    full64 replay = BLOCKED

Representative replay unblocks only after:

    robust p16 method + training lock

Full64 p16 unblocks only after:

    representative layer gate

Official holdout unblocks only for a selected whole-model finalist under the
existing project policy.

======================================================================
23. PLANNING / PREDICTION DISCIPLINE
======================================================================

For each phase during Plan:

choose prediction depth:

    none
    compact
    expanded

Use expanded prediction for:

    new distillation method
    corpus/split changes
    representative transfer
    full64 conversion
    model assembly
    quantization

During Validate compare observations to the activated Prediction Contract and
record exactly one semantic Prediction Result classifier.

A:

    scope-discovery
    counterexample
    invalid-validation

result reopens/blocks Discovery and requires repair/replan.

Do not advance while a material contradiction remains unresolved.

======================================================================
24. SUCCESS GOAL
======================================================================

Near-term success:

    prove the V2.1 oracle-routed p16 training method with a real 32k GPU pilot.

Medium-term success:

    robust-green p16 layer-0
    training-method lock
    p32/top5 strong candidate
    12 representative layers

Product success:

    complete 64-layer p16 fallback
    validated BF16 sparse Qwen3.8 derivative
    at least one validated quantized artifact

Preferred product success:

    p32/top5 also reaches robust-green and produces a validated higher-sparsity
    BF16 + quantized model.

The project must always retain a clear path to the p16 fallback.

======================================================================
25. FIRST EXECUTION PHASE
======================================================================

Begin with PHASE 0 only.

Do not immediately start the 750k capture.

PHASE 0 output must include:

1. NSP discovery receipt / Repository Fact Ledger.
2. Reconciled current state and HEAD.
3. Corpus V2.1 freeze.
4. Benchmark-derived task quarantine.
5. Agent-task diversity expansion.
6. Final split/overlap/tokenization receipts.
7. Recovered and pinned native-Windows ML environment.
8. Environment doctor output.
9. Full relevant ML tests.
10. Exact PHASE 1 commands for:
       small balanced teacher capture
       p16 oracle smoke
       2k-4k method smoke
       32k method-proof pilot

Then return control/evidence to the EPIC owner.

Do NOT skip directly into large training before PHASE 0 gates are green.