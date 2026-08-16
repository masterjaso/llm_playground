You are taking over an active dense-to-MoE conversion research project after
the previous agent session became stale/hung during repository bookkeeping.

PROJECT
=======

Repository:
    C:\workplace\llm_playground

Branch:
    agent/windows-dense2moe-real-pipeline

Remote:
    masterjaso/llm_playground

Last known PUSHED commit:
    290370ff1f44b4f45fbf651988350facab6087c0
    "enforce clean validation and gate-aware routing selection"

Primary run:
    runs/20260815-184644-windows-real-d2m-v4-streaming

Source model:
    Qwen/Qwen3.8-27B

Pinned source revision:
    1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

Source geometry:
    hidden size = 5120
    dense FFN width = 17408
    decoder layers = 64

Execution environment:
    Native Windows for D2M work.
    Preserve existing captures/checkpoints/reports.
    Do not restart teacher capture unless genuinely necessary.

======================================================================
MISSION
======================================================================

Finish the research needed to identify a TRAINABLE sparse-MoE architecture
and training recipe suitable for conversion of all 64 dense Qwen FFNs.

The product objective remains:

    >= 70% reduction in ACTIVE dense-FFN width/compute

while preserving the Qwen backbone and achieving strong reconstruction and
whole-model quality.

DO NOT allow an easier low-sparsity p8 solution to redefine the product goal.

The conversion remains FFN-only.

Preserve unchanged:
    attention
    RoPE / positional behavior
    norms
    residual topology
    embeddings
    LM head
    tokenizer
    special tokens
    chat template / reasoning interface
    other Qwen-specific non-FFN behavior

Only the dense SwiGLU FFNs are being replaced by sparse MoE FFNs.

======================================================================
FIRST: RECOVER LOCAL STATE SAFELY
======================================================================

The previous session appears to have hung during commands like:

    git status --short --ignored

against the very large runs tree.

Those were repository-inspection commands, NOT model training.

Do NOT repeat whole-tree ignored-file enumeration.

If stale processes remain, inspect them with something targeted such as:

    ps -eo pid,ppid,etime,stat,%cpu,%mem,cmd | rg \
      'git status|git check-ignore|git log|clean-validation|train_|python'

If an old multi-hour process is only:

    git status --short --ignored

or equivalent read-only repository scanning, terminate it safely.

Do not terminate genuine Python/CUDA training jobs until their purpose is
identified.

Use lightweight Git inspection:

    git log -1 --oneline --decorate
    git rev-parse HEAD
    git rev-parse origin/agent/windows-dense2moe-real-pipeline
    git status --short --untracked-files=no

For ignored checkpoint verification use targeted:

    git check-ignore -v <specific checkpoint path>

Do NOT enumerate all ignored files.

Safetensors model/checkpoint blobs are intentionally ignored. Preserve them
locally. Reports/manifests/metadata should remain commit-able.

Before doing new science:

1. determine whether local HEAD is ahead of the pushed HEAD;
2. determine whether meaningful uncommitted CODE or REPORT changes exist;
3. determine whether a genuine experiment is still running;
4. preserve all useful existing artifacts;
5. do not blindly reset or clean the worktree.

If there is valuable unpushed work from the stale session, inspect and preserve
it first.

======================================================================
CURRENT SCIENTIFIC STATE
======================================================================

A true FIT / VALIDATION / HOLDOUT protocol has now been established.

Original TRAIN:
    131,508 rows

TRUE FIT:
    115,124 rows

TRUE VALIDATION:
    16,384 rows
    excluded from ALL optimizer updates

Validation identity hash:
    5a7739c753dae98698a8c1a22c6a10409230f0750b33cfc9631594f16d8b8e1c

Full holdout:
    16,598 rows

The holdout must remain closed during new architecture/router development.

Current product layer-level green gate:

    normalized MSE <= 0.05
    cosine >= 0.98
    dead experts = 0
    load CV <= 0.50

Checkpoint selection is intended to be gate-aware:

FIRST satisfy:
    NMSE <= .05
    dead = 0
    load CV <= .50

THEN maximize:
    cosine

Tie-break:
    lower NMSE
    then lower load CV

If nothing is fully feasible:
    preserve the Pareto frontier
    select the highest-cosine relevant fallback

Validation should be measured every epoch and the best intermediate checkpoint
must be retained.

======================================================================
CURRENT CANDIDATES
======================================================================

PRIMARY PRODUCTION RESEARCH CANDIDATE:

    p16/top4 independent-positive

Geometry:
    16 routed experts
    expert width = 1024
    top-k = 4
    shared width = 1024

Active FFN width:
    1024 + 4*1024 = 5120

FFN reduction:
    70.59%

Latest CLEAN validation result:

    NMSE        0.04251     PASS
    cosine      0.97297     FAIL
    dead        0           PASS
    load CV     0.2978      PASS

This is the current primary architecture.

The problem is now specifically ANGULAR FIDELITY / EXPERT SELECTION.

Do not waste effort re-solving NMSE or load balance unless a new method
regresses them.

OTHER RESULTS:

p16/top4 unordered BCE:
    reduction 70.59%
    NMSE      0.04549
    cosine    0.97276
    dead      0
    load CV   0.2447

Conclusion:
    BCE improves load balance but does NOT improve angular quality.
    Do not pursue plain BCE as the main route.

p32/top5:
    32 routed experts
    width 512
    top-k 5
    shared 1024
    active width 3584
    reduction 79.41%

Clean validation:
    NMSE      0.04485
    cosine    0.96729
    dead      0
    load CV   0.5366

Conclusion:
    attractive sparsity but currently inferior to p16/top4.
    Keep as a frontier candidate, not the immediate primary target.

Other aggressive candidates remain:
    p16/top3  -> 76.47% reduction
    p32/top6  -> 76.47% reduction
    p32/top4  -> 82.35% reduction

p8/top6 remains ONLY:
    TRAINABILITY / ROUTER QUALITY CONTROL

Do not promote p8/top6 to production.

======================================================================
MOST IMPORTANT DIAGNOSTIC RESULT
======================================================================

For clean-validation p16/top4, comparison to the residual-correlation oracle
showed:

    mean selector recall      ~= 0.719
    exact top-4 set match     ~= 23.4%
    mean Jaccard              ~= 0.604

Conditional reconstruction:

    STUDENT IDs + STUDENT amplitudes:
        cosine ~= 0.97297

    STUDENT IDs + ORACLE amplitudes:
        cosine ~= 0.97253

    ORACLE IDs + STUDENT amplitudes:
        cosine ~= 0.97936

    ORACLE IDs + ORACLE amplitudes:
        cosine ~= 0.97929

This is highly important.

Interpretation:

1. Amplitude prediction is NOT the primary cosine blocker.
2. Expert selection is the largest recoverable source of angular error.
3. Replacing student expert IDs with oracle IDs recovers almost the entire
   gap to 0.98.
4. Therefore stop spending large budgets on amplitude-loss tuning while NMSE
   remains green.
5. The current selector/router deserves primary investigation.

The current oracle used for diagnostics is still bounded:

    residual-correlation search
    beam width = 4
    pool size = 10
    positive exact final coefficients

It is NOT proof that p16/top4 itself tops out at cosine ~0.9793.

======================================================================
HARD TOKEN DIAGNOSIS
======================================================================

Validation residual-norm quartiles approximately showed:

easiest quartile:
    student cosine ~0.98937
    oracle cosine  ~0.98974
    selector recall ~0.747

Q2:
    student cosine ~0.98840
    oracle cosine  ~0.99024

Q3:
    student cosine ~0.97242
    oracle cosine  ~0.97664

hardest quartile:
    student cosine ~0.94169
    oracle cosine  ~0.96052
    selector recall ~0.684

The difficult/high-residual token population dominates the remaining failure.

Do NOT assume all tokens need equal treatment.

======================================================================
PROVENANCE ISSUE TO FIX
======================================================================

The latest clean-validation implementation was committed at:

    290370ff1f44b4f45fbf651988350facab6087c0

but several generated clean-validation receipts record:

    code_commit = 123ae0dcc7dd114337752ed97be0638d54f65a98

because the experiments were apparently run before the implementation was
committed.

This is useful WIP evidence but not ideal reproducibility.

For every NEW decisive experiment:

    1. implement the needed code;
    2. run tests;
    3. commit the implementation;
    4. ensure the code worktree is clean;
    5. run the experiment from that exact committed HEAD;
    6. record that exact HEAD in all experiment receipts;
    7. commit/push reports and metadata afterward.

Do not knowingly create another decisive result whose code_commit cannot be
checked out and reproduced.

======================================================================
NEXT SCIENTIFIC QUESTION
======================================================================

We need to determine whether the remaining p16/top4 gap is primarily:

    A. insufficient oracle/search quality;
    B. insufficient ROUTER EXPRESSIVITY;
    C. expert/shared basis limitation;
    D. genuine top4 capacity limitation.

Resolve these in that order.

======================================================================
PHASE 1 — STRONG / EXACT p16/top4 ORACLE
======================================================================

This is the next highest-priority experiment.

For p16/top4 there are only:

    C(16,4) = 1820

possible routed-expert sets per token.

The existing bounded beam oracle is no longer strong enough to decide whether
the architecture itself can satisfy cosine >= .98.

Implement a stronger oracle.

Preferred progression:

A. First run an EXACT all-1820-combination oracle on the hardest validation
   quartile (~4096 tokens).

For each token:
    evaluate all 4-of-16 expert combinations
    solve positive coefficients exactly/bounded
    include the shared branch exactly as deployed
    compute reconstruction NMSE/cosine

Measure:

    student
    existing bounded oracle
    exact oracle

Report both:
    global energy-weighted NMSE
    mean token relative error

Do NOT call both metrics "normalized_mse".

Suggested naming:
    global_nmse
    mean_token_relative_mse

B. If computationally reasonable, extend the exact oracle to all 16,384
   validation tokens.

C. Compare exact-oracle expert sets to:
    current residual beam oracle
    learned router

Record:
    exact-set match
    top-k recall
    Jaccard
    error by residual quartile
    selected expert frequency

DECISION GATE:

If exact-oracle validation cosine is clearly >= .98:
    current p16/top4 capacity is sufficient.
    Focus on learning the selector.

If exact-oracle cosine remains < .98:
    do not waste weeks making the router imitate an insufficient basis.
    Move to PHASE 3 basis/topology work.

Do not open holdout.

======================================================================
PHASE 2 — ROUTER EXPRESSIVITY
======================================================================

Run this phase if the strong oracle demonstrates that p16/top4 CAN clear the
target.

Current selector is effectively a single linear map:

    hidden(5120) -> 16 expert logits

The oracle decision depends on nonlinear shared/expert contribution geometry.

Test whether the linear selector is underpowered.

Preserve the existing linear router as BASELINE.

Add one deliberately small nonlinear router, preferably:

    5120
      ->
    low-rank hidden size 128 or 256
      ->
    SiLU
      ->
    16 selection logits

Do not create a giant router.

Router overhead must remain tiny relative to sparse FFN compute.

Also consider, only if cleanly implemented:

    shared small trunk
       -> selection head
       -> amplitude head

but keep the first A/B minimal.

Compare:

    linear selector
    nonlinear selector

Same:
    partition
    experts
    shared branch
    FIT rows
    validation rows
    seed
    optimizer budget
    routing mode
    checkpoint rule

The only intentional variable should initially be router architecture.

Training target:

Use a selector objective that reflects expert SET / RANK quality.

Plain multi-label BCE already failed to improve cosine.

Prefer one bounded experiment with either:

    soft/listwise target distribution derived from exact/strong oracle scores

or

    ranking loss over oracle expert scores

or

    top-k set objective with useful score ordering

Do not perform a giant selector-loss sweep.

Track:

    validation cosine/NMSE
    selector recall
    exact set match
    Jaccard
    hard-quartile cosine
    hard-quartile selector recall
    load CV
    dead experts

SUCCESS:

    p16/top4 true-validation:
        NMSE <= .05
        cosine >= .98
        dead = 0
        load CV <= .50

If achieved:
    freeze the candidate
    strict-reload it
    then perform ONE full holdout confirmation.

======================================================================
PHASE 3 — BASIS / TOPOLOGY ONLY IF STRONG ORACLE SAYS NECESSARY
======================================================================

If even the exact/strong p16/top4 oracle cannot reach cosine >= .98, treat
that as evidence that routing alone cannot solve the problem.

Then investigate basis/topology while preserving >=70% FFN reduction.

Priority ideas:

1. improve the shared/routed neuron partition specifically for difficult
   residual tokens;

2. residual-aware local swap/move refinement using TRUE FIT only;

3. contribution/correlation clustering rather than only activation magnitude;

4. optimize shared branch assignment for dense residual coverage;

5. consider modestly increasing shared capacity while remaining at >=70%
   active FFN reduction;

6. consider a different routed expert width/count geometry only when the
   active-compute comparison is explicit.

Do not use validation gradients.

Validation may select between FIT-trained candidates.

Holdout remains closed.

For every candidate, compute a strong frozen oracle BEFORE expensive training.

Do not train architectures whose strong oracle cannot plausibly meet the gate.

======================================================================
P32 / MORE AGGRESSIVE SPARSITY
======================================================================

The ultimate product objective is not merely 70.59% if higher sparsity can
retain quality.

However:

    first solve the METHOD on p16/top4.

Once p16/top4 demonstrates a generalizable green recipe, apply the same method
fairly to:

    p16/top3  76.47%
    p32/top6  76.47%
    p32/top5  79.41%
    p32/top4  82.35%

For p32/top5 remember the existing result reused the p32/top6 refined
partition; it was a fair k interpolation but not necessarily a k5-specific
optimal partition.

Do not invest heavily in p32-specific refinement before the selector/basis
method is understood.

We want a quality/compute Pareto curve, not architecture churn.

Preferred final product hierarchy:

    >=76% reduction with green quality
        BEST

    ~70% reduction with excellent/green quality
        ACCEPTABLE PRODUCTION TARGET

    lower-sparsity p8
        CONTROL / FALLBACK ONLY

======================================================================
STOP BROAD HYPERPARAMETER SWEEPS
======================================================================

We already learned that:

    cosine weight changes
    more epochs
    shared-basis adaptation
    BCE vs repeated CE
    amplitude-supervision changes

can move NMSE/load somewhat but did not independently solve ~.97 cosine.

Do not launch broad blind sweeps.

Every new experiment must answer one falsifiable question.

Before running it, record:

    hypothesis
    expected result
    falsifier
    compute budget
    decision enabled by the result

======================================================================
REPRESENTATIVE-LAYER TRAINING GATE
======================================================================

Do NOT resume the old 64-layer rolling replay yet.

Durable historical replay is preserved through layer 29.

Do not continue layer 30+ merely because compute is available.

First obtain ONE >=70% layer-0 architecture that passes on TRUE validation and
then full holdout:

    NMSE <= .05
    cosine >= .98
    dead experts = 0
    load CV <= .50

Then train the same frozen architecture/training recipe on representative
layers:

    0
    16
    32
    48
    63

Use layer-specific training data but do NOT redesign the architecture per
layer unless a representative-layer failure demonstrates that it is necessary.

Representative-layer success should demonstrate:

    no systematic layer-depth collapse
    acceptable reconstruction metrics
    stable routing
    no dead experts
    bounded load imbalance

If the recipe fails one representative layer:
    diagnose before full replay.

Only after representative-layer success should full 64-layer conversion be
authorized.

======================================================================
FULL MODEL GOAL
======================================================================

Once all FFNs are converted:

Preserve every non-FFN Qwen tensor unchanged.

Assemble a real sparse MoE checkpoint with:
    shared experts
    routed experts
    learned selector
    independent-positive amplitudes
    routing metadata
    source-compatible tokenizer/config packaging

Then validate end-to-end behavior.

Whole-model quality gates already tracked by this project include:

    perplexity increase:
        green <= 5%
        yellow <= 10%

    token KL:
        green <= 0.10
        yellow <= 0.20

    top-1 token agreement:
        >= 85%

Also evaluate:
    normal instruction following
    reasoning / thinking behavior
    short context
    long context
    multi-turn generation
    long-context retrieval

Attention, tokenizer, chat template, etc. should be inherited from Qwen and
must not be modified simply to compensate for FFN conversion quality.

======================================================================
ENGINEERING / SAFETY RULES
======================================================================

- Preserve all captured activations.
- Preserve all existing checkpoints and reports.
- Do not delete old evidence just because a new method wins.
- Large safetensors stay ignored from Git.
- Commit reports/manifests/code, not giant blobs.
- Do not run expensive whole-tree `git status --ignored`.
- Do not repeatedly open the holdout.
- Do not silently lower quality gates.
- Do not redefine p8 as production success.
- Do not resume deep replay prematurely.
- Do not change attention or unrelated backbone architecture.
- Keep exact source revision and dataset hashes in receipts.
- Use deterministic seeds where practical.
- Strict-reload finalists before confirmation.
- Keep train/validation/holdout provenance explicit.

======================================================================
COMMITS / HANDOFF
======================================================================

At each meaningful decision boundary:

1. commit code;
2. run the experiment from a clean committed implementation;
3. save concise machine-readable + human-readable reports;
4. update:
       HANDOFF.md
       state.json
       decision-register.json
   so they all agree;
5. commit/push reports and metadata;
6. include exact next action.

Do not leave the project sitting on a vague status.

Avoid spending hours on repository bookkeeping.

If Git inspection takes unexpectedly long, diagnose the filesystem command
rather than waiting indefinitely.

======================================================================
AUTONOMY
======================================================================

Proceed autonomously through the bounded experiments above.

Do NOT stop to ask me for permission between routine steps.

Pause and report before expensive representative-layer or 64-layer execution,
or when a result changes the architecture decision materially.

The immediate desired sequence is:

    recover stale local state
        ->
    clean/reproducible implementation baseline
        ->
    exact/strong p16/top4 oracle
        ->
    determine ROUTER vs BASIS ceiling
        ->
    nonlinear/listwise selector experiment if oracle supports p16/top4
        ->
    obtain >=70% green layer-0 finalist
        ->
    strict holdout confirmation
        ->
    representative layers 0/16/32/48/63
        ->
    freeze trainable architecture + recipe
        ->
    full 64-layer conversion
        ->
    whole-model evaluation

======================================================================
DEFINITION OF SUCCESS FOR THIS TAKEOVER
======================================================================

The near-term goal is NOT merely "run more experiments."

The near-term goal is to produce a defensible, reproducible answer to:

    Which >=70%-reduction MoE architecture and training recipe should we
    apply across Qwen's 64 FFNs?

A TRAINABLE SET is reached when we have:

    1. frozen expert/shared geometry;
    2. frozen top-k;
    3. frozen routing architecture;
    4. frozen routing mode;
    5. frozen loss/training schedule;
    6. true FIT/VALIDATION separation;
    7. layer-0 full green holdout confirmation;
    8. representative-layer evidence that the recipe generalizes.

Only then begin the full 64-layer production conversion.

Start by inspecting the local state and determining whether anything valuable
exists beyond pushed HEAD 290370ff1f44b4f45fbf651988350facab6087c0.
Do not assume the stale session completed cleanly.