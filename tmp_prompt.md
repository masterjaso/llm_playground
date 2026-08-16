You are taking over an active dense-to-MoE conversion research project and
are responsible for driving it from the current research state through a
defensible TRAINABLE SET, representative-layer validation, full 64-layer
conversion, and whole-model evaluation.

Do not merely run experiments. Drive the project toward a final architecture
and training recipe using explicit gates and falsifiable decisions.

======================================================================
PROJECT
======================================================================

Repository:
    C:\workplace\llm_playground

Branch:
    agent/windows-dense2moe-real-pipeline

Remote:
    masterjaso/llm_playground

Current pushed HEAD at takeover:
    a22e713088237a70c82b9404ba009c2b0d87a81a

Commit:
    "Make validation A-B exclusions explicit"

Primary run:
    runs/20260815-184644-windows-real-d2m-v4-streaming

Source model:
    Qwen/Qwen3.8-27B

Pinned source revision:
    1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

Implementation/model class:
    qwen3_5_text

Source geometry:
    hidden size              5120
    dense FFN width          17408
    decoder layers           64

Execution environment:
    Native Windows is authoritative for D2M execution.
    WSL/Linux may be used for lightweight inspection but must not be treated as
    proof that native PowerShell/CUDA integration works.

======================================================================
PRIMARY MISSION
======================================================================

Convert the model's 64 dense SwiGLU FFNs to real sparse MoE FFNs while
preserving the Qwen backbone and as much model quality as possible.

Product priority:

    BEST:
        p32/top4 or equivalent
        ~82.35% active FFN reduction

    EXCELLENT:
        p32/top5 or equivalent
        ~79.41% active FFN reduction

    ACCEPTABLE PRODUCTION BASELINE:
        p16/top4
        ~70.59% active FFN reduction

Do not allow lower-sparsity p8 candidates to redefine the product target.
p8 is only a trainability/router-quality control.

The final answer does NOT have to use one identical topology for all 64 FFNs
if evidence shows that different Qwen attention-cycle classes tolerate
different sparsity.

A heterogeneous but structurally simple layer-class policy is acceptable if
it materially improves the quality/compute Pareto frontier.

======================================================================
NON-FFN BACKBONE MUST REMAIN UNCHANGED
======================================================================

Preserve:

    Gated DeltaNet / linear attention
    full gated attention
    Q/K/V/O projections
    attention gating
    RoPE / positional behavior
    norms
    residual topology
    embeddings
    LM head
    tokenizer
    special tokens
    chat template
    generation semantics
    cache/state semantics
    all other non-FFN source-model tensors

Only the dense SwiGLU FFNs are being replaced.

Do NOT MoE-ify attention.
Do NOT modify Gated DeltaNet.
Do NOT modify full attention to compensate for FFN errors.

======================================================================
CRITICAL QWEN ARCHITECTURE FACT — 3:1 ATTENTION CYCLE
======================================================================

The exact pinned source geometry records:

    full_attention_interval = 4

with the 64-layer pattern:

    linear_attention
    linear_attention
    linear_attention
    full_attention

repeated sixteen times.

Therefore define attention-cycle positions as:

    layer % 4 == 0:
        LINEAR_A
        Gated DeltaNet / linear-attention-associated FFN

    layer % 4 == 1:
        LINEAR_B
        Gated DeltaNet / linear-attention-associated FFN

    layer % 4 == 2:
        LINEAR_C
        Gated DeltaNet / linear-attention-associated FFN

    layer % 4 == 3:
        FULL_ATTENTION
        full-attention-associated FFN

And:

    cycle_index = layer // 4

IMPORTANT:

Every one of the 64 decoder layers STILL HAS AN FFN.

The 3:1 architecture does not reduce the number of FFNs being converted.
It changes the sequence-mixing context feeding those FFNs.

Do not assume all four cycle positions have identical MoE compressibility.

This is now a first-class experimental dimension.

======================================================================
CURRENT p16/top4 STATE
======================================================================

p16/top4 geometry:

    routed experts            16
    routed expert width       1024
    shared width              1024
    top-k                     4

Active width:

    1024 + 4*1024 = 5120

FFN reduction:

    70.59%

Best current clean validation:

    global/NMSE        ~0.022887
    cosine             ~0.981599
    dead experts       0
    load CV            ~0.4421

Historical strict holdout confirmation:

    NMSE               ~0.027239
    cosine             ~0.977044
    dead experts       0
    load CV            ~0.4713

The same refined basis has shown much higher target-assisted reconstruction
capacity than the learned selector.

Historical exact/bounded oracle analysis has shown roughly:

    validation oracle cosine > .98
    holdout oracle cosine     ~.9822

while learned routing remains lower.

Therefore p16/top4 is a credible proof-of-method architecture.

The remaining unresolved issue is simultaneous:

    reconstruction quality
    +
    robust selector generalization
    +
    acceptable load balance

Do not reopen architecture search for p16 without evidence that its basis
cannot satisfy these jointly.

======================================================================
CURRENT p32 PRODUCT TARGETS
======================================================================

p32/top5:

    routed experts            32
    expert width              512
    shared width              1024
    top-k                     5

Active width:

    1024 + 5*512 = 3584

Active FFN fraction:

    20.59%

FFN reduction:

    79.41%


p32/top4:

    routed experts            32
    expert width              512
    shared width              1024
    top-k                     4

Active width:

    1024 + 4*512 = 3072

Active FFN fraction:

    17.65%

FFN reduction:

    82.35%

The old p32/top5 result was approximately:

    NMSE       .04485
    cosine     .96729
    dead       0
    load CV    .5366

but that result predates the current refined methodology and historically
reused a partition designed around another p32 top-k.

It is NOT sufficient evidence that a properly optimized p32/top5 cannot work.

Current code now supports topology-specific p32/top5 materialization and
explicit p32/top4/top5 search.

Use those paths.

======================================================================
LAYER RECONSTRUCTION GREEN GATE
======================================================================

A candidate is GREEN only when the applicable clean validation/shadow split
satisfies:

    global NMSE <= 0.05
    cosine      >= 0.98
    dead experts = 0
    load CV     <= 0.50

Checkpoint selection:

FIRST require:

    NMSE <= .05
    dead = 0
    load CV <= .50

THEN:

    maximize cosine

Tie-break:

    lower NMSE
    lower load CV

If no fully feasible point exists:

    retain the Pareto frontier
    clearly label the result non-green
    select the most decision-useful fallback

Never choose a lower-cosine checkpoint merely because its already-green NMSE
is slightly smaller.

======================================================================
VALIDATION / HOLDOUT CONTRACT
======================================================================

Validation-A:

    architecture/checkpoint selection

Validation-B:

    shadow generalization confirmation

FIT:

    all optimizer updates

Holdout:

    finalist confirmation only

For NEW experiments:

    validation-A subset_of fit_exclude_indices
    validation-B subset_of fit_exclude_indices
    A and B must be disjoint

Persist hashes for:

    validation-A
    validation-B
    complete fit exclusions
    resulting FIT rows
    dataset identity

The current code supports this explicit A/B exclusion contract.

Important distinction:

A validation-B chosen from historical FIT rows is NOT an untouched end-to-end
validation set for the already-trained p16 basis because that basis may have
previously seen those rows.

It may be used only for selector-only confirmation when:

    basis is frozen
    validation-A excluded
    validation-B excluded

For a truly untouched p16 end-to-end shadow test, prefer a fresh
layer-0-only capture from new text.

For new p32 work, create/freeze validation-B before p32 optimization and keep
it out of all optimizer updates.

Do NOT repeatedly use the existing holdout as an optimization signal.

======================================================================
CLI / PROCESS SUPERVISION IS A HARD REQUIREMENT
======================================================================

The project previously wasted hours on silent shell/Git processes.

That must not happen again.

Every normal command must execute through the guarded command infrastructure
or an equivalent bounded supervisor.

Required terminal contract:

    __CMD_START__

followed by exactly one:

    __CMD_DONE__
    __CMD_FAILED__
    __CMD_TIMEOUT__

with:

    command name
    return code where applicable
    elapsed time

FAST commands:

    normal timeout <= 60 sec

MEDIUM commands:

    normal timeout <= 5 min

LONG_RUNNING commands:

    must be explicitly classified
    must have a heartbeat receipt
    must periodically emit __HEARTBEAT__
    must spool complete logs
    must retain bounded output tails only

Long-running heartbeat receipts must expose at minimum:

    status
    PID
    elapsed time
    stdout bytes
    stderr bytes
    last child output time
    child output age
    child_output_stale
    stdout log
    stderr log

Silence is NOT proof of useful work.

Never use expensive whole-tree commands such as:

    git status --short --ignored

against the large runs tree.

Prefer:

    git rev-parse HEAD
    git log -1
    git status --short --untracked-files=no
    git diff --name-only
    git check-ignore -v <specific-path>

Disable:

    pagers
    editors
    credential prompts
    stdin-dependent confirmations

On timeout:

    terminate full process tree
    emit terminal receipt
    retry identical command no more than once
    then change strategy

======================================================================
PHASE 0 — VERIFY NATIVE WINDOWS PROCESS SUPERVISION
======================================================================

The native Windows smoke harness exists but the checked-in receipt still says:

    NOT_RUN_LINUX_ENVIRONMENT

Run on native Windows:

    scripts\Invoke-GuardedCommand-Smoke.ps1

Required cases:

    success
    nonzero failure
    timeout + descendant process-tree termination
    long-running heartbeat
    guarded git log

Do not proceed to expensive runs until all cases pass.

Persist the actual native Windows receipt.

If the smoke test fails:

    fix it
    rerun
    commit the fix and receipt

Do not bypass the guard layer.

======================================================================
PHASE 1 — FINISH REAL-SCALE LOAD-AWARE ORACLE INFRASTRUCTURE
======================================================================

The load-aware oracle is now much safer:

    batched float32 Gram/correlation scoring
    compact candidate matrices
    optional memmap
    cosine-aware Pareto selection
    no giant Python per-token candidate object graph

However the input path still needs to be safe at real scale.

The current validation contribution sizes are potentially enormous.

For example routed contributions alone are approximately:

p16:

    16384 * 16 * 5120 float32
    ~5 GiB

p32:

    16384 * 32 * 5120 float32
    ~10 GiB

Do NOT require those arrays to be eagerly decompressed from a giant compressed
NPZ into RAM.

Prefer one of:

A.

    shared.npy
    routed.npy
    target.npy
    manifest.json

loaded with:

    mmap_mode="r"

or:

B.

    a batch-producing contribution source that never materializes the complete
    contribution cube.

Preserve a small NPZ path only for tests if useful.

The real validation path must be memory-bounded.

======================================================================
PHASE 2 — CALIBRATE THE NEW STREAMING COEFFICIENT SOLVER
======================================================================

The real-scale oracle now uses:

    exhaustive candidate SETS for p16/top4

but candidate coefficient fitting is a projected float32 approximation rather
than the old full active-face exact positive solver.

Therefore do NOT casually label the result:

    exact oracle

Preferred terminology:

    exhaustive-set projected-positive oracle

or:

    exhaustive candidate-set load-aware oracle

Calibrate before using tiny .001-.003 cosine differences for decisions.

Use a deterministic stratified p16 subset, ideally including:

    easy residual tokens
    medium residual tokens
    hardest quartile tokens

Target:

    512-1024 tokens if practical

Compare:

    old exact active-face positive solve

versus:

    streaming projected float32 solve

Report:

    reconstruction cosine delta
    global NMSE delta
    selected-set agreement
    coefficient error
    hard-quartile delta
    runtime
    peak memory

Acceptance should be driven primarily by QUALITY ERROR, not set identity.

A reasonable target is approximately:

    mean cosine discrepancy <= 2e-4
    global NMSE absolute discrepancy <= 5e-4

If the projected scorer is materially worse:

    improve it

or:

    exact-rescore a bounded shortlist of promising candidates/token

and ALWAYS consider exact-refitting the final selected route IDs before
publishing final metrics.

Do not let an approximate coefficient solver decide a .98 gate incorrectly.

======================================================================
PHASE 3 — p16/top4 LOAD-AWARE FEASIBILITY
======================================================================

Run the final refined p16/top4 basis through the real-scale load-aware
validation oracle.

p16 has:

    C(16,4) = 1820

expert sets.

Use all candidate SETS.

The global assignment question is:

Does there exist an assignment satisfying simultaneously:

    cosine >= .98
    global NMSE <= .05
    load CV <= .50
    dead experts = 0

The load-aware oracle must report:

    unconstrained reconstruction point
    load-constrained selected point
    full cosine/NMSE/load Pareto frontier
    hard-quartile cosine
    expert usage
    dead experts
    candidate-set assurance
    coefficient solver assurance
    memory/storage mode

DECISION:

If p16 load-aware oracle is GREEN:

    freeze p16 basis geometry
    treat selector learning/generalization as the remaining problem

If p16 load-aware oracle cannot simultaneously reach the gates:

    inspect the Pareto frontier
    determine whether the blocker is genuinely basis/load coupling
    do not blindly tune the router toward an impossible target

======================================================================
PHASE 4 — p32/top5 PRODUCT SEARCH
======================================================================

p32/top5 is the primary high-sparsity product target.

Run a current-HEAD FIT/dev partition search specifically supporting top5.

Do not silently reuse a top6-tuned partition as the definitive top5 basis.

Use:

    topology-specific p32/top5 evaluation
    residual-aware/refined basis search
    hard-token-aware evidence where useful

Do not open holdout.

Once the best FIT/dev p32/top5 basis is selected:

    freeze it
    materialize it with provenance
    run bounded load-aware oracle analysis

Current p32 default oracle search is bounded.

That is acceptable.

Report explicitly:

    candidate pool
    number of combinations/token
    max combinations
    search assurance
    global NMSE
    cosine
    load CV
    hard-quartile cosine
    dead experts
    expert usage
    quality/load Pareto

DECISION:

If p32/top5 reaches or exceeds the green gate at oracle level:

    PROMOTE to actual student/router training

If p32/top5 lands near target, approximately:

    cosine .975-.98

while NMSE/load are plausible:

    strengthen the bounded candidate search
    refine basis
    do not reject prematurely

If materially below target after a stronger bounded oracle:

    retain evidence
    deprioritize for now

======================================================================
PHASE 5 — p32/top4 STRETCH TARGET
======================================================================

Screen p32/top4 in parallel after top5 infrastructure is established.

Initial evaluation may use the best p32 basis produced during top5 search.

However:

If p32/top4 is close to the target, it MUST receive its own top4-specific basis
refinement before rejection.

Do not reject:

    82.35% FFN reduction

simply because a top5-tuned partition is imperfect for top4.

Use the same bounded-oracle escalation policy as top5.

If p32/top4 becomes green or convincingly near-green:

    promote to student training

======================================================================
PHASE 6 — ROUTER TRAINING POLICY
======================================================================

Do not launch broad hyperparameter sweeps.

We already have evidence that simply changing:

    BCE
    repeated CE
    generic cosine weights
    more epochs
    larger generic nonlinear router widths
    amplitude supervision

does not automatically solve selector generalization.

Every new router experiment must answer a specific hypothesis.

For p16 and promoted p32 candidates track:

    validation cosine
    NMSE
    load CV
    dead experts

plus:

    top-k recall vs oracle
    exact route-set match
    Jaccard
    router margin
    router entropy
    hard-quartile recall
    hard-quartile cosine

Amplitude quality should still be recorded but should not dominate work while
NMSE remains green.

======================================================================
SELECTOR DATA BEFORE SELECTOR BLOAT
======================================================================

If selector generalization remains the blocker, prefer increasing high-quality
router training examples before repeatedly increasing router size.

Consider a fresh LAYER-0-ONLY activation corpus.

Do NOT restart full 64-layer capture.

Initial useful scale:

    several hundred thousand diverse token states

If throughput/storage make it cheap:

    approximately 1M

Use new data for:

    expanded selector FIT
    genuinely untouched shadow validation

Do not contaminate current holdout.

Strong-oracle label the additional selector examples.

Freeze the expert/shared basis during selector-only experiments when the
scientific question is router generalization.

======================================================================
SHARED-OUTPUT ROUTER EXPERIMENT
======================================================================

The code now supports an opt-in selector using:

    x
    +
    already-computed shared FFN output

This is scientifically interesting because the shared branch exposes nonlinear
information related to the dense residual.

Run at most a bounded A/B initially:

A:

    current best selector

B:

    shared-output-feature selector

Keep fixed:

    basis
    FIT rows
    validation A/B
    top-k
    routing mode
    optimizer budget
    seed where practical

Measure:

    cosine
    NMSE
    load CV
    selector recall
    exact set match
    router margin
    router entropy
    hard-token quality
    router parameter count
    router FLOPs
    measured layer latency

Important:

The shared-output selector creates a dependency where routed dispatch waits for
the shared branch output.

Therefore it must win enough quality to justify any runtime serialization.

Do not promote from reconstruction quality alone.

======================================================================
ATTENTION-CYCLE-AWARE REPRESENTATIVE LAYER GATE
======================================================================

Do NOT use only:

    0
    16
    32
    48
    63

as representative layers.

Layers 0/16/32/48 all occupy the same LINEAR_A cycle position.

That sampling is biased.

Once one or more layer-0 candidates are convincingly green, representative
validation must cover every attention-cycle position across depth.

Required representative set:

EARLY:

    layer 0   LINEAR_A
    layer 1   LINEAR_B
    layer 2   LINEAR_C
    layer 3   FULL_ATTENTION

MIDDLE:

    layer 28  LINEAR_A
    layer 29  LINEAR_B
    layer 30  LINEAR_C
    layer 31  FULL_ATTENTION

LATE:

    layer 60  LINEAR_A
    layer 61  LINEAR_B
    layer 62  LINEAR_C
    layer 63  FULL_ATTENTION

This 12-layer matrix is the preferred representative gate.

For every result persist:

    layer number
    depth fraction
    cycle_index
    cycle_position
    attention_type
    MoE topology
    partition hash
    training recipe hash
    validation-A metrics
    validation-B metrics
    oracle metrics
    router metrics

Do not assume LINEAR_A/B/C are equivalent until evidence shows they are.

======================================================================
HETEROGENEOUS TOPOLOGY IS ALLOWED
======================================================================

The final 64-layer architecture may choose topology by attention-cycle class if
the data justifies it.

For example:

    48 linear-attention-associated FFNs:
        p32/top4

    16 full-attention-associated FFNs:
        p16/top4

would yield approximately:

    79.41% average active FFN reduction

across all 64 equal-width FFNs.

Likewise:

    48 linear-attention FFNs:
        p32/top5

    16 full-attention FFNs:
        p16/top4

would yield approximately:

    77.21% average active FFN reduction.

These are highly attractive product outcomes.

But do NOT assume linear-attention FFNs are easier to sparsify.

Measure it.

Possible final policies include:

    homogeneous p32/top4
    homogeneous p32/top5
    homogeneous p16/top4

or a simple cycle-class mapping such as:

    LINEAR_A/B/C -> p32/top4 or p32/top5
    FULL_ATTENTION -> p16/top4

A more complex per-layer topology should require strong evidence because it
increases implementation and serving complexity.

Prefer the simplest policy on the best quality/compute frontier.

======================================================================
REPRESENTATIVE-LAYER SUCCESS CRITERIA
======================================================================

For each representative layer:

    NMSE <= .05
    cosine >= .98
    dead experts = 0
    load CV <= .50

Also examine:

    router recall
    hard-quartile cosine
    hard-quartile recall
    expert usage
    FIT->validation gap
    A->B validation gap

Do not require perfectly identical metrics by depth.

Do require absence of systematic failure by:

    attention-cycle class
    depth
    router collapse
    load collapse

If one attention class systematically requires less aggressive sparsity:

    adopt a heterogeneous class-level policy

rather than forcing all layers into the same topology.

======================================================================
FUNDAMENTAL TELEMETRY TO START COLLECTING NOW
======================================================================

Do NOT perform reasoning-efficiency post-training.

Do NOT modify:

    thinking behavior
    chain-of-thought length
    stopping behavior
    reasoning templates
    generation policy

Do NOT introduce a loss encouraging shorter reasoning.

That work is explicitly OUT OF SCOPE for this phase.

However, begin collecting low-cost fundamental telemetry now because it is
useful for:

    router diagnostics
    sparse-compute analysis
    layer-class analysis
    future behavioral analysis
    detecting routing stagnation/collapse
    latency analysis

For reconstruction/training runs record per layer:

    attention_type
    cycle_position
    cycle_index

    router logit statistics
    router entropy
    top-k margin
    selected expert IDs
    expert usage frequency
    load CV
    dead experts

    routing-amplitude statistics
    shared output norm
    routed output norm
    shared:routed norm ratio

    target norm
    residual norm
    reconstruction norm
    cosine
    global NMSE
    mean-token-relative MSE

    route-set agreement with oracle where available
    Jaccard with oracle
    top-k recall

For sequential/generation runs, if already being performed for normal model
validation, passively record:

    prompt token count
    total generated token count
    final answer token count when identifiable
    explicit thinking-segment token count when identifiable from the model's
        public generation format
    whether thinking mode was enabled
    time to first token
    prefill latency
    decode latency
    tokens/sec
    stop reason

Do NOT optimize against thinking-token counts now.

They are telemetry only.

Do not store giant raw hidden-state histories by default.

Prefer bounded aggregate statistics and sampled representative-token traces.

For selected representative layers also collect:

    route-set Jaccard vs previous generated token
    route repetition streak length
    router entropy over generation
    router margin over generation
    shared:routed contribution ratio over generation

These metrics are useful NOW for detecting:

    router collapse
    expert monopolization
    repetitive routing
    unstable routing
    attention-cycle differences

and may also support later behavioral work without requiring the current
distillation objective to change.

======================================================================
COMPUTE / LATENCY TELEMETRY
======================================================================

For each production candidate report:

    theoretical active FFN width
    FFN reduction
    router parameters
    router FLOP estimate
    active FFN parameter estimate

and measured where practical:

    peak VRAM
    layer forward latency
    short-context decode speed
    prefill speed
    routing overhead

For shared-output routers separately measure:

    dispatch dependency latency

Do not equate theoretical FFN-width reduction directly with whole-model speedup.

======================================================================
NO BLIND SWEEPS
======================================================================

Before every new experiment record:

    hypothesis
    intentional variable
    expected outcome
    falsifier
    compute budget
    decision enabled

One experiment should answer one question.

Do not burn hours repeating slightly different loss weights without a clear
decision boundary.

======================================================================
PROVENANCE CONTRACT
======================================================================

For every decisive experiment:

1. implement code
2. run unit/focused tests
3. commit implementation
4. ensure science code is clean
5. run experiment from exact committed HEAD
6. receipt records exact SHA
7. persist dataset/split hashes
8. persist partition hash
9. persist config/training recipe
10. commit/push reports afterward

Do not publish decisive experiments whose `code_commit` points to a state that
cannot reproduce the run.

Large safetensors/checkpoint blobs remain ignored from Git.

Preserve local checkpoints and captures.

Do not delete historical evidence.

======================================================================
DECISION REGISTER / HANDOFF
======================================================================

At every material decision boundary update the authoritative project state,
including as applicable:

    HANDOFF.md
    state.json
    decision-register.json

They must agree on:

    current code SHA
    current winning candidate
    current gates
    rejected candidates
    reason for rejection
    holdout status
    exact next action

Do not leave stale handoff text behind after a major decision.

======================================================================
PHASE 7 — FREEZE THE TRAINABLE SET
======================================================================

A TRAINABLE SET exists only when we have frozen:

    topology policy
    shared width
    expert count
    expert width
    top-k
    routing mode
    router architecture
    amplitude behavior
    partition/refinement method
    loss schedule
    optimizer schedule
    checkpoint-selection rule
    validation protocol

and demonstrated:

    green layer-0 evidence
    generalization confirmation
    representative-layer success across all four attention-cycle positions
    across early/middle/late depth

If topology differs by attention-cycle class, freeze that mapping explicitly.

Examples:

    LINEAR_A/B/C -> p32/top5
    FULL_ATTENTION -> p16/top4

or:

    all positions -> p32/top5

Do not begin an uncontrolled full 64-layer conversion before this definition is
satisfied.

======================================================================
PHASE 8 — FULL 64-LAYER CONVERSION
======================================================================

Once the TRAINABLE SET is frozen:

    convert all 64 FFNs

Use the established exact layer-streaming teacher/capture mechanism.

Preserve durable existing captures/checkpoints.

Historical rolling replay artifacts must not be destroyed.

For each converted layer persist:

    attention type
    cycle position
    topology
    partition hash
    checkpoint hash
    validation metrics
    training receipt
    code SHA

Apply the frozen recipe without per-layer architecture tinkering unless a layer
fails a predeclared gate.

If a layer fails:

    diagnose by:
        depth
        attention-cycle type
        residual difficulty
        router behavior
        basis capacity

Do not silently weaken the gate.

======================================================================
PHASE 9 — FULL MODEL ASSEMBLY
======================================================================

Assemble a real sparse model while copying every non-MLP source tensor
unchanged.

Strictly preserve:

    embeddings
    norms
    attention
    Gated DeltaNet
    RoPE
    LM head
    tokenizer
    chat template
    special tokens
    generation config/semantics

Strict-reload the final assembled model.

Verify deterministic source-vs-assembly inventories.

No missing/unexpected tensor namespaces.

======================================================================
WHOLE-MODEL QUALITY GATES
======================================================================

Existing tracked product gates:

Perplexity increase:

    GREEN <= 5%
    YELLOW <= 10%

Token KL:

    GREEN <= 0.10
    YELLOW <= 0.20

Top-1 token agreement:

    >= 85%

Also test:

    short-context generation
    long-context generation
    instruction following
    coding
    reasoning capability
    multi-turn behavior
    long-context retrieval
    tool-use sanity where available

At this stage we are checking PRESERVATION.

Do NOT post-train the model to shorten reasoning.
Do NOT intentionally alter thinking behavior.

Passive token/routing telemetry may be collected as described above.

======================================================================
ATTENTION-CYCLE WHOLE-MODEL ANALYSIS
======================================================================

After representative/full conversion, summarize quality grouped by:

    LINEAR_A
    LINEAR_B
    LINEAR_C
    FULL_ATTENTION

and grouped by depth:

    early
    middle
    late

Answer explicitly:

    Does sparsification error correlate with attention-cycle position?

    Are full-attention-associated FFNs harder to sparsify?

    Are one or more linear-attention positions harder than the others?

    Does selector recall degrade with depth?

    Does load balance degrade with depth?

    Does hard-token error concentrate in a particular cycle class?

These results should determine whether the final homogeneous or heterogeneous
topology policy is justified.

======================================================================
AUTONOMY
======================================================================

Proceed autonomously.

Do NOT stop for permission between routine bounded steps.

Do not ask me to choose between experiments when the existing gates already
determine the next action.

Pause only when:

    a destructive operation is required
    source/capture provenance is at risk
    a major architecture decision has two genuinely equivalent evidence paths
    an unexpected blocker invalidates the decision tree
    compute requirements materially exceed the planned research budget

Otherwise continue.

Long experiments must remain observable through the heartbeat system.

======================================================================
IMMEDIATE EXECUTION SEQUENCE
======================================================================

Start from current committed HEAD:

    a22e713088237a70c82b9404ba009c2b0d87a81a

Then execute:

1.
    inspect local state safely
    preserve any valuable uncommitted artifacts
    confirm HEAD/remote

2.
    run native Windows guarded-command smoke suite
    fix until green
    save receipt

3.
    make load-aware contribution input genuinely mmap/stream safe

4.
    calibrate projected streaming coefficient solve vs old exact solve

5.
    run full p16/top4 validation load-aware feasibility analysis

6.
    make the p16 decision:
        simultaneous quality/load feasible?
        yes -> selector generalization work
        no  -> inspect basis/load Pareto

7.
    rerun current-HEAD p32/top5-specific FIT/dev basis/refinement search

8.
    run bounded load-aware p32/top5 oracle

9.
    run bounded load-aware p32/top4 oracle

10.
    if p32/top4 is close:
        run top4-specific basis refinement
        rerun stronger oracle

11.
    promote only oracle-plausible architectures to student training

12.
    solve/freeze the layer-0 candidate set

13.
    create/capture any required clean selector/shadow data

14.
    train/evaluate the 12 attention-cycle representative layers:

        0 1 2 3
        28 29 30 31
        60 61 62 63

15.
    determine whether final topology should be:

        homogeneous p32/top4
        homogeneous p32/top5
        homogeneous p16/top4
        or a simple attention-cycle heterogeneous mapping

16.
    freeze the TRAINABLE SET

17.
    perform full 64-layer conversion

18.
    assemble strict-reloadable sparse model

19.
    run whole-model quality and performance evaluation

20.
    publish final synthesis with:
        chosen topology policy
        quality
        sparsity
        actual active FFN reduction
        attention-cycle findings
        routing findings
        latency
        remaining risks
        exact reproducibility receipts

======================================================================
FINAL SUCCESS DEFINITION
======================================================================

This takeover is complete only when we have a defensible answer to:

    What sparse FFN architecture and training recipe should replace all 64
    dense Qwen FFNs?

The preferred answer is:

    p32/top4 or p32/top5 wherever quality allows

with:

    p16/top4 available as the quality-preserving safety topology.

A heterogeneous attention-cycle policy is considered a valid SUCCESS if it
gives a substantially better average FFN reduction while satisfying quality
gates.

The final deliverable must include:

    frozen architecture/topology policy
    frozen training recipe
    complete provenance
    representative-layer evidence
    attention-cycle analysis
    full 64-layer conversion
    strict model assembly
    whole-model quality evaluation
    compute/latency measurements
    passive routing/generation telemetry

Do not optimize reasoning length or perform reasoning-efficiency post-training
during this takeover.

First preserve and prove the model.

Begin now with native-state recovery and the Windows guarded-command smoke
gate, then proceed through the sequence above without unnecessary pauses.