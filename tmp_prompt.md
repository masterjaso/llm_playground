Current branch head is f1e7f162409d78b0e219e2d44c6d6df78f9d10b2; the authoritative state still has replay blocked because no >=70% candidate is green on unseen holdout.

Take over the D2M project from current HEAD:

    f1e7f162409d78b0e219e2d44c6d6df78f9d10b2

Repository:
    C:\workplace\llm_playground

Branch:
    agent/windows-dense2moe-real-pipeline

Read first and reconcile:
    runs/20260815-184644-windows-real-d2m-v4-streaming/HANDOFF.md
    runs/20260815-184644-windows-real-d2m-v4-streaming/state.json
    runs/20260815-184644-windows-real-d2m-v4-streaming/decision-register.json
    runs/20260815-184644-windows-real-d2m-v4-streaming/reports/takeover-decision-20260816.json

The project now has exactly TWO active architecture targets.

======================================================================
TARGETS
======================================================================

TARGET A — SAFE FALLBACK / MUST LOCK DOWN

    p16/top4
    16 routed experts
    expert width 1024
    shared width 1024
    top4
    active width 5120
    FFN reduction 70.59%

Goal:
    turn p16/top4 into a robust, validated, usable fallback architecture.

TARGET B — PRIMARY PRODUCT TARGET

    p32/top5
    32 routed experts
    expert width 512
    shared width 1024
    top5
    active width 3584
    FFN reduction 79.41%

Goal:
    push p32/top5 to a valid usable candidate using the lessons learned from
    p16/top4.

DO NOT spend additional research time on p32/top4 for now.

Keep its existing artifacts and evidence, but remove it from the active
decision path.

======================================================================
FIX THE VALIDATION PROTOCOL FIRST
======================================================================

The current selector cross-validation incorrectly uses validation-B inside the
A+B checkpoint-selection union while also describing B as an independent
confirmation set.

Fix this.

Required protocol:

    FIT:
        optimizer updates only

    validation-A:
        checkpoint selection only

    validation-B:
        independent confirmation only
        MUST NOT participate in checkpoint selection

    holdout:
        finalist confirmation only

Therefore:

    selection_indices = validation-A
    fit_exclude_indices includes validation-A + validation-B
    validation-B evaluated only AFTER checkpoint selection
    no selection_union containing validation-B

Update receipts and metadata so they truthfully describe which split influenced
selection.

Add regression tests enforcing:

    B cannot participate in checkpoint selection
    A and B are excluded from gradients
    A and B are disjoint

======================================================================
CREATE A TRUE FRESH SHADOW DATASET
======================================================================

Historical validation-B was drawn from previous FIT data and therefore is not
a fully untouched end-to-end test for the already-trained p16 basis.

Create a fresh LAYER-0-ONLY activation capture from new/diverse text.

Do NOT restart 64-layer replay.

Use the fresh capture for:

    expanded selector FIT
    genuinely untouched shadow validation-B

Initial target:

    several hundred thousand new token states

If throughput/storage are reasonable:

    approximately 500k-1M selector-training states

Freeze a new validation-B before optimization.

Never use it for:

    gradient updates
    checkpoint selection
    loss tuning

Only use it as confirmation.

Do not touch existing holdout during this work.

======================================================================
p16/top4 — FREEZE THE BASIS
======================================================================

The current p16 basis has sufficient reconstruction quality to justify treating
the expert/shared basis as frozen while solving selection.

Do NOT keep jointly changing:

    shared basis
    routed expert basis
    selector

during selector-generalization research.

Freeze the strongest verified p16/top4 expert/shared basis.

Record its:

    checkpoint path
    tensor SHA256
    partition hash
    code SHA

Then train only:

    selection router
    positive amplitude router where applicable

until there is evidence the frozen basis itself is the blocker.

======================================================================
p16/top4 — ESTABLISH JOINT QUALITY + LOAD FEASIBILITY
======================================================================

Current evidence shows:

    student holdout cosine ~0.9772
    student holdout load CV ~0.5003

while:

    exact reconstruction oracle cosine ~0.9822
    exact oracle load CV ~0.6674

Therefore unconstrained oracle quality alone is NOT enough.

We need to establish whether there exists a routing assignment satisfying
SIMULTANEOUSLY:

    cosine >= 0.98
    NMSE <= 0.05
    load CV <= 0.50
    dead experts = 0

Implement/run the strongest practical LOAD-CONSTRAINED oracle on clean,
non-holdout data.

Use a global pricing/Lagrangian or equivalent assignment mechanism.

Report the Pareto curve:

    cosine
    NMSE
    load CV
    dead experts

The key question is:

    Can p16/top4 reach cosine >= .98 while load CV <= .50?

If YES:
    selector learning is the remaining blocker.

If NO:
    quantify the closest Pareto point before spending more selector budget.

Do not use holdout to tune load prices.

======================================================================
FIX THE LOAD-BALANCE OBJECTIVE
======================================================================

Current evidence shows increasing the differentiable load coefficient either:

    produced the same p16 checkpoint

or:

    made p32 hard top-k load balance much worse.

Therefore do NOT keep sweeping the same coefficient.

The selector objective must align with ACTUAL HARD TOP-K DISPATCH.

Investigate a bounded replacement/addition such as:

    expert-use prices / dual variables from load-constrained oracle targets

    hard-dispatch-aware batch penalties

    straight-through hard top-k load statistics

    per-expert capacity/usage targets

    regret-aware oracle labels that already include expert-use prices

The training target should approximate:

    reconstruction regret
    +
    global expert-use cost

rather than learning the unconstrained reconstruction oracle and hoping a
soft load loss fixes dispatch later.

For every new objective compare:

    predicted soft load
    actual hard top-k load
    validation-A load CV
    untouched validation-B load CV

Reject objectives whose soft balance improves while hard dispatch worsens.

======================================================================
p16 SELECTOR — DATA BEFORE MODEL SIZE
======================================================================

Do not resume generic larger-MLP router sweeps.

We already know that simply increasing nonlinear router capacity did not solve
generalization.

Use the expanded fresh layer-0 dataset first.

Train the selector on substantially more diverse states while keeping:

    basis frozen
    architecture fixed
    validation-A fixed
    fresh validation-B untouched

Track:

    cosine
    NMSE
    load CV
    dead experts
    exact-set match
    top-k recall
    mean Jaccard
    router entropy
    top-k margin
    hard-quartile recall
    hard-quartile cosine

Prefer REGRET-WEIGHTED selector supervision:

A routing mistake that barely changes reconstruction should matter less than a
routing mistake with large cosine/reconstruction regret.

======================================================================
p16 SUCCESS CONDITION
======================================================================

Do not call p16/top4 locked until:

validation-A:

    NMSE <= .05
    cosine >= .98
    dead = 0
    load CV <= .50

fresh untouched validation-B:

    NMSE <= .05
    cosine >= .98
    dead = 0
    load CV <= .50

Only then authorize ONE new post-selection holdout confirmation.

If holdout also clears all four gates:

    classify p16/top4 as SAFE FALLBACK
    freeze its complete TRAINABLE SET

The trainable set includes:

    partition method
    expert/shared geometry
    top-k
    routing mode
    router architecture
    router inputs
    amplitude behavior
    oracle/selector targets
    load objective
    loss coefficients
    optimizer schedule
    selector dataset protocol
    checkpoint-selection rule

Once frozen, do not keep tinkering with p16 unless representative-layer
evidence falsifies it.

======================================================================
p32/top5 — PRIMARY TARGET
======================================================================

Treat p32/top5 as the only active high-sparsity target.

Current validation is approximately:

    NMSE       .04484
    cosine     .96728
    load CV    .53660
    dead       0

This is not green, but it is close enough to remain worth serious research.

The current router-only load refinement was harmful.

Do NOT repeat it.

Use lessons from p16:

1. topology-specific p32/top5 basis
2. hard-token-aware basis refinement
3. load-constrained oracle targets
4. regret-aware selector targets
5. expanded fresh selector data
6. actual hard top-k load objective
7. untouched A/B validation protocol

======================================================================
p32/top5 ORACLE-FIRST DEVELOPMENT
======================================================================

Before expensive student training, establish stronger p32/top5 capacity
evidence.

Current p32 oracle is bounded.

Increase candidate-pool strength methodically when justified.

For each oracle run report:

    candidate pool size
    combinations/token
    candidate-generation method
    reconstruction cosine
    NMSE
    load CV
    hard-quartile cosine
    dead experts
    quality/load Pareto frontier

Do not describe bounded p32 evidence as an impossibility proof.

Decision:

If strongest practical load-aware p32/top5 oracle reaches:

    cosine >= .98
    NMSE <= .05
    load CV <= .50

then aggressively promote p32/top5 to selector training.

If it reaches approximately:

    cosine .975-.98

with acceptable NMSE/load:

    continue topology-specific basis/refinement work.

If it remains materially below target after strong basis + candidate search:

    document the limitation
    preserve p16 fallback
    avoid burning uncontrolled training budget.

======================================================================
ATTENTION-CYCLE REQUIREMENT
======================================================================

Qwen uses the repeating 3:1 sequence-mixing cycle:

    LINEAR_A
    LINEAR_B
    LINEAR_C
    FULL_ATTENTION

Every layer still contains an FFN.

Only FFNs are replaced.

Once either architecture is ready for representative replay, use:

    0  1  2  3
    28 29 30 31
    60 61 62 63

with labels:

    layer % 4 == 0 -> LINEAR_A
    layer % 4 == 1 -> LINEAR_B
    layer % 4 == 2 -> LINEAR_C
    layer % 4 == 3 -> FULL_ATTENTION

Do not return to the biased 0/16/32/48/63 sampling.

======================================================================
FUTURE TWO-TARGET REPRESENTATIVE PLAN
======================================================================

Once p16/top4 is locked:

    run p16/top4 across the 12 representative layers.

Once p32/top5 reaches layer-0 green:

    run p32/top5 across the same 12-layer matrix.

Compare by:

    depth
    attention-cycle position

We may ultimately choose:

    all layers p32/top5

or:

    p32/top5 where green
    p16/top4 for cycle classes that need additional capacity

A simple heterogeneous topology by attention class is allowed.

Do not introduce arbitrary per-layer topology differences unless evidence
requires them.

======================================================================
PASSIVE TELEMETRY
======================================================================

Continue collecting low-cost fundamental metrics:

    router entropy
    top-k margin
    selected expert IDs
    expert usage
    hard dispatch load CV
    dead experts
    shared output norm
    routed output norm
    shared:routed ratio
    residual norm
    reconstruction cosine/NMSE
    oracle route recall/Jaccard
    attention-cycle class
    layer depth
    latency
    peak VRAM

For normal future generation validation also record passively:

    prompt tokens
    generated tokens
    thinking-segment token count when identifiable
    answer token count
    time to first token
    decode speed
    stop reason

DO NOT optimize reasoning effort.
DO NOT perform reasoning post-training.
These are telemetry only.

======================================================================
HOLDOUT / REPLAY POLICY
======================================================================

Holdout is CLOSED during research.

Do not use it for:

    architecture selection
    loss tuning
    load tuning
    oracle pricing
    checkpoint selection
    dataset decisions

Only open it for a candidate that is already green on:

    validation-A
    untouched validation-B

Representative and full 64-layer replay stay blocked until at least p16/top4
is genuinely locked.

======================================================================
COMMAND / PROVENANCE REQUIREMENTS
======================================================================

Use guarded commands for all CLI work.

Every command must terminate with:

    DONE
    FAILED
    TIMEOUT

Long jobs require:

    heartbeat
    bounded output memory
    spool logs
    child-output freshness

No silent waits.

Every decisive experiment:

1. state hypothesis
2. state falsifier
3. state compute budget
4. state decision enabled
5. implement
6. test
7. commit code
8. run from committed HEAD
9. record exact code SHA
10. record tensor/partition/split hashes
11. commit results
12. update HANDOFF/state/decision-register

Do not run decisive experiments from dirty science code.

======================================================================
IMMEDIATE EXECUTION ORDER
======================================================================

1. Safely inspect local state and reconcile current HEAD.

2. Fix validation protocol:
       A selects
       B confirms
       B never participates in selection.

3. Add tests proving that contract.

4. Freeze/document the current best p16 expert/shared basis.

5. Build fresh layer-0 selector dataset and untouched shadow validation-B.

6. Establish p16 load-constrained oracle Pareto feasibility.

7. Design one hard-top-k-aligned load-aware selector target/objective.

8. Train p16 selector with expanded data and frozen basis.

9. Iterate only through hypothesis-driven selector experiments until p16 is
   green on A and fresh B.

10. Run one authorized p16 holdout confirmation.

11. If green:
        LOCK p16/top4 SAFE FALLBACK.

12. In parallel where compute permits, improve p32/top5 basis/oracle evidence.

13. Apply the successful p16 selector/load methodology to p32/top5.

14. Bring p32/top5 to green A+B evidence, then holdout confirmation.

15. Only after p16 is locked begin the 12-layer representative gate.

======================================================================
FINAL PRIORITY
======================================================================

Do not optimize three architectures.

We have two:

    p16/top4 = SAFE FALLBACK
    p32/top5 = PRIMARY TARGET

The immediate mission is:

    make p16/top4 unquestionably usable

while developing the methodology required to move:

    p32/top5 from .967 validation cosine toward >= .98

without sacrificing:

    NMSE
    load balance
    generalization

Keep working autonomously through these gates.

Do not stop merely to report incremental progress.

Stop only for:
    destructive-risk decisions,
    provenance risk,
    a genuine architectural fork,
    or after p16/top4 has been locked and the next major p32/top5 decision
    boundary is reached.