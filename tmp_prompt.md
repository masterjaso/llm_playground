MID-RUN COURSE CORRECTION
TOP-K / EXPERT-GRANULARITY SEARCH BEFORE FURTHER DEEP TOP-2 TRAINING

IMPORTANT
=========

Do NOT throw away the work currently in progress.

If a training epoch/stage is actively running:

    finish the current safe checkpoint/epoch if reasonably short,
    persist its checkpoint and metrics,
    classify it as the p8/top2 baseline,
    then apply this directive.

If a long multi-hour top2-only optimization sequence has not yet begun,
do not start it.

Do NOT restart teacher capture.
Do NOT regenerate the corpus.
Do NOT discard existing layer-0 train or holdout activations.

The new priority is to determine the best TOP-K / EXPERT-GRANULARITY /
GATING design before spending substantial compute optimizing top-2 alone.

======================================================================
WHY THIS DIRECTION CHANGED
======================================================================

Current evidence suggests:

    p8/top2 positive/non-normalized oracle:
        NMSE ~0.0465

    current learned p8/top2 student:
        NMSE ~0.18+

Therefore:

    the dense→expert decomposition has meaningful potential,

but:

    top2 + current routing/gating may not be the best architecture.

We now want to answer:

    How many experts should fire per token?

and:

    For the same active-compute budget, are fewer large experts or more
    smaller experts better?

Do not assume top2 is optimal.

======================================================================
SCIENTIFIC SEARCH POLICY
======================================================================

Treat the architecture as a two-dimensional design space:

    expert granularity
        ×
    top-k

Initial profiles:

    p8:
        8 routed experts
        2048 width/expert
        1024 shared

    p16:
        16 routed experts
        1024 width/expert
        1024 shared

    p32:
        32 routed experts
        512 width/expert
        1024 shared

Evaluate top-k values:

    1
    2
    3
    4
    5
    6

Do NOT immediately train every combination.

Use inexpensive oracle/diagnostic evaluation first.

======================================================================
ACTIVE WIDTH REFERENCE
======================================================================

Dense source FFN width:

    17408

Active width is:

    shared_width + top_k * expert_width

Therefore:

p8:
    k=2:  5120   = 29.4% dense FFN
    k=3:  7168   = 41.2%
    k=4:  9216   = 52.9%
    k=5: 11264   = 64.7%
    k=6: 13312   = 76.5%

p16:
    k=2: 3072    = 17.6%
    k=3: 4096    = 23.5%
    k=4: 5120    = 29.4%
    k=5: 6144    = 35.3%
    k=6: 7168    = 41.2%

p32:
    k=2: 2048    = 11.8%
    k=3: 2560    = 14.7%
    k=4: 3072    = 17.6%
    k=5: 3584    = 20.6%
    k=6: 4096    = 23.5%

These are FFN active-width ratios, NOT exact whole-model active parameter
counts.

Calculate exact active parameter counts from the real tensor inventory for
reports.

======================================================================
PHASE A — FREEZE CURRENT TOP2 RESULT AS BASELINE
======================================================================

Record the best current p8/top2 checkpoint and metrics.

It becomes:

    BASELINE_P8_TOP2

Record:

    initialization
    router formulation
    gating formulation
    NMSE
    cosine
    dead experts
    load CV
    runtime
    VRAM
    training duration

Do not overwrite it.

Then suspend further deep top2-only tuning until the architecture search below
is complete.

======================================================================
PHASE B — AVOID HOLDOUT OVERFITTING
======================================================================

Do NOT repeatedly select architectures using the final holdout split.

We already have full layer-0 training activations.

Create a deterministic ARCHITECTURE_DEV subset from TRAIN data only.

Suggested:

    16k–32k representative train tokens

stratified across the known corpus domains if practical.

Use:

    ARCHITECTURE_DEV
        for oracle sweeps and architecture selection

Preserve:

    FULL HOLDOUT
        for confirmation of finalists only.

Do not change the existing train/holdout identity.

Record the deterministic dev-token IDs/hash as a derived research artifact.

Existing historical holdout oracle measurements remain valid evidence, but do
not continue repeatedly optimizing architecture directly against holdout.

======================================================================
PHASE C — p8 EXACT TOP-K ORACLE CURVE
======================================================================

Start with p8 because only eight experts exist and exact combination search is
tractable.

For top-k:

    1
    2
    3
    4
    5
    6

evaluate on ARCHITECTURE_DEV:

1. normalized/simplex routing oracle

2. exact positive/non-negative coefficient oracle

3. learned global expert-scale formulation

4. token-dependent positive-amplitude formulation/proxy if currently
   available

For p8, exhaustively enumerate expert combinations where computationally
reasonable.

Examples:

    C(8,2) = 28
    C(8,3) = 56
    C(8,4) = 70
    C(8,5) = 56
    C(8,6) = 28

No approximate search is necessary for these candidate counts.

Produce:

    reports/topk-p8-oracle-curve.json

with for each k:

    active FFN width
    active FFN ratio
    exact active parameter estimate
    NMSE
    cosine
    coefficient formulation
    expert usage
    oracle compute time

Identify the QUALITY ELBOW:

    the point beyond which another active expert yields little meaningful
    NMSE/cosine improvement.

Do not automatically prefer the lowest k.

======================================================================
PHASE D — SEPARATE TOP-K FROM GATING FORMULATION
======================================================================

Current evidence indicates coefficient semantics may matter almost as much as
expert count.

Therefore compare for each promising k:

A.
    top-k + normalized softmax
    selected coefficients sum to 1

B.
    top-k selection + independent positive amplitudes

C.
    top-k selection + normalized routing + learned output scale

D.
    if useful:
    token-dependent scale/amplitude head

We specifically need to answer:

    Is top2 bad because two experts are insufficient?

or:

    Is top2 bad because forcing the two coefficients onto a unit simplex is
    too restrictive?

or both?

Do not conflate these.

======================================================================
PHASE E — SAME-COMPUTE EXPERT GRANULARITY TEST
======================================================================

This is a priority experiment.

Compare architectures with approximately identical active FFN width but
different numbers/sizes of experts.

PAIR 1:

    p16 / top2
        active width = 3072

versus

    p32 / top4
        active width = 3072

PAIR 2:

    p16 / top3
        active width = 4096

versus

    p32 / top6
        active width = 4096

PAIR 3:

    p8 / top2
        active width = 5120

versus

    p16 / top4
        active width = 5120

These comparisons answer:

    At equal nominal FFN compute, do more smaller experts provide better
    reconstruction than fewer larger experts?

Use identical ARCHITECTURE_DEV examples.

Run the same oracle/coefficient formulations.

Produce:

    reports/equal-compute-expert-granularity.json

======================================================================
PHASE F — BUILD A PARETO FRONTIER
======================================================================

For every candidate record:

    profile
    number of total experts
    top_k
    shared width
    expert width
    active FFN width
    active FFN fraction
    exact active parameters/token
    oracle NMSE
    oracle cosine
    gating formulation
    expected dispatch count

Build the Pareto frontier for:

    quality
        versus
    active compute

A candidate is dominated if another candidate has:

    equal or better NMSE/cosine
    AND
    equal or lower active width.

Do not train dominated candidates.

======================================================================
PHASE G — CHOOSE ONLY 2–3 TRAINING CANDIDATES
======================================================================

After the oracle search, select only the strongest 2–3 candidates.

Likely categories, but DO NOT prejudge results:

    quality-first candidate
    balanced candidate
    maximum-sparsity viable candidate

Potential examples might be:

    p8/top3
    p16/top3
    p32/top4

but choose from measurements, not this suggestion.

Confirm the finalists once on the FULL HOLDOUT oracle before training.

======================================================================
PHASE H — TRAIN THE FINALISTS ON LAYER 0
======================================================================

For each selected architecture:

use identical:

    training activations
    holdout
    seed set
    optimizer budget
    stage schedule

Train actual students.

Do not give one candidate substantially more optimization budget than another
during the first comparison.

Record:

    initial NMSE/cosine
    oracle ceiling
    final trained NMSE/cosine
    oracle regret
    dead experts
    load CV
    wall-clock training time
    peak VRAM

If independent positive amplitudes were strongly favored by the oracle,
implement/train that gating formulation rather than evaluating only softmax
routing.

======================================================================
PHASE I — DO NOT USE NMSE ALONE
======================================================================

Architecture selection must consider:

PRIMARY:
    holdout NMSE
    holdout cosine

SECONDARY:
    dead experts
    load CV
    oracle regret
    reproducibility

EFFICIENCY:
    active FFN width
    active parameters/token
    number of expert dispatches

LATER:
    real inference tokens/sec

Do not select a design solely because its nominal active-parameter count is
lowest.

======================================================================
PHASE J — SEARCH STOP CONDITION
======================================================================

Stop increasing top-k when either:

1. another expert produces only marginal reconstruction gain,

or

2. active FFN compute becomes too close to dense to justify the complexity.

For p8, top5/top6 may primarily be diagnostic.

For p16/p32, top4/top6 can still represent strong sparsity and should not be
dismissed merely because k is larger.

In particular:

    p32/top4:
        ~17.6% of dense FFN width active

    p32/top6:
        ~23.5% of dense FFN width active

Both remain substantially sparse.

======================================================================
PHASE K — PRESERVE THE TRUE PRODUCT GOAL
======================================================================

The objective is NOT:

    "top2 at all costs"

and NOT:

    "minimum active parameters at all costs."

The objective is:

    retain as much Qwen3.8 quality as possible
        while
    removing a large majority of dense FFN work
        and
    producing a runtime architecture that performs well on consumer hardware.

A model using 20–25% of original FFN compute but preserving substantially more
quality may be preferable to one using 12% but suffering meaningful model
degradation.

Likewise, a model with four small expert dispatches may or may not outperform
one with two larger GEMMs despite identical nominal FLOPs.

Runtime benchmarking will decide that later.

======================================================================
DO NOT DO YET
======================================================================

Until this top-k search finishes:

DO NOT:

    deeply optimize p8/top2 for many additional runs
    propagate the full 131k train corpus to every representative layer
    train layers 16/32/48/63
    train all 64 layers
    commit to p16/top2 or p32/top2
    change the quality gates

Layer-0 cached real activations are sufficient for the current architectural
decision.

======================================================================
EXPECTED NEXT DELIVERABLE
======================================================================

Produce a compact architecture-search report:

    TOP_K_ARCHITECTURE_SEARCH.md/json

containing:

1. p8 k=1..6 oracle curves
2. normalized vs positive/token-amplitude gating comparison
3. equal-active-width p16/p32 comparisons
4. active parameter estimates
5. Pareto frontier
6. selected 2–3 training finalists
7. trained layer-0 comparison for those finalists
8. recommended architecture for representative-layer testing
9. exact reasoning for rejecting alternatives

Only after this decision is evidence-backed should deep representative-layer
training resume.

Continue autonomously after selecting the winner.
Do not stop merely to report the sweep if the next experiment is locally
executable.