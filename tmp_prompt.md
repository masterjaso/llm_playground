Yes. I’d make **corpus-v2 construction a hard Phase 0 and forbid any further meaningful basis refinement until it is frozen**.

Also, your inability to run the dense checkpoint autoregressively to create agent trajectories is **not a blocker**. We can obtain existing code/agent trajectories externally, then feed those fixed sequences through the dense Qwen checkpoint only for activation/teacher capture. The external trajectory supplies *where in token/context space we sample*; Qwen still supplies the actual FFN reconstruction target.

There are good public sources for this. NVIDIA’s current Open-SWE-Traces contains large numbers of SWE-agent/OpenHands trajectories, including a Qwen3.5-generated non-thinking split, which is particularly attractive as an input-distribution source for us. ([Hugging Face][1]) CodeSearchNet gives repository-separated code/documentation data, and BigCode provides code plus issue/PR-oriented sources with provenance/licensing metadata, though The Stack requires careful compliance with its current access and source-license terms. ([GitHub][2])

The branch is still at `b750733f3c2c337410f466c9bfc5c7db3042f941`, so this can supersede the previous takeover prompt cleanly.

```text
D2M TAKEOVER — CORPUS-V2 FIRST, THEN GENERALIZATION-FIRST BASIS REFINEMENT

Repository:
    C:\workplace\llm_playground

Branch:
    agent/windows-dense2moe-real-pipeline

Observed remote HEAD at prompt creation:
    b750733f3c2c337410f466c9bfc5c7db3042f941

Execution:
    Native Windows.

Before doing anything:
- pull/inspect actual HEAD
- reconcile HANDOFF.md, state.json, decision-register.json, and latest takeover report
- preserve all previous captures/checkpoints/reports
- do not open official holdout
- do not begin representative replay
- do not begin full64 replay

======================================================================
ACTIVE TARGETS
======================================================================

Exactly two sparse targets remain active:

SAFE FALLBACK:
    p16/top4
    16 experts x 1024
    shared width 1024
    top4
    70.59% FFN reduction

PRIMARY PRODUCT TARGET:
    p32/top5
    32 experts x 512
    shared width 1024
    top5
    79.41% FFN reduction

Do NOT work p32/top4.

Current working blocker for both:
    BASIS_QUALITY

However, current ~0.922 oracle results were produced from tiny bounded
continuations and must NOT be interpreted as topology capacity limits.

======================================================================
CRITICAL CHANGE: STOP REFINEMENT UNTIL CORPUS-V2 EXISTS
======================================================================

DO NOT launch another meaningful basis-refinement campaign on the existing
WikiText/Gutenberg-heavy fresh corpus.

The previous project trajectory demonstrated that we can appear very close
on one activation distribution while being far from robust on genuinely new
data.

We will not repeat that.

Corpus-v2 must be created, audited, split, fingerprinted, and frozen BEFORE
the next serious p16/p32 basis campaign.

Small implementation smoke tests are allowed.

No serious optimization is allowed before corpus-v2 is ready.

======================================================================
IMPORTANT CONSTRAINT: NO LOCAL DENSE-MODEL TRAJECTORY GENERATION
======================================================================

The user CANNOT afford/run the dense source model to autoregressively generate
a large coding/agent trajectory corpus.

Do NOT make source-model generation a prerequisite.

Instead:

1. Acquire static coding/agentic text and trajectories from external public
   sources.

2. Normalize those trajectories into sequences compatible with the source
   model chat/tool format where practical.

3. Feed those FIXED sequences through the existing dense Qwen teacher/capture
   pipeline.

4. Use the source checkpoint ONLY to obtain:
       hidden states
       dense FFN outputs
       reconstruction targets

The externally sourced model responses/tool traces are NOT teacher labels.

The Qwen dense FFN output remains the teacher target.

This distinction is fundamental:

    external corpus = WHERE we sample the activation manifold
    dense Qwen oracle = WHAT the sparse FFN must reproduce

We do NOT need Qwen itself to author the corpus.

======================================================================
WHY STATIC AGENT TRAJECTORIES ARE VALID
======================================================================

A coding-agent trajectory can already contain:

    user request
    repository context
    assistant action
    shell command
    shell output
    file read
    patch
    test result
    compiler error
    retry
    tool call
    tool response
    final answer

Feed the complete causal sequence through the dense teacher.

This exposes the sparse-conversion training pipeline to the kinds of token
contexts encountered during real agent execution without requiring the local
dense model to generate those trajectories itself.

Prefer trajectories structurally similar to the actual production workload.

======================================================================
CORPUS-V2 PRODUCTION OBJECTIVE
======================================================================

This converted model will be used predominantly for:

    coding
    software engineering
    repository reasoning
    tool calling
    terminal interaction
    agentic multi-step work

Therefore corpus-v2 should be PRODUCTION-WEIGHTED, while retaining enough
general-domain material to prevent narrow specialization.

Initial target mixture by activation-token budget:

    40-50% REAL SOURCE CODE / REPOSITORY CONTEXT

        Python
        TypeScript / JavaScript
        C / C++
        Rust
        Go
        Java / C#
        GDScript
        SQL
        Bash
        PowerShell
        HTML/CSS/config formats

        Include:
            implementations
            tests
            configs
            build files
            CI
            package manifests
            schemas
            migrations

    25-30% AGENTIC SOFTWARE-ENGINEERING TRAJECTORIES

        Include sequences containing:
            issue/problem statement
            repository exploration
            grep/search
            file reads
            tool calls
            shell commands
            command output
            stack traces
            compiler errors
            test failures
            patches/diffs
            retries
            validation
            final responses

    10-15% SOFTWARE-ENGINEERING NATURAL LANGUAGE

        READMEs
        API documentation
        architecture docs
        issue descriptions
        PR descriptions
        code reviews
        commit messages
        specifications

    5-10% STRUCTURED / TOOL MATERIAL

        JSON
        JSON Schema
        XML
        YAML
        TOML
        shell transcripts
        function/tool calls
        tool results
        git output
        compiler output
        logs

    10-15% GENERAL/STEM PRESERVATION

        technical prose
        normal natural language
        factual text
        math/scientific text
        existing broad-text corpus

Do not treat these percentages as immutable hyperparameters.

They are an initial production-weighted mixture.

No single repository, book, dataset, or source family may dominate.

======================================================================
EXTERNAL CORPUS SOURCES
======================================================================

Prioritize sources whose provenance/license/terms can be recorded clearly.

Strong initial candidates include:

A. OPEN-SWE-TRACES

Use as a major source of agentic software-engineering trajectories.

Prefer, where practical:

    Qwen3.5 non-thinking trajectories
    OpenHands trajectories
    SWE-agent trajectories

The Qwen3.5 trajectory split is particularly interesting because its style is
closer to the model family we are converting.

IMPORTANT:

Do NOT treat another model's reasoning as our teacher.

Use trajectory text/actions/observations only as activation contexts.

Prefer:
    actions
    observations
    commands
    code
    diffs
    tool interactions
    assistant responses

Do not intentionally train reasoning-efficiency behavior.

Do not add a loss over external model answers.

If a source contains explicit private/internal chain-of-thought-like fields,
exclude those by default unless they are ordinary visible model output that
would genuinely exist in our intended runtime format.

B. PERMISSIVELY LICENSED REAL REPOSITORIES

Curate real repositories across our target language families.

Prefer active, nontrivial projects containing:
    tests
    docs
    issue-related changes
    build tooling
    CI
    multiple interacting modules

Apply a license whitelist.

Record:
    repo URL
    commit SHA
    license
    language
    file list/hash
    split assignment

C. CODESEARCHNET OR COMPARABLE CODE/DOC SOURCES

Useful for:
    code
    documentation
    code-comment relationships

Do not depend solely on it because agent behavior requires much more than
function bodies.

D. BIGCODE / THE STACK FAMILY

May be used as a source of additional code/issues/PR-style material only if:

    current access terms are satisfied
    source licenses are acceptable
    provenance is retained
    removals/opt-out requirements are respected

Do not blindly bulk-ingest it.

E. REAL REPOSITORY HISTORY

For selected permissive repositories, derive static examples from:

    issues
    commits
    diffs
    PR descriptions
    test changes
    bug fixes

This is valuable because it provides real software-maintenance structure
without requiring local LLM generation.

======================================================================
BENCHMARK CONTAMINATION POLICY
======================================================================

Do NOT train on benchmark instances that we intend to use as final evaluation.

Maintain a denylist for intended downstream benchmarks.

At minimum, avoid training directly on held-out evaluation tasks from any
benchmark we plan to report later.

Repository-level overlap counts as contamination even if the exact issue is
different when we are claiming repository-generalization evidence.

Track:
    repo
    task
    issue/PR identifier
    commit
    benchmark membership if known

======================================================================
CORPUS QUALITY > RAW SIZE
======================================================================

Do not solve this by simply downloading millions of random code files.

Favor:

    real maintained projects
    tests
    buildable projects
    realistic diffs
    tool interactions
    error/recovery sequences
    cross-file context
    useful documentation

Reject or heavily downweight:

    generated boilerplate
    vendored dependencies
    minified code
    lockfiles dominating the mix
    enormous generated sources
    duplicated forks
    obvious spam
    binary-derived text
    repeated templates

Near-deduplicate corpus-v2 before capture.

======================================================================
REPOSITORY/DOCUMENT DISJOINTNESS IS MANDATORY
======================================================================

Random token splits are NOT acceptable generalization evidence.

Build the split BEFORE activation capture where practical.

No repository may span:

    FIT
    GATE-A
    SHADOW-B
    SHADOW-C

For non-code sources:

    no document may span those roles

For agent trajectories:

    avoid sharing the same underlying issue/PR/task across roles

Where possible, also keep related forks out of opposing splits.

======================================================================
GENERALIZATION MATRIX
======================================================================

Create and freeze:

1. FIT-TRAIN

Optimizer data.

Production-weighted coding/agentic mixture.

2. FIT-DEV

From FIT source families but independent examples.

Can be evaluated frequently.

Used for:
    implementation iteration
    pilot checkpointing
    LR/loss debugging

NOT evidence of broad generalization.

3. GATE-A

Repository/document-disjoint.

Used for real checkpoint selection.

Evaluate only at meaningful milestones.

4. SHADOW-B

Different repositories and partially different task/language composition.

Confirmation only.

Do not use for hyperparameter tuning.

5. SHADOW-C

A harder, independently sourced coding/agentic shadow.

Different repositories and preferably different trajectory source/framework.

Do not open until a candidate has already survived A and B.

6. GENERAL-PRESERVATION CANARY

Use the existing WikiText/Gutenberg/general fresh data and selected historical
data as regression canaries.

These sets have already influenced project decisions and are NOT untouched
generalization evidence.

Their purpose is to detect catastrophic narrowing.

7. OFFICIAL HISTORICAL HOLDOUT

CLOSED.

Do not open.

======================================================================
SOURCE-FAMILY DIVERSITY INSIDE EACH MAJOR SPLIT
======================================================================

Within FIT/A/B/C record metrics by:

    language
    repository
    source family
    task type
    trajectory framework
    code vs prose vs tool/log
    sequence length bucket

Important task families:

    code generation
    bug fixing
    test repair
    test creation
    refactoring
    repository navigation
    code explanation
    dependency problems
    configuration problems
    build failures
    type errors
    compiler errors
    runtime exceptions
    API usage
    git operations
    multi-file changes
    long-context repository analysis
    JSON/tool calling
    shell interaction
    iterative diagnose-edit-test loops

Do not let "Python code completion" become a proxy for coding-agent coverage.

======================================================================
SAMPLING
======================================================================

Do not sample only proportional to corpus size.

Use capped/balanced source-family sampling.

Cap:
    per repository
    per source
    per language where necessary

so large repositories/datasets cannot dominate.

Record the ACTUAL optimizer mixture in every run.

======================================================================
CORPUS-V2 SIZE
======================================================================

Do not immediately create a gigantic activation store.

First produce a source corpus large enough to support staged captures.

Suggested activation progression:

    CORRECTNESS/SMOKE:
        2k-4k states

    PILOT:
        ~32k states

    SERIOUS:
        ~128k states

    BROAD FIT:
        500k-1M+ states if trajectory continues improving

The corpus source itself may be considerably larger.

Activation capture can be progressively expanded without changing the frozen
split identities.

======================================================================
CORPUS-V2 REQUIRED RECEIPT
======================================================================

Before basis refinement resumes, create a machine-readable corpus-v2 receipt.

Record:

    corpus version
    creation code SHA
    all source URLs/identifiers
    source revisions
    repository commit SHAs
    licenses/terms metadata
    source hashes
    document hashes
    language
    source family
    task family
    trajectory framework/model if known
    token counts
    split assignments
    dedup statistics
    repo/document overlap checks
    benchmark denylist checks

Report final token/state budget per:

    FIT
    FIT-DEV
    A
    B
    C
    preservation canary

Do not proceed if overlap/provenance checks fail.

======================================================================
ONLY AFTER CORPUS-V2 IS FROZEN:
ORACLE-ROUTED BASIS REFINEMENT
======================================================================

The current bounded basis trainer freezes the selector but still routes basis
training through the selector's selected experts.

Stop doing that for capacity learning.

The selector is already known to be weak.

Implement ORACLE-ROUTED / EM-LIKE basis refinement.

E STEP:

For current basis and each training state:

    obtain a strong sparse assignment
    obtain positive route coefficients where applicable

For p16/top4:
    exhaustive top4 assignment where practical

For p32/top5:
    validated bounded strong candidate search
    expand candidate pool when evidence suggests candidate limitation

M STEP:

Using frozen oracle assignments for a bounded interval, update only:

    shared gate/up/down
    expert gate/up/down
    expert scales

against the dense source FFN target.

Then recompute assignments.

The learned selector does NOT control which experts receive basis gradients.

======================================================================
SHARED → RESIDUAL → JOINT REFINEMENT
======================================================================

Stage 1:
    shared-foundation refinement

Train the shared 1024 branch toward broadly useful dense FFN behavior.

Stage 2:
    routed residual specialization

Define:

    residual = dense_teacher - shared_output

Train experts to reconstruct residual under oracle routing.

Stage 3:
    joint basis refinement

Jointly refine:
    shared
    experts
    scales

under periodically refreshed oracle assignments.

======================================================================
DO NOT OVERFIT CORPUS-V2 EITHER
======================================================================

The existence of a better corpus does not remove overfitting risk.

Use this experiment ladder:

SMOKE:
    FIT only

PILOT:
    FIT + FIT-DEV

SERIOUS:
    FIT + FIT-DEV
    occasional A

PROMOTION:
    select using A

CONFIRMATION:
    B exactly once per selected finalist

ROBUST CONFIRMATION:
    C only after A+B success

Do not look at B/C after every training run.

If B or C fails:

    record failure
    return to FIT/design
    formulate a hypothesis
    train a NEW candidate

Do not tune interactively while repeatedly observing shadow metrics.

======================================================================
PER-DOMAIN METRICS
======================================================================

For all meaningful evaluation cohorts report:

    aggregate cosine
    aggregate NMSE
    hard-quartile cosine

plus per:

    language
    repository family
    source family
    task family

Report:

    worst-domain cosine
    best-domain cosine
    domain spread
    FIT-DEV -> A gap
    A -> B gap
    B -> C gap

Flag:

    any material domain collapse
    any independent-cohort cosine drop > ~0.01
    any important production domain below ~0.975 when aggregate is near green

These are investigation triggers.

Do not lower the global production gate.

======================================================================
P16/TOP4 FIRST
======================================================================

p16/top4 remains the methodology proving ground and safe fallback.

After corpus-v2 and oracle-routing are ready:

1. 2k-4k smoke
2. 16k-32k GPU pilot
3. 64k-128k serious run if trajectory is strong
4. scale toward broader FIT only after material improvement

Primary basis criterion:

    trained-basis unconstrained oracle

Production reconstruction target:

    cosine >= .98
    NMSE <= .05

Do not focus on selector quality until basis capacity is demonstrated.

Do not focus aggressively on load CV while basis cosine remains ~.92.

======================================================================
MATERIAL-IMPROVEMENT FALSIFIER
======================================================================

The previous tiny continuations improved only ~0.0004 cosine.

A real 16k-32k oracle-routed diverse-data pilot should materially outperform
that trajectory.

If the first serious pilot does NOT improve oracle cosine by approximately
0.01 absolute or show a comparably convincing trajectory:

DO NOT blindly scale to the full corpus.

Investigate:

    partition initialization
    shared capacity
    residual decomposition
    assignment quality
    coefficient fitting
    optimization scale/LR
    expert specialization
    loss normalization

before spending large compute.

======================================================================
QUALITY FIRST, LOAD SECOND
======================================================================

Current oracle reconstruction is too poor for load balancing to be the main
optimization target.

Phase order:

1. reconstruction capacity
2. robust cross-domain reconstruction
3. quality/load joint geometry
4. selector imitation/generalization

Only when unconstrained basis oracle approaches >= .97 should load-aware
basis pressure become a major objective.

Final gate remains:

    cosine >= .98
    NMSE <= .05
    load CV <= .50
    dead experts = 0

Evaluate load on hundreds/thousands of states, not tiny 16-token samples.

======================================================================
SELECTOR ONLY AFTER BASIS CAPACITY
======================================================================

Once a basis is robust across the required independent cohorts:

    freeze basis

Generate/cache oracle assignments over diverse FIT.

Train selector against:

    oracle route IDs
    route coefficients
    reconstruction regret
    cosine regret

Use actual reconstruction cost to weight mistakes.

Selection:
    FIT-DEV for frequent telemetry
    A for finalist selection

Confirmation:
    B
    then C

Only after robust A/B/C evidence should an official holdout confirmation be
considered.

======================================================================
P32/TOP5 AFTER P16 RECIPE IS WORKING
======================================================================

Do not make p32 independently rediscover the basis recipe.

Once p16 refinement clearly works:

Transfer:
    p16 shared 1024 branch

Investigate structured initialization:

    each p16 1024 expert
        ->
    two p32 512 experts

Prefer contribution-aware/neuron-clustered split over arbitrary halves.

Then run p32-specific:

    top5 oracle routing
    residual specialization
    joint refinement

using corpus-v2.

p32/top5 remains the primary product objective.

Decision guidance:

    >= .98 / <= .05
        proceed aggressively

    .975-.98
        continue refinement

    still .92-.94 after strong p16-derived initialization +
    meaningful diverse GPU training
        investigate real topology capacity limitation

Do not reject p32 using current tiny-pilot evidence.

======================================================================
GENERALIZATION STATUS TERMINOLOGY
======================================================================

Use explicit labels:

    FIT-GREEN
    A-GREEN
    B-GREEN
    C-GREEN
    ROBUST-GREEN

Do not call a candidate "close" merely because FIT or A is green.

ROBUST-GREEN requires independent evidence.

======================================================================
REPLAY REMAINS CLOSED
======================================================================

Do not begin representative replay.

Do not begin full64 conversion.

Do not open official holdout.

Future representative set remains:

    0 1 2 3
    28 29 30 31
    60 61 62 63

with:

    LINEAR_A
    LINEAR_B
    LINEAR_C
    FULL_ATTENTION

Only SwiGLU FFNs may be replaced.

======================================================================
NO REASONING POST-TRAINING
======================================================================

Do not optimize reasoning length.

Do not perform reasoning-efficiency post-training.

External agent trajectories are being used for activation-distribution
coverage, not to introduce a new reasoning objective.

Passive telemetry remains acceptable.

======================================================================
IMMEDIATE EXECUTION ORDER
======================================================================

1. Reconcile HEAD/state/HANDOFF.

2. Update stale project state:
       targets = p16/top4 + p32/top5
       blocker = BASIS_QUALITY
       replay = blocked

3. STOP serious existing-corpus basis refinement.

4. Design corpus-v2 manifest/schema and benchmark denylist.

5. Acquire a coding/agent-heavy external corpus.

6. Prioritize:
       externally sourced agent trajectories
       permissively licensed real repositories
       issues/PRs/diffs/tests/docs
       structured tool/log material

7. Preserve a smaller general/STEM corpus.

8. Deduplicate and license/provenance audit.

9. Freeze repo/document-disjoint:
       FIT
       FIT-DEV
       A
       B
       C

10. Publish corpus-v2 receipt.

11. Capture a small dense-teacher activation smoke from corpus-v2.

12. Implement/test oracle-routed basis training.

13. Run p16 16k-32k GPU pilot.

14. Evaluate FIT-DEV and meaningful A sample.

15. Scale only if materially improving.

16. Establish robust p16 basis methodology.

17. Transfer methodology/initialization to p32/top5.

18. Do not open B/C casually.

19. Do not open official holdout.

20. Do not start representative/full64 replay.

======================================================================
TAKEOVER SUCCESS CONDITION
======================================================================

The next handoff should NOT simply say:

    "trained longer and gained +0.0004"

It should answer:

A. CORPUS

    Is corpus-v2 coding/agent-heavy?
    Are sources/repositories documented?
    Are splits truly repo/document disjoint?
    Is benchmark contamination controlled?
    Is general-domain preservation retained?

B. P16

    Did oracle-routed diverse training materially improve the basis?
    What is the cross-domain trajectory?
    What is the A generalization gap?

C. P32

    Has the p16 methodology been transferred?
    Is p32 responding to the improved training method?

D. SCIENTIFIC QUESTION

    Are we learning a basis that reconstructs the dense Qwen FFN across the
    production coding/agent manifold,

    OR

    are we merely fitting another convenient corpus?

Do not proceed to expensive later-stage conversion until the evidence supports
the former.

Proceed autonomously through the bounded gates above.
```

One addition I feel strongly about: **prefer the Qwen3.5 non-thinking portion of Open-SWE-Traces over importing arbitrary visible “reasoning” from other models**. It gives us realistic software-agent conversations/tool interactions without making Minimax-style reasoning prose a dominant part of the activation distribution. Open-SWE-Traces explicitly separates those kinds of traces, which makes that feasible. ([Hugging Face][3])

And I would keep the old WikiText/Gutenberg capture rather than delete it—it has become a useful **regression/OOD canary**, just not the corpus we should optimize around anymore.

[1]: https://huggingface.co/datasets/nvidia/Open-SWE-Traces?utm_source=chatgpt.com "nvidia/Open-SWE-Traces · Datasets at Hugging Face"
[2]: https://github.com/github/CodeSearchNet?utm_source=chatgpt.com "GitHub - github/CodeSearchNet: Datasets, tools, and benchmarks for representation learning of code. · GitHub"
[3]: https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/81ad5141cac45fccfc5af0528fea819c6989fc05/README.md?utm_source=chatgpt.com "README.md · nvidia/Open-SWE-Traces at 81ad5141cac45fccfc5af0528fea819c6989fc05"
