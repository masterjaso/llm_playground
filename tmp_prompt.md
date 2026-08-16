Yes. I would now explicitly run **two research tracks in parallel**:

* **p16/top4 = proof-of-method / near-term production candidate.** It already has green validation at `0.981599` cosine and misses holdout by only ~`0.003`, while the same holdout basis has exact-oracle cosine `0.982185`. That says the 70.6% sparse architecture is viable and the remaining issue is learned routing/generalization.
* **p32/top5 and p32/top4 = stretch product targets.** Top5 activates `3584 / 17408 = 20.59%` of the dense FFN, i.e. **79.41% FFN reduction**. Top4 activates `3072 / 17408 = 17.65%`, i.e. **82.35% reduction**. Those are absolutely worth pursuing once we apply the lessons learned from p16.

And yes, I would add a **hard command-execution contract** to Luna. The old behavior where a shell/Git command silently sits for hours is unacceptable for this workflow.

One correction on the Qwen 35B-A3B comparison: `3B active / 35B total` is roughly an 8.6% whole-model active-parameter fraction, whereas our `17.6–20.6%` numbers are **FFN-width activation fractions only**, so they are not apples-to-apples. But your broader point is right: **p32/top4 or top5 gets us into a much more compelling sparse-compute regime** than p16/top4.

I would append this to Luna's takeover prompt:

```text
======================================================================
NEW PRODUCT PRIORITY — KEEP p16/top4, PUSH p32/top5 AND p32/top4
======================================================================

We now have two explicit product tracks.

TRACK A — PROVE THE METHOD

    p16/top4
    active width = 5120
    FFN reduction = 70.59%

This remains the proof-of-method architecture because it has already reached:

    validation:
        NMSE       0.022887
        cosine     0.981599
        dead       0
        load CV    0.4421

    holdout:
        NMSE       0.027239
        cosine     0.977044
        dead       0
        load CV    0.4713

and its exact holdout oracle reaches:

        cosine     0.982185

Therefore p16/top4 reconstruction capacity is sufficient.
Its current remaining blocker is robust selector generalization / the
quality-vs-load selection tradeoff.

Continue solving that.

TRACK B — HIGH-SPARSITY PRODUCT TARGET

We ALSO want to aggressively test:

    p32/top5
        32 routed experts
        expert width = 512
        shared width = 1024
        top-k = 5

        active width:
            1024 + 5*512 = 3584

        active FFN fraction:
            3584 / 17408 = 20.59%

        FFN reduction:
            79.41%

and:

    p32/top4
        active width:
            1024 + 4*512 = 3072

        active FFN fraction:
            17.65%

        FFN reduction:
            82.35%

The preferred product outcome is now:

    p32/top4 or p32/top5 green
        BEST

    p16/top4 green
        ACCEPTABLE / PROOF PRODUCTION TARGET

Do not discard p16/top4 while pursuing p32.

Use p16/top4 to solve the TRAINING METHOD.
Use p32/top5/top4 to determine how far sparsity can be pushed.

======================================================================
P32 MUST BENEFIT FROM EVERYTHING LEARNED ON p16
======================================================================

Do NOT judge p32/top5 from the earlier result alone.

That earlier p32/top5 experiment:

    validation cosine ~0.9673
    NMSE ~0.0449
    load CV ~0.5366

was performed before the full p16 hard-token/basis/refined-routing progress
and reused the p32/top6 partition.

It is not sufficient evidence that a properly optimized p32/top5 cannot work.

For p32/top5 and p32/top4:

1. build topology-specific FIT-only partitions/refinements where justified;

2. apply the hard-token/basis lessons learned from p16;

3. evaluate exact or strongest practical frozen oracle BEFORE expensive
   student training;

4. measure a LOAD-AWARE oracle as well as unconstrained reconstruction oracle;

5. train selectors only for architectures whose oracle evidence makes the
   green gate plausible.

For each architecture report:

    exact/global oracle cosine
    global NMSE
    hard-quartile cosine
    oracle load CV
    load-constrained oracle cosine
    load-constrained oracle CV
    expert usage
    dead experts

Do not simply ask:

    "Can an unconstrained oracle reconstruct?"

Ask:

    "Does there exist a top-k assignment with BOTH:
         cosine >= .98
         load CV <= .50
     while keeping NMSE <= .05?"

This is especially important because the current p16 exact oracle has excellent
cosine but load CV above the production threshold.

======================================================================
LOAD-CONSTRAINED ORACLE — HIGH PRIORITY
======================================================================

The current unconstrained exact oracle optimizes reconstruction independently
per token.

That gives p16/top4 roughly:

    learned holdout:
        cosine .9770
        CV     .471

    exact oracle holdout:
        cosine .9822
        CV     .667

Therefore perfect imitation of the unconstrained oracle is NOT the actual
production objective.

Implement a bounded load-aware oracle / assignment diagnostic.

A reasonable formulation is a Lagrangian:

    token_reconstruction_loss
        +
    sum(selected expert prices)

where expert prices are adjusted iteratively to reduce global imbalance.

Alternatively use another tractable constrained assignment approximation.

Goal:

    maximize reconstruction/cosine
    subject to approximately:
        load CV <= .50

Report the quality-vs-load Pareto curve.

Perform this for:

    p16/top4
    p32/top5
    p32/top4

This tells us whether each topology actually has a simultaneous
quality+balance solution before spending large training budgets.

======================================================================
SELECTOR GENERALIZATION — DATA BEFORE PARAMETER COUNT
======================================================================

We have already tested:

    linear selector
    low-rank nonlinear hidden-128
    nonlinear hidden-512
    listwise/exact-set objectives
    set-margin objectives
    soft top-k surrogate

No generic larger selector has yet clearly solved holdout generalization.

Therefore do not repeatedly make the router larger.

Investigate whether ROUTER TRAINING DATA is the limiter.

The selector currently learns a high-dimensional combinatorial decision from
~115k FIT activation states.

Consider capturing a much larger LAYER-0-ONLY router-training corpus:

    initial target:
        several hundred thousand diverse token states

    potentially:
        ~1M if storage/throughput is reasonable

This is much cheaper than resuming 64-layer replay.

Keep:
    validation untouched
    current holdout untouched

Exact/strong oracle-label the additional FIT router examples.

Freeze the good expert/shared basis for selector-only experiments where
possible.

Measure whether selector exact-set recall / top-k recall improves on the
existing validation set.

======================================================================
SELECTOR FEATURES — USE EXISTING NONLINEAR SIGNAL
======================================================================

One promising bounded experiment is to give the selector information from the
always-computed shared expert.

Current selector is primarily:

    x -> router

But the routing decision is trying to approximate a nonlinear residual:

    dense FFN target - shared expert response

The shared branch already computes useful nonlinear SwiGLU features.

Test ONE bounded feature-enhanced selector such as:

    features =
        x
        +
        compressed/shared-branch feature

or:

    concat(
        low-rank projection of x,
        low-rank projection of shared activation/output
    )
        ->
    small selection head

Do not make the selector enormous.

The purpose is to expose nonlinear residual-relevant information, not add a
second FFN.

Measure actual selector overhead.

======================================================================
VALIDATION DISCIPLINE
======================================================================

The existing validation split has now been used for many adaptive research
decisions.

Do not repeatedly tune until the same 16,384 rows barely pass.

Create an additional untouched SHADOW VALIDATION set if source/corpus capture
allows it without contaminating existing holdout semantics.

Preferred contract:

    ROUTER FIT / expanded FIT
        -> optimizer updates

    validation-A
        -> ordinary experiment selection

    shadow-validation-B
        -> confirmation of selector generalization

    existing full holdout
        -> finalist confirmation only

Do not use the existing opened holdout as an optimization signal.

======================================================================
MANDATORY CLI / TOOL EXECUTION CONTRACT
======================================================================

No command is allowed to silently consume hours.

Every shell/CLI invocation must have:

    1. START marker
    2. bounded timeout unless explicitly classified LONG_RUNNING
    3. SUCCESS / FAILURE / TIMEOUT marker
    4. exit code
    5. elapsed time

Example required terminal semantics:

    __CMD_START__ name=<name> timestamp=<time>

    ...

    __CMD_DONE__ name=<name> rc=0 elapsed=<seconds>

or:

    __CMD_FAILED__ name=<name> rc=<code> elapsed=<seconds>

or:

    __CMD_TIMEOUT__ name=<name> elapsed=<seconds>

A command that produces no terminal marker is NOT considered complete.

======================================================================
NORMAL COMMAND TIMEOUTS
======================================================================

Default categories:

FAST:
    git log
    git rev-parse
    git check-ignore
    git diff --stat
    gh API metadata lookup
    file existence checks
    JSON inspection

    timeout <= 60 seconds

MEDIUM:
    pytest subset
    repository searches
    report generation
    checkpoint metric reload

    timeout <= 5 minutes unless justified

LONG_RUNNING:
    actual CUDA training
    large oracle enumeration
    teacher capture
    deliberate large dataset generation

Only LONG_RUNNING work may exceed those bounds.

Long-running work must produce heartbeat/progress output.

======================================================================
LONG-RUNNING HEARTBEAT CONTRACT
======================================================================

Any intentional process expected to run >5 minutes must print or update at
least every 30-60 seconds:

    __HEARTBEAT__
    task=<task>
    elapsed=<seconds>
    progress=<completed>/<total if known>
    last_checkpoint=<path if applicable>
    gpu=<device if applicable>

Also persist a small progress/status JSON file such as:

    {
      "status": "RUNNING",
      "task": "...",
      "started": "...",
      "last_heartbeat": "...",
      "completed": ...,
      "total": ...,
      "pid": ...,
      "code_commit": "..."
    }

On completion write:

    "status": "SUCCESS"

On failure:

    "status": "FAILED"

The agent must never infer that silence means useful progress.

======================================================================
GIT / GITHUB COMMAND RULES
======================================================================

Do NOT use expensive whole-worktree commands such as:

    git status --short --ignored

against the giant runs tree.

Prefer:

    git status --short --untracked-files=no
    git rev-parse HEAD
    git rev-parse origin/<branch>
    git diff --name-only
    git check-ignore -v <specific-path>

For GitHub CLI / gh:

    disable paging:
        GH_PAGER=cat
        PAGER=cat
        GIT_PAGER=cat

Do not launch interactive authentication, editors, pagers, or prompts.

Use machine-readable output where possible:

    --json
    --jq

Bound every metadata/query call with the FAST timeout.

If a GitHub command times out:

    kill it
    emit __CMD_TIMEOUT__
    use a more targeted API request

Do not simply wait indefinitely.

======================================================================
NO INTERACTIVE CLI
======================================================================

Commands must be non-interactive.

Disable:

    pagers
    editors
    credential prompts
    confirmations requiring stdin

where appropriate.

Examples:

    GIT_TERMINAL_PROMPT=0
    GIT_PAGER=cat
    GH_PAGER=cat

If authentication is unavailable, FAIL FAST and report it.

Do not sit waiting for credential input.

======================================================================
SHELL WRAPPER
======================================================================

Implement or use one reusable guarded-command wrapper rather than relying on
discipline at every call.

It should accept:

    command
    timeout
    task name

and guarantee one of:

    DONE
    FAILED
    TIMEOUT

with elapsed time and exit code.

For native Windows, implement this in PowerShell or Python if necessary.

Use it for:
    git
    gh
    pytest
    report scripts
    lightweight diagnostics

Do NOT wrap actual GPU training with a short timeout; training instead uses
the long-running heartbeat contract.

======================================================================
AGENT BEHAVIOR ON TIMEOUT
======================================================================

When a command reaches its timeout:

DO:
    terminate the process tree
    record the timeout
    inspect why it was slow
    choose a narrower command
    continue useful work

DO NOT:
    issue the same command repeatedly
    wait several hours
    interpret silence as progress
    block all research on repository bookkeeping

Maximum retry for an identical timed-out command:
    ONE

After that, change strategy.

======================================================================
CURRENT PRIORITY ORDER
======================================================================

1. p16/top4 load-aware oracle and selector-generalization work

2. p32/top5 exact + load-aware oracle using an appropriately refined basis

3. p32/top4 exact + load-aware oracle

4. expand selector training data / selector feature experiment

5. obtain a robust green candidate

6. only then representative layers

Do NOT resume full 64-layer replay yet.
```

### I would make one architectural adjustment to our strategy

We previously treated p16/top4 as something we needed to finish before really touching p32. I would relax that now.

We have learned enough from p16 that we can cheaply run **oracle-level p32 research in parallel**. We don't need to spend a full training campaign on p32 yet, but exact oracle + refined partition + load-constrained oracle is cheap compared with 64-layer training and can tell us whether `79–82%` is realistically reachable.

The decision tree I want now is:

```text
                   refined method
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
      p16/top4                    p32 family
      70.6%                       │
          │                  ┌────┴────┐
          │                  ▼         ▼
 load-aware oracle       p32/top5   p32/top4
 selector generalize       79.4%      82.4%
          │                  │          │
          └──────────────┬───┴──────────┘
                         ▼
              quality/load Pareto
                         │
                         ▼
             choose TRAINABLE SET
```

And I strongly support the **DONE/FAILED/TIMEOUT contract**. For an autonomous agent, a command that can silently wait forever is effectively a workflow bug. The new session should treat bounded command execution and heartbeats as part of the engineering requirements, not optional convenience. Current pushed head remains `0db04c2`.
