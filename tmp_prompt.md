TAKEOVER TASK — FIX BASIS CAPACITY DIAGNOSTICS AND RETURN TO REFINEMENT

Repository:
C:\workplace\llm_playground

Branch:
agent/windows-dense2moe-real-pipeline

Execution:
Native Windows only.

Current observed remote HEAD:
f80d1262a24d98232b356681546c721b34a73e15

Before doing anything:
- Pull/inspect the actual branch HEAD and record it.
- Preserve all existing runs, captures, checkpoints, reports, partitions, and provenance.
- Do not reset or overwrite useful artifacts.
- Do not run representative-layer replay.
- Do not run 64-layer replay.
- Do not open or tune against the official holdout.
- Do not broaden this task into unrelated architecture experiments.

We have exactly TWO active sparse targets for this task:

1. p16/top4
   - 16 routed experts
   - routed expert width 1024
   - shared width 1024
   - top_k = 4
   - active FFN width = 5120
   - FFN reduction = 70.5882%
   - current safe/proof production target

2. p32/top5
   - 32 routed experts
   - routed expert width 512
   - shared width 1024
   - top_k = 5
   - active FFN width = 3584
   - FFN reduction = 79.41%
   - preferred aggressive product target

Do NOT work p32/top4 in this takeover.
Do NOT restart representative/full-model replay.
The goal is to get p16/top4 and p32/top5 back into productive layer-0 refinement.

============================================================
MISSION
============================================================

Resolve one scientific ambiguity first:

The recently reported load-constrained oracle appears to have evaluated
expert contributions reconstructed from the original dense weights and
partition plan rather than the ACTUAL TRAINED/FROZEN MoE BASIS.

That means the recent ~0.9468 oracle result cannot yet be used to decide
whether the trained p16 basis lacks capacity.

Fix this correctly, validate the oracle/load machinery, determine whether
each trained basis has sufficient capacity on clean fresh data, and then
immediately resume the appropriate refinement path for each of our two
targets.

Do not spend another cycle merely adding diagnostics.

The desired endpoint of this takeover is:

  p16/top4 actively refining again
  AND
  p32/top5 actively refining again

with the correct refinement objective selected from measured basis/oracle
evidence.

============================================================
PHASE 1 — MAKE CONTRIBUTION/ORACLE EVALUATION CHECKPOINT-AWARE
============================================================

Inspect the existing contribution-store and oracle code.

Currently, contribution generation can reconstruct expert outputs from:
- original dense Qwen FFN tensors
- partition definition

That is useful for raw-partition diagnostics, but it is NOT equivalent to
evaluating a trained/refined MoE checkpoint.

Implement explicit support for a trained basis/checkpoint.

The contribution/oracle path must be able to accept:

  --checkpoint <trained MoE checkpoint>

or an equivalent unambiguous basis artifact.

When checkpoint mode is selected, shared/expert outputs MUST come from the
actual frozen learned tensors in that checkpoint.

Do not silently fall back to raw dense partition reconstruction.

A trained-checkpoint contribution manifest must record at least:

- basis_source = trained_checkpoint
- checkpoint path
- checkpoint tensor SHA256/fingerprint
- partition path and SHA256/fingerprint
- topology:
    expert_count
    expert_width
    shared_width
    top_k
- source model revision
- dataset/capture identity
- split identity
- code commit
- dtype
- row count

Keep raw-partition mode if useful, but label it explicitly:

  basis_source = raw_dense_partition

There must never again be ambiguity between these two modes.

============================================================
PHASE 2 — NUMERICALLY PROVE CHECKPOINT CONTRIBUTIONS ARE CORRECT
============================================================

Before running a large oracle:

Take a small deterministic batch of states and compare:

A. direct forward execution through the frozen trained MoE basis
versus
B. reconstruction from the checkpoint-aware contribution store

Verify independently:

- shared contribution
- every routed expert contribution
- arbitrary selected top-k expert sums
- final reconstructed FFN output

Use dtype-appropriate numerical tolerances.

Record:
- max absolute error
- mean absolute error
- MSE
- cosine agreement

This equivalence test is a HARD prerequisite.

If checkpoint-store reconstruction is not numerically equivalent to direct
checkpoint inference, fix it before continuing.

Add regression tests so raw-partition and trained-checkpoint modes cannot
be confused later.

============================================================
PHASE 3 — VERIFY THE LOAD-CONSTRAINED ORACLE ACTUALLY MOVES LOAD
============================================================

The previous p16 load-constrained result had approximately:

  cosine = 0.9468
  NMSE = 0.0458
  load CV = 1.2523

and appeared largely unchanged across penalty settings.

Before trusting that algorithm as a feasibility test, prove that its
pricing mechanism works.

Add/report, per pricing iteration or penalty point:

- reconstruction objective
- cosine
- NMSE
- load CV
- dead experts
- expert loads
- assignment-change fraction from the zero-price assignment
- assignment-change fraction from previous iteration
- price min/mean/max
- convergence reason

Test it first on:
1. synthetic fixture where load prices MUST alter assignment;
2. small real p16 slice;
3. full p16 candidate-error table only after the first two work.

Check objective scaling carefully.

If reconstruction costs dwarf price terms numerically, normalize or
otherwise fix the pricing scale rather than merely increasing arbitrary
constants.

The search must be capable of expressing the actual gate question:

  cosine >= 0.98
  NMSE <= 0.05
  load CV <= 0.50
  dead experts == 0

Final candidate ranking must be:

1. require CV <= .50, NMSE <= .05, dead == 0 when such points exist;
2. among gate-feasible points maximize cosine;
3. tie-break with lower NMSE;
4. then lower CV.

Do not rank primarily by NMSE and accidentally discard the best cosine
solution.

For p16/top4:
- use exhaustive C(16,4) = 1820 candidate sets per token;
- retain the scalable/vectorized/memmapped implementation;
- do not recreate tens of millions of Python candidate objects.

For p32/top5:
- use a bounded candidate search suitable for 32 experts;
- start with a strong practical pool, e.g. 15 experts => C(15,5)=3003
  candidate sets per token when feasible;
- if a near-gate result appears candidate-bound, expand search before
  declaring topology failure;
- report candidate-pool construction and coverage explicitly.

============================================================
PHASE 4 — P16/TOP4: ANSWER THE TRAINED-BASIS CAPACITY QUESTION
============================================================

Use the ACTUAL frozen final refined p16/top4 checkpoint.

Known important provenance includes the final continuation tensor SHA:

6693d65b1cf2bc731c2ec3d84b7872bc6f79fe8c1363ba422848269e3a7a6acb

Verify the actual artifact/fingerprint rather than trusting this prompt.

The latest fresh selector result was roughly:

  validation A:
    cosine ~0.9372
    CV ~0.2330

  validation B:
    cosine ~0.9379
    CV ~0.2357

This means the frozen model itself generalizes poorly to the new corpus,
but we do NOT yet know whether the cause is:

A. selector failure,
B. basis reconstruction-capacity failure,
or
C. joint quality/load-capacity failure.

Run on fresh validation-A first:

1. frozen student evaluation
2. trained-basis unconstrained exhaustive reconstruction oracle
3. trained-basis load-constrained oracle

Record:
- cosine
- NMSE
- CV
- dead experts
- hard-quartile metrics
- expert usage
- student/oracle top-k recall where applicable

Do not use the official holdout.

Do not use validation-B for iterative hyperparameter tuning.

============================================================
P16 DECISION RULE
============================================================

CASE P16-A:

If trained-basis unconstrained oracle achieves:

  cosine >= .98
  NMSE <= .05

AND the load-constrained oracle can simultaneously achieve:

  cosine >= .98
  NMSE <= .05
  CV <= .50
  dead == 0

then the p16 BASIS IS GOOD.

Freeze it.

Return directly to SELECTOR REFINEMENT.

Use the expanded diverse layer-0 FIT corpus while excluding validation A/B
from optimizer updates.

Focus selector training on oracle regret / hard routing rather than generic
capacity increases.

Strong candidates include:
- hard-dispatch regret weighting
- reconstruction/cosine regret weighting
- load prices from the corrected oracle
- shared-output router feature [x, shared_output]

Do not change the frozen basis during this branch.

Optimize on FIT.
Select on A.
Use B only as confirmation of selected finalists.

Goal:
  A cosine >= .98
  A NMSE <= .05
  A CV <= .50
  dead == 0

Then confirm on B without further tuning.

------------------------------------------------------------

CASE P16-B:

If unconstrained trained-basis oracle is >= .98 but no assignment can
remain >= .98 while satisfying CV <= .50:

This is a JOINT BASIS/LOAD GEOMETRY problem.

Return to BASIS REFINEMENT, not selector-only work.

Refine the current p16 basis using FIT only with explicit pressure toward:

- multiple reconstructively competitive experts per token
- lower oracle load concentration
- hard-token reconstruction
- expert diversity
- preserving total quality

Use the existing basis as initialization.

Do not restart from scratch unless measurements show it is necessary.

After each meaningful basis refinement:
- freeze/fingerprint the candidate
- rerun trained-basis oracle on A
- require joint quality/load feasibility before selector refinement

------------------------------------------------------------

CASE P16-C:

If the trained-basis unconstrained oracle itself is below .98 on fresh A:

The basis does not generalize sufficiently.

Return immediately to BASIS REFINEMENT using the newly expanded/diverse
fresh FIT corpus.

Use:
- current refined basis as initialization
- hard-token emphasis
- contribution/reconstruction losses
- cosine-aware objective
- broader fresh token-state distribution

Keep A/B excluded from optimization.

The objective is NOT merely to improve the current router.

The first milestone is:

  trained-basis oracle on A >= .98 cosine
  NMSE <= .05

Then solve joint CV <= .50.

Only after basis capacity is proven should selector refinement resume.

============================================================
PHASE 5 — P32/TOP5: APPLY THE SAME CORRECT METHODOLOGY
============================================================

Once the checkpoint-aware path and corrected oracle have been validated on
p16, apply the same infrastructure to p32/top5.

Do not reuse a p32/top6 basis and call it a p32/top5 result.

p32/top5 requires its own topology-specific partition/refinement.

Geometry:

  routed experts = 32
  expert width = 512
  shared width = 1024
  top_k = 5
  active FFN width = 3584
  FFN reduction = 79.41%

Use the current clean FIT/fresh corpus.

Use the lessons learned from p16:

- hard-token basis refinement
- cosine-aware reconstruction
- independent positive routing/coefficients where applicable
- broad fresh token-state distribution
- explicit load-aware oracle diagnostics
- no holdout tuning

If a current trained p32/top5 basis exists:
- fingerprint it;
- evaluate it with the checkpoint-aware oracle first.

If there is no valid current trained p32/top5 basis:
- rerun/consume the current topology-specific p32/top5 partition search;
- select a small number of promising partition candidates using FIT/A only;
- initialize p32/top5 refinement from those candidates;
- do not perform a giant architecture sweep.

For p32/top5, answer the same questions:

1. Can the TRAINED BASIS achieve >= .98 cosine / <= .05 NMSE?
2. Can it do so while also satisfying CV <= .50 / dead == 0?
3. If yes, can a learned selector recover that assignment robustly?

Use exactly the same decision logic:

- oracle quality failure -> basis refinement
- oracle quality green but load feasibility failure -> load-friendly basis refinement
- oracle joint green -> freeze basis and refine selector

============================================================
PHASE 6 — ACTUALLY RETURN BOTH TARGETS TO REFINEMENT
============================================================

Do not stop after generating oracle reports.

For EACH target, once the diagnostic identifies the correct blocker,
launch a bounded refinement continuation on the appropriate component.

p16/top4:
- basis continuation if basis/oracle says basis is blocker
OR
- selector continuation if basis is jointly feasible

p32/top5:
- basis continuation if basis/oracle says basis is blocker
OR
- selector continuation if basis is jointly feasible

Use short/bounded scientific continuations first.

Require validation telemetry frequently enough to detect direction before
burning a large GPU budget.

Do not run large multi-hour sweeps until a bounded continuation shows an
actual improvement.

The refinement loop should retain the best checkpoint by the clean gate
criteria, not simply final epoch.

============================================================
DATA SPLIT DISCIPLINE
============================================================

Maintain strict roles:

FIT:
- optimizer updates
- basis refinement
- selector training

Validation A:
- candidate selection
- refinement decisions
- oracle diagnostics

Validation B:
- confirmation of selected finalists
- no optimizer updates
- do not repeatedly tune against B

Official holdout:
- CLOSED during this takeover

If existing A/B provenance has been compromised or ambiguous, create a
replacement clean split from the fresh corpus before spending serious GPU
budget and document it explicitly.

Every report must identify:
- FIT rows
- excluded indices
- A rows
- B rows
- hashes/fingerprints
- overlap checks

A∩FIT = 0
B∩FIT = 0
A∩B = 0

============================================================
REQUIRED REPORTS
============================================================

Produce one takeover report summarizing:

INFRASTRUCTURE
- checkpoint-aware contribution implementation
- numerical equivalence receipt
- load-price movement verification
- tests

P16/TOP4
- frozen checkpoint fingerprint
- frozen student A metrics
- unconstrained trained-basis oracle A metrics
- load-constrained trained-basis oracle A frontier
- identified blocker:
    BASIS_QUALITY
    BASIS_LOAD_GEOMETRY
    SELECTOR
- refinement continuation launched
- before/after refinement metrics

P32/TOP5
- topology-specific partition/basis fingerprint
- student metrics if available
- trained-basis oracle frontier
- identified blocker
- refinement continuation launched
- before/after refinement metrics

DECISION TABLE

Target      Basis cosine   Joint-gate oracle   Blocker        Next state
p16/top4    ...            ...                 ...            REFINING
p32/top5    ...            ...                 ...            REFINING

Also state explicitly:

  HOLDOUT_OPENED = false
  REPRESENTATIVE_REPLAY_STARTED = false
  FULL64_REPLAY_STARTED = false

============================================================
TESTING / CODE QUALITY
============================================================

Run:
- focused unit tests for changed oracle/contribution code
- regression tests distinguishing raw-partition vs trained-checkpoint basis
- synthetic load-price movement test
- compilation
- focused Ruff/static checks
- native-Windows smoke for the affected commands

Use guarded command execution for potentially long tasks.

Do not allow a silent hung command.

Preserve terminal receipts.

============================================================
COMMITS
============================================================

Make clean commits at useful boundaries:

1. checkpoint-aware contribution/oracle correctness
2. p16 trained-basis diagnosis + refinement state
3. p32/top5 trained-basis diagnosis + refinement state
4. final reports/bookkeeping

Push all commits to:
agent/windows-dense2moe-real-pipeline

Do not leave decisive science only in an uncommitted working tree.

============================================================
SUCCESS CRITERION FOR THIS TAKEOVER
============================================================

This takeover is successful when:

1. We can prove that oracle metrics correspond to the ACTUAL trained basis,
   not reconstructed raw partition weights.

2. Load-constrained assignment has been numerically validated as capable of
   trading reconstruction quality against expert balance.

3. p16/top4 has a measured blocker classification and is back in the
   correct refinement loop.

4. p32/top5 has a measured blocker classification and is back in the
   correct refinement loop.

5. Neither target has been rejected based on a bounded/raw/incorrect oracle.

6. Holdout and representative/full64 replay remain blocked.

7. The agent finishes with concrete before/after refinement numbers, not
   merely infrastructure changes.

Do not declare either topology solved until the full layer-0 green gate is:

  cosine >= .98
  NMSE <= .05
  load CV <= .50
  dead experts == 0

Proceed autonomously through these phases unless an unrecoverable
provenance/data error prevents scientifically valid continuation.
