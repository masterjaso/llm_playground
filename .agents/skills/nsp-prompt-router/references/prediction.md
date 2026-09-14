# Prediction inside PIV

Acceptance defines the destination; prediction describes the mechanism and its
falsifiers. The agent selects `none | compact | expanded` during Plan.

- DIRECT and clear low-risk mechanical BOUNDED work use `none`, with no placeholder.
- Uncertain, risky, high-impact, or costly-to-reverse BOUNDED work uses `compact`.
- Use `expanded` only when alternatives or diagnostic probes improve the decision.
- WORKSTREAM records package invariants plus selected child/integration predictions.
  EPIC records the active phase prediction; do not require one for every PIV.

Reassess at phase boundaries and after a counterexample. A prior counterexample
requires reassessment before selecting `none` again.

An activated contract records mechanism, expected observations, blast radius,
material invariants, falsifiers, and mismatch response. Optional experiments must
be bounded and decision-relevant; stop when the decision is resolved.

During Validate compare actual observations with the activated contract and
record exactly one agent-owned Prediction Result classifier:

| Classifier | Required action |
|---|---|
| `expected-match` | Record supporting observations. |
| `benign-deviation` | Explain the difference and prove unchanged scope, material invariants, and a still-usable hypothesis. |
| `scope-discovery` | Update affected plan/context/claims/validation before continuing. |
| `counterexample` | Stop the affected slice, revise or abandon the hypothesis, and replan. |
| `invalid-validation` | Repair the validation method and rerun it. |

The last three reopen affected discovery evidence and decisions. Do not restart
unrelated phases. Missing/unclassified results, material contradictions, or
pending mismatch actions block PIV, phase, maintain, and PR readiness. Relabeling
a mismatch without evidence cannot satisfy a gate.

Resume from accepted hypothesis, mode, mismatch status, unresolved contradictions,
next action, and evidence references. Do not copy transcripts or create prediction
records for `none`. The CLI checks form, budgets, provenance, and lifecycle only.
