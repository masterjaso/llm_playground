<!-- nsp:meta
id: docs.evaluation.v2.test.matrix
kind: document
scope: technical
persona: scientific-evaluation
status: active
source: human
confidence: high
reviewStatus: reviewed
graphNode: document:docs/EVALUATION_V2_TEST_MATRIX.md
graphTags: evaluation,tests,contracts
validation: manifest-check,secret-scan
owner: scientific-evaluation
lastReviewed: 2026-08-19
replaces:
replacedBy:
-->

# Evaluation V2 test-alignment contract

The authoritative metric IDs and policy semantics live in
`src/dense2moe/evaluation/registry.py` (`METRIC_REGISTRY`, `TEST_MATRIX`). The
matrix is validated by `validate_metric_registry()` and the deterministic
contract meta-test. Every gate-bearing entry requires these coverage classes:

| Coverage class | Required proof |
|---|---|
| `reference_formula` | independent deterministic teacher/candidate fixture |
| `streaming_reference` | streaming reduction equals the independent reference |
| `masking` | padding, invalid, and non-finite positions are excluded |
| `slice_aggregation` | source/domain/target-norm/hard-token reconciliation |
| `serialization_schema` | canonical ID and versioned receipt field |
| `decision_boundary` | immediately below/at/above GREEN/YELLOW thresholds |
| `runner_integration` | producer/consumer path emits registered IDs |
| `missing_insufficient_evidence` | explicit status, never zero or PASS |
| `non_finite_input` | fail closed on NaN/Inf |
| `receipt_round_trip` | immutable write/read/hash validation |
| `policy_hash_sensitivity` | gate-bearing policy change changes the hash |

The focused fixture suite is `tests/test_evaluation_v2.py`; future metrics are
added by registering the ID and semantics, adding an independent fixture and
boundary/failure cases, adding receipt/schema and slice reconciliation fields,
integrating the runner, incrementing the appropriate policy/schema version,
recomputing the policy hash, and rerunning or explicitly classifying affected
historical candidates.

Required decision invariants include independent FIT/DEV metrics, no averaged
promotion score, GREEN FIT + RED DEV ⇒ `GENERALIZATION_REJECT`, no oracle or LM
rescue of structural fresh RED, non-overridable learned loadCV/dead experts/
source-slice collapse, exact forward KL direction, and protected confirmation
for any actual override.
