# Human review output

Use when writing a full human PR review. Lead with concrete blockers and evidence.
Keep command traces and bot progress in run records, not report appendices.

1. Review Scope: verified base/head/range, parent-branch status, changed-file count,
   maintain state, available CI/tests, guidance, and material limits.
2. File Review Summary: every changed file accounted for as reviewed, not reviewed,
   or justified N/A. Group simple files for prose only; the manifest remains exhaustive.
3. Detailed Findings: severity, exact location, failure trigger/consequence, evidence,
   smallest useful fix, and confidence. Do not invent a finding to populate a category.
4. Compliance Checklist: PASS/FAIL/PARTIAL/N/A with evidence for applicable repository
   rules, public contracts, security, correctness, tests, context/CCB, and release needs.
5. Test and Validation Review: separate claimed tests from fresh visible evidence and
   remaining checks.
6. PR Hygiene Review: scope/motivation, behavior, compatibility, testing, and rollout
   detail proportional to the change.
7. Release Readiness Verdict: one of `Approved / approvable`, `Approvable after listed
   confirmations`, `Request changes`, or `Not enough information to approve`.

Approval requires a verified PR integration-base comparison, complete maintain and
review accounting, fresh required checks, and resolved critical/high blockers.
Prior-commit fallback supports an explicitly requested single-commit review only;
it cannot establish PR merge-target approval. Missing evidence means insufficient
information; a large queue means continue, not abandon coverage. Label partial
work INTERIM PROGRESS ONLY. Review permission is not merge permission.
