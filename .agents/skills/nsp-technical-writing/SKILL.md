---
name: nsp-technical-writing
description: Keeps human-facing technical prose clear and precise. Use when writing docs, RFCs, READMEs, plans, PR descriptions, or developer explanations. Do not use for bot-only records, machine handoffs, source code, product UI copy, schemas, or quoted material.
user-invocable: false
---

# Technical writing

## Human surfaces
Apply to human-facing docs, plans, RFCs, READMEs, PR descriptions, commit messages, and explanations. Bot-only records and coordination use the smallest supported structured form.

## Pass
Choose the mode the reader needs: tutorial, how-to, reference, or explanation. Separate modes when combining them would obscure the task. State the problem and resulting behavior first, then evidence, exact commands or paths, and material limits. Use real domain terms and consistent names. Explain prerequisites and consequences at the point of use. Label assumptions and inference; link canonical guidance instead of copying it. For changes, include relevant validation and residual risk. Apply plain speech before delivery.

## Boundaries
Preserve technical meaning, evidence, caveats, quotations, and requested style. Source code, schemas, product UI copy, and machine state are outside this pass unless assigned. Do not add comments that paraphrase code; document non-obvious intent or constraints. Wording alone needs no execution artifacts.
