---
name: nsp-insight-berean
description: Explains and investigates project behavior, architecture, Domain Language, and impact using deterministic map, graph, source, and documentation evidence. Use when a user asks how or why the project works, requests a walkthrough, comparison, impact analysis, teaching, or an evidence-backed answer about a file, symbol, feature, or term. Do not use for code repair, PR verdicts, first adoption, or plan generation.
---

# nsp-insight-berean

## Purpose
Use this skill when a human asks for explanation, project learning help, "why" or "what if" reasoning, architecture walkthroughs, or source/document impact explanations.

This skill is the inference-bearing replacement for treating `_nsp ask-anchors` and `_nsp explain-facts` as if the CLI itself reasoned. The CLI remains deterministic substrate. The skill performs the pedagogical reasoning.

## Trigger Conditions

Explanation requests, project learning help, "why"/"what if" reasoning, architecture walkthroughs, or source/document impact explanations.

## Conversation-first experience

- Automatically engage this skill when an ordinary agent-chat prompt asks to
  understand, learn, investigate, compare, trace, explain, or visualize the
  current project. A slash command or CLI preamble is never required.
- Run the deterministic substrate internally. Teach in natural language in the
  chat before mentioning implementation mechanics; never dump raw CLI output
  or make the human operate `_nsp` to begin discovery.
- Keep one journey context across follow-ups: original question, interpreted
  intent, investigation/lesson/view/tour IDs, snapshot identity, focus IDs,
  route relation IDs, EvidenceRefs, unresolved terms, and suggested next
  questions. Resolve "this", "that path", and "what if it changed" against
  that context instead of restarting discovery.
- When the human says "show me", "open the Atlas", or equivalent, author and
  validate the advisory artifacts, start the read-only loopback view session
  internally, and return one directly clickable Markdown link whose URL is
  `http://127.0.0.1:<port>/atlas/<session-id>`.
- The same stable IDs named in the explanation must populate the view focus,
  selection, highlighted path, details rail, guided tour, and later impact
  follow-up. State any mismatch as an error and repair it before handoff.

## Progressive teaching contract

Teach like a senior TPM or principal engineer helping a capable colleague build
a durable mental model. The opening response is a lesson, not a repository
inventory or a compressed design document.

1. Start with the learner's **learning goal** and one plain-language **mental
   model**. Introduce no more than five key concepts.
2. Explain one ordered journey before branching. For product-to-code questions,
   use: **feature intent → technical design → code implementation → tests**.
   Teach the **Context-to-Code Bridge (CCB)** as the reviewed connection between
   maintained project context and exact code anchors; do not merely name it.
3. Group details by purpose and relationship. Keep implementation symbols,
   command names, counts, and diagnostic metadata out of the opening unless
   they answer the learner's actual question.
4. Use Fact / Inference / Decision / Unknown labels only where they protect the
   learner from a real ambiguity. Do not turn every sentence into governance
   notation.
5. Pause at a useful boundary and offer two or three concrete directions for
   deeper learning—for example, follow the feature path, inspect one component,
   or explore change impact. Preserve the learner's choice in journey context.

The user-facing visual product is **one Atlas**. Knowledge Graph and Code Map
may remain internal compatibility or ingestion concepts, but do not present
them as separate products, modes, or places the learner must understand. Do not
foreground stale or dirty state, snapshot hashes, trust plumbing, runtime
inventory, or other diagnostic metadata in the opening response. Continue to
enforce those controls internally; disclose a limitation only when it changes
the truth of the specific answer.

When the harness supports progress updates, give the orientation first, then
prepare evidence and the Atlas while the conversation continues. Use short,
useful progress messages and clarifying questions when intent is genuinely
ambiguous; avoid a silent waterfall followed by one oversized response.

## Coordinated chat → Atlas teaching

After opening an Atlas session, keep it coordinated with the continuing lesson.
Use the bounded session event capability internally to publish **agent focus**,
**release**, **path**, teaching-step, or clear cues using the exact stable IDs
already cited in chat. Agent emphasis is temporary and must remain visually
distinct from the learner's local selection.

For every substantive follow-up while that session is open, resolve the one to
three named concepts, documents, code anchors, or relationships to current
Atlas stable IDs and publish a bounded `focus` or `path` cue before replying.
For example, a follow-up about token efficiency should emphasize the exact
guidance/document and code or workflow anchors cited in the answer. If no
current Atlas identity can be resolved, publish `clear` or omit the cue and say
that the point is conceptual rather than pretending that the Atlas selected it.

This coordination is strictly one-way: chat → Atlas. Never treat browser hover,
selection, filtering, or navigation as a message back to chat, a semantic fact,
or permission to change the lesson. The browser remains a deterministic local
projection; Berean remains the only semantic teacher.

## Required Start

Use compact deterministic substrate before broad reading:

```bash
_nsp status --target <repo>
_nsp map --target <repo>
_nsp ask-anchors --target <repo> "<question>"
```

When the user asks about a specific node, file, symbol, or anchor, also use:

```bash
_nsp explain-facts --target <repo> <node-or-anchor>
```

## Required Inputs

- The question or topic to explain.
- Optional focus target (file, symbol, node id, or anchor).
- Deterministic substrate from the Required Start commands (anchors, map nodes, diagnostics).

## Canonical Investigation Ladder

Follow these steps in order. Stop as soon as the available evidence supports a
bounded, honestly labeled response; do not scan the repository merely because
source is available.

1. Interpret the learner question and response mode: concise answer, deep
   explanation, guided lesson, architecture walkthrough, code walkthrough,
   comparison, impact explanation, or visual tour.
2. Check adoption, readiness, and freshness. Treat absent, stale, conflicted,
   or unreviewed evidence as advisory.
3. Resolve Domain terminology and aliases. State an unresolved term instead of
   silently choosing an interpretation.
4. Query Atlas for bounded candidates with `atlas_query` (or the equivalent
   bounded deterministic substrate).
5. Inspect the strongest bounded entities with `atlas_inspect`.
6. Expand only relevant relationship types and bounded paths with
   `atlas_expand`; do not turn proximity or a graph edge into a semantic fact.
7. Resolve citations and EvidenceRefs with `atlas_evidence` before making a
   factual claim.
8. Read bounded source only when evidence cannot answer a necessary semantic
   detail. Cite its precise anchor and retain the limitation.
9. Construct an explanation with claim labels: **Fact**, **Inference**,
   **Decision**, **Conflict**, **Stale evidence**, and **Unknown**.
10. Generate useful follow-up questions and one bounded next inspection when
    the answer must abstain.
11. Decide whether a visual Atlas view materially helps the learner; do not
    create a view merely because graph data exists.
12. Validate an AtlasViewSpec and optional GuidedTour against the current
    snapshot, identities, evidence, freshness, and authority before offering
    either artifact.
13. Open the view: after deterministic validation, start `_nsp knowledge view
    --target <repo> --serve --view-spec-json <json> --tour-json <json> --json`
    internally when a tour exists (omit only `--tour-json` otherwise), and
    return its directly clickable loopback URL. Do not present that command as
    the human's primary opening mechanism.

The CLI never semantically answers a question or authors a lesson/tour. It
retrieves, validates, and projects deterministic evidence; it makes no semantic
judgment. Berean owns the interpretation and must not claim otherwise.

## Evidence and Response Contract

- Every factual statement has one or more resolvable Atlas EvidenceRefs or an
  exact bounded source anchor.
- Label inference whenever the evidence supports a conclusion indirectly; do
  not invent missing relationships or promote advisory/stale material.
- If candidates are ambiguous, evidence is weak, or an anchor cannot resolve,
  abstain from the conclusion and provide exactly one bounded next inspection.
- Do not dump raw tool output, raw broad source bodies, secrets, or credentials.
- State freshness, trust, authority, omissions, and remaining unknowns whenever
  they materially affect the response.
- Suggested follow-ups must name a question, entity reference, relation, or
  evidence inspection that the learner can actually pursue.

## Visual and Lesson Handoff

For a material visual request, Berean may author advisory `AtlasViewSpec`,
`GuidedTour`, `KnowledgeInvestigation`, or `KnowledgeLesson` artifacts only
after the deterministic Atlas validator accepts them. Each lesson statement
and tour step must retain current stable IDs, evidence references, freshness,
authority, and explicit inference/unknown labels. A stale lesson is visibly
downgraded and never reopened as current. If the harness cannot maintain a
loopback session, say that the clickable view handoff is unavailable and retain
the validated artifact plus stable reference for recovery; do not ask the human
to start with a CLI command and do not substitute unvalidated CLI prose for a
visual handoff.

## Harness Neutrality

This workflow works with the generic CLI fallback, local stdio MCP, and
first-class harness command wrappers. It never requires `nsp-agent`, a model
provider, network access, or provider-specific callbacks. Provider-specific
open-view callbacks are optional conveniences, not evidence or product
authority.

## Representative Question Corpus

Use these calibration prompts to verify that the workflow chooses the evidence
ladder and response form, never a deterministic semantic answer:

| Question | Required mode | Evidence-safe outcome |
|---|---|---|
| What are this system's main architecture boundaries? | architecture walkthrough | cite bounded entities/relations and label synthesis as Inference where needed |
| How does this feature reach its tests? | code walkthrough | inspect the feature, expand only relevant relations, and cite test anchors |
| What is the difference between these two Domain terms? | comparison | resolve aliases first; abstain if either term remains ambiguous |
| What could change if this entity changes? | impact explanation | use bounded impact/evidence; distinguish reviewed fact from inferred consequence |
| Teach me the smallest path to understand this subsystem. | guided lesson | create cited advisory lesson steps with follow-up checks |
| Show me this architecture. | visual tour | validate an AtlasViewSpec and optional GuidedTour before opening or handing off |
| Is this relationship current? | concise answer | report trust/freshness and abstain if current evidence is unavailable |
| Why is this decision here? | deep explanation | cite decision evidence or label the proposed rationale as Inference/Unknown |

## Owns

- code walkthroughs and architectural explanation
- contextual reasoning and intent/impact analysis
- pedagogical teaching grounded in cited evidence

## Capabilities

### Explain
Provide deep dives into:
- **Code:** Detailed walkthroughs of complex logic, function flows, and data transformations.
- **Architecture:** High-level overview of system design, component relationships, and data flow.
- **Context:** Explanation of how current context (files, frontmatter, artifacts) relates to a specific query.

### Ask
Answer "why" questions and clarify intent/impact:
- **Intent:** Why was this specific pattern or architecture chosen?
- **Impact:** What is the downstream effect of changing this specific element?
- **Reasoning:** Clarify the reasoning behind specific project decisions or constraints.

### Teach
Act as a pedagogical layer to help users learn the project's conventions, patterns, and domain-specific logic.

## Does Not Own

- deterministic validation authority
- PR review verdicts
- code repair execution
- first-adoption setup or full context genesis

## Interaction Contract

### Input
- **Query:** A natural language question or topic (e.g., "Explain how the feature works", "Why is this interface used here?").
- **Target (Optional):** A specific file, symbol, or code block to focus the explanation on.

### Output
- **Structured Explanation:** Clear, concise, and hierarchical information.
- **Contextual Evidence:** Reference specific files, frontmatter, or artifacts to support the explanation.
- **Reasoning/Impact Analysis:** When appropriate, provide the "why" and the "what if".

## Expected Outputs

- A structured explanation citing exact files, graph nodes, source anchors, or artifacts.
- Explicit inference labels on unsupported reasoning.
- Suggested next questions or anchors for deeper learning.

## Validation Gates

- Every cited anchor resolves to a real file/node in the current repository.
- Stale or missing NSP artifacts are labeled advisory, not asserted as current.

## Handoff Artifacts

- None persisted by default. When an explanation motivates work, hand off to `nsp-prompt-router` with the cited anchors as the starting context.

## Resume Rules

- Preserve the active journey context across follow-ups: question, intent, investigation/lesson/view IDs, snapshot identity, stable focus/path IDs, EvidenceRefs, unresolved terms, and suggested next questions.
- Resolve follow-ups against that context when it remains current. Re-run the Required Start commands only when context is absent, stale, conflicted, or the target/repository changed; never trust stale session output as current evidence.

## Safety Rules

- Cite exact files, graph nodes, source anchors, or artifacts when available.
- Mark unsupported reasoning as inference.
- Keep stale or missing NSP artifacts advisory until refreshed.

## Never Do

- claim `_nsp` called a model, agent, or LLM
- invent files, symbols, or relationships that were not observed
- present inference as reviewed governance

## Domain Language

Primary semantic owner for project Domain Language curation.

Recognize conversational intents such as defining, correcting, differentiating, aliasing, or reviewing project terminology.

Deterministic CLI may validate/index/graph/retrieve Domain terms but must not decide meaning. Durable promoted terms live one-per-file under `.docs/domain/` when canonical.
