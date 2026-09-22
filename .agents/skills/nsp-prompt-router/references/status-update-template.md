# Work status update template

## Presentation
Open with a brief current-status paragraph. State the supported number of unique formally closed gates over the full accepted scope, including deferred gates. If scope or acceptance authority is missing or contradictory, state that the count is unknown. Name active work only when current records establish activity. Do not invent percentages or approval.

Use this legend: ✅ Completed · 🔄 In progress/partial · ⬜ Todo/deferred.

Then show one complete nested bullet hierarchy, never a table or a flat replacement. Preserve exact original IDs and names. Use milestone → original phase → gate → substantive sub-work for epics that have those levels. For workflows use the actual workflow → phase/step → gate structure, omitting absent levels. Repeat split phases/shared gates where the accepted map places them, but count each gate ID once. Do not add artificial levels or filler subtasks.

Every status bullet at every depth begins with its icon. The following is a shape, not data to copy:

- 🔄 **[existing milestone or workflow ID — exact name]**
  - 🔄 **[original phase/step ID — exact name]**
    - 🔄 **[original gate ID — exact name]** — [formal status and uncertainty]
      - ✅ [supported completed sub-work]
      - 🔄 [partial sub-work and remaining uncertainty]
      - ⬜ [remaining requirement or exact prerequisite]

Replace all placeholders in a finished report. Include all accepted gates and deferred work, not only the current milestone. Preserve material contractual bounds such as numeric limits and required durations.

## Status meaning
- ✅ Completed: the stated scope is formally accepted. For accepted gates whose evidence is now stale or invalidated, explicitly qualify historical acceptance and current applicability separately; never silently reset them. Completed sub-work does not close a parent gate. Milestones require all required gates and milestone conditions.
- 🔄 In progress/partial: evidence of partial work exists, not necessarily a running executor. Formal NOT_RUN with partial evidence uses 🔄 and explicitly says not formally passed. Failed gates use 🔄 and name the unresolved failure. Unavailable evidence is not a known failure.
- ⬜ Todo/deferred: pending or deferred work. Blocked items use ⬜ and the exact dependency, without calling independent work blocked. Deferred parents can contain partial gates and completed sub-work.

Subjective owner acceptance cannot be replaced by tests or inferred approval. Distinguish historical acceptance, current evidence applicability, and continuation readiness even for closed capsules with blocked continuation. Where records conflict, preserve the conflicting claims and mark the unresolved authority; do not fabricate a resolution or a count.

## Closure and sources
After the hierarchy, give the concise next authorized closure sequence using exact gate IDs. Use a short icon-prefixed priority list describing remaining outcomes, not a command to start work. Respect accepted amendments, deferrals, owner-review stops, and current work boundaries. Do not imply a handoff or a green suite closes the whole epic. If sequencing is unknown, say so rather than invent priorities.

End with verified clickable absolute paths or source URLs for the plan, acceptance authority, and checkpoint used. Verify existence and the cited content; do not output placeholder or guessed links. Mention snapshot limits when the report is not live verification. Do not announce old achievements as new changes without a comparison record. Keep routine test evidence concise and the report self-contained.
