# FlashMini sources (v4)

Candidate families and roles; verified revision/license/gating come from
`registry/source_snapshot.lock.json` after `source-lock`. Gated NVIDIA
sources stay recipe-only unless terms explicitly permit redistribution.

- General/educational web: `fineweb_edu`, `fineweb`, `nemotron_cc_v2` (gated)
- Books/reference/science: `common_pile_books` (per-component licensing)
- Code: `stackv2_edu`, `stack_edu`, `nemotron_code` (gated, recipe-only)
- Math/STEM: `finemath`, `open_web_math`, `nemotron_math` (gated, recipe-only)
- Synthetic educational: `cosmopedia`
- Reasoning/post-training: `openthoughts` (reasoning vs final kept separate)

Post-training pools (separate from base): code/engineering, reasoning,
instruction-following (verifiable constraints), tool use, operational safety
(read-only < reversible < recoverable < irreversible/external/credential),
concise neutral style (minimum words to fully satisfy; no canned
friendliness), general assistant, preference pairs.
